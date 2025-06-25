# FILE: src/ray_scripts/modes/refinement_tuner.py (FINAL, CORRECTED)

import ray
import optuna
import torch
import logging
import pickle
import copy
from pathlib import Path
import time
import warnings

from ...config_loader import MDetectorConfigAccessor
from ...core.m_detector.base import MDetector
from ..shared_utils import deep_update_dict, calculate_iou, setup_trial_file_logging
from ..tuning_manager import define_search_space
from ...data_utils.validation_utils_torch import calculate_metrics
from ...core.constants import DYNAMIC_LABEL_VALUE

class MockDI:
    """A lightweight mock of a DepthImage for the refinement function."""
    def __init__(self, points_global: torch.Tensor):
        self.original_points_global_coords = points_global

def _calculate_metrics_for_baked_frame(pred_labels: torch.Tensor, frame_data: dict, mdet_label_val: int) -> dict:
    """Calculates metrics for a single frame from baked data."""
    device = pred_labels.device
    pred_is_dyn = (pred_labels == mdet_label_val)
    num_filtered_points = len(pred_labels)
    gt_is_dyn = torch.zeros(num_filtered_points, dtype=torch.bool, device=device)
    map_tensor = torch.from_numpy(frame_data['original_indices_map'].copy()).long().to(device) 
    gt_sparse_tensor = torch.from_numpy(frame_data['gt_sparse_indices'].copy()).long().to(device)
    is_in_gt_mask = torch.isin(map_tensor, gt_sparse_tensor)
    gt_is_dyn[is_in_gt_mask] = True
    return calculate_metrics(pred_is_dyn, gt_is_dyn)

@ray.remote
def run_trial_on_ray(trial_params: dict, trial_number: int, base_config: dict, run_timestamp: str, cached_data: list):
    """
    A lightweight Ray worker that tunes ONLY the refinement stage using pre-baked data.
    This function's logs are configured to go ONLY to files, not the console.
    """
    # --- Filter Warnings within the remote worker ---
    warnings.filterwarnings("ignore", message="The cuda.cuda module is deprecated", category=FutureWarning)
    warnings.filterwarnings("ignore", message="The cuda.cudart module is deprecated", category=FutureWarning)
    warnings.filterwarnings(
        "ignore",
        message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.",
        category=FutureWarning
    )

    # 1. Setup Worker-Specific File Logging
    logger_name = f"worker_trial_{trial_number}"
    logger = logging.getLogger(logger_name)
    accessor = MDetectorConfigAccessor(config_dict=base_config)
    setup_trial_file_logging(
        logger_name,
        accessor.get_study_name(), 
        run_timestamp, 
        trial_number, 
        accessor.get_logging_settings()
    )
    logger.info(f"Worker starting refinement trial {trial_number} with params: {trial_params.get('m_detector', {}).get('frame_refinement', {})}")
    
    # 2. Configuration and Setup
    trial_config = deep_update_dict(copy.deepcopy(base_config), trial_params)
    trial_accessor = MDetectorConfigAccessor(config_dict=trial_config)
    device = torch.device("cuda:0" if trial_accessor.get_processing_settings().get('device') == 'cuda' and torch.cuda.is_available() else "cpu")
    detector = MDetector(config_accessor=trial_accessor, device=device, logger_name=logger_name)
    mdet_label_val = DYNAMIC_LABEL_VALUE

    # 3. Processing and Metrics Calculation
    total_tp, total_fp, total_fn = 0, 0, 0
    for frame_data in cached_data:
        labels_before = frame_data['labels_before_refinement'].to(device)
        points_global = frame_data['points_global'].to(device)
        mock_di = MockDI(points_global)
        
        refined_labels = detector._apply_frame_refinement(labels_before, mock_di)
        
        metrics = _calculate_metrics_for_baked_frame(refined_labels, frame_data, mdet_label_val)
        total_tp += metrics['tp']
        total_fp += metrics['fp']
        total_fn += metrics['fn']

    # 4. Final Result
    final_iou = calculate_iou(total_tp, total_fp, total_fn)
    logger.info(f"Worker finished trial {trial_number}. Final IoU: {final_iou:.4f}")
    return trial_number, final_iou, {}

def run(config: dict, study_name: str, n_trials: int, bake_id: str, db_dir: Path, run_timestamp: str):
    """
    Manages an Optuna study for the refinement stage.
    This function acts as the "driver" and logs formatted progress to the console.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting Tune-Refinement Mode for study '[bold green]{study_name}[/bold green]' ---")
    logger.info(f"Using baked results from: '[bold cyan]{bake_id}[/bold cyan]'")

    cache_path = Path("cache") / f"{bake_id}.pkl"
    if not cache_path.exists():
        logger.error(f"[bold red]Bake file not found: {cache_path}. Please run bake mode first.[/bold red]")
        return
        
    logger.info(f"Deserializing baked data from [yellow]{cache_path}[/yellow]...")
    with open(cache_path, 'rb') as f:
        cached_data = pickle.load(f)
    
    logger.info("Data loaded. Placing into Ray Object Store for worker access...")
    cached_data_ref = ray.put(cached_data)
    logger.info(f"Successfully loaded and broadcasted [bold cyan]{len(cached_data)}[/bold cyan] baked data entries.")
    
    storage_path = f"sqlite:///{db_dir / study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_path, direction="maximize", load_if_exists=True)
    
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus == 0:
        logger.error("[bold red]No GPUs available in the Ray cluster. Refinement tuning requires GPUs.[/bold red]")
        return

    workers_per_gpu = config['processing_settings']['workers_per_gpu']
    gpus_per_trial = 1.0 / workers_per_gpu
    max_concurrent_trials = int(available_gpus / gpus_per_trial)
    logger.info(f"Running with a maximum of [bold cyan]{max_concurrent_trials}[/bold cyan] parallel trials ({gpus_per_trial} GPUs per trial).")

    running_tasks = {}
    existing_trials = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED)))
    completed_trials = existing_trials
    logger.info(f"Found {existing_trials} existing trials in the study.")

    while completed_trials < n_trials:
        if (completed_trials + len(running_tasks)) < n_trials and len(running_tasks) < max_concurrent_trials:
            trial = study.ask()
            params = define_search_space(trial, mode='tune-refinement')
            
            future = run_trial_on_ray.options(num_gpus=gpus_per_trial).remote(params, trial.number, config, run_timestamp, cached_data_ref)
            running_tasks[future] = trial
            logger.info(f"Dispatched Trial #[bold cyan]{trial.number}[/bold cyan]...")
        else:
            if not running_tasks:
                time.sleep(1) # Avoid busy-waiting
                continue
                
            ready_futures, _ = ray.wait(list(running_tasks.keys()), num_returns=1)
            for ready_future in ready_futures:
                trial_obj = running_tasks.pop(ready_future)
                try:
                    trial_num, iou_score, metadata = ray.get(ready_future)
                    if "error" in metadata:
                        study.tell(trial_obj, state=optuna.trial.TrialState.FAIL)
                        logger.warning(f"[yellow]Trial {trial_obj.number} FAILED. Reason: {metadata['error']}[/yellow]")
                    else:
                        study.tell(trial_obj, iou_score)
                        logger.info(f"Trial {trial_obj.number} finished. Score (IoU): [bold yellow]{iou_score:.4f}[/bold yellow]")
                except Exception as e:
                    study.tell(trial_obj, state=optuna.trial.TrialState.FAIL)
                    logger.error(f"[bold red]Trial {trial_obj.number} crashed with exception: {e}[/bold red]", exc_info=True)

                completed_trials += 1
                best_score_str = f"{study.best_value:.4f}" if study.best_value is not None else "N/A"
                logger.info(f"Progress: {completed_trials}/{n_trials}. Best score: [bold green]{best_score_str}[/bold green]")
                
    logger.info(f"\n[bold green]--- Optuna Study '{study_name}' Complete ---[/bold green]")