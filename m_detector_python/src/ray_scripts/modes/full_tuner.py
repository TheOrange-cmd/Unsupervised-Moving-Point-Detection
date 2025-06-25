# FILE: src/ray_scripts/modes/full_tuner.py (FINAL, CORRECTED)

import ray
import optuna
import torch
import logging
import os
import copy
from pathlib import Path
import time
from typing import Union
import warnings

from ...config_loader import MDetectorConfigAccessor
from ...core.m_detector.base import MDetector
from ...data_utils.nuscenes_helper import NuScenesProcessor
from ...data_utils.validation_utils_torch import calculate_metrics_for_optuna_trial_in_memory
from ..shared_utils import deep_update_dict, calculate_iou, setup_trial_file_logging
from ..tuning_manager import define_search_space

@ray.remote
class PruningManagerActor:
    """A central actor to manage all interactions with the Optuna study."""
    def __init__(self, study_name: str, storage_path: str):
        self.study_name = study_name
        self.storage_path = storage_path

    def report(self, trial_number: int, value: float, step: int) -> Union[bool, None]:
        """
        Reports an intermediate value for a trial.
        """
        try:
            # Load a fresh instance of the study to get the most up-to-date trial list.
            study = optuna.load_study(study_name=self.study_name, storage=self.storage_path)
            trial = optuna.trial.Trial(study, trial_number)
            
            trial.report(value, step)
            if trial.should_prune():
                return True # Prune the trial
                
        except KeyError:
            return None
        except optuna.exceptions.UpdateFinishedTrialError:
            # The trial has already been completed or pruned by another mechanism.
            return True
            
        return False # Do not prune

@ray.remote
def run_trial_on_ray(trial_params: dict, trial_number: int, base_config: dict, run_timestamp: str, pruning_manager: PruningManagerActor):
    """A Ray worker task that reports progress to the central PruningManagerActor."""
    warnings.filterwarnings("ignore", message="The cuda.cuda module is deprecated", category=FutureWarning)

    logger_name = f"worker_trial_{trial_number}"
    logger = logging.getLogger(logger_name)
    setup_trial_file_logging(logger_name, base_config['study_name'], run_timestamp, trial_number, base_config['logging_settings'])
    logger.info(f"Worker starting trial {trial_number}.")

    trial_config = deep_update_dict(copy.deepcopy(base_config), trial_params)
    accessor = MDetectorConfigAccessor(config_dict=trial_config)
    data_actor_handle = ray.get_actor("nuscenes_data_service")

    try:
        device = torch.device("cuda:0" if accessor.get_processing_settings().get('device') == 'cuda' and torch.cuda.is_available() else "cpu")
        nusc_handle = ray.get(data_actor_handle.get_nusc_handle.remote())
        detector = MDetector(config_accessor=accessor, device=device, logger_name=logger_name)
        processor = NuScenesProcessor(
            data_actor=data_actor_handle, config_accessor=accessor,
            progress_actor=None, worker_id=trial_number, logger_name=logger_name
        )
    except Exception as e:
        logger.error(f"Trial {trial_number} failed during setup: {e}", exc_info=True)
        raise

    scene_indices = accessor.get_processing_settings()['scene_indices_to_run']

    if isinstance(scene_indices, str) and scene_indices.lower() == 'all':
        total_scenes = ray.get(data_actor_handle.get_scene_count.remote())
        scene_indices = set(range(total_scenes))
    else:
        scene_indices = set(scene_indices)
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for i, scene_idx in enumerate(scene_indices):
        result_dict = processor.process_scene(scene_idx, detector, f"T{trial_number}_S{scene_idx}")
        logger.info(f"Trial {trial_number}, done processing Scene {scene_idx}")
        if not result_dict:
            logger.warning(f"Trial {trial_number}, Scene {scene_idx} produced no result dictionary. Skipping.")
            continue

        metrics = calculate_metrics_for_optuna_trial_in_memory(
            mdet_results_dict=result_dict, eval_params={"mdet_dynamic_label_value": 0},
            nusc=nusc_handle, scene_idx=scene_idx
        ) 
        total_tp += metrics['tp']; total_fp += metrics['fp']; total_fn += metrics['fn']


        step = i + 1
        intermediate_iou = calculate_iou(total_tp, total_fp, total_fn)

        logger.info(f"Trial {trial_number}, Intermediate IoU for Scene {scene_idx}: {intermediate_iou}")
        
        pruning_decision = ray.get(pruning_manager.report.remote(trial_number, intermediate_iou, step))

        if pruning_decision is None and trial_number != 0:
            raise RuntimeError(f"PruningManagerActor could not find Trial #{trial_number} in the database.")
        
        if pruning_decision is True:
            logger.warning(f"Worker for trial {trial_number} is being pruned at step {step} with IoU {intermediate_iou:.4f}. Another trial was better.")
            raise optuna.TrialPruned()

    final_iou = calculate_iou(total_tp, total_fp, total_fn)
    logger.info(f"Worker finished trial {trial_number}. Final IoU: {final_iou:.4f}")
    return final_iou


def run(config: dict, study_name: str, n_trials: int, run_timestamp: str, db_dir: Path):
    """Manages an Optuna study for end-to-end tuning."""
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting Tune-Full Mode for study '[bold green]{study_name}[/bold green]' ---")

    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3)

    storage_path = f"sqlite:///{db_dir / study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name, storage=storage_path, 
        direction="maximize", load_if_exists=True, pruner=pruner
    )
    
    pruning_manager = PruningManagerActor.remote(study_name, storage_path)
    logger.info(f"Using Storage: [bold cyan]SQLite via PruningManagerActor[/bold cyan]")
    logger.info(f"Using Pruner: [bold cyan]HyperbandPruner[/bold cyan]")

    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus == 0:
        logger.error("[bold red]No GPUs available...[/bold red]"); return

    workers_per_gpu = config['processing_settings']['workers_per_gpu']
    gpus_per_trial = 1.0 / workers_per_gpu
    max_concurrent_trials = int(available_gpus / gpus_per_trial)
    logger.info(f"Running with a maximum of [bold cyan]{max_concurrent_trials}[/bold cyan] parallel trials...")

    running_tasks = {}
    completed_trials = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED)))
    logger.info(f"Found {completed_trials} existing trials in the study.")

    while completed_trials < n_trials:
        if (completed_trials + len(running_tasks)) < n_trials and len(running_tasks) < max_concurrent_trials:
            trial_obj = study.ask()
            params = define_search_space(trial_obj, mode='tune-full')
            
            future = run_trial_on_ray.options(num_gpus=gpus_per_trial).remote(
                params, trial_obj.number, config, run_timestamp, pruning_manager
            )
            running_tasks[future] = trial_obj
            logger.info(f"Dispatched Trial #[bold cyan]{trial_obj.number}[/bold cyan]...")
        else:
            if not running_tasks: time.sleep(1); continue
            ready_futures, _ = ray.wait(list(running_tasks.keys()), num_returns=1)
            for ready_future in ready_futures:
                trial_obj = running_tasks.pop(ready_future)
                try:
                    score = ray.get(ready_future)
                    study.tell(trial_obj, score)
                    logger.info(f"Trial {trial_obj.number} finished. Score (IoU): [bold yellow]{score:.4f}[/bold yellow]")
                except optuna.TrialPruned:
                    study.tell(trial_obj, state=optuna.trial.TrialState.PRUNED)
                    logger.info(f"[yellow]Trial {trial_obj.number} was PRUNED.[/yellow]")
                except Exception as e:
                    study.tell(trial_obj, state=optuna.trial.TrialState.FAIL)
                    logger.error(f"[bold red]Trial {trial_obj.number} crashed: {e}[/bold red]", exc_info=False)
                
                completed_trials += 1
                
                try:
                    best_score_str = f"{study.best_value:.4f} (T#{study.best_trial.number})"
                except ValueError:
                    best_score_str = "N/A"
                
                logger.info(f"Progress: {completed_trials}/{n_trials}. Best score: [bold green]{best_score_str}[/bold green]")
    
    logger.info(f"\n[bold green]--- Optuna Study '{study_name}' Complete ---[/bold green]")