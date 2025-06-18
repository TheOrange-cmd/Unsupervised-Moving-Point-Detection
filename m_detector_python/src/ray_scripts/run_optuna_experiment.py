# src/ray_scripts/run_optuna_experiment.py

import ray
import yaml
import os
import numpy as np
import sys
import copy
import time
import torch
import optuna
import argparse
from pathlib import Path
from rich.console import Console
import logging
from nuscenes.nuscenes import NuScenes

# --- Profiling Imports ---
import cProfile
import pstats

# Custom imports
from .tuning_manager import define_search_space
from ..config_loader import MDetectorConfigAccessor
from ..data_utils.validation_utils_torch import calculate_metrics_for_optuna_trial_in_memory
from ..data_utils.seeding_utils import set_seed
from ..core.m_detector.base import MDetector
from ..data_utils.nuscenes_helper import NuScenesProcessor
from .ray_actors import NuScenesDataActor, load_all_gt_data_in_background

# --- Setup Paths and Environment ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKING_DIR = PROJECT_ROOT.parent
CONFIG_PATH_ABSOLUTE = WORKING_DIR / 'config' / 'm_detector_config.yaml'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Helper Functions ---
def deep_update_dict(base_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update_dict(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def calculate_iou(metrics: dict) -> float:
    tp, fp, fn = metrics.get('tp', 0), metrics.get('fp', 0), metrics.get('fn', 0)
    denominator = tp + fp + fn
    return tp / denominator if denominator > 0 else 0.0


@ray.remote(num_gpus=0.25) 
def run_trial_on_ray(trial_params: dict, trial_number: int, base_config: dict, phase: int) -> tuple[int, float, dict]:
    """
    This Ray task executes a complete Optuna trial entirely in memory and includes profiling.
    """
    # --- 1. Profiling Setup ---
    profiles_dir = WORKING_DIR / "profiles"
    os.makedirs(profiles_dir, exist_ok=True)
    profile_filename = profiles_dir / f"trial_{trial_number}.prof"
    profiler = cProfile.Profile()
    profiler.enable()

    # --- 2. In-Memory Configuration Setup ---
    console = Console()
    trial_config = copy.deepcopy(base_config)
    deep_update_dict(trial_config, trial_params)

    data_actor_handle = ray.get_actor("nuscenes_data_service")
    
    try:
        # Create accessor directly from the config dictionary
        accessor = MDetectorConfigAccessor(config_dict=trial_config)
        
        # This line is now inside the try block, which is correct.
        nusc_handle = ray.get(data_actor_handle.get_nusc_handle.remote())
        
        seed = accessor.get_random_seed()
        if seed is not None:
            set_seed(seed)
        
        detector = MDetector(config_accessor=accessor, device=torch.device("cuda:0"))
        # The processor is created with the guaranteed handle
        processor = NuScenesProcessor(
            data_actor=data_actor_handle, config_accessor=accessor,
            progress_actor=None, worker_id=trial_number, logger_name=f"optuna_{trial_number}"
        )
        
    except Exception as e:
        profiler.disable() # Ensure profiler is disabled on error
        console.log(f"[bold red]Trial {trial_number} failed during setup: {e}[/bold red]")
        return trial_number, 0.0, {"error": f"setup_failed: {e}"}

    # --- 3. Scene Processing and Metrics Calculation ---
    all_metrics = []
    scene_indices = accessor.get_mdetector_output_paths()['scene_indices_to_run']
    
    for scene_idx in scene_indices:
        try:
            # Run the main algorithm; returns the result dictionary directly
            result_dict = processor.process_scene(scene_idx, detector, f"T{trial_number}_S{scene_idx}")
            if not result_dict:
                all_metrics.append({'tp': 0, 'fp': 1, 'fn': 1})
                continue
            
            # NO MORE FILE I/O
            
            eval_params_for_trial = {
                "mdet_label_field_name": "mdet_label",
                "mdet_dynamic_label_value": 0, # OcclusionResult.OCCLUDING_IMAGE.value
                "coordinate_tolerance_for_verification": 1e-3,
                "evaluate_only_keyframes": False,
                "gt_velocity_threshold": accessor.get_validation_params()['gt_velocity_threshold']
            }
            
            # Call the new IN-MEMORY worker logic
            metrics = calculate_metrics_for_optuna_trial_in_memory(
                mdet_results_dict=result_dict,
                eval_params=eval_params_for_trial,
                nusc=nusc_handle, 
                scene_idx=scene_idx
            )

            
            if "error" in metrics:
                all_metrics.append({'tp': 0, 'fp': 1, 'fn': 1})
            else:
                all_metrics.append(metrics)
        except Exception as e:
            import traceback
            console.log(f"[bold red]Trial {trial_number} Scene {scene_idx} crashed: {e}\n{traceback.format_exc()}[/bold red]")
            all_metrics.append({'tp': 0, 'fp': 1, 'fn': 1})

    # --- 4. Aggregate Results ---
    if not all_metrics:
        final_iou = 0.0
    else:
        final_tp = sum(m['tp'] for m in all_metrics)
        final_fp = sum(m['fp'] for m in all_metrics)
        final_fn = sum(m['fn'] for m in all_metrics)
        final_iou = calculate_iou({'tp': final_tp, 'fp': final_fp, 'fn': final_fn})
    
    # --- 5. Finalize Profiling and Return ---
    profiler.disable()
    profiler.dump_stats(profile_filename)
    
    return trial_number, final_iou, {}

def main():
    # --- 1. Argument Parser (Simplified) ---
    # We remove the --gpu-ids argument as it's no longer needed.
    parser = argparse.ArgumentParser(description="Run M-Detector hyperparameter tuning with Optuna and Ray.")
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True, help="Tuning phase (1: broad, 2: fine-tune).")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials to run.")
    parser.add_argument("--study-name", type=str, required=True, help="Name for the Optuna study.")
    parser.add_argument("--n-parallel-trials", type=int, default=4, help="Number of trials to run in parallel on Ray.")
    args = parser.parse_args()

    console = Console()

    # --- 2. GPU Configuration (Simplified: reads ONLY from config file) ---
    temp_accessor = MDetectorConfigAccessor(CONFIG_PATH_ABSOLUTE)
    base_config = temp_accessor.get_raw_config()
    
    processing_settings = temp_accessor.get_processing_settings()
    config_gpu_ids = processing_settings.get('gpu_ids')

    if config_gpu_ids: # This handles None and empty lists
        visible_gpus = ",".join(map(str, config_gpu_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus
        console.log(f"[bold cyan]CONFIG FILE:[/bold cyan] Limiting Ray to use only GPU(s): {visible_gpus}")
    else:
        console.log("[bold green]DEFAULT:[/bold green] No GPU IDs found in config. Ray will use all available GPUs.")

    # --- 3. Ray and Optuna Setup (No changes here) ---
    ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)
    console.log("[bold green]Ray initialized.[/bold green]")

    try:
        # This part remains the same, using the simplified actor logic
        nuscenes_cfg = temp_accessor.get_nuscenes_params()
        data_actor = NuScenesDataActor.options(name="nuscenes_data_service", get_if_exists=True).remote(
            nuscenes_cfg['version'], nuscenes_cfg['dataroot'], CONFIG_PATH_ABSOLUTE
        )

        # 2. Launch the standalone background task to load all data.
        console.log("Starting background caching of Ground Truth files...")
        gt_data_future = load_all_gt_data_in_background.remote(CONFIG_PATH_ABSOLUTE)

        # 3. Wait for the data to be loaded and set it in the actor.
        #    This part is now synchronous in the main script, which is clearer.
        gt_data = ray.get(gt_data_future)
        ray.get(data_actor.set_cache.remote(gt_data)) # Wait for the actor's state to be set.
        console.log("[bold green]Ground Truth caching and actor setup complete.[/bold green]")
        
        # --- Optuna Study Setup (the rest of the script is the same) ---
        storage_path = f"sqlite:///{args.study_name}.db"
        study = optuna.create_study(
            study_name=args.study_name, storage=storage_path,
            direction="maximize", pruner=optuna.pruners.MedianPruner(),
            load_if_exists=True
        )
        
        console.log(f"--- Starting Optuna Study ---")
        console.log(f"Study Name: {args.study_name}, Metric: [bold cyan]IoU[/bold cyan]")
        console.log(f"Ray Parallelism: {args.n_parallel_trials} trials")

        # --- Custom Optimization Loop ---
        running_tasks = {}
        # Correctly count completed trials from the loaded study
        existing_trials = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED)))
        completed_trials = existing_trials

        console.log(f"[yellow]Found {existing_trials} existing trials in the study.[/yellow]")

        while completed_trials < args.n_trials:
            # --- THIS IS THE CORRECTED LOGIC ---
            # Only dispatch a new trial if the total number of trials we have created
            # (completed + currently running) is less than our target.
            if (completed_trials + len(running_tasks)) < args.n_trials:
                # And if we have a free worker slot.
                while len(running_tasks) < args.n_parallel_trials:
                    # Break out of this inner loop if we've dispatched enough trials.
                    if (completed_trials + len(running_tasks)) >= args.n_trials:
                        break
                        
                    trial = study.ask()
                    params = define_search_space(trial)
                    
                    future = run_trial_on_ray.remote(params, trial.number, base_config, args.phase)
                    running_tasks[future] = trial
                    console.log(f"Dispatched Trial {trial.number} to Ray.")

            # Wait for at least one of the running tasks to complete
            ready_futures, _ = ray.wait(list(running_tasks.keys()), num_returns=1)
            ready_future = ready_futures[0]
            
            # Process the completed trial
            trial_obj = running_tasks.pop(ready_future)
            trial_num, iou_score, metadata = ray.get(ready_future)

            if "error" in metadata:
                study.tell(trial_obj, state=optuna.trial.TrialState.FAIL)
                console.log(f"[bold red]Trial {trial_obj.number} FAILED. Reason: {metadata['error']}[/bold red]")
            else:
                study.tell(trial_obj, iou_score)
                console.log(f"[bold green]Trial {trial_obj.number} finished. Score (IoU): {iou_score:.4f}[/bold green]")
            
            completed_trials += 1
            console.log(f"Progress: {completed_trials}/{args.n_trials} trials complete. Best score so far: {f'{study.best_value:.4f}' if study.best_value is not None else 'N/A'}")


        console.log("\n--- Optuna Study Complete ---")
        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        console.log(df)

        best_trial = study.best_trial
        if best_trial:
            console.log(f"Best trial:")
            console.log(f"  Value (IoU): {best_trial.value}")
            console.log(f"  Params: ")
            for key, value in best_trial.params.items():
                console.log(f"    {key}: {value}")
    
    finally:
        ray.shutdown()
        console.log("[bold green]Ray shut down.[/bold green]")

if __name__ == '__main__':
    main()