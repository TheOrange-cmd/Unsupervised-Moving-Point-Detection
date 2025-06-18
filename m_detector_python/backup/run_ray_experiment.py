# src/ray_scripts/run_ray_experiment.py

import ray
import yaml
import os
import json
import numpy as np
import sys
import logging
import copy
import time
import torch
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from pathlib import Path

# Custom imports
from .ray_actors import ProgressActor, SceneProcessorActor, NuScenesDataActor, ProfilingFlagActor
from .tuning_manager import get_experiments
from ..config_loader import MDetectorConfigAccessor
from ..data_utils.seeding_utils import set_seed

# --- Module-Level Constants and Environment Setup ---

# This path logic is now critical for defining the working directory for Ray.
# This file is in: .../Unsupervised-Moving-Point-Detection/m_detector_python/src/ray_scripts/
# SCRIPT_DIR is .../src/ray_scripts
SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT is .../src
PROJECT_ROOT = SCRIPT_DIR.parent
# The working_dir for Ray should be the parent of 'src', which is the project's root directory.
WORKING_DIR = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# CONFIG_PATH_ABSOLUTE = PROJECT_ROOT.parent / 'config' / 'm_detector_config.yaml'
CONFIG_PATH_ABSOLUTE = PROJECT_ROOT.parent / 'config' / 'config_best_trial.yaml'

def configure_environment_from_config(config_path: str) -> tuple[str, list]:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"FATAL: Main configuration file not found at {config_path}. Cannot configure environment.")

    processing_cfg = config['processing_settings']
    device_str = processing_cfg['device']
    gpu_ids = []

    if device_str == 'cuda':
        gpu_ids = processing_cfg.get('gpu_ids', [])
        if not isinstance(gpu_ids, list):
            raise TypeError("Config error: 'gpu_ids' must be a list (e.g., [0, 2, 5] or [] for all).")
        
        if gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_ids))
            print(f"--- Environment configured to use specific GPUs: {os.environ['CUDA_VISIBLE_DEVICES']} ---")
        else:
            if torch.cuda.is_available():
                gpu_ids = list(range(torch.cuda.device_count()))
            print(f"--- Environment configured to use ALL available GPUs: {gpu_ids} ---")
    
    return device_str, gpu_ids

DEVICE_STR_FROM_CONFIG, GPU_IDS_FROM_CONFIG = configure_environment_from_config(CONFIG_PATH_ABSOLUTE)

def setup_worker_logging(log_dir: str, summary_level: int, detail_level: int):
    worker_pid = os.getpid()
    logger_name = f"worker_{worker_pid}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(min(summary_level, detail_level))
    logger.propagate = False
    
    if logger.hasHandlers():
        logger.handlers.clear()

    summary_log_path = os.path.join(log_dir, "run_summary.log")
    summary_handler = logging.FileHandler(summary_log_path)
    summary_handler.setLevel(summary_level)
    summary_formatter = logging.Formatter(f'%(asctime)s - PID:{worker_pid} - %(levelname)s - %(message)s')
    summary_handler.setFormatter(summary_formatter)
    logger.addHandler(summary_handler)

    detail_log_path = os.path.join(log_dir, "run_detailed.log")
    detail_handler = logging.FileHandler(detail_log_path)
    detail_handler.setLevel(detail_level)
    detail_formatter = logging.Formatter(f'%(asctime)s - PID:{worker_pid} - %(name)s - %(levelname)s - %(message)s')
    detail_handler.setFormatter(detail_formatter)
    logger.addHandler(detail_handler)

class NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpySafeEncoder, self).default(obj)

def deep_update_dict(base_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update_dict(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def main():
    start_time = time.time()
    console = Console(stderr=True)
    temp_accessor = MDetectorConfigAccessor(CONFIG_PATH_ABSOLUTE)

    seed = temp_accessor.get_random_seed()
    if seed is not None:
        set_seed(seed)
        console.log(f"[bold yellow]Global random seed set to: {seed}[/bold yellow]")

    log_dir = WORKING_DIR / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_level_str = temp_accessor.get_processing_settings().get('detailed_log_level', 'INFO').upper()
    log_level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    detail_log_level = log_level_map.get(log_level_str, logging.INFO)
    
    console.log(f"Log files will be saved in: [cyan]{log_dir}[/cyan]")
    console.log(f"Detailed log level set to: [bold yellow]{log_level_str}[/bold yellow]")

    ray.init(
        runtime_env={
            "working_dir": str(WORKING_DIR),
            "worker_process_setup_hook": lambda: setup_worker_logging(
                log_dir=str(log_dir), 
                summary_level=logging.INFO, 
                detail_level=detail_log_level
            )
        }
    )
    console.log("[bold green]Ray initialized.[/bold green]")

    console.log("Preparing tuning configurations...")
    base_config_dict = temp_accessor.get_raw_config()
    tuning_experiments = get_experiments(mode="static", num_trials=0)
    console.log(f"Generated {len(tuning_experiments)} tuning configurations.")

    if not tuning_experiments:
        console.log("[bold yellow]No tuning experiments generated. Creating a single static run based on the loaded config file.[/bold yellow]")
        tuning_experiments = [{
            "name": "static_validation_run",  # This will be the output folder name
            "overrides": {}                   # An empty override dict means "use the config as-is"
        }]
    
    main_save_path = temp_accessor.get_mdetector_output_paths()['save_path']
    nuscenes_cfg = temp_accessor.get_nuscenes_params()
    workers_per_gpu = temp_accessor.get_processing_settings().get('workers_per_gpu', 1)
    gpu_fraction = 1.0 / workers_per_gpu if workers_per_gpu > 0 else 0
    console.log(f"Configured for [bold cyan]{workers_per_gpu}[/bold cyan] workers per GPU (each using {gpu_fraction:.2f} of a GPU).")
    
    console.log("Creating NuScenesDataActor service...")
    data_actor = NuScenesDataActor.remote(nuscenes_cfg['version'], nuscenes_cfg['dataroot'])
    
    scene_indices_config = temp_accessor.get_mdetector_output_paths()['scene_indices_to_run']
    if isinstance(scene_indices_config, str) and scene_indices_config.lower() == 'all':
        total_scenes = ray.get(data_actor.get_scene_count.remote())
        all_scene_indices = set(range(total_scenes))
    else:
        all_scene_indices = set(scene_indices_config)
    
    all_individual_tasks = []
    for tuning_exp in tuning_experiments:
        tuning_name = tuning_exp["name"]
        overrides = tuning_exp["overrides"]
        current_tuned_config = copy.deepcopy(base_config_dict)
        deep_update_dict(current_tuned_config, overrides)
        output_dir = os.path.join(main_save_path, tuning_name)
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, f"config_tuned_{tuning_name}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(current_tuned_config, f, sort_keys=False)
        for scene_idx in all_scene_indices:
            all_individual_tasks.append({
                "scene_idx": scene_idx, "tuning_name": tuning_name,
                "config_path": config_path, "output_dir": output_dir,
            })

    num_gpus = len(GPU_IDS_FROM_CONFIG) if GPU_IDS_FROM_CONFIG else 1
    total_workers = num_gpus * workers_per_gpu
    console.log(f"Creating a pool of [bold cyan]{total_workers}[/bold cyan] worker actors...")
    
    progress_actor = ProgressActor.remote(
        total_tasks=len(all_individual_tasks),
        num_workers=total_workers
    )

    profiling_flag_actor = ProfilingFlagActor.remote()
    
    worker_actors = []
    for i in range(total_workers):
        actor_options = {"num_gpus": gpu_fraction} if DEVICE_STR_FROM_CONFIG == 'cuda' else {}
        actor = SceneProcessorActor.options(**actor_options).remote(
            worker_id=i, 
            data_actor=data_actor, 
            progress_actor=progress_actor,
            profiling_flag_actor=profiling_flag_actor,
            log_dir=str(log_dir) 
        )
        worker_actors.append(actor)
    console.log(f"Created {len(worker_actors)} worker actors.")

    console.log(f"Dispatching {len(all_individual_tasks)} tasks to the worker pool...")
    ui_future = progress_actor.run.remote()
    result_futures = [
        worker_actors[i % total_workers].process_single_tuning.remote(task_info)
        for i, task_info in enumerate(all_individual_tasks)
    ]

    def save_result_local(result_tuple):
        scene_idx, tuning_name, dict_of_arrays, output_dir, config_path, error = result_tuple
        if error:
            console.log(f"[bold red]ERROR CONFIRMED:[/bold red] Worker failed for Scene {scene_idx}, Tuning '{tuning_name}'. Check logs.")
            return False
        if dict_of_arrays is None:
            console.log(f"[yellow]WARNING:[/yellow] No data returned for Scene {scene_idx}, Tuning '{tuning_name}'.")
            return False
        
        output_filename_pt = f"mdet_results_{tuning_name}_scene_{scene_idx}.pt"
        output_filepath_pt = os.path.join(output_dir, output_filename_pt)
        try:
            torch.save(dict_of_arrays, output_filepath_pt)
            return True
        except Exception as e:
            console.log(f"[bold red]SAVE FAILURE:[/bold red] Failed to save for Scene {scene_idx}, Tuning '{tuning_name}'. Error: {e}")
            return False

    total_saved, total_failed = 0, 0
    with Progress(
        TextColumn("[bold blue]Collecting Results"), BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("({task.completed} of {task.total})"),
        TimeRemainingColumn(), TimeElapsedColumn(), console=console
    ) as progress_bar:
        main_task = progress_bar.add_task("Processing", total=len(result_futures))
        while total_saved + total_failed < len(all_individual_tasks):
            ready_futures, result_futures = ray.wait(result_futures, num_returns=1)
            result = ray.get(ready_futures[0])
            if save_result_local(result):
                total_saved += 1
            else:
                total_failed += 1
            progress_actor.task_complete.remote()
            progress_bar.update(main_task, advance=1)

    console.log("[bold green]All tasks complete. Shutting down.[/bold green]")
    progress_actor.shutdown.remote()
    time.sleep(1) 
    ray.shutdown()

    console.log("\n--- Run Summary ---")
    console.log(f"Total duration: {time.time() - start_time:.2f} seconds.")
    console.log(f"Total successful scene-tuning instances saved: {total_saved}.")
    if total_failed > 0:
        console.log(f"[bold red]Total failed instances: {total_failed} (check logs for details).[/bold red]")

if __name__ == '__main__':
    main()