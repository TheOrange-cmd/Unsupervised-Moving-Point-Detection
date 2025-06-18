import ray
import yaml
import os
import json
import numpy as np
import sys
import logging
import copy
import time
import h5py
import torch
from nuscenes.nuscenes import NuScenes
from rich.console import Console

# Custom imports
from scripts.ray_actors import ProgressActor, SceneProcessorActor, LoggingActor, NuScenesDataActor
from scripts.tuning_manager import get_experiments
from src.config_loader import MDetectorConfigAccessor

# --- Part 1: Module-Level Constants and Environment Setup ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Define the absolute path to the config file as a module constant
CONFIG_PATH_ABSOLUTE = os.path.join(PROJECT_ROOT, 'config', 'm_detector_config.yaml')

def configure_environment_from_config(config_path: str) -> tuple[str, list]:
    """
    Loads the config, sets CUDA_VISIBLE_DEVICES, and returns the device string and list of GPU IDs.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"FATAL: Main configuration file not found at {config_path}. Cannot configure environment.")

    processing_cfg = config['processing_settings']
    device_str = processing_cfg['device']
    gpu_ids = []

    if device_str == 'cuda':
        gpu_ids = processing_cfg['gpu_ids']
        if not isinstance(gpu_ids, list):
            raise TypeError("Config error: 'gpu_ids' must be a list (e.g., [0, 2, 5] or [] for all).")
        
        if gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_ids))
            print(f"--- Environment configured to use specific GPUs: {os.environ['CUDA_VISIBLE_DEVICES']} ---")
        else:
            # If gpu_ids is empty, we need to find out how many GPUs are actually available
            import torch
            if torch.cuda.is_available():
                gpu_ids = list(range(torch.cuda.device_count()))
            print(f"--- Environment configured to use ALL available GPUs: {gpu_ids} ---")
    
    return device_str, gpu_ids

# Execute the environment setup at import time
# This sets CUDA_VISIBLE_DEVICES before any torch operations
DEVICE_STR_FROM_CONFIG, GPU_IDS_FROM_CONFIG = configure_environment_from_config(CONFIG_PATH_ABSOLUTE)

# --- Helper functions can live here or be imported ---
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

def save_and_log_result(result_tuple, console: Console):
    """
    Handles saving results and logging outcomes using the provided console.
    """
    scene_idx, tuning_name, dict_of_arrays, output_dir, config_path, error = result_tuple
    
    if error:
        console.log(f"[bold red]ERROR:[/bold red] Worker failed for Scene {scene_idx}, Tuning '{tuning_name}'. Error: {error}")
        return False

    if dict_of_arrays is None:
        console.log(f"[yellow]WARNING:[/yellow] Task returned no data for Scene {scene_idx}, Tuning '{tuning_name}'.")
        return False

    output_filename_h5 = f"mdet_results_{tuning_name}_scene_{scene_idx}.h5"
    output_filepath_h5 = os.path.join(output_dir, output_filename_h5)

    try:
        with h5py.File(output_filepath_h5, 'w') as hf:
            for key, array_data in dict_of_arrays.items():
                if isinstance(array_data, np.ndarray):
                    hf.create_dataset(key, data=array_data)
                else:
                    hf.create_dataset(key, data=np.array(array_data))
        
        console.log(f"Saved: [cyan]{output_filename_h5}[/cyan]")
        return True
    except Exception as e:
        console.log(f"[bold red]ERROR:[/bold red] Failed to save HDF5 for Scene {scene_idx}, Tuning '{tuning_name}'. Error: {e}")
        return False

def main():
    # --- 1. Initialization and Config Loading ---
    start_time = time.time()
    console = Console(stderr=True)
    ray.init(logging_level=logging.FATAL) # Keep Ray's own console output quiet
    console.log("[bold green]Ray initialized.[/bold green]")

    # --- 2. Configuration & Task Preparation ---
    console.log("Preparing tuning configurations...")
    base_config_dict = yaml.safe_load(open(CONFIG_PATH_ABSOLUTE, 'r'))
    tuning_experiments = get_experiments(mode="static")
    if not tuning_experiments:
        tuning_experiments = [{"name": "baseline_config", "overrides": {}}]

    temp_accessor = MDetectorConfigAccessor(CONFIG_PATH_ABSOLUTE)
    main_save_path = temp_accessor.get_mdetector_output_paths()['save_path']
    nuscenes_cfg = temp_accessor.get_nuscenes_params()
    workers_per_gpu = temp_accessor.get_processing_settings().get('workers_per_gpu', 1)
    gpu_fraction = 1.0 / workers_per_gpu
    console.log(f"Configured for [bold cyan]{workers_per_gpu}[/bold cyan] workers per GPU (each using {gpu_fraction:.2f} of a GPU).")

    # Setup logging

    log_dir = os.path.join(PROJECT_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Map string from config to actual logging level
    log_level_str = temp_accessor.get_processing_settings().get('detailed_log_level', "INFO").upper()
    log_level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
    detail_log_level = log_level_map.get(log_level_str, logging.INFO)
    
    console.log(f"Log files will be saved in: [cyan]{log_dir}[/cyan]")
    console.log(f"Detailed log level set to: [bold yellow]{log_level_str}[/bold yellow]")

    console.log("Creating NuScenesDataActor service...")
    data_actor = NuScenesDataActor.remote(nuscenes_cfg['version'], nuscenes_cfg['dataroot'])

    # # This NuScenes instance is only used to get the scene indices.
    # # It is loaded into shared memory to prevent a memory explosing when running big jobs.
    # nusc_main = NuScenes(version=nuscenes_cfg['version'], dataroot=nuscenes_cfg['dataroot'], verbose=False)
    # nusc_ref = ray.put(nusc_main)
    # console.log("NuScenes data placed in shared memory.")

    # console.log("Creating NuScenesDataActor service...")
    # data_actor = NuScenesDataActor.remote(nuscenes_cfg['version'], nuscenes_cfg['dataroot'])

    # # Get scene indices we want to process
    # scene_indices_config = temp_accessor.get_mdetector_output_paths().get('scene_indices_to_run', 'all')
    # if isinstance(scene_indices_config, str) and scene_indices_config.lower() == 'all':
    #     all_scene_indices = set(range(len(nusc_main.scene)))
    # else:
    #     all_scene_indices = set(scene_indices_config)

    scene_indices_config = temp_accessor.get_mdetector_output_paths()['scene_indices_to_run']
    if isinstance(scene_indices_config, str) and scene_indices_config.lower() == 'all':
        total_scenes = ray.get(data_actor.get_scene_count.remote())
        all_scene_indices = set(range(total_scenes))
    else:
        all_scene_indices = set(scene_indices_config)

    # --- Create a list of all individual tasks ---
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
                "scene_idx": scene_idx,
                "tuning_name": tuning_name,
                "config_path": config_path,
                "output_dir": output_dir,
            })

    # --- 3. Create Actors ---
    num_gpus = len(GPU_IDS_FROM_CONFIG)
    total_workers = num_gpus * workers_per_gpu
    console.log(f"Creating a pool of [bold cyan]{total_workers}[/bold cyan] worker actors...")

    logging_actor = LoggingActor.remote(log_dir, detail_log_level)
    progress_actor = ProgressActor.remote(total_tasks=len(all_individual_tasks), logging_actor=logging_actor)

    worker_actors = []
    for i in range(total_workers):
        actor = SceneProcessorActor.options(num_gpus=gpu_fraction).remote(
            worker_id=i,
            data_actor=data_actor, 
            progress_actor=progress_actor,
            logging_actor=logging_actor
        )
        worker_actors.append(actor)
    console.log(f"Created {len(worker_actors)} worker actors.")

    # --- 4. Dispatch Tasks ---
    console.log(f"Dispatching {len(all_individual_tasks)} tasks to the worker pool...")
    # Start the UI actor's rendering loop
    ui_future = progress_actor.run.remote()

    result_futures = []
    # Distribute tasks in a round-robin fashion to the workers
    for i, task_info in enumerate(all_individual_tasks):
        actor = worker_actors[i % total_workers]
        future = actor.process_single_tuning.remote(task_info)
        result_futures.append(future)

    # --- 5. Collect Results ---
    total_saved = 0
    total_failed = 0

    while result_futures:
        ready_futures, result_futures = ray.wait(result_futures, num_returns=1)
        result = ray.get(ready_futures[0])

        if save_and_log_result(result, console):
            total_saved += 1
        else:
            total_failed += 1

    # --- 6. Shutdown ---
    console.log("[bold green]All tasks complete. Shutting down.[/bold green]")
    progress_actor.shutdown.remote()
    ray.get(logging_actor.shutdown.remote())
    time.sleep(1) # Give the UI time to shut down gracefully
    ray.shutdown()

    # --- Final Summary ---
    total_duration = time.time() - start_time
    print("\n--- Run Complete ---")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours).")
    print(f"Total successful scene-tuning instances saved: {total_saved}.")
    if total_failed > 0:
        print(f"Total failed instances: {total_failed} (check logs for details).")

if __name__ == '__main__':
    main()