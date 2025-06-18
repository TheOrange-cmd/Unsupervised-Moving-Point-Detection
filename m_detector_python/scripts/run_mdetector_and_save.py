# --- Part 1: Initial, minimal imports for environment configuration ---
import yaml
import os
import json
import numpy as np
import sys
import logging
import multiprocessing
import copy
import time
import h5py
from logging.handlers import QueueHandler

from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.logging import RichHandler
from rich.console import Console

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
            print(f"--- Environment configured by script to use specific GPUs: {os.environ['CUDA_VISIBLE_DEVICES']} ---")
        else:
            # If gpu_ids is empty, we need to find out how many GPUs are actually available
            import torch
            if torch.cuda.is_available():
                gpu_ids = list(range(torch.cuda.device_count()))
            print(f"--- Environment configured to use ALL available GPUs: {gpu_ids} ---")
    
    return device_str, gpu_ids

def logging_listener_process(log_queue, ui_queue):
    while True:
        record = log_queue.get()
        if record is None:
            ui_queue.put(None) # Signal UI to exit
            break
        if hasattr(record, 'progress'):
            ui_queue.put(record.progress)

def configure_main_logging(verbose=False):
    """
    This configures logging for the main process and the listener process.
    """
    level = logging.INFO if verbose else logging.WARNING
    
    root = logging.getLogger()
    h = logging.StreamHandler()
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)
    root.setLevel(level)

def configure_worker_logging(queue):
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(logging.INFO)

# --- Some globals ... ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH_ABSOLUTE = os.path.join(PROJECT_ROOT, 'config', 'm_detector_config.yaml')

DEVICE_STR_FROM_CONFIG, GPU_IDS_FROM_CONFIG = configure_environment_from_config(CONFIG_PATH_ABSOLUTE)

# --- Import project libraries ---
import torch
from nuscenes.nuscenes import NuScenes
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from src.core.m_detector.base import MDetector
from src.data_utils.nuscenes_helper import NuScenesProcessor
from src.config_loader import MDetectorConfigAccessor
from scripts.tuning_manager import get_experiments

# --- Helper Functions ---
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

# --- Listener and Worker Configuration ---

def ui_listener_process(ui_queue, total_scenes):
    """Renders the rich progress UI."""
    progress = Progress(
        TextColumn("[bold blue]PID {task.fields[pid]}"),
        TextColumn("{task.description}", justify="left"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        SpinnerColumn(),
    )
    
    # Add a placeholder 'pid' to the overall task so the column renderer doesn't fail.
    overall_task = progress.add_task("[green]All Scenes", total=total_scenes, pid="Overall")
    worker_tasks = {}

    layout = Table.grid(expand=True)
    layout.add_row(progress)
    
    # Redirect to stderr to not interfere with stdout from other processes if any
    with Live(layout, refresh_per_second=10, console=Console(stderr=True)):
        finished = False
        while not finished:
            while not ui_queue.empty():
                msg = ui_queue.get()
                if msg is None:
                    finished = True
                    break
                
                pid = msg['pid']
                if msg['type'] == 'start':
                    worker_tasks[pid] = progress.add_task(
                        msg['description'], total=msg['total'], pid=pid, start=True
                    )
                elif msg['type'] == 'update' and pid in worker_tasks:
                    progress.update(worker_tasks[pid], advance=msg['advance'])
                elif msg['type'] == 'stop' and pid in worker_tasks:
                    # Use remove_task to make it disappear cleanly
                    progress.remove_task(worker_tasks[pid])
                    del worker_tasks[pid]
                    progress.update(overall_task, advance=1)
            time.sleep(0.05) # Sleep briefly to prevent pegging the CPU

def logging_listener_process(log_queue, ui_queue, verbose):
    """Receives all logs, forwards progress logs, and prints other logs based on verbosity."""
    handler = RichHandler(rich_tracebacks=True, show_path=False)
    
    while True:
        try:
            record = log_queue.get()
            if record is None:
                ui_queue.put(None)
                break
            
            if hasattr(record, 'progress'):
                ui_queue.put(record.progress)
            else:
                # Only handle the log if in verbose mode OR if it's a warning/error.
                if verbose or record.levelno >= logging.WARNING:
                    logger = logging.getLogger(record.name)
                    logger.propagate = False
                    logger.handlers.clear()
                    logger.addHandler(handler)
                    logger.handle(record)
        except Exception:
            import sys, traceback
            traceback.print_exc(file=sys.stderr)


def configure_worker_logging(queue):
    """Configures each worker to send all its logs to the queue."""
    h = QueueHandler(queue)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(logging.INFO) # Workers should always send all info


# --- Worker and Save Functions  ---

def process_scene_with_tunings_worker(task_info_tuple):
    """
    Worker function that processes ONE scene with MULTIPLE tuning configurations.
    """
    scene_idx, list_of_tuned_config_paths, output_dirs, nuscenes_version, nuscenes_dataroot, base_device_str, gpu_index = task_info_tuple
    
    scene_results = []
    try:
        if base_device_str == 'cuda':
            device = torch.device(f'cuda:{gpu_index}')
        else:
            device = torch.device('cpu')

        nusc_worker = NuScenes(version=nuscenes_version, dataroot=nuscenes_dataroot, verbose=False)
        scene_record = nusc_worker.scene[scene_idx]
        scene_name = scene_record['name']
        
        for i, config_path in enumerate(list_of_tuned_config_paths):
            try:
                config_accessor = MDetectorConfigAccessor(config_path)
                detector_worker = MDetector(config_accessor=config_accessor, device=device)
                processor_worker = NuScenesProcessor(nusc_worker, config_accessor=config_accessor)

                dict_of_arrays = processor_worker.process_scene(
                    scene_index=scene_idx, detector=detector_worker, with_progress=False
                )

                if dict_of_arrays is not None:
                    raw_config = config_accessor.get_raw_config()
                    config_json_str = json.dumps(raw_config, sort_keys=True, indent=4, cls=NumpySafeEncoder)
                    dict_of_arrays['_config_json_str'] = np.array(config_json_str)
                
                scene_results.append((scene_idx, scene_name, dict_of_arrays, output_dirs[i], config_path, None))
            except Exception as e_tuning:
                # Return the exception to be logged by the main process
                scene_results.append((scene_idx, scene_name, None, output_dirs[i], config_path, e_tuning))
        return scene_results
    except Exception as e_scene:
        # Return a list of failures with the exception
        failed_results = [(scene_idx, f"scene_idx_{scene_idx}_FAILED", None, output_dirs[i], config_path, e_scene) for i, config_path in enumerate(list_of_tuned_config_paths)]
        return failed_results

def save_and_log_result(result_tuple):
    """
    MODIFICATION: New function in the main process to handle all I/O.
    This function is called for each result returned by the workers.
    """
    scene_idx, scene_name, dict_of_arrays, output_dir, config_path, error = result_tuple
    
    if error:
        logging.error(f"Worker failed for Scene '{scene_name}' (idx {scene_idx}), Config: {os.path.basename(config_path)}. Error: {error}", exc_info=isinstance(error, Exception))
        return False

    if dict_of_arrays is None:
        logging.warning(f"Task returned no data: Scene '{scene_name}' (idx {scene_idx}), Config: {os.path.basename(config_path)}")
        return False

    output_filename_h5 = f"mdet_results_{scene_name}.h5"
    output_filepath_h5 = os.path.join(output_dir, output_filename_h5)

    try:
        with h5py.File(output_filepath_h5, 'w') as hf:
            for key, array_data in dict_of_arrays.items():
                if isinstance(array_data, np.ndarray):
                    if array_data.ndim == 0: hf.create_dataset(key, data=array_data.item())
                    else: hf.create_dataset(key, data=array_data)
                else: hf.create_dataset(key, data=np.array(array_data))
        
        logging.info(f"Successfully saved HDF5 for Scene '{scene_name}', Config: {os.path.basename(config_path)}")
        return True
    except Exception as e:
        logging.error(f"Error saving HDF5 for Scene '{scene_name}', Config: {os.path.basename(config_path)}. Error: {e}", exc_info=True)
        return False

# --- main ---
def main(verbose=False): 
    # --- Setup Listeners and Queues ---
    log_queue = multiprocessing.Manager().Queue()
    ui_queue = multiprocessing.Manager().Queue()

    # --- Task Preparation  ---
    base_config_dict = yaml.safe_load(open(CONFIG_PATH_ABSOLUTE, 'r'))
    tuning_experiments = get_experiments(mode="static")
    if not tuning_experiments:
        tuning_experiments = [{"name": "baseline_config", "overrides": {}}]

    temp_accessor = MDetectorConfigAccessor(CONFIG_PATH_ABSOLUTE)
    main_save_path = temp_accessor.get_mdetector_output_paths()['save_path']
    nuscenes_cfg = temp_accessor.get_nuscenes_params()
    
    nusc_main = NuScenes(version=nuscenes_cfg['version'], dataroot=nuscenes_cfg['dataroot'], verbose=False)
    
    scene_indices_config = temp_accessor.get_mdetector_output_paths().get('scene_indices_to_run', 'all')
    if isinstance(scene_indices_config, str) and scene_indices_config.lower() == 'all':
        all_scene_indices = set(range(len(nusc_main.scene)))
    else:
        all_scene_indices = set(scene_indices_config)

    tasks_by_scene = {scene_idx: [] for scene_idx in all_scene_indices}
    
    print("Preparing tuning configurations...")
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
            tasks_by_scene[scene_idx].append({"config_path": config_path, "output_dir": output_dir})

    # Create the final list of tasks for the multiprocessing pool
    all_grouped_tasks = []
    num_gpus = len(GPU_IDS_FROM_CONFIG)
    for i, (scene_idx, tunings) in enumerate(tasks_by_scene.items()):
        if not tunings: continue
        gpu_index = i % num_gpus if num_gpus > 0 else 0
        all_grouped_tasks.append((
            scene_idx, [t['config_path'] for t in tunings], [t['output_dir'] for t in tunings],
            nuscenes_cfg['version'], nuscenes_cfg['dataroot'], DEVICE_STR_FROM_CONFIG, gpu_index
        ))
        
    if not all_grouped_tasks:
        print("No valid processing tasks generated. Exiting.")
        return

    # --- Start Listeners and Worker Pool ---
    overall_start_time = time.time()
    
    ui_process = multiprocessing.Process(target=ui_listener_process, args=(ui_queue, len(all_grouped_tasks)))
    ui_process.start()

    # Pass the verbose flag to the logging listener
    logging_listener = multiprocessing.Process(target=logging_listener_process, args=(log_queue, ui_queue, verbose))
    logging_listener.start()

    num_cores_to_use = temp_accessor.get_mdetector_output_paths().get('max_parallel_scenes', 1)
    num_cores_to_use = min(num_cores_to_use, multiprocessing.cpu_count(), len(all_grouped_tasks))
    
    total_saved_instances = 0
    total_failed_instances = 0
    
    with multiprocessing.Pool(processes=num_cores_to_use, 
                              initializer=configure_worker_logging, 
                              initargs=(log_queue,)) as pool:
        for list_of_results in pool.imap_unordered(process_scene_with_tunings_worker, all_grouped_tasks):
            for result_tuple in list_of_results:
                scene_idx, scene_name, dict_of_arrays, output_dir, config_path, error = result_tuple
                if error:
                    # Log the error. The listener will print it.
                    logging.error(f"Worker task failed for Scene '{scene_name}'", exc_info=error)
                    total_failed_instances += 1
                    continue
                if save_and_log_result(result_tuple):
                    total_saved_instances += 1
                else:
                    total_failed_instances += 1

    # --- Clean Shutdown ---
    log_queue.put(None) # Signal logging listener to stop
    logging_listener.join() # Wait for it to finish
    ui_process.join() # Wait for the UI to finish

    # --- Final Summary ---
    total_duration = time.time() - overall_start_time
    # Use a final print statement as the rich UI is now closed.
    print("\n--- Run Complete ---")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours).")
    print(f"Total successful scene-tuning instances saved: {total_saved_instances}.")
    if total_failed_instances > 0:
        print(f"Total failed instances: {total_failed_instances} (check logs for details).")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    IS_DEBUG_MODE = False 
    main(verbose=IS_DEBUG_MODE)