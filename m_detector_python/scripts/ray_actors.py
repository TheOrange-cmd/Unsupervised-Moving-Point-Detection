import ray
import os
import logging
import time
import torch
import traceback 
import time # for timestamp string
import numpy as np
from pyquaternion import Quaternion

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.nuscenes import NuScenes

from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich.console import Console

# Import project-specific classes
from src.core.m_detector.base import MDetector
from src.data_utils.nuscenes_helper import NuScenesProcessor
from src.config_loader import MDetectorConfigAccessor

@ray.remote
class ProgressActor:
    """A UI actor that shows overall progress and the live status of each worker."""
    def __init__(self, total_tasks: int, logging_actor: ray.actor.ActorHandle):
        self.logging_actor = logging_actor
        self.console = Console(stderr=True)
        self.total_tasks = total_tasks
        self.worker_status = {}
        self.shutdown_event = False

        # --- UI Components ---
        self.layout = Table.grid(expand=True, padding=(0, 1))
        self.layout.add_column(ratio=1)

        self.overall_progress = Progress(
            TextColumn("[bold green]Overall Progress"),
            BarColumn(),
            TextColumn("{task.completed} / {task.total} Tasks")
        )
        self.overall_task = self.overall_progress.add_task("Total", total=self.total_tasks)
        self.layout.add_row(self.overall_progress)

        self.worker_table = Table(title="Live Worker Status", show_header=True, header_style="bold magenta", expand=True)
        self.worker_table.add_column("Worker ID", justify="center", style="cyan")
        self.worker_table.add_column("PID", justify="center")
        self.worker_table.add_column("Status", justify="left", no_wrap=True, ratio=1)
        self.layout.add_row(self.worker_table)

        self.live = Live(self.layout, console=self.console, refresh_per_second=4)

    def update_status(self, worker_id: int, pid: int, status: str):
        """Called by a worker to report what it's currently doing."""
        self.worker_status[worker_id] = {"pid": pid, "status": status}
        # Also send a high-level log message
        self.logging_actor.log_summary.remote(logging.INFO, f"UI STATUS: Worker {worker_id} -> {status}")

    def task_complete(self):
        """Called by a worker after it completes one task."""
        self.overall_progress.update(self.overall_task, advance=1)

    def run(self):
        """Starts the Live display loop."""
        with self.live:
            while not self.shutdown_event:
                # Rebuild the table rows on each refresh
                self.worker_table.rows = []
                sorted_workers = sorted(self.worker_status.items())
                for worker_id, info in sorted_workers:
                    self.worker_table.add_row(str(worker_id), str(info['pid']), info['status'])
                time.sleep(0.25)

    def shutdown(self):
        self.shutdown_event = True


@ray.remote
class SceneProcessorActor:
    def __init__(self, worker_id: int, data_actor: ray.actor.ActorHandle,
                 progress_actor: ray.actor.ActorHandle, logging_actor: ray.actor.ActorHandle):
        self.worker_id = worker_id
        self.data_actor = data_actor
        self.progress_actor = progress_actor
        self.logging_actor = logging_actor
        self.logger_name = f"Worker-{worker_id}"
        self.device = torch.device("cuda:0")
        self.pid = os.getpid()

    def process_single_tuning(self, task_info: dict):
        scene_idx = task_info['scene_idx']
        tuning_name = task_info['tuning_name']
        config_path = task_info['config_path']
        output_dir = task_info['output_dir']

        try:
            # Create a NuScenesProcessor instance for this specific task
            # It no longer holds a nusc object, but the data_actor handle.
            processor = NuScenesProcessor(self.data_actor, processor.config_accessor)
            
            # The detector is created here as it's stateful per task
            detector = MDetector(config_accessor=processor.config_accessor, device=self.device)
            
            # The main processing call, now fully self-contained
            dict_of_arrays = processor.process_scene(scene_idx, detector)
            
            self.progress_actor.task_complete.remote()
            return (scene_idx, tuning_name, dict_of_arrays, output_dir, config_path, None)
            
        except Exception as e:
            # --- Log the error to BOTH files ---
            error_summary = f"WORKER FAILED on Scene {scene_idx}, Tuning '{tuning_name}'. Error: {e}"
            # High-level summary log
            self.logging_actor.log_summary.remote(logging.ERROR, error_summary)
            # Detailed log with full traceback
            detailed_error = f"{error_summary}\n{traceback.format_exc()}"
            self.logging_actor.log_detail.remote(logging.ERROR, detailed_error, self.logger_name)

            # Update UI and return
            error_msg_ui = f"FAILED on Scene {scene_idx} with Tuning {tuning_name}"
            self.progress_actor.update_status.remote(self.worker_id, self.pid, f"[bold red]{error_msg_ui}[/bold red]")
            self.progress_actor.task_complete.remote()
            return (scene_idx, tuning_name, None, output_dir, config_path, e)

        
@ray.remote
class LoggingActor:
    """An actor that handles writing all logs to files to prevent race conditions."""
    def __init__(self, log_dir: str, detail_level: int):
        # Create the log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # --- Summary Logger (for high-level info and errors) ---
        self.summary_logger = logging.getLogger("summary")
        self.summary_logger.setLevel(logging.INFO) # Always capture info and above
        summary_handler = logging.FileHandler(os.path.join(log_dir, f"run_summary{timestamp}.log"))
        summary_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        summary_handler.setFormatter(summary_formatter)
        self.summary_logger.addHandler(summary_handler)
        self.summary_logger.propagate = False

        # --- Detail Logger (for verbose, level-controlled info) ---
        self.detail_logger = logging.getLogger("detail")
        self.detail_logger.setLevel(detail_level) # Use the level from the config
        detail_handler = logging.FileHandler(os.path.join(log_dir, f"detailed_run{timestamp}.log"))
        detail_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        detail_handler.setFormatter(detail_formatter)
        self.detail_logger.addHandler(detail_handler)
        self.detail_logger.propagate = False
        
        self.log_summary(logging.INFO, "LoggingActor initialized.")

    def log_summary(self, level, message):
        self.summary_logger.log(level, message)

    def log_detail(self, level, message, logger_name="worker"):
        # Temporarily set the logger's name for this message
        original_name = self.detail_logger.name
        self.detail_logger.name = logger_name
        self.detail_logger.log(level, message)
        self.detail_logger.name = original_name

    def shutdown(self):
        logging.shutdown()

@ray.remote
class NuScenesDataActor:
    """
    A service actor that holds the single NuScenes instance and provides
    methods to access its data, preventing memory duplication.
    """
    def __init__(self, version: str, dataroot: str):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        # This print will confirm it's only initialized once
        print(f"NuScenesDataActor (PID: {os.getpid()}) initialized with NuScenes version {version}.")

    def get_scene_record(self, scene_index: int) -> dict:
        """Returns the scene record dictionary for a given index."""
        return self.nusc.scene[scene_index]
    
    def get_scene_count(self) -> int:
        """Returns the total number of scenes."""
        return len(self.nusc.scene)

    def _get_lidar_sweep_data(self, lidar_sd_token: str) -> dict:
        """
        Private helper that mirrors the logic from the old get_lidar_sweep_data.
        It returns a serializable dictionary instead of a complex tuple.
        """
        sweep_rec = self.nusc.get('sample_data', lidar_sd_token)
        cs_rec = self.nusc.get('calibrated_sensor', sweep_rec['calibrated_sensor_token'])
        pose_rec = self.nusc.get('ego_pose', sweep_rec['ego_pose_token'])
        pc_filepath = os.path.join(self.nusc.dataroot, sweep_rec['filename'])

        if not os.path.exists(pc_filepath):
            points_sensor_frame = np.empty((0, 5), dtype=np.float32)
        else:
            pc = LidarPointCloud.from_file(pc_filepath)
            points_sensor_frame = pc.points.T  # Shape (N, 5)

        sens_to_ego_rot = Quaternion(cs_rec['rotation']).rotation_matrix
        sens_to_ego_trans = np.array(cs_rec['translation'])
        T_sensor_ego = np.eye(4, dtype=np.float32)
        T_sensor_ego[:3, :3] = sens_to_ego_rot
        T_sensor_ego[:3, 3] = sens_to_ego_trans

        ego_to_glob_rot = Quaternion(pose_rec['rotation']).rotation_matrix
        ego_to_glob_trans = np.array(pose_rec['translation'])
        T_ego_global = np.eye(4, dtype=np.float32)
        T_ego_global[:3, :3] = ego_to_glob_rot
        T_ego_global[:3, 3] = ego_to_glob_trans

        T_global_sensor = T_ego_global @ T_sensor_ego

        return {
            'points_sensor_frame': points_sensor_frame[:, :3], # xyz for processing
            'T_global_lidar': T_global_sensor,
            'timestamp': sweep_rec['timestamp'],
            'calibrated_sensor_token': sweep_rec['calibrated_sensor_token'],
            'lidar_sd_token': lidar_sd_token,
            'is_key_frame': sweep_rec['is_key_frame'],
            'sample_token': sweep_rec['sample_token'],
        }

    def get_scene_sweep_sequence(self, scene_token: str, lidar_name: str) -> list:
        """
        Mirrors the logic from get_scene_sweep_data_sequence, but returns a
        list of dictionaries, which is easily passed between actors.
        """
        scene_rec = self.nusc.get('scene', scene_token)
        first_sample_token = scene_rec['first_sample_token']
        first_sample_rec = self.nusc.get('sample', first_sample_token)
        
        # Find the first valid LIDAR_TOP token in the scene
        current_sd_token = first_sample_rec['data'].get(lidar_name)
        if not current_sd_token:
            _s_token = first_sample_token
            while _s_token:
                _s_rec = self.nusc.get('sample', _s_token)
                if lidar_name in _s_rec['data']:
                    current_sd_token = _s_rec['data'][lidar_name]
                    break
                _s_token = _s_rec['next']
            if not current_sd_token:
                return []

        # Go to the first sweep in the sequence
        while True:
            sd_rec = self.nusc.get('sample_data', current_sd_token)
            if sd_rec['prev']:
                current_sd_token = sd_rec['prev']
            else:
                break
        
        # Iterate forwards and collect all sweep data
        all_sweeps = []
        while current_sd_token:
            sweep_data = self._get_lidar_sweep_data(current_sd_token)
            all_sweeps.append(sweep_data)
            current_sd_token = self.nusc.get('sample_data', current_sd_token)['next']
            
        return all_sweeps