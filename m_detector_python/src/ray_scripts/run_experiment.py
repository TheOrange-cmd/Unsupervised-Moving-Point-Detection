# src/ray_scripts/run_experiment.py 

# Filter annoying warnings
import warnings
warnings.filterwarnings("ignore", message="The cuda.cuda module is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="The cuda.cudart module is deprecated", category=FutureWarning)

import os
import argparse
import ray
import logging
from pathlib import Path
import time
from rich.logging import RichHandler

from ..config_loader import MDetectorConfigAccessor
from .modes import bake_runner, refinement_tuner, full_tuner
from .ray_actors import NuScenesDataActor, load_all_gt_data_in_background

# # toggle this to see all logs for all trials - 
# can lead to an explosion of terminal output when running big experiments 
# os.environ["RAY_DEDUP_LOGS"] = "0" 

# --- Setup Paths and Environment ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKING_DIR = PROJECT_ROOT.parent
CONFIG_PATH_ABSOLUTE = WORKING_DIR / 'config' / 'm_detector_config.yaml'
DB_DIR = WORKING_DIR / "optuna_studies"

def main():
    parser = argparse.ArgumentParser(description="M-Detector Experiment Runner")
    parser.add_argument("--mode", type=str, required=True, choices=['bake', 'tune-refinement', 'tune-full'], help="The mode to run.")
    parser.add_argument("--study-name", type=str, required=True, help="Name for the Optuna study.")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials for tuning modes.")
    parser.add_argument("--source-study-name", type=str, help="For 'bake' mode: the geometric study to get best params from.")
    parser.add_argument("--bake-id", type=str, help="For 'tune-refinement' mode: the ID of the baked results to use.")
    args = parser.parse_args()

    # --- Centralized Logging Configuration ---
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)] # markup=True is default but explicit
    )
    # Get a logger for this file, which will now use the RichHandler.
    logger = logging.getLogger(__name__)

    accessor = MDetectorConfigAccessor(CONFIG_PATH_ABSOLUTE)
    base_config = accessor.get_raw_config()
    base_config['study_name'] = args.study_name

    # Configure CUDA_VISIBLE_DEVICES based on config ---
    processing_settings = accessor.get_processing_settings()
    if processing_settings['device'] == 'cuda':
        gpu_ids = processing_settings['gpu_ids']
        if gpu_ids: # If gpu_ids is not empty
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            logger.info(f"Setting CUDA_VISIBLE_DEVICES=[bold yellow]{os.environ['CUDA_VISIBLE_DEVICES']}[/bold yellow]")
        else:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            logger.info("CUDA_VISIBLE_DEVICES not explicitly set, Ray will use all available GPUs.")

    ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)
    nuscenes_cfg = accessor.get_nuscenes_params()
    config_path_str = str(CONFIG_PATH_ABSOLUTE)
    
    data_actor = NuScenesDataActor.options(name="nuscenes_data_service", get_if_exists=True).remote(
        nuscenes_cfg['version'], nuscenes_cfg['dataroot'], config_path_str
    )
    gt_data_future = load_all_gt_data_in_background.remote(config_path_str)
    ray.get(data_actor.set_cache.remote(ray.get(gt_data_future)))
    logger.info("Ray initialized and data actor is ready.")

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")

    try:
        if args.mode == 'bake':
            if not args.source_study_name:
                raise ValueError("--source-study-name is required for 'bake' mode.")
            bake_runner.run(base_config, args.source_study_name, DB_DIR)

        elif args.mode == 'tune-refinement':
            if not args.bake_id:
                raise ValueError("--bake-id is required for 'tune-refinement' mode.")
            refinement_tuner.run(base_config, args.study_name, args.n_trials, args.bake_id, DB_DIR, run_timestamp)

        elif args.mode == 'tune-full':
            full_tuner.run(base_config, args.study_name, args.n_trials, run_timestamp, DB_DIR)
            
    finally:
        ray.shutdown()
        logger.info("Ray shut down.")

if __name__ == '__main__':
    main()