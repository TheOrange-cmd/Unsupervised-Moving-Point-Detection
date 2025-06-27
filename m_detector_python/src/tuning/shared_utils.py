# src/tuning/shared_utils.py

import logging
from pathlib import Path

def deep_update_dict(base_dict, update_dict):
    """Recursively updates a dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update_dict(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def calculate_iou(tp: int, fp: int, fn: int) -> float:
    """Calculates IoU from true positives, false positives, and false negatives."""
    denominator = tp + fp + fn
    return tp / denominator if denominator > 0 else 0.0

def setup_trial_file_logging(logger_name: str, study_name: str, run_timestamp: str, trial_number: int, logging_config: dict):
    """
    Sets up structured, ISOLATED file logging for a specific trial worker.
    This logger outputs to a file to prevent clogging the terminal output. 
    """
    log_dir = Path(logging_config["log_dir"])
    study_log_dir = log_dir / study_name / run_timestamp
    study_log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get the SPECIFIC logger for this worker.
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) # Capture all levels of messages for the files.

    # 2. IMPORTANT: Prevent messages from bubbling up to the root logger.
    # This stops worker logs from appearing in the main script's console.
    logger.propagate = False

    # 3. Clear any existing handlers from previous runs in the same reused worker process.
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. Create Summary File Handler (INFO level)
    # This file gets a high-level summary from all trials.
    summary_handler = logging.FileHandler(study_log_dir / "run_summary.log", mode='a')
    summary_handler.setLevel(logging.INFO)
    summary_formatter = logging.Formatter(f'%(asctime)s - T{trial_number:03d} - %(levelname)s - %(message)s')
    summary_handler.setFormatter(summary_formatter)
    logger.addHandler(summary_handler)

    # 5. Create Detailed Trial-Specific File Handler (DEBUG level)
    detailed_trials = logging_config["log_detailed_for_trials"]
    if trial_number in detailed_trials or "all" in detailed_trials:
        detailed_handler = logging.FileHandler(study_log_dir / f"detailed_trial_{trial_number}.log", mode='w')
        detailed_handler.setLevel(logging.DEBUG)
        detailed_formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
        detailed_handler.setFormatter(detailed_formatter)
        logger.addHandler(detailed_handler)

    # NOTE: The console handler has been intentionally removed.

    # Silence other verbose libraries for this worker
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)