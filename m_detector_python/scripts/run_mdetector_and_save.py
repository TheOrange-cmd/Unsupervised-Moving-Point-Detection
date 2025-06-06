# scripts/run_mdetector_and_save.py
import yaml
import os
import json
import numpy as np
from nuscenes.nuscenes import NuScenes
import sys
from tqdm import tqdm
import logging
import h5py
import multiprocessing
from functools import partial
import copy
import shutil
import time
import itertools # For generating combinations for systematic sweeps

# BASE OVERRIDE FOR ALL experiments in this run
# This incorporates your best finding as the new default for this tuning session.
NEW_BASELINE_OVERRIDES = {
    "m_detector": {
        "map_consistency": {
            "adaptive_epsilon_forward_config": {"dmin": 0.1} # Your best so far
        }
    }
}

def generate_tuning_experiments_v2(target_count=116):
    experiments = []

    # Helper to add experiment if name is unique
    def add_experiment(name, override_dict):
        # Ensure the NEW_BASELINE_OVERRIDES are part of this experiment's overrides
        # Start with a deepcopy of the new baseline
        final_overrides = copy.deepcopy(NEW_BASELINE_OVERRIDES)
        
        # Deeply merge the specific override_dict for this experiment
        # This function needs to handle nested keys properly
        def deep_merge(source, destination):
            for key, value in source.items():
                if isinstance(value, dict):
                    node = destination.setdefault(key, {})
                    deep_merge(value, node)
                else:
                    destination[key] = value
            return destination

        deep_merge(override_dict, final_overrides)
        
        # Check for uniqueness if desired, though for generated names it's usually fine
        experiments.append({"name": name, "overrides": final_overrides})

    # 0. Add the New Baseline Itself
    add_experiment("ref_mccFwdDmin0p1", {}) # Empty override as it's handled by NEW_BASELINE_OVERRIDES logic

    # --- Group 1: Occlusion Determination Epsilon (Base for Adaptive Occ) ---
    # Current in config: 0.1
    # Values to try: 0.07, 0.13 (2 experiments)
    for val in [0.07, 0.13]:
        add_experiment(f"Occ_BaseEps_{str(val).replace('.', 'p')}", 
                       {"m_detector": {"occlusion_determination": {"epsilon_depth": val}}})

    # --- Group 2: Systematic Adaptive Occlusion (`occlusion_determination.adaptive_epsilon_depth_config`) ---
    # Config defaults: dthr: 5.0, kthr: 0.01, dmax: 0.5, dmin: 0.05
    # We'll try 2 values for each, creating 2^4 = 16 experiments
    adap_occ_params = {
        "dthr": [3.0, 7.0],      # vs 5.0
        "kthr": [0.005, 0.015],  # vs 0.01
        "dmax": [0.3, 0.7],      # vs 0.5
        "dmin": [0.03, 0.07]     # vs 0.05
    }
    param_names = list(adap_occ_params.keys())
    for values_combo in itertools.product(*[adap_occ_params[k] for k in param_names]):
        current_adap_occ_override = dict(zip(param_names, values_combo))
        current_adap_occ_override["enabled"] = True # Ensure it's enabled
        name_parts = [f"{k}{str(v).replace('.', 'p')}" for k, v in zip(param_names, values_combo)]
        add_experiment(f"AdapOcc_{'_'.join(name_parts)}", 
                       {"m_detector": {"occlusion_determination": {"adaptive_epsilon_depth_config": current_adap_occ_override}}})
    # Count: 2 (Group 1) + 16 (Group 2) = 18 experiments so far (plus baseline = 19)

    # --- Group 3: Event Test Parameters ---
    # M2 (current 2): [1, 3] (2 exp)
    for val in [1, 3]:
        add_experiment(f"Event_M2_{val}", {"m_detector": {"event_detection_logic": {"test2_parallel_away": {"num_historical_DIs_M2": val}}}})
    # M3 (current 3): [2, 4] (2 exp)
    for val in [2, 4]:
        add_experiment(f"Event_M3_{val}", {"m_detector": {"event_detection_logic": {"test3_parallel_towards": {"num_historical_DIs_M3": val}}}})
    # N (current 10): [7, 13] (2 exp)
    for val in [7, 13]:
        add_experiment(f"Event_N_{val}", {"m_detector": {"event_detection_logic": {"test4_perpendicular": {"num_historical_DIs_N": val}}}})
    # M4 (current 4, assuming N=10): [3, 5] (2 exp)
    for val in [3, 5]: # These assume N is still around 10 for M4 to make sense
        add_experiment(f"Event_M4_{val}_N10", {"m_detector": {"event_detection_logic": {"test4_perpendicular": {"min_occluding_DIs_M4": val}}}})
    # Count: 19 + 8 = 27 experiments

    # --- Group 4: Map Consistency (General) ---
    # time_window_past_s (current 0.5): [0.25, 0.75] (2 exp)
    for val in [0.25, 0.75]:
        add_experiment(f"MCC_Time_{str(val).replace('.', 'p')}s", {"m_detector": {"map_consistency": {"time_window_past_s": val}}})
    # epsilon_phi_deg (current 0.57): [0.3, 0.8] (2 exp)
    for val in [0.3, 0.8]:
        add_experiment(f"MCC_EpsPhi_{str(val).replace('.', 'p')}", {"m_detector": {"map_consistency": {"epsilon_phi_deg": val}}})
    # epsilon_theta_deg (current 1.72): [1.0, 2.5] (2 exp)
    for val in [1.0, 2.5]:
        add_experiment(f"MCC_EpsTheta_{str(val).replace('.', 'p')}", {"m_detector": {"map_consistency": {"epsilon_theta_deg": val}}})
    # threshold_value_count (current 2): [1, 3] (2 exp)
    for val in [1, 3]:
        add_experiment(f"MCC_ThreshCount_{val}", {"m_detector": {"map_consistency": {"threshold_value_count": val}}})
    # interpolation_enabled (current true): [false] (1 exp)
    add_experiment("MCC_InterpDisabled", {"m_detector": {"map_consistency": {"interpolation_enabled": False}}})
    # interpolation_min_depth_m (current 15.0): [10.0, 20.0] (2 exp)
    for val in [10.0, 20.0]:
        add_experiment(f"MCC_InterpMinD_{str(val).replace('.', 'p')}", {"m_detector": {"map_consistency": {"interpolation_min_depth_m": val}}})
    # interpolation_max_neighbors_to_consider (current 20): [10, 30] (2 exp)
    for val in [10, 30]:
        add_experiment(f"MCC_InterpMaxN_{val}", {"m_detector": {"map_consistency": {"interpolation_max_neighbors_to_consider": val}}})
    # Count: 27 + 2+2+2+2+1+2+2 = 27 + 13 = 40 experiments

    # --- Group 5: Systematic Adaptive MCC Backward (`map_consistency.adaptive_epsilon_backward_config`) ---
    # Config defaults: dthr: 20.0, kthr: 0.03, dmax: 1.0, dmin: 0.05
    adap_mccB_params = {
        "dthr": [15.0, 25.0],     # vs 20.0
        "kthr": [0.015, 0.045],   # vs 0.03
        "dmax": [0.7, 1.3],       # vs 1.0
        "dmin": [0.025, 0.075]    # vs 0.05
    }
    param_names_mccB = list(adap_mccB_params.keys())
    for values_combo_mccB in itertools.product(*[adap_mccB_params[k] for k in param_names_mccB]):
        current_adap_mccB_override = dict(zip(param_names_mccB, values_combo_mccB))
        current_adap_mccB_override["enabled"] = True
        name_parts_mccB = [f"{k}{str(v).replace('.', 'p')}" for k, v in zip(param_names_mccB, values_combo_mccB)]
        add_experiment(f"AdapMCCB_{'_'.join(name_parts_mccB)}", 
                       {"m_detector": {"map_consistency": {"adaptive_epsilon_backward_config": current_adap_mccB_override}}})
    # Count: 40 + 16 = 56 experiments

    # --- Group 6: Systematic Adaptive MCC Forward (dthr, kthr, dmax) ---
    # dmin is fixed at 0.1 by NEW_BASELINE_OVERRIDES
    # Config defaults: dthr: 20.0, kthr: 0.07, dmax: 3.0
    adap_mccF_params = {
        "dthr": [15.0, 25.0],     # vs 20.0
        "kthr": [0.05, 0.09],     # vs 0.07
        "dmax": [2.5, 3.5]        # vs 3.0
    }
    param_names_mccF = list(adap_mccF_params.keys())
    for values_combo_mccF in itertools.product(*[adap_mccF_params[k] for k in param_names_mccF]):
        current_adap_mccF_override = dict(zip(param_names_mccF, values_combo_mccF))
        current_adap_mccF_override["enabled"] = True
        current_adap_mccF_override["dmin"] = 0.1 # Explicitly ensure dmin is set with these other changes
        name_parts_mccF = [f"{k}{str(v).replace('.', 'p')}" for k, v in zip(param_names_mccF, values_combo_mccF)]
        add_experiment(f"AdapMCCF_dmin0p1_{'_'.join(name_parts_mccF)}", 
                       {"m_detector": {"map_consistency": {"adaptive_epsilon_forward_config": current_adap_mccF_override}}})
    # Count: 56 + 2*2*2 = 56 + 8 = 64 experiments

    # --- Group 7: RANSAC Ground Parameters ---
    # xyradius_threshold (current 25.0): [15.0, 35.0] (2 exp)
    for val in [15.0, 35.0]:
        add_experiment(f"RANSAC_XYRad_{str(val).replace('.', 'p')}", {"m_detector": {"ransac_ground_params": {"xyradius_threshold": val}}})
    # num_trials (current 20): [10, 30] (2 exp)
    for val in [10, 30]:
        add_experiment(f"RANSAC_Trials_{val}", {"m_detector": {"ransac_ground_params": {"num_trials": val}}})
    # inlier_threshold (current 0.3): [0.2, 0.4] (2 exp)
    for val in [0.2, 0.4]:
        add_experiment(f"RANSAC_Inlier_{str(val).replace('.', 'p')}", {"m_detector": {"ransac_ground_params": {"inlier_threshold": val}}})
    # ground_threshold (current 0.3): [0.2, 0.4] (2 exp)
    for val in [0.2, 0.4]:
        add_experiment(f"RANSAC_GroundTh_{str(val).replace('.', 'p')}", {"m_detector": {"ransac_ground_params": {"ground_threshold": val}}})
    # Count: 64 + 8 = 72 experiments

    # --- Group 8: Initialization Phase ---
    # num_sweeps_for_initial_map (current 10): [5, 15, 20] (3 exp)
    for val in [5, 15, 20]:
        add_experiment(f"Init_Sweeps_{val}", {"m_detector": {"initialization_phase": {"num_sweeps_for_initial_map": val}}})
    # Count: 72 + 3 = 75 experiments

    # --- Group 9: Occlusion Determination (Pixel/Angular Neighborhoods) ---
    # pixel_neighborhood_h (current 2): [1, 3] (2 exp)
    for val in [1, 3]:
        add_experiment(f"Occ_PixelH_{val}", {"m_detector": {"occlusion_determination": {"pixel_neighborhood_h": val}}})
    # pixel_neighborhood_v (current 1): [0, 2] (2 exp)
    for val in [0, 2]:
        add_experiment(f"Occ_PixelV_{val}", {"m_detector": {"occlusion_determination": {"pixel_neighborhood_v": val}}})
    # angular_neighborhood_h_deg (current 0.29): [0.15, 0.45] (2 exp)
    for val in [0.15, 0.45]:
        add_experiment(f"Occ_AngularH_{str(val).replace('.', 'p')}", {"m_detector": {"occlusion_determination": {"angular_neighborhood_h_deg": val}}})
    # angular_neighborhood_v_deg (current 0.4): [0.2, 0.6] (2 exp)
    for val in [0.2, 0.6]:
        add_experiment(f"Occ_AngularV_{str(val).replace('.', 'p')}", {"m_detector": {"occlusion_determination": {"angular_neighborhood_v_deg": val}}})
    # Count: 75 + 8 = 83 experiments

    # --- Fill up to target_count with more variations or combinations ---
    # We have 116 - 1 (baseline) - 82 (current group experiments) = 33 slots left
    
    # Add more variations to Test4 (Perpendicular Event) as it's complex
    # N (current 10): add 5, 15 (2 more)
    for val in [5, 15]: # N=10 is default, 7 and 13 already added
        add_experiment(f"Event_N_{val}_v2", {"m_detector": {"event_detection_logic": {"test4_perpendicular": {"num_historical_DIs_N": val}}}})
    # M4 (current 4, for N=10): add 2, 6 (2 more)
    for val in [2, 6]: 
        add_experiment(f"Event_M4_{val}_N10_v2", {"m_detector": {"event_detection_logic": {"test4_perpendicular": {"min_occluding_DIs_M4": val}}}})
    # Count: 83 + 4 = 87

    # Try some combined variations for Test4:
    # N=7 with M4=[2,3,4] (3 exp)
    for m4_val in [2,3,4]:
         add_experiment(f"Event_N7_M4_{m4_val}", {"m_detector": {"event_detection_logic": {"test4_perpendicular": {"num_historical_DIs_N": 7, "min_occluding_DIs_M4": m4_val}}}})
    # N=13 with M4=[3,4,5,6] (4 exp)
    for m4_val in [3,4,5,6]:
         add_experiment(f"Event_N13_M4_{m4_val}", {"m_detector": {"event_detection_logic": {"test4_perpendicular": {"num_historical_DIs_N": 13, "min_occluding_DIs_M4": m4_val}}}})
    # Count: 87 + 3 + 4 = 94

    # More variations for MCC static labels (this is a bit manual to define variations)
    # Variation 1: Only OCCLUDED_BY_IMAGE
    add_experiment("MCC_StaticLabels_Strict", {"m_detector": {"map_consistency": {"static_labels_for_map_check": ["OCCLUDED_BY_IMAGE", "PRELABELED_STATIC_GROUND"]}}})
    # Variation 2: Add EMPTY_IN_IMAGE
    add_experiment("MCC_StaticLabels_WithEmpty", {"m_detector": {"map_consistency": {"static_labels_for_map_check": ["OCCLUDED_BY_IMAGE", "UNDETERMINED", "PRELABELED_STATIC_GROUND", "EMPTY_IN_IMAGE"]}}})
    # Count: 94 + 2 = 96

    # More variations for adaptive parameters (single param changes from their block's defaults)
    # Adaptive Occlusion (defaults: dthr: 5.0, kthr: 0.01, dmax: 0.5, dmin: 0.05)
    add_experiment(f"AdapOcc_Single_dthr_4p0", {"m_detector": {"occlusion_determination": {"adaptive_epsilon_depth_config": {"enabled":True, "dthr": 4.0}}}})
    add_experiment(f"AdapOcc_Single_kthr_0p02", {"m_detector": {"occlusion_determination": {"adaptive_epsilon_depth_config": {"enabled":True, "kthr": 0.02}}}})
    add_experiment(f"AdapOcc_Single_dmax_0p4", {"m_detector": {"occlusion_determination": {"adaptive_epsilon_depth_config": {"enabled":True, "dmax": 0.4}}}})
    add_experiment(f"AdapOcc_Single_dmin_0p06", {"m_detector": {"occlusion_determination": {"adaptive_epsilon_depth_config": {"enabled":True, "dmin": 0.06}}}})
    # Count: 96 + 4 = 100

    # Adaptive MCC Backward (defaults: dthr: 20.0, kthr: 0.03, dmax: 1.0, dmin: 0.05)
    add_experiment(f"AdapMCCB_Single_dthr_18p0", {"m_detector": {"map_consistency": {"adaptive_epsilon_backward_config": {"enabled":True, "dthr": 18.0}}}})
    add_experiment(f"AdapMCCB_Single_kthr_0p02", {"m_detector": {"map_consistency": {"adaptive_epsilon_backward_config": {"enabled":True, "kthr": 0.02}}}})
    add_experiment(f"AdapMCCB_Single_dmax_0p8", {"m_detector": {"map_consistency": {"adaptive_epsilon_backward_config": {"enabled":True, "dmax": 0.8}}}})
    add_experiment(f"AdapMCCB_Single_dmin_0p06", {"m_detector": {"map_consistency": {"adaptive_epsilon_backward_config": {"enabled":True, "dmin": 0.06}}}})
    # Count: 100 + 4 = 104

    # Adaptive MCC Forward (dmin fixed at 0.1; defaults: dthr: 20.0, kthr: 0.07, dmax: 3.0)
    add_experiment(f"AdapMCCF_Single_dthr_18p0", {"m_detector": {"map_consistency": {"adaptive_epsilon_forward_config": {"enabled":True, "dmin":0.1, "dthr": 18.0}}}})
    add_experiment(f"AdapMCCF_Single_kthr_0p06", {"m_detector": {"map_consistency": {"adaptive_epsilon_forward_config": {"enabled":True, "dmin":0.1, "kthr": 0.06}}}})
    add_experiment(f"AdapMCCF_Single_dmax_2p8", {"m_detector": {"map_consistency": {"adaptive_epsilon_forward_config": {"enabled":True, "dmin":0.1, "dmax": 2.8}}}})
    # Count: 104 + 3 = 107

    # Fill remaining ~9 slots with more single param variations
    # Example:
    add_experiment(f"Occ_BaseEps_0p085", {"m_detector": {"occlusion_determination": {"epsilon_depth": 0.085}}})
    add_experiment(f"Occ_BaseEps_0p115", {"m_detector": {"occlusion_determination": {"epsilon_depth": 0.115}}})
    add_experiment(f"Event_M2_v2", {"m_detector": {"event_detection_logic": {"test2_parallel_away": {"num_historical_DIs_M2": 2}}}}) # Re-test default M2 with new baseline
    add_experiment(f"Event_M3_v2", {"m_detector": {"event_detection_logic": {"test3_parallel_towards": {"num_historical_DIs_M3": 3}}}}) # Re-test default M3
    add_experiment(f"MCC_Time_0p5s_v2", {"m_detector": {"map_consistency": {"time_window_past_s": 0.5}}}) # Re-test default
    add_experiment(f"MCC_EpsPhi_0p57_v2", {"m_detector": {"map_consistency": {"epsilon_phi_deg": 0.57}}})
    add_experiment(f"MCC_EpsTheta_1p72_v2", {"m_detector": {"map_consistency": {"epsilon_theta_deg": 1.72}}})
    add_experiment(f"RANSAC_Inlier_0p3_v2", {"m_detector": {"ransac_ground_params": {"inlier_threshold": 0.3}}})
    add_experiment(f"Init_Sweeps_10_v2", {"m_detector": {"initialization_phase": {"num_sweeps_for_initial_map": 10}}})
    # Count: 107 + 9 = 116

    print(f"Generated {len(experiments)} tuning configurations.")
    if len(experiments) > target_count:
        print(f"Warning: Generated {len(experiments)}, which is more than target {target_count}. Consider trimming.")
    elif len(experiments) < target_count:
        print(f"Info: Generated {len(experiments)}, which is less than target {target_count}. You can add more variations.")
        
    return experiments[:target_count] # Ensure we don't exceed the target


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    tqdm.write(f"Added project root to sys.path: {PROJECT_ROOT}")

progress_logger = logging.getLogger('ProgressLogger')
progress_logger.setLevel(logging.INFO)
progress_logger.propagate = False
logs_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(logs_dir, exist_ok=True)
progress_log_file_path = os.path.join(logs_dir, "progress_summary.log")
progress_file_handler = logging.FileHandler(progress_log_file_path, mode='w')
progress_file_handler.setLevel(logging.INFO)
progress_formatter = logging.Formatter('%(asctime)s - %(message)s')
progress_file_handler.setFormatter(progress_formatter)
progress_logger.addHandler(progress_file_handler)

from src.core.m_detector.base import MDetector
from src.data_utils.nuscenes_helper import NuScenesProcessor
from src.config_loader import MDetectorConfigAccessor

# Global for workers
worker_nusc_instance = None
worker_nusc_version = None
worker_nusc_dataroot = None

def init_worker(version, dataroot, verbose_load):
    global worker_nusc_instance, worker_nusc_version, worker_nusc_dataroot
    # Only load if it's not already loaded or if params changed (less likely with imap)
    if worker_nusc_instance is None or worker_nusc_version != version or worker_nusc_dataroot != dataroot:
        logging.info(f"Worker {os.getpid()}: Initializing NuScenes {version} from {dataroot}")
        worker_nusc_instance = NuScenes(version=version, dataroot=dataroot, verbose=verbose_load)
        worker_nusc_version = version
        worker_nusc_dataroot = dataroot
    else:
        logging.info(f"Worker {os.getpid()}: Reusing existing NuScenes instance.")

def represent_none(dumper, _):
    return dumper.represent_scalar('tag:yaml.org,2002:null', 'null')

class NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpySafeEncoder, self).default(obj)

def load_base_config(config_path_absolute: str) -> dict:
    try:
        with open(config_path_absolute, 'r') as f: return yaml.safe_load(f)
    except Exception as e:
        tqdm.write(f"FATAL: Error loading base YAML config file from {config_path_absolute}: {e}")
        raise

def deep_update_dict(base_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update_dict(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def process_single_scene_worker(scene_idx: int,
                                tuned_config_file_path: str,
                                nuscenes_version: str,
                                nuscenes_dataroot: str,
                                nuscenes_verbose_load: bool,
                                output_dir_for_tuning: str):
    try:
        config_accessor = MDetectorConfigAccessor(tuned_config_file_path)
        nusc_worker = worker_nusc_instance 
        detector_worker = MDetector(config_accessor=config_accessor)
        processor_worker = NuScenesProcessor(nusc_worker, config_accessor=config_accessor)
        scene_record = nusc_worker.scene[scene_idx]
        scene_name = scene_record['name']
        dict_of_arrays_for_scene = processor_worker.process_scene(
            scene_index=scene_idx, detector=detector_worker, with_progress=False
        )
        if dict_of_arrays_for_scene is None:
            return scene_idx, scene_name, None, output_dir_for_tuning, tuned_config_file_path # Added tuned_config_file_path for logging
        try:
            raw_config_for_saving = config_accessor.get_raw_config()
            config_json_str = json.dumps(raw_config_for_saving, sort_keys=True, indent=4, cls=NumpySafeEncoder)
            dict_of_arrays_for_scene['_config_json_str'] = np.array(config_json_str)
        except Exception as e_json_dump:
            logging.error(f"Worker (scene_idx {scene_idx}): Error serializing config to JSON: {e_json_dump}", exc_info=True)
            error_config_dict = {"error_while_saving_config": True, "original_serialization_error": str(e_json_dump)}
            config_json_str_error = json.dumps(error_config_dict)
            dict_of_arrays_for_scene['_config_json_str'] = np.array(config_json_str_error)
        return scene_idx, scene_name, dict_of_arrays_for_scene, output_dir_for_tuning, tuned_config_file_path # Added tuned_config_file_path
    except Exception as e:
        logging.error(f"Error processing scene_idx {scene_idx} in worker with config {tuned_config_file_path}: {e}", exc_info=True)
        failed_scene_name = f"scene_idx_{scene_idx}" # Default
        if worker_nusc_instance: # Check if the global instance exists
            try:
                failed_scene_name = worker_nusc_instance.scene[scene_idx]['name']
            except Exception as e_name:
                logging.error(f"Could not get scene name for failed idx {scene_idx} from worker nusc instance: {e_name}")
        return scene_idx, failed_scene_name, None, output_dir_for_tuning, tuned_config_file_path

def global_worker_wrapper(task_info_tuple):
    scene_idx, tuned_config_path, output_dir, base_nuscenes_version, base_nuscenes_dataroot = task_info_tuple
    return process_single_scene_worker(
        scene_idx=scene_idx,
        tuned_config_file_path=tuned_config_path,
        nuscenes_version=base_nuscenes_version,
        nuscenes_dataroot=base_nuscenes_dataroot,
        nuscenes_verbose_load=False,
        output_dir_for_tuning=output_dir
    )

def save_single_result(scene_idx, scene_name, dict_of_arrays_for_scene, task_output_dir, config_path_for_log):
    """
    Saves the result for a single processed scene.
    Returns True if save was successful, False otherwise.
    """
    if dict_of_arrays_for_scene is None:
        # Log failure for this specific task to progress_logger and main logger
        log_msg = f"Task failed (no data returned): Scene '{scene_name}' (idx {scene_idx}), Config: {os.path.basename(config_path_for_log)}"
        progress_logger.warning(log_msg)
        tqdm.write(log_msg) # Keep tqdm.write for immediate console feedback if desired
        return False

    output_filename_h5 = f"mdet_results_{scene_name}.h5"
    output_filepath_h5 = os.path.join(task_output_dir, output_filename_h5)

    try:
        with h5py.File(output_filepath_h5, 'w') as hf:
            for key, array_data in dict_of_arrays_for_scene.items():
                if isinstance(array_data, np.ndarray):
                    if array_data.ndim == 0: hf.create_dataset(key, data=array_data.item())
                    else: hf.create_dataset(key, data=array_data)
                elif isinstance(array_data, (str, bytes, int, float, bool)):
                    hf.create_dataset(key, data=array_data)
                else:
                    try:
                        converted_array = np.array(array_data)
                        hf.create_dataset(key, data=converted_array)
                    except Exception as e_conv:
                        tqdm.write(f"  Warning: Could not save key '{key}' for scene '{scene_name}' (type: {type(array_data)}). Error: {e_conv}")
        
        num_frames_saved = len(dict_of_arrays_for_scene.get('sweep_lidar_sd_tokens', []))
        log_msg = f"Successfully saved HDF5 ({num_frames_saved} frames) for Scene '{scene_name}' (idx {scene_idx}), Config: {os.path.basename(config_path_for_log)} to {output_filepath_h5}"
        progress_logger.info(log_msg)
        tqdm.write(log_msg)
        return True
    except Exception as e:
        log_msg = f"Error saving HDF5 for Scene '{scene_name}' (idx {scene_idx}), Config: {os.path.basename(config_path_for_log)}. Error: {e}"
        progress_logger.error(log_msg)
        tqdm.write(log_msg)
        return False

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    # --- (Your existing logger level settings) ---
    logging.getLogger('src.core.m_detector.base').setLevel(logging.INFO)
    logging.getLogger('src.core.m_detector.processing').setLevel(logging.INFO)
    logging.getLogger('src.core.m_detector.map_consistency').setLevel(logging.INFO)
    logging.getLogger('src.core.m_detector.interpolation_utils').setLevel(logging.INFO)

    yaml.add_representer(type(None), represent_none, Dumper=yaml.SafeDumper)

    progress_logger.info("=== M-Detector Processing Run Started ===")
    overall_start_time = time.time()

    base_config_file_path_relative = 'config/m_detector_config.yaml'
    base_config_file_path_absolute = os.path.join(PROJECT_ROOT, base_config_file_path_relative) if not os.path.isabs(base_config_file_path_relative) and PROJECT_ROOT else base_config_file_path_relative
    
    progress_logger.info(f"Loading BASE config from: {base_config_file_path_absolute}")
    tqdm.write(f"Loading BASE config from: {base_config_file_path_absolute}")
    base_config_dict = load_base_config(base_config_file_path_absolute)

    tuning_experiments = generate_tuning_experiments_v2(target_count=2)
    # tuning_experiments = [] # don't try any tunings, just use the config as is. 

    if not tuning_experiments:
        tuning_experiments = [{"name": "baseline_best_config", "overrides": {}}]
        print("No tuning experiments specified. Running with base config only.")

    temp_base_config_accessor = MDetectorConfigAccessor(base_config_file_path_absolute)
    mdet_output_cfg_base = temp_base_config_accessor.get_mdetector_output_paths()
    base_nuscenes_cfg = temp_base_config_accessor.get_nuscenes_params()
    main_save_path = mdet_output_cfg_base.get('save_path')
    if not os.path.isabs(main_save_path) and PROJECT_ROOT:
        main_save_path = os.path.join(PROJECT_ROOT, main_save_path)

    try:
        tqdm.write("Loading main NuScenes object for task preparation...")
        nusc_main = NuScenes(version=base_nuscenes_cfg.get('version'), 
                             dataroot=base_nuscenes_cfg.get('dataroot'), 
                             verbose=False)
        tqdm.write("NuScenes object loaded.")
    except Exception as e_nusc_main_load:
        tqdm.write(f"FATAL: Could not load main NuScenes object. Exiting. Error: {e_nusc_main_load}")
        return

    all_processing_tasks = []

    progress_logger.info(f"Preparing {len(tuning_experiments)} tuning configurations...")
    tqdm.write("Preparing tuning configurations and output directories...")
    for i, tuning_experiment in enumerate(tqdm(tuning_experiments, desc="Preparing Tunings")):
        tuning_name = tuning_experiment["name"]
        progress_logger.info(f"  Preparing tuning [{i+1}/{len(tuning_experiments)}]: {tuning_name}")
        overrides = tuning_experiment["overrides"]
        current_tuned_config_dict = copy.deepcopy(base_config_dict)
        deep_update_dict(current_tuned_config_dict, overrides)
        output_dir_for_this_tuning = os.path.join(main_save_path, tuning_name)
        os.makedirs(output_dir_for_this_tuning, exist_ok=True)
        tuned_config_filename = f"config_tuned_{tuning_name}.yaml"
        tuned_config_filepath = os.path.join(output_dir_for_this_tuning, tuned_config_filename)
        try:
            with open(tuned_config_filepath, 'w') as f_tuned:
                yaml.dump(current_tuned_config_dict, f_tuned, sort_keys=False, Dumper=yaml.SafeDumper)
        except Exception as e_yaml_save:
            tqdm.write(f"Error saving tuned YAML for '{tuning_name}': {e_yaml_save}. Skipping this tuning.")
            progress_logger.error(f"Error saving tuned YAML for '{tuning_name}': {e_yaml_save}. Skipping.")
            continue
        
        # tuned_nuscenes_cfg = current_tuned_config_dict.get('nuscenes', {})
        # nusc_version_for_list = tuned_nuscenes_cfg.get('version', base_nuscenes_cfg.get('version'))
        # nusc_dataroot_for_list = tuned_nuscenes_cfg.get('dataroot', base_nuscenes_cfg.get('dataroot'))
        # try:
        #     nusc_temp_for_list = NuScenes(version=nusc_version_for_list, dataroot=nusc_dataroot_for_list, verbose=False)
        # except Exception as e_nusc_load:
        #     tqdm.write(f"Error loading NuScenes for tuning '{tuning_name}' to get scene list: {e_nusc_load}. Skipping.")
        #     progress_logger.error(f"Error loading NuScenes for tuning '{tuning_name}': {e_nusc_load}. Skipping.")
        #     continue

        tuned_mdet_output_cfg = current_tuned_config_dict.get('mdetector_output_paths', {})
        scene_indices_to_process_config = current_tuned_config_dict.get('mdetector_output_paths', {}).get('scene_indices_to_run', 'all')
        if isinstance(scene_indices_to_process_config, str) and scene_indices_to_process_config.lower() == 'all':
            current_scene_indices_list = list(range(len(nusc_main.scene)))
        elif isinstance(scene_indices_to_process_config, list):
            current_scene_indices_list = scene_indices_to_process_config
        else:
            tqdm.write(f"Invalid 'scene_indices_to_run' for '{tuning_name}'. Defaulting to all.")
            progress_logger.warning(f"Invalid 'scene_indices_to_run' for '{tuning_name}'. Defaulting to all.")
            current_scene_indices_list = list(range(len(nusc_main.scene)))

        for scene_idx in current_scene_indices_list:
            if isinstance(scene_idx, int) and 0 <= scene_idx < len(nusc_main.scene):
                all_processing_tasks.append(
                    (scene_idx, tuned_config_filepath, output_dir_for_this_tuning,
                     base_nuscenes_cfg.get('version'), base_nuscenes_cfg.get('dataroot'))
                )
            else:
                tqdm.write(f"Skipping invalid scene_idx {scene_idx} for tuning '{tuning_name}'")
                progress_logger.warning(f"Skipping invalid scene_idx {scene_idx} for tuning '{tuning_name}'")
    
    if not all_processing_tasks:
        progress_logger.warning("No valid processing tasks generated. Exiting.")
        tqdm.write("No valid processing tasks generated. Exiting.")
        return

    num_cores_to_use = mdet_output_cfg_base.get('max_parallel_scenes', 1)
    num_cores_to_use = max(1, min(num_cores_to_use, multiprocessing.cpu_count(), len(all_processing_tasks)))
    
    progress_logger.info(f"Total processing tasks to run: {len(all_processing_tasks)}")
    progress_logger.info(f"Using up to {num_cores_to_use} parallel processes.")
    tqdm.write(f"Total tasks to process: {len(all_processing_tasks)}. Using up to {num_cores_to_use} parallel processes.")

    total_saved_scenes = 0
    total_failed_tasks = 0
    tasks_processed_count = 0 # Renamed from tasks_completed_count for clarity
    pool_initargs = (base_nuscenes_cfg.get('version'), base_nuscenes_cfg.get('dataroot'), False)

    with multiprocessing.Pool(processes=num_cores_to_use, initializer=init_worker, initargs=pool_initargs) as pool:
        with tqdm(total=len(all_processing_tasks), desc="Overall Task Progress") as pbar:
            for worker_result_tuple in pool.imap_unordered(global_worker_wrapper, all_processing_tasks):
                # Unpack the result from the worker
                # Ensure worker returns config_path for logging if it's not already part of worker_result_tuple
                scene_idx, scene_name, dict_of_arrays_for_scene, task_output_dir, config_path_for_log = worker_result_tuple
                
                # Save the result immediately
                save_successful = save_single_result(
                    scene_idx, scene_name, dict_of_arrays_for_scene, task_output_dir, config_path_for_log
                )
                if save_successful:
                    total_saved_scenes += 1
                else:
                    total_failed_tasks += 1
                
                pbar.update(1)
                tasks_processed_count += 1
                if tasks_processed_count % 10 == 0 or tasks_processed_count == len(all_processing_tasks):
                    progress_logger.info(f"Task processing progress: {tasks_processed_count}/{len(all_processing_tasks)} tasks processed (saved/failed).")

    progress_logger.info("All worker processes and incremental saving finished.")
    tqdm.write("\nAll worker processes and incremental saving finished.") # Keep for console

    overall_end_time = time.time()
    total_duration_seconds = overall_end_time - overall_start_time
    progress_logger.info(f"=== M-Detector Processing Run Finished ===")
    progress_logger.info(f"Total duration: {total_duration_seconds:.2f} seconds ({total_duration_seconds/3600:.2f} hours).")
    progress_logger.info(f"Total successful scene-tuning instances saved: {total_saved_scenes}.")
    if total_failed_tasks > 0:
        progress_logger.warning(f"Total failed tasks: {total_failed_tasks} (check detailed logs).")
    
    # Final tqdm write for console
    tqdm.write(f"\nAll tuning experiments and scene processing complete.")
    tqdm.write(f"Total successful scene-tuning instances saved: {total_saved_scenes}.")
    if total_failed_tasks > 0:
        tqdm.write(f"Total failed tasks: {total_failed_tasks} (check logs for details).")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()