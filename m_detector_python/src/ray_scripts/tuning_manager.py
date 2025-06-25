# src/ray_scripts/tuning_manager.py

import optuna
from ..core.m_detector.refinement_algorithms import ClusteringAlgorithm
from .shared_utils import deep_update_dict

def define_geometric_search_space(trial: optuna.Trial) -> dict:
    """Defines the search space for the expensive geometric parameters."""
    overrides = {}
    # These parameters are critical as they happen first.
    overrides['ransac_ground_params'] = {
        # How close a point must be to a candidate plane during trial fits.
        "inlier_threshold": trial.suggest_float("ransac_inlier_thresh", 0.1, 0.5),
        # How close a point must be to the FINAL best plane to be labeled ground.
        "ground_threshold": trial.suggest_float("ransac_ground_thresh", 0.1, 0.5),
        # Define the vertical window to search for ground.
        "z_min_threshold": trial.suggest_float("ransac_z_min", -3.0, -2.0),
        "z_max_threshold": trial.suggest_float("ransac_z_max", -1.0, 0.0),
    }

    # === Occlusion Determination (Initial Detection) ===
    overrides['occlusion_determination'] = {
        "epsilon_depth": trial.suggest_float("occ_epsilon_depth", 0.1, 0.25),
        "pixel_neighborhood_h": 2,  # Fixed
        "pixel_neighborhood_v": 2,  # Fixed
    }

    overrides['detailed_occlusion_check'] = {
        "epsilon_depth": trial.suggest_float("detail_epsilon_depth", 0.1, 0.25),
        "angular_neighborhood_h_deg": trial.suggest_float("detail_angular_h", 0.05, 0.2), # Narrow, precise range
        "angular_neighborhood_v_deg": trial.suggest_float("detail_angular_v", 0.1, 0.4),  # Narrow, precise range
    }
    
    # Allow the optimizer to enable/disable adaptive epsilon for occlusion checks.
    if trial.suggest_categorical("occ_adaptive_enabled", [True, False]):
        overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {
            "enabled": True,
            "dthr": trial.suggest_float("occ_adaptive_dthr", 5.0, 20.0),
            "kthr": trial.suggest_float("occ_adaptive_kthr", 0.01, 0.1, log=True),
            "dmax": trial.suggest_float("occ_adaptive_dmax", 0.3, 1.0),
            "dmin": 0.05  # Fixed safety net
        }
    else:
        overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {"enabled": False}

    # === Event Detection Logic ===
    # Keep these ranges wide as well.
    event_n = trial.suggest_int("event_N", 15, 25)
    overrides['event_detection_logic'] = {
        "test2_parallel_away": {"num_historical_DIs_M2": trial.suggest_int("event_M2", 3, 10)},
        "test3_parallel_towards": {"num_historical_DIs_M3": 1}, # Fixed
        "test4_perpendicular": {
            "num_historical_DIs_N": event_n,
            "min_occluding_DIs_M4": trial.suggest_int("event_M4", 5, 10)
        }
    }

    # === Map Consistency Check (MCC) ===
    overrides['map_consistency'] = {
        "enabled": True,  
        "interpolation_enabled": False,  # Keep disabled for faster tuning
        "num_past_sweeps_for_mcc": trial.suggest_int("mcc_num_sweeps", 15, 25),
        
        # Use lenient angular windows to give the optimizer a good starting point.
        "epsilon_phi_deg": trial.suggest_float("mcc_eps_phi", 0.1, 0.5),
        "epsilon_theta_deg": trial.suggest_float("mcc_eps_theta", 0.5, 1.5),

        # A low value is lenient (few points marked static).
        # A high value is strict (many points marked static).
        "static_confidence_threshold": trial.suggest_float("mcc_score_thresh", 0.1, 0.9),
    }
    
    # Also allow tuning of the adaptive epsilon for the MCC, using forgiving ranges.
    overrides['map_consistency']['adaptive_epsilon_forward_config'] = {
        "enabled": True,
        "dthr": trial.suggest_float("mcc_fwd_adaptive_dthr", 15.0, 30.0),
        "kthr": trial.suggest_float("mcc_fwd_adaptive_kthr", 0.01, 0.05, log=True),
        "dmax": trial.suggest_float("mcc_fwd_adaptive_dmax", 0.2, 0.8),
        "dmin": trial.suggest_float("mcc_fwd_adaptive_dmin", 0.01, 0.1)
    }
    overrides['map_consistency']['adaptive_epsilon_backward_config'] = {
        "enabled": True,
        "dthr": trial.suggest_float("mcc_bwd_adaptive_dthr", 15.0, 30.0),
        "kthr": trial.suggest_float("mcc_bwd_adaptive_kthr", 0.01, 0.05, log=True),
        "dmax": trial.suggest_float("mcc_bwd_adaptive_dmax", 0.2, 0.8),
        "dmin": trial.suggest_float("mcc_bwd_adaptive_dmin", 0.01, 0.1)
    }

    # === Derived Initialization Parameter ===
    # This logic sets the number of sweeps needed before processing can start.
    mcc_sweeps = overrides.get('map_consistency', {}).get('num_past_sweeps_for_mcc', 0)
    required_init_sweeps = max(event_n, mcc_sweeps)
    overrides['initialization_phase'] = {
        "num_sweeps_for_initial_map": required_init_sweeps
    }

    return {"m_detector": overrides}


def define_refinement_search_space(trial: optuna.Trial) -> dict:
    """
    Defines the search space for the refinement stage using the stable
    CPU-based hdbscan library.
    """
    algo_choice = trial.suggest_categorical(
        "cluster_algorithm", ['none', ClusteringAlgorithm.HDBSCAN.value]
    )

    if algo_choice == 'none':
        return {"m_detector": {"frame_refinement": {"enabled": False}}}

    overrides = {"m_detector": {"frame_refinement": {"enabled": True}}}
    clustering_params = {"algorithm": algo_choice}

    if algo_choice == ClusteringAlgorithm.HDBSCAN.value:
        min_points = trial.suggest_int("hdbscan_min_points", 3, 25)
        min_cluster_size = trial.suggest_int("hdbscan_min_cluster_size", min_points, 50)
        
        cluster_selection_epsilon = trial.suggest_float(
            "hdbscan_cluster_selection_epsilon", 0.0, 5.0 
        )
        
        clustering_params[algo_choice] = {
            "min_cluster_size": min_cluster_size,
            "min_points": min_points,
            "cluster_selection_epsilon": cluster_selection_epsilon,
        }
        
    overrides["m_detector"]["frame_refinement"]["clustering"] = clustering_params
    return overrides


def define_search_space(trial: optuna.Trial, mode: str):
    """Defines the search space based on the current run mode."""
    if mode == 'tune-full':
        # FOR THE GEOMETRIC RUN, WE ONLY CALL THIS
        params = define_geometric_search_space(trial)
        # AND FORCE REFINEMENT TO BE OFF
        params["m_detector"]["frame_refinement"] = {"enabled": False}
        return params


    elif mode == 'tune-refinement':
        return define_refinement_search_space(trial)
    else:
        return {}



# # src/ray_scripts/tuning_manager.py

# import copy
# import itertools
# import optuna

# def define_search_space(trial: optuna.Trial) -> dict:
#     """
#     PHASE 7: Tuning with Convex Hull Frame Refinement.

#     GOAL: Break the plateau using a geometrically precise refinement method.

#     STRATEGY:
#     1.  REVERT the core geometric parameters to the ranges from the best run (Phase 5).
#     2.  ENABLE the new Convex Hull refinement logic.
#     3.  TUNE the clustering parameters (`eps`, `min_samples`) to find optimal clusters
#         for this new, more precise refinement method.
#     """
#     overrides = {}

#     # === Core Geometric Parameters (based on the 0.475-scoring run) ===
#     overrides['occlusion_determination'] = {
#         "epsilon_depth": trial.suggest_float("occ_epsilon_depth", 0.2, 0.3),
#         "adaptive_epsilon_depth_config": {
#             "enabled": True,
#             "dthr": trial.suggest_float("occ_adaptive_dthr", 7.0, 12.0),
#             "kthr": trial.suggest_float("occ_adaptive_kthr", 0.07, 0.1),
#             "dmax": trial.suggest_float("occ_adaptive_dmax", 0.4, 0.6),
#             "dmin": 0.05
#         }
#     }
#     overrides['detailed_occlusion_check'] = {
#         "epsilon_depth": trial.suggest_float("detail_epsilon_depth", 0.18, 0.25),
#         "angular_neighborhood_h_deg": trial.suggest_float("detail_angular_h", 0.12, 0.18),
#         "angular_neighborhood_v_deg": trial.suggest_float("detail_angular_v", 0.25, 0.35),
#     }
#     overrides['map_consistency'] = {
#         "enabled": True,
#         "num_past_sweeps_for_mcc": 19, # Fixed
#         "epsilon_phi_deg": trial.suggest_float("mcc_eps_phi", 0.35, 0.55),
#         "epsilon_theta_deg": trial.suggest_float("mcc_eps_theta", 1.0, 1.5),
#         "static_confidence_threshold": trial.suggest_float("mcc_score_thresh", 0.45, 0.75),
#     }
    
#     # === Event Logic (FIXED based on best results) ===
#     overrides['event_detection_logic'] = {
#         "test2_parallel_away": {"num_historical_DIs_M2": 8},
#         "test4_perpendicular": {"min_occluding_DIs_M4": 7, "num_historical_DIs_N": 19}
#     }

#     # === NEW: Convex Hull Refinement (Tuning this is now the priority) ===
#     overrides['frame_refinement'] = {
#         "enabled": True,
#         "clustering": {
#             # Widen the search space again, as the logic is new
#             "dbscan_eps": trial.suggest_float("cluster_eps", 0.3, 1.5),
#             "dbscan_min_samples": trial.suggest_int("cluster_min_samples", 3, 15)
#         }
#     }

#     return {"m_detector": overrides}

    
# def define_search_space(trial: optuna.Trial) -> dict:
#     """
#     Defines the hyperparameter search space for parameters we want to tune.
#     """
#     overrides = {}

#     # === Occlusion Determination ===
#     overrides['occlusion_determination'] = {
#         "epsilon_depth": trial.suggest_float("occ_epsilon_depth", 0.05, 0.2, log=True),
#         "pixel_neighborhood_h": trial.suggest_int("occ_pixel_h", 1, 3),
#         "pixel_neighborhood_v": trial.suggest_int("occ_pixel_v", 1, 2),
#         "angular_neighborhood_h_deg": trial.suggest_float("occ_angular_h", 0.1, 0.6),
#         "angular_neighborhood_v_deg": trial.suggest_float("occ_angular_v", 0.2, 0.8)
#     }
#     # Optional: Add adaptive epsilon for occlusion here 
#     # ...

#     # === Event Detection Logic ===
#     event_n = trial.suggest_int("event_N", 5, 15)
#     overrides['event_detection_logic'] = {
#         "test2_parallel_away": {"num_historical_DIs_M2": trial.suggest_int("event_M2", 1, 5)},
#         "test3_parallel_towards": {"num_historical_DIs_M3": trial.suggest_int("event_M3", 1, 5)},
#         "test4_perpendicular": {
#             "num_historical_DIs_N": event_n,
#             "min_occluding_DIs_M4": trial.suggest_int("event_M4", 2, event_n)
#         }
#     }

#     # === Map Consistency Check (MCC) ===
#     mcc_sweeps = 0 # Default to 0 if MCC is disabled
#     if trial.suggest_categorical("mcc_enabled", [True, ]):
#         mcc_sweeps = trial.suggest_int("mcc_num_sweeps", 5, 20)
#         overrides['map_consistency'] = {
#             "enabled": True,
#             "num_past_sweeps_for_mcc": mcc_sweeps,
#             "epsilon_phi_deg": trial.suggest_float("mcc_eps_phi", 0.3, 1.0),
#             "epsilon_theta_deg": trial.suggest_float("mcc_eps_theta", 1.0, 3.0),
#             "threshold_value_count": trial.suggest_int("mcc_thresh_count", 1, 4),
#             # Note: We are not tuning the adaptive MCC params for simplicity here, but they could be added.
#         }

#     required_init_sweeps = max(event_n, mcc_sweeps)
    
#     overrides['initialization_phase'] = {
#         "num_sweeps_for_initial_map": required_init_sweeps
#     }

#     # The final dictionary must be nested under the top-level 'm_detector' key
#     return {"m_detector": overrides}

# def define_search_space(trial: optuna.Trial) -> dict:
#     """
#     PHASE 2 (Take 3): A hybrid "Anchor and Explore" strategy.
#     Start with the original working space and make minimal, high-confidence changes.
#     """
#     overrides = {}

#     # === Occlusion Determination ===
#     # ANCHOR: Keep the original, wide search space for these parameters.
#     # The risk of narrowing them all at once is too high. Let the optimizer
#     # explore freely here, now that we are guiding it elsewhere.
#     overrides['occlusion_determination'] = {
#         "epsilon_depth": trial.suggest_float("occ_epsilon_depth", 0.05, 0.2, log=True),
#         # FIX: We can confidently fix the pixel neighborhoods based on the analysis.
#         # This is a low-risk change that simplifies the search space.
#         "pixel_neighborhood_h": 2,
#         "pixel_neighborhood_v": 2,
#         "angular_neighborhood_h_deg": trial.suggest_float("occ_angular_h", 0.1, 0.6),
#         "angular_neighborhood_v_deg": trial.suggest_float("occ_angular_v", 0.2, 0.8)
#     }

#     # === Event Detection Logic ===
#     # EXPLORE: This is our most confident change. The analysis showed a clear
#     # desire for higher values. We extend the upper bound but keep the lower
#     # bound within the original "known good" territory.
#     event_n = trial.suggest_int("event_N", 10, 25) # Changed from (5, 15)
#     overrides['event_detection_logic'] = {
#         "test2_parallel_away": {"num_historical_DIs_M2": trial.suggest_int("event_M2", 3, 10)}, # Changed from (1, 5)
#         # ANCHOR: Keep M3 the same.
#         "test3_parallel_towards": {"num_historical_DIs_M3": trial.suggest_int("event_M3", 1, 5)},
#         "test4_perpendicular": {
#             "num_historical_DIs_N": event_n,
#             "min_occluding_DIs_M4": trial.suggest_int("event_M4", 2, event_n)
#         }
#     }

#     # === Map Consistency Check (MCC) ===
#     # ANCHOR: Keep the main MCC parameters broad.
#     mcc_sweeps = 0
#     if trial.suggest_categorical("mcc_enabled", [True, ]):
#         mcc_sweeps = trial.suggest_int("mcc_num_sweeps", 5, 20)
#         overrides['map_consistency'] = {
#             "enabled": True,
#             "num_past_sweeps_for_mcc": mcc_sweeps,
#             "epsilon_phi_deg": trial.suggest_float("mcc_eps_phi", 0.3, 1.0),
#             "epsilon_theta_deg": trial.suggest_float("mcc_eps_theta", 1.0, 3.0),
#             # FIX: This is our other high-confidence change.
#             "threshold_value_count": 1, # Changed from suggest_int(..., 1, 4)
#         }

#     # This logic remains the same.
#     required_init_sweeps = max(event_n, mcc_sweeps)
#     overrides['initialization_phase'] = {
#         "num_sweeps_for_initial_map": required_init_sweeps
#     }

#     return {"m_detector": overrides}

# def define_search_space(trial: optuna.Trial) -> dict:
#     """
#     PHASE 3: Exploit and Verify.
#     - Exploit the highly consistent results from the last run by creating a very narrow search space.
#     - Verify if the adaptive epsilon feature can provide an additional performance boost.
#     """
#     overrides = {}

#     # === Occlusion Determination ===
#     # EXPLOIT: Zoom in on the optimal regions found in the last run.
#     # The best values for epsilon_depth and angular_h were pinned at the boundaries,
#     # so we will center our new search there and give a little room to explore.
#     overrides['occlusion_determination'] = {
#         "epsilon_depth": trial.suggest_float("occ_epsilon_depth", 0.19, 0.25), # Centered around the old max of 0.2
#         "pixel_neighborhood_h": 2, # Fixed
#         "pixel_neighborhood_v": 2, # Fixed
#         "angular_neighborhood_h_deg": trial.suggest_float("occ_angular_h", 0.08, 0.15), # Centered around the old min of 0.1
#         "angular_neighborhood_v_deg": trial.suggest_float("occ_angular_v", 0.3, 0.7), # Keep this range wider as the signal wasn't as strong
#     }
    
#     # --- VERIFY: Re-introduce the adaptive epsilon feature ---
#     # We now test if this feature, added to our new high-performing baseline, can improve the score.
#     if trial.suggest_categorical("occ_adaptive_enabled", [True, False]):
#         overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {
#             "enabled": True,
#             # Let's give these reasonable ranges to start.
#             "dthr": trial.suggest_float("occ_adaptive_dthr", 3.0, 15.0),
#             "kthr": trial.suggest_float("occ_adaptive_kthr", 0.01, 0.05, log=True),
#             "dmax": trial.suggest_float("occ_adaptive_dmax", 0.4, 1.0),
#             "dmin": 0.05 # Fixed, as it's a safety net
#         }
#     else:
#         # Ensure it's explicitly disabled if the categorical is False
#         overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {"enabled": False}


#     # === Event Detection Logic ===
#     # EXPLOIT: This is our most important parameter set. We will create a very
#     # tight search space around the optimal values found.
#     event_n = trial.suggest_int("event_N", 18, 22) # Tight range around 19-20
#     overrides['event_detection_logic'] = {
#         "test2_parallel_away": {"num_historical_DIs_M2": trial.suggest_int("event_M2", 5, 8)}, # Tight range around 5-7
#         "test3_parallel_towards": {"num_historical_DIs_M3": 1}, # FIX: Confidently fix to 1.
#         "test4_perpendicular": {
#             "num_historical_DIs_N": event_n,
#             "min_occluding_DIs_M4": trial.suggest_int("event_M4", 6, 8) # Best M4 was 7, let's explore around it.
#         }
#     }

#     # === Map Consistency Check (MCC) ===
#     # EXPLOIT: Refine the ranges based on the new data.
#     overrides['map_consistency'] = {
#         "enabled": True, # Fixed
#         "num_past_sweeps_for_mcc": trial.suggest_int("mcc_num_sweeps", 15, 20), # Nudge range up
#         "epsilon_phi_deg": trial.suggest_float("mcc_eps_phi", 0.45, 0.7), # Refined range
#         "epsilon_theta_deg": trial.suggest_float("mcc_eps_theta", 1.0, 2.8), # Keep wide
#         "threshold_value_count": 1, # Fixed
#     }
    
#     # This logic remains the same.
#     mcc_sweeps = overrides.get('map_consistency').get('num_past_sweeps_for_mcc')
#     required_init_sweeps = max(event_n, mcc_sweeps)
#     overrides['initialization_phase'] = {
#         "num_sweeps_for_initial_map": required_init_sweeps
#     }

#     return {"m_detector": overrides}


# def define_search_space(trial: optuna.Trial) -> dict:
#     """
#     PHASE 4a: Fast Calibration (Interpolation DISABLED).
    
#     The goal is to rapidly find the optimal parameters for the fast, GPU-only
#     path of the Map Consistency Check. By disabling the slow CPU interpolation,
#     we can run many more trials and quickly establish a strong baseline.
#     """
#     overrides = {}

#     # === Occlusion Determination ===
#     # We keep the search space wide to allow the optimizer to find the best
#     # parameters for this specific (non-interpolated) system behavior.
#     overrides['occlusion_determination'] = {
#         "epsilon_depth": trial.suggest_float("occ_epsilon_depth", 0.1, 0.25),
#         "pixel_neighborhood_h": 2,
#         "pixel_neighborhood_v": 2,
#         "angular_neighborhood_h_deg": trial.suggest_float("occ_angular_h", 0.1, 0.5),
#         "angular_neighborhood_v_deg": trial.suggest_float("occ_angular_v", 0.2, 0.8),
#     }
    
#     if trial.suggest_categorical("occ_adaptive_enabled", [True, False]):
#         overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {
#             "enabled": True,
#             "dthr": trial.suggest_float("occ_adaptive_dthr", 5.0, 20.0),
#             "kthr": trial.suggest_float("occ_adaptive_kthr", 0.01, 0.1, log=True),
#             "dmax": trial.suggest_float("occ_adaptive_dmax", 0.3, 1.0),
#             "dmin": 0.05
#         }
#     else:
#         overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {"enabled": False}

#     # === Event Detection Logic ===
#     event_n = trial.suggest_int("event_N", 15, 25)
#     overrides['event_detection_logic'] = {
#         "test2_parallel_away": {"num_historical_DIs_M2": trial.suggest_int("event_M2", 3, 10)},
#         "test3_parallel_towards": {"num_historical_DIs_M3": 1},
#         "test4_perpendicular": {
#             "num_historical_DIs_N": event_n,
#             "min_occluding_DIs_M4": trial.suggest_int("event_M4", 5, 10)
#         }
#     }

#     # === Map Consistency Check (MCC) ===
#     overrides['map_consistency'] = {
#         "enabled": True,
#         # --- THIS IS THE KEY CHANGE FOR THIS STAGE ---
#         "interpolation_enabled": False, # Disable slow CPU path for fast tuning.
        
#         "num_past_sweeps_for_mcc": trial.suggest_int("mcc_num_sweeps", 15, 25),
#         "epsilon_phi_deg": trial.suggest_float("mcc_eps_phi", 0.3, 1.0),
#         "epsilon_theta_deg": trial.suggest_float("mcc_eps_theta", 1.0, 3.0),
#         "threshold_value_count": trial.suggest_int("mcc_thresh_count", 1, 3),
#     }
    
#     # We still tune the adaptive epsilon configs, as they affect the GPU path.
#     overrides['map_consistency']['adaptive_epsilon_forward_config'] = {
#         "enabled": True,
#         "dthr": trial.suggest_float("mcc_fwd_adaptive_dthr", 10.0, 25.0),
#         "kthr": trial.suggest_float("mcc_fwd_adaptive_kthr", 0.02, 0.1, log=True),
#         "dmax": trial.suggest_float("mcc_fwd_adaptive_dmax", 0.4, 1.5),
#         "dmin": trial.suggest_float("mcc_fwd_adaptive_dmin", 0.05, 0.2)
#     }
#     overrides['map_consistency']['adaptive_epsilon_backward_config'] = {
#         "enabled": True,
#         "dthr": trial.suggest_float("mcc_bwd_adaptive_dthr", 10.0, 25.0),
#         "kthr": trial.suggest_float("mcc_bwd_adaptive_kthr", 0.02, 0.1, log=True),
#         "dmax": trial.suggest_float("mcc_bwd_adaptive_dmax", 0.4, 1.5),
#         "dmin": trial.suggest_float("mcc_bwd_adaptive_dmin", 0.05, 0.2)
#     }

#     # This logic remains the same.
#     mcc_sweeps = overrides.get('map_consistency').get('num_past_sweeps_for_mcc')
#     required_init_sweeps = max(event_n, mcc_sweeps)
#     overrides['initialization_phase'] = {
#         "num_sweeps_for_initial_map": required_init_sweeps
#     }

#     return {"m_detector": overrides}

# Tight tuning space with old code:

# def define_search_space(trial: optuna.Trial) -> dict:
#     """
#     PHASE 3: Exploit and Verify.
#     - Exploit the highly consistent results from the last run by creating a very narrow search space.
#     - Verify if the adaptive epsilon feature can provide an additional performance boost.
#     """
#     overrides = {}

#     # === Occlusion Determination ===
#     # EXPLOIT: Zoom in on the optimal regions found in the last run.
#     # The best values for epsilon_depth and angular_h were pinned at the boundaries,
#     # so we will center our new search there and give a little room to explore.
#     overrides['occlusion_determination'] = {
#         "epsilon_depth": trial.suggest_float("occ_epsilon_depth", 0.19, 0.25), # Centered around the old max of 0.2
#         "pixel_neighborhood_h": 2, # Fixed
#         "pixel_neighborhood_v": 2, # Fixed
#         "angular_neighborhood_h_deg": trial.suggest_float("occ_angular_h", 0.08, 0.15), # Centered around the old min of 0.1
#         "angular_neighborhood_v_deg": trial.suggest_float("occ_angular_v", 0.3, 0.7), # Keep this range wider as the signal wasn't as strong
#     }
    
#     # --- VERIFY: Re-introduce the adaptive epsilon feature ---
#     # We now test if this feature, added to our new high-performing baseline, can improve the score.
#     if trial.suggest_categorical("occ_adaptive_enabled", [True, False]):
#         overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {
#             "enabled": True,
#             # Let's give these reasonable ranges to start.
#             "dthr": trial.suggest_float("occ_adaptive_dthr", 3.0, 15.0),
#             "kthr": trial.suggest_float("occ_adaptive_kthr", 0.01, 0.05, log=True),
#             "dmax": trial.suggest_float("occ_adaptive_dmax", 0.4, 1.0),
#             "dmin": 0.05 # Fixed, as it's a safety net
#         }
#     else:
#         # Ensure it's explicitly disabled if the categorical is False
#         overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {"enabled": False}


#     # === Event Detection Logic ===
#     # EXPLOIT: This is our most important parameter set. We will create a very
#     # tight search space around the optimal values found.
#     event_n = trial.suggest_int("event_N", 18, 22) # Tight range around 19-20
#     overrides['event_detection_logic'] = {
#         "test2_parallel_away": {"num_historical_DIs_M2": trial.suggest_int("event_M2", 5, 8)}, # Tight range around 5-7
#         "test3_parallel_towards": {"num_historical_DIs_M3": 1}, # FIX: Confidently fix to 1.
#         "test4_perpendicular": {
#             "num_historical_DIs_N": event_n,
#             "min_occluding_DIs_M4": trial.suggest_int("event_M4", 6, 8) # Best M4 was 7, let's explore around it.
#         }
#     }

#     # === Map Consistency Check (MCC) ===
#     # EXPLOIT: Refine the ranges based on the new data.
#     overrides['map_consistency'] = {
#         "enabled": False, # Fixed
#         "num_past_sweeps_for_mcc": trial.suggest_int("mcc_num_sweeps", 15, 20), # Nudge range up
#         "epsilon_phi_deg": trial.suggest_float("mcc_eps_phi", 0.45, 0.7), # Refined range
#         "epsilon_theta_deg": trial.suggest_float("mcc_eps_theta", 1.0, 2.8), # Keep wide
#         "threshold_value_count": 1, # Fixed
#     }
    
#     # This logic remains the same.
#     mcc_sweeps = overrides.get('map_consistency').get('num_past_sweeps_for_mcc')
#     required_init_sweeps = max(event_n, mcc_sweeps)
#     overrides['initialization_phase'] = {
#         "num_sweeps_for_initial_map": required_init_sweeps
#     }

#     return {"m_detector": overrides}

# def define_search_space(trial: optuna.Trial) -> dict:
#     """
#     PHASE 4b: Forgiving Calibration (with Hyperband Pruner).
    
#     The goal is to make the MCC filter less aggressive by default, to ensure
#     the optimizer can find a non-zero IoU signal to work with. We are making
#     the epsilon values for the MCC SMALLER, meaning it will be more strict
#     about what it considers a "match" and therefore will filter FEWER points.
#     """
#     overrides = {}

#     # === Occlusion Determination ===
#     # Keep this part wide
#     overrides['occlusion_determination'] = {
#         "epsilon_depth": trial.suggest_float("occ_epsilon_depth", 0.1, 0.25),
#         "pixel_neighborhood_h": 2, "pixel_neighborhood_v": 2,
#         "angular_neighborhood_h_deg": trial.suggest_float("occ_angular_h", 0.1, 0.5),
#         "angular_neighborhood_v_deg": trial.suggest_float("occ_angular_v", 0.2, 0.8),
#     }
#     # ... (adaptive epsilon config for occlusion remains the same)

#     # === Event Detection Logic ===
#     # Keep this part wide
#     event_n = trial.suggest_int("event_N", 15, 25)
#     overrides['event_detection_logic'] = {
#         "test2_parallel_away": {"num_historical_DIs_M2": trial.suggest_int("event_M2", 3, 10)},
#         "test3_parallel_towards": {"num_historical_DIs_M3": 1},
#         "test4_perpendicular": { "num_historical_DIs_N": event_n, "min_occluding_DIs_M4": trial.suggest_int("event_M4", 5, 10) }
#     }

#     # === Map Consistency Check (MCC) ===
#     overrides['map_consistency'] = {
#         "enabled": True,
#         "interpolation_enabled": False, # Still disabled for speed
#         "num_past_sweeps_for_mcc": trial.suggest_int("mcc_num_sweeps", 15, 25),
        
#         # --- KEY CHANGE: Make the filter LESS AGGRESSIVE ---
#         # Use smaller angular windows. A smaller window means a point has to be
#         # VERY close to a static point to be filtered out. This will let more
#         # true positives through.
#         "epsilon_phi_deg": trial.suggest_float("mcc_eps_phi", 0.1, 0.5), # Was (0.3, 1.0)
#         "epsilon_theta_deg": trial.suggest_float("mcc_eps_theta", 0.5, 1.5), # Was (1.0, 3.0)
        
#         "threshold_value_count": trial.suggest_int("mcc_thresh_count", 1, 2), # More likely to be 1 now
#     }
    
#     # Also make the adaptive epsilon for MCC less aggressive by default
#     overrides['map_consistency']['adaptive_epsilon_forward_config'] = {
#         "enabled": True,
#         "dthr": trial.suggest_float("mcc_fwd_adaptive_dthr", 15.0, 30.0),
#         "kthr": trial.suggest_float("mcc_fwd_adaptive_kthr", 0.01, 0.05, log=True), # Smaller k
#         "dmax": trial.suggest_float("mcc_fwd_adaptive_dmax", 0.2, 0.8), # Smaller max
#         "dmin": trial.suggest_float("mcc_fwd_adaptive_dmin", 0.01, 0.1) # Smaller min
#     }
#     overrides['map_consistency']['adaptive_epsilon_backward_config'] = {
#         "enabled": True,
#         "dthr": trial.suggest_float("mcc_bwd_adaptive_dthr", 15.0, 30.0),
#         "kthr": trial.suggest_float("mcc_bwd_adaptive_kthr", 0.01, 0.05, log=True),
#         "dmax": trial.suggest_float("mcc_bwd_adaptive_dmax", 0.2, 0.8),
#         "dmin": trial.suggest_float("mcc_bwd_adaptive_dmin", 0.01, 0.1)
#     }

#     # ... (initialization logic remains the same) ...
#     mcc_sweeps = overrides.get('map_consistency').get('num_past_sweeps_for_mcc')
#     required_init_sweeps = max(event_n, mcc_sweeps)
#     overrides['initialization_phase'] = { "num_sweeps_for_initial_map": required_init_sweeps }

#     return {"m_detector": overrides}

# def define_search_space(trial: optuna.Trial) -> dict:
#     """
#     PHASE 5: Tuning the Normalized Score MCC.

#     GOAL: Beat the 0.47 IoU baseline by finding the optimal configuration for the now-working
#     Map Consistency Check (MCC) filter.

#     STRATEGY:
#     1.  RE-ENABLE the MCC.
#     2.  TUNE the new `static_confidence_threshold`, giving it a wide range to explore. This is
#         now the primary control for the filter's strictness.
#     3.  KEEP the other parameters in wide, forgiving ranges to allow the optimizer to find
#         the best combination between the initial detection and the new filter.
#     """
#     overrides = {}

#     # === Occlusion Determination (Initial Detection) ===
#     # Keep this search space wide. The baseline is strong, but there may be
#     # better configurations when combined with a working MCC.
#     overrides['occlusion_determination'] = {
#         "epsilon_depth": trial.suggest_float("occ_epsilon_depth", 0.1, 0.25),
#         "pixel_neighborhood_h": 2,  # Fixed
#         "pixel_neighborhood_v": 2,  # Fixed
#     }

#     overrides['detailed_occlusion_check'] = {
#         "epsilon_depth": trial.suggest_float("detail_epsilon_depth", 0.1, 0.25),
#         "angular_neighborhood_h_deg": trial.suggest_float("detail_angular_h", 0.05, 0.2), # Narrow, precise range
#         "angular_neighborhood_v_deg": trial.suggest_float("detail_angular_v", 0.1, 0.4),  # Narrow, precise range
#     }
    
    
#     # Allow the optimizer to enable/disable adaptive epsilon for occlusion checks.
#     if trial.suggest_categorical("occ_adaptive_enabled", [True, False]):
#         overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {
#             "enabled": True,
#             "dthr": trial.suggest_float("occ_adaptive_dthr", 5.0, 20.0),
#             "kthr": trial.suggest_float("occ_adaptive_kthr", 0.01, 0.1, log=True),
#             "dmax": trial.suggest_float("occ_adaptive_dmax", 0.3, 1.0),
#             "dmin": 0.05  # Fixed safety net
#         }
#     else:
#         overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {"enabled": False}


#     # === Event Detection Logic ===
#     # Keep these ranges wide as well.
#     event_n = trial.suggest_int("event_N", 15, 25)
#     overrides['event_detection_logic'] = {
#         "test2_parallel_away": {"num_historical_DIs_M2": trial.suggest_int("event_M2", 3, 10)},
#         "test3_parallel_towards": {"num_historical_DIs_M3": 1}, # Fixed
#         "test4_perpendicular": {
#             "num_historical_DIs_N": event_n,
#             "min_occluding_DIs_M4": trial.suggest_int("event_M4", 5, 10)
#         }
#     }

#     # === Map Consistency Check (MCC) ===
#     # This is the most critical section for this phase.
#     overrides['map_consistency'] = {
#         "enabled": True,  # RE-ENABLE THE MCC
#         "interpolation_enabled": False,  # Keep disabled for faster tuning
        
#         "num_past_sweeps_for_mcc": trial.suggest_int("mcc_num_sweeps", 15, 25),
        
#         # Use lenient angular windows to give the optimizer a good starting point.
#         "epsilon_phi_deg": trial.suggest_float("mcc_eps_phi", 0.1, 0.5),
#         "epsilon_theta_deg": trial.suggest_float("mcc_eps_theta", 0.5, 1.5),

#         # This is the NEW primary tuning parameter for the MCC.
#         # A low value is lenient (few points marked static).
#         # A high value is strict (many points marked static).
#         "static_confidence_threshold": trial.suggest_float("mcc_score_thresh", 0.1, 0.9),

#         # The old count threshold is no longer used, so it's removed.
#         # "threshold_value_count": ...
#     }
    
#     # Also allow tuning of the adaptive epsilon for the MCC, using forgiving ranges.
#     overrides['map_consistency']['adaptive_epsilon_forward_config'] = {
#         "enabled": True,
#         "dthr": trial.suggest_float("mcc_fwd_adaptive_dthr", 15.0, 30.0),
#         "kthr": trial.suggest_float("mcc_fwd_adaptive_kthr", 0.01, 0.05, log=True),
#         "dmax": trial.suggest_float("mcc_fwd_adaptive_dmax", 0.2, 0.8),
#         "dmin": trial.suggest_float("mcc_fwd_adaptive_dmin", 0.01, 0.1)
#     }
#     overrides['map_consistency']['adaptive_epsilon_backward_config'] = {
#         "enabled": True,
#         "dthr": trial.suggest_float("mcc_bwd_adaptive_dthr", 15.0, 30.0),
#         "kthr": trial.suggest_float("mcc_bwd_adaptive_kthr", 0.01, 0.05, log=True),
#         "dmax": trial.suggest_float("mcc_bwd_adaptive_dmax", 0.2, 0.8),
#         "dmin": trial.suggest_float("mcc_bwd_adaptive_dmin", 0.01, 0.1)
#     }

#     # === Derived Initialization Parameter ===
#     # This logic correctly sets the number of sweeps needed before processing can start.
#     mcc_sweeps = overrides.get('map_consistency', {}).get('num_past_sweeps_for_mcc', 0)
#     required_init_sweeps = max(event_n, mcc_sweeps)
#     overrides['initialization_phase'] = {
#         "num_sweeps_for_initial_map": required_init_sweeps
#     }

#     return {"m_detector": overrides}


