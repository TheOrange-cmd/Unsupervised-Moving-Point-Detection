# src/ray_scripts/tuning_manager.py

import copy
import itertools
import optuna
    
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

def define_search_space(trial: optuna.Trial) -> dict:
    """
    PHASE 3: Exploit and Verify.
    - Exploit the highly consistent results from the last run by creating a very narrow search space.
    - Verify if the adaptive epsilon feature can provide an additional performance boost.
    """
    overrides = {}

    # === Occlusion Determination ===
    # EXPLOIT: Zoom in on the optimal regions found in the last run.
    # The best values for epsilon_depth and angular_h were pinned at the boundaries,
    # so we will center our new search there and give a little room to explore.
    overrides['occlusion_determination'] = {
        "epsilon_depth": trial.suggest_float("occ_epsilon_depth", 0.19, 0.25), # Centered around the old max of 0.2
        "pixel_neighborhood_h": 2, # Fixed
        "pixel_neighborhood_v": 2, # Fixed
        "angular_neighborhood_h_deg": trial.suggest_float("occ_angular_h", 0.08, 0.15), # Centered around the old min of 0.1
        "angular_neighborhood_v_deg": trial.suggest_float("occ_angular_v", 0.3, 0.7), # Keep this range wider as the signal wasn't as strong
    }
    
    # --- VERIFY: Re-introduce the adaptive epsilon feature ---
    # We now test if this feature, added to our new high-performing baseline, can improve the score.
    if trial.suggest_categorical("occ_adaptive_enabled", [True, False]):
        overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {
            "enabled": True,
            # Let's give these reasonable ranges to start.
            "dthr": trial.suggest_float("occ_adaptive_dthr", 3.0, 15.0),
            "kthr": trial.suggest_float("occ_adaptive_kthr", 0.01, 0.05, log=True),
            "dmax": trial.suggest_float("occ_adaptive_dmax", 0.4, 1.0),
            "dmin": 0.05 # Fixed, as it's a safety net
        }
    else:
        # Ensure it's explicitly disabled if the categorical is False
        overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {"enabled": False}


    # === Event Detection Logic ===
    # EXPLOIT: This is our most important parameter set. We will create a very
    # tight search space around the optimal values found.
    event_n = trial.suggest_int("event_N", 18, 22) # Tight range around 19-20
    overrides['event_detection_logic'] = {
        "test2_parallel_away": {"num_historical_DIs_M2": trial.suggest_int("event_M2", 5, 8)}, # Tight range around 5-7
        "test3_parallel_towards": {"num_historical_DIs_M3": 1}, # FIX: Confidently fix to 1.
        "test4_perpendicular": {
            "num_historical_DIs_N": event_n,
            "min_occluding_DIs_M4": trial.suggest_int("event_M4", 6, 8) # Best M4 was 7, let's explore around it.
        }
    }

    # === Map Consistency Check (MCC) ===
    # EXPLOIT: Refine the ranges based on the new data.
    overrides['map_consistency'] = {
        "enabled": True, # Fixed
        "num_past_sweeps_for_mcc": trial.suggest_int("mcc_num_sweeps", 15, 20), # Nudge range up
        "epsilon_phi_deg": trial.suggest_float("mcc_eps_phi", 0.45, 0.7), # Refined range
        "epsilon_theta_deg": trial.suggest_float("mcc_eps_theta", 1.0, 2.8), # Keep wide
        "threshold_value_count": 1, # Fixed
    }
    
    # This logic remains the same.
    mcc_sweeps = overrides.get('map_consistency', {}).get('num_past_sweeps_for_mcc', 0)
    required_init_sweeps = max(event_n, mcc_sweeps)
    overrides['initialization_phase'] = {
        "num_sweeps_for_initial_map": required_init_sweeps
    }

    return {"m_detector": overrides}
