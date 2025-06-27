# src/tuning/tuning_manager.py

import optuna
from ..core.m_detector.refinement_algorithms import ClusteringAlgorithm
from .shared_utils import deep_update_dict

def define_geometric_search_space(trial: optuna.Trial) -> dict:
    """Defines the search space for the expensive geometric parameters."""
    overrides = {}
    # === RANSAC ground filtering ===
    # overrides['ransac_ground_params'] = {
    #     # How close a point must be to a candidate plane during trial fits.
    #     "inlier_threshold": trial.suggest_float("ransac_inlier_thresh", 0.1, 0.5),
    #     # How close a point must be to the FINAL best plane to be labeled ground.
    #     "ground_threshold": trial.suggest_float("ransac_ground_thresh", 0.1, 0.5),
    #     # Define the vertical window to search for ground.
    #     "z_min_threshold": trial.suggest_float("ransac_z_min", -3.0, -2.0),
    #     "z_max_threshold": trial.suggest_float("ransac_z_max", -1.0, 0.0),
    # }

    # === Occlusion Determination (Initial Detection) ===
    overrides['occlusion_determination'] = {
        "epsilon_depth": trial.suggest_float("occ_epsilon_depth", 0.2, 0.4),
        "pixel_neighborhood_h": 2,
        "pixel_neighborhood_v": 2,
    }
    overrides['detailed_occlusion_check'] = {
        "epsilon_depth": trial.suggest_float("detail_epsilon_depth", 0.1, 0.25),
        "angular_neighborhood_h_deg": trial.suggest_float("detail_angular_h", 0.05, 0.2),
        "angular_neighborhood_v_deg": trial.suggest_float("detail_angular_v", 0.1, 0.4),
    }
    if trial.suggest_categorical("occ_adaptive_enabled", [True, False]):
        overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {
            "enabled": True,
            "dthr": trial.suggest_float("occ_adaptive_dthr", 5.0, 20.0),
            "kthr": trial.suggest_float("occ_adaptive_kthr", 0.01, 0.1, log=True),
            "dmax": trial.suggest_float("occ_adaptive_dmax", 0.3, 1.0),
            "dmin": 0.05
        }
    else:
        overrides['occlusion_determination']['adaptive_epsilon_depth_config'] = {"enabled": False}

    # === Event Detection Logic ===
    event_n = trial.suggest_int("event_N", 15, 25)
    overrides['geometric_tests'] = {
        'initial_occlusion_pass': {
            'history_length': trial.suggest_int("initial_pass_history_length", 15, 25),
            'min_occlusion_count': trial.suggest_int("initial_pass_min_occ_count", 2, 10)
        },
        'event_tests': {
            'parallel_motion_away': {
                'history_length': trial.suggest_int("parallel_motion_hist_len", 2, 8),
                'apply_mcc_filter': trial.suggest_categorical("parallel_motion_mcc", [True, False])
            },
            'perpendicular_motion': {
                'history_length': trial.suggest_int("perp_motion_hist_len", 1, 5),
                'apply_mcc_filter': trial.suggest_categorical("perp_motion_mcc", [True, False])
            }
        }
    }

    # === Map Consistency Check (MCC) ===
    overrides['map_consistency'] = {
        "enabled": True,  
        "interpolation_enabled": False,
        "num_past_sweeps_for_mcc": trial.suggest_int("mcc_num_sweeps", 15, 25),
        "epsilon_phi_deg": trial.suggest_float("mcc_eps_phi", 0.1, 0.5),
        "epsilon_theta_deg": trial.suggest_float("mcc_eps_theta", 0.5, 1.5),
        "static_confidence_threshold": trial.suggest_float("mcc_score_thresh", 0.1, 0.9),
    }
    
    # REASONING: Also keep the adaptive epsilon for MCC flexible.
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
    mcc_sweeps = overrides.get('map_consistency', {}).get('num_past_sweeps_for_mcc', 0)
    required_init_sweeps = max(
        overrides.get('geometric_tests', {}).get('initial_occlusion_pass', {}).get('history_length', 0),
        overrides.get('map_consistency', {}).get('num_past_sweeps_for_mcc', 0)
    )

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