# analyze_study.py (Corrected)

import optuna
import pandas as pd
import argparse
import yaml
from pathlib import Path
from src.ray_scripts.run_optuna_experiment import deep_update_dict

def analyze_study(study_name: str, db_path: str, top_n: int = 10):
    """
    Loads an Optuna study, prints the results, and saves the best trial's parameters to a YAML file.
    """
    db_file = Path(db_path)
    if not db_file.exists():
        print(f"Error: Database file not found at '{db_file.resolve()}'")
        return

    storage_uri = f"sqlite:///{db_file.resolve()}"
    study = optuna.load_study(study_name=study_name, storage=storage_uri)

    print(f"--- Study: {study.study_name} ---")
    print(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print("\n--- Best Trial ---")
    print(f"  Value (IoU): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # --- Reconstruct the nested dictionary from Optuna's flat params ---
    nested_params = {}
    for key, value in best_trial.params.items():
        parts = key.split('_', 1)
        category = parts[0]
        param_name = parts[1]
        
        if category == "occ":
            target_dict = nested_params.setdefault('occlusion_determination', {})
            if param_name == "adaptive_enabled":
                target_dict.setdefault('adaptive_epsilon_depth_config', {})['enabled'] = value
            elif param_name.startswith("adaptive"):
                sub_dict = target_dict.setdefault('adaptive_epsilon_depth_config', {})
                sub_dict[param_name.replace("adaptive_", "")] = value
            else:
                if param_name == "angular_h": param_name = "angular_neighborhood_h_deg"
                if param_name == "angular_v": param_name = "angular_neighborhood_v_deg"
                target_dict[param_name] = value
        elif category == "event":
            target_dict = nested_params.setdefault('event_detection_logic', {})
            if param_name == "N":
                target_dict.setdefault('test4_perpendicular', {})['num_historical_DIs_N'] = value
            elif param_name == "M2":
                target_dict.setdefault('test2_parallel_away', {})['num_historical_DIs_M2'] = value
            elif param_name == "M4":
                target_dict.setdefault('test4_perpendicular', {})['min_occluding_DIs_M4'] = value
        elif category == "mcc":
            target_dict = nested_params.setdefault('map_consistency', {})
            # CORRECTED: Use the full key name from the config
            if param_name == "num_sweeps": param_name = "num_past_sweeps_for_mcc"
            if param_name == "eps_phi": param_name = "epsilon_phi_deg"
            if param_name == "eps_theta": param_name = "epsilon_theta_deg"
            target_dict[param_name] = value

    # --- THIS IS THE FIX: Re-calculate the derived parameter ---
    if 'event_detection_logic' in nested_params and 'map_consistency' in nested_params:
        event_n = nested_params.get('event_detection_logic', {}).get('test4_perpendicular', {}).get('num_historical_DIs_N', 0)
        mcc_sweeps = nested_params.get('map_consistency', {}).get('num_past_sweeps_for_mcc', 0)
        required_init_sweeps = max(event_n, mcc_sweeps)
        nested_params.setdefault('initialization_phase', {})['num_sweeps_for_initial_map'] = required_init_sweeps
    # --- END FIX ---

    output_filename = f"best_params_{study.study_name}.yaml"
    with open(output_filename, 'w') as f:
        yaml.dump({"m_detector": nested_params}, f, default_flow_style=False, sort_keys=False)
    print(f"\nBest parameters saved to '{output_filename}'")

    print(f"\n--- Top {top_n} Trials ---")
    df = study.trials_dataframe()
    pd.set_option('display.max_rows', top_n + 5)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 160)
    print(df.sort_values(by="value", ascending=False).head(top_n))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze an Optuna study.")
    parser.add_argument("study_name", type=str, help="The name of the study to analyze.")
    parser.add_argument("--db", type=str, required=True, help="Path to the SQLite database file.")
    parser.add_argument("--top", type=int, default=10, help="Number of top trials to display.")
    args = parser.parse_args()
    
    analyze_study(study_name=args.study_name, db_path=args.db, top_n=args.top)