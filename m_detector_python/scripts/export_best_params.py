import argparse
import optuna
import yaml

def main():
    """
    Connects to an Optuna database, automatically finds the best trial in a study,
    and saves its parameters to a YAML file.
    """
    parser = argparse.ArgumentParser(description="Extract the best Optuna trial's parameters to a YAML file.")
    parser.add_argument('--storage', type=str, required=True, help="Path to the Optuna database (e.g., 'sqlite:///studies.db').")
    parser.add_argument('--study-name', type=str, required=True, help="The name of the study to load.")
    parser.add_argument('--output-file', type=str, required=True, help="Path to save the output YAML file.")
    args = parser.parse_args()

    print(f"Loading study '{args.study_name}' from '{args.storage}'...")
    try:
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    except KeyError:
        print(f"Error: Study '{args.study_name}' not found in the database.")
        return

    try:
        # Optuna's study.best_trial property automatically finds the best one!
        best_trial = study.best_trial
        print(f"Found best trial: Number {best_trial.number} with value: {best_trial.value:.6f}")
    except ValueError:
        print(f"Error: No completed trials found in study '{args.study_name}'. Cannot determine the best trial.")
        return

    # The parameters are stored in a flat dictionary, e.g., {'m_detector.mcc.max_dist': 0.5}
    # We need to convert this to a nested dictionary for our config system.
    nested_params = {}
    for key, value in best_trial.params.items():
        parts = key.split('.')
        d = nested_params
        # Traverse the path, creating keys if they don't exist
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        # Set the final value
        d[parts[-1]] = value

    print(f"Saving nested parameters to '{args.output_file}'...")
    with open(args.output_file, 'w') as f:
        # Use indent to make the YAML file readable
        yaml.dump(nested_params, f, default_flow_style=False, indent=4)
        
    print(f"Successfully exported parameters from trial {best_trial.number} to {args.output_file}.")

if __name__ == '__main__':
    main()