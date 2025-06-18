# src/ray_scripts/extract_best_trial.py

import optuna
import yaml
import argparse
import copy
from pathlib import Path
from rich.console import Console

from .tuning_manager import define_search_space

# --- Helper Function (copied from run_optuna_experiment.py) ---
def deep_update_dict(base_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update_dict(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

class DummyTrial:
    def __init__(self, params):
        self.params = params
    def suggest_float(self, name, low, high, log=False): return self.params[name]
    def suggest_int(self, name, low, high): return self.params[name]
    def suggest_categorical(self, name, choices): return self.params[name]

def main():
    parser = argparse.ArgumentParser(description="Extract the best trial from an Optuna study and generate its config file.")
    parser.add_argument("--study-name", type=str, required=True, help="Name of the Optuna study.")
    parser.add_argument("--base-config", type=str, required=True, help="Path to the base config YAML file.")
    parser.add_argument("--output-file", type=str, default="config_best_trial.yaml", help="Name for the output YAML file.")
    args = parser.parse_args()

    console = Console()
    storage_path = f"sqlite:///{args.study_name}.db"

    try:
        study = optuna.load_study(study_name=args.study_name, storage=storage_path)
    except KeyError:
        console.log(f"[bold red]Error: Study '{args.study_name}' not found in '{storage_path}'.[/bold red]")
        return

    best_trial = study.best_trial
    if not best_trial:
        console.log("[bold red]Error: No completed trials found in the study.[/bold red]")
        return

    console.log("--- Best Trial Found ---")
    console.log(f"Trial Number: {best_trial.number}")
    console.log(f"IoU Score: [bold green]{best_trial.value:.6f}[/bold green]")
    console.log("Parameters:")
    console.log(best_trial.params)

    # Load the base config
    with open(args.base_config, 'r') as f:
        base_config_dict = yaml.safe_load(f)
    
    # Create the final config by applying the best trial's overrides
    final_config = copy.deepcopy(base_config_dict)
    dummy_trial = DummyTrial(best_trial.params)
    reconstructed_overrides = define_search_space(dummy_trial)
    deep_update_dict(final_config, reconstructed_overrides)

    # Save the new config file
    with open(args.output_file, 'w') as f:
        yaml.dump(final_config, f, sort_keys=False, indent=2)
    
    console.log(f"\n[bold green]Successfully generated config file '{args.output_file}' from best trial.[/bold green]")


if __name__ == '__main__':
    main()