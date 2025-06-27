import argparse
import sys
import pandas as pd
import optuna
from pathlib import Path
from typing import Optional

# Set pandas display options for better text-based output
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200) # Increased width for better table display
pd.set_option('display.precision', 5)

def analyze_study(db_path: str, study_name: Optional[str] = None):
    """
    Connects to an Optuna SQLite database, loads a study, and performs
    a text-based analysis to guide future search space adjustments.
    """
    if not study_name:
        study_name = Path(db_path).stem
        print(f"INFO: --study-name not provided. Inferred study name '{study_name}' from database path.")

    storage_url = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except KeyError:
        print(f"\nError: Study '{study_name}' not found in the database '{db_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the study: {e}")
        sys.exit(1)

    df = study.trials_dataframe()
    print("\n" + "="*80)
    print(f"Optuna Study Analysis for: '{study.study_name}'")
    print(f"Direction: {study.direction.name}")
    print("="*80)

    complete_trials = df[df.state == 'COMPLETE']
    if complete_trials.empty:
        print("\nNo trials have been completed yet. Cannot perform analysis.")
        sys.exit(0)
        
    print(f"\n[1] Overall Summary\n" + "-"*20)
    print(f"Total trials in DB: {len(df)}")
    print(f"Completed trials:   {len(complete_trials)}")
    print(f"Pruned trials:      {len(df[df.state == 'PRUNED'])}")
    print(f"Failed trials:      {len(df[df.state == 'FAIL'])}")

    best_trial = study.best_trial
    print(f"\n[2] Best Trial (Score: {best_trial.value:.5f})\n" + "-"*20)
    for key, value in best_trial.params.items():
        print(f"  - {key:<35}: {value}")

    # Parameter Importance Calculation
    top_importances = []
    try:
        # Check for low variance in the *target value*
        score_variance = complete_trials['value'].var()
        if score_variance < 1e-10:
            print("\n[3] Parameter Importance\n" + "-"*20)
            print("INFO: Could not calculate parameter importance.")
            print(f"REASON: The scores of all {len(complete_trials)} completed trials are nearly identical (variance < 1e-10).")
            print("This often means the pruner is very effective and only high-performing trials are completing.")
        else:
            print(f"\n[3] Parameter Importance\n" + "-"*20)
            print("This shows which parameters have the biggest impact on the score.")
            param_importance = optuna.importance.get_param_importances(study, target=lambda t: t.state == optuna.trial.TrialState.COMPLETE)
            
            max_importance = max(param_importance.values()) if param_importance else 0
            sorted_importance = sorted(param_importance.items(), key=lambda item: item[1], reverse=True)
            
            for param, importance in sorted_importance:
                bar_length = int(50 * importance / max_importance) if max_importance > 0 else 0
                bar = '█' * bar_length
                print(f"  - {param:<35}: {importance:.5f} |{bar}")
            
            top_importances = sorted_importance

    except Exception as e:
        # This catch-all is still good practice
        print(f"\n[3] Parameter Importance\n" + "-"*20)
        print(f"Could not calculate parameter importance: {e}")

    # --- Top Performing Trials Table ---
    print(f"\n[4] Top 10 Performing Trials\n" + "-"*20)
    df_display = complete_trials.copy()
    df_display.columns = df_display.columns.str.replace('params_', '')
    sort_ascending = study.direction != optuna.study.StudyDirection.MAXIMIZE
    df_display.sort_values(by='value', ascending=sort_ascending, inplace=True)
    param_cols = list(study.best_params.keys())
    
    # Show all columns for the top 10 trials
    print(df_display.head(10)[['value'] + param_cols].to_string())

    csv_filename = f"{study.study_name}_all_completed_trials.csv"
    df_display[['value'] + param_cols].to_csv(csv_filename, index=False)
    print(f"\nFull data for all {len(df_display)} completed trials exported to: {csv_filename}")

    # --- Search Space Suggestions ---
    print(f"\n[5] Search Space Suggestions\n" + "-"*20)
    if not top_importances:
        print("Cannot generate suggestions without parameter importances.")
        sys.exit(0)

    print("Based on the distribution of parameters in the TOP 20% of trials.")
    top_quantile_threshold = complete_trials['value'].quantile(0.8 if study.direction == optuna.study.StudyDirection.MAXIMIZE else 0.2)
    df_top_trials = complete_trials[complete_trials['value'] >= top_quantile_threshold] if study.direction == optuna.study.StudyDirection.MAXIMIZE else complete_trials[complete_trials['value'] <= top_quantile_threshold]
    print(f"(Analyzing {len(df_top_trials)} trials with scores better than {top_quantile_threshold:.5f})\n")

    for param_name, _ in top_importances[:7]:
        df_col_name = f"params_{param_name}"
        if df_col_name in df_top_trials.columns:
            stats = df_top_trials[df_col_name].describe()
            original_range_low = df[df_col_name].min()
            original_range_high = df[df_col_name].max()

            print(f"-> For parameter '{param_name}':")
            print(f"   Original Search Range: [{original_range_low:.4f}, {original_range_high:.4f}]")
            print(f"   Best Trials' Range:    [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"   Mean: {stats['mean']:.4f}, Median: {df_top_trials[df_col_name].median():.4f}")
            
            if stats['max'] / original_range_high > 0.98:
                print("   NOTE: Best values are hitting the UPPER bound. Consider EXPANDING the range upwards.")
            if original_range_low > 0 and stats['min'] / original_range_low < 1.02:
                 print("   NOTE: Best values are hitting the LOWER bound. Consider EXPANDING the range downwards.")
            print(f"   SUGGESTION: Consider narrowing the next search range to approx. [{stats['25%']:.4f}, {stats['75%']:.4f}]\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze an Optuna study from an SQLite database.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("db_path", help="Path to the Optuna sqlite .db file.")
    parser.add_argument("--study-name", required=False, help="Name of the study to analyze. If not provided, it will be inferred from the db_path filename.")
    args = parser.parse_args()
    analyze_study(args.db_path, args.study_name)