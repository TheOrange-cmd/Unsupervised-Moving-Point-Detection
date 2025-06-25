import argparse
import sys
import pandas as pd
import optuna

# Set pandas display options for better text-based output
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 180)
pd.set_option('display.precision', 5)

def analyze_study(db_path: str, study_name: str):
    """
    Connects to an Optuna SQLite database, loads a study, and performs
    a text-based analysis to guide future search space adjustments.
    """
    # 1. Connect to the Optuna database
    storage_url = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except KeyError:
        print(f"Error: Study '{study_name}' not found in the database '{db_path}'.")
        # Optional: List available studies
        try:
            all_studies = optuna.get_all_study_summaries(storage=storage_url)
            if all_studies:
                print("\nAvailable studies are:")
                for s in all_studies:
                    print(f"- {s.study_name}")
        except Exception:
            pass # Ignore if listing fails
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the study: {e}")
        sys.exit(1)

    # 2. Get all trials into a pandas DataFrame
    df = study.trials_dataframe()

    print("\n" + "="*80)
    print(f"Optuna Study Analysis for: '{study.study_name}'")
    print(f"Direction: {study.direction.name}")
    print("="*80)

    # --- Overall Summary ---
    total_trials = len(df)
    complete_trials = df[df.state == 'COMPLETE']
    
    if complete_trials.empty:
        print("\nNo trials have been completed yet. Cannot perform analysis.")
        sys.exit(0)
        
    print(f"\n[1] Overall Summary\n" + "-"*20)
    print(f"Total trials in DB: {total_trials}")
    print(f"Completed trials:   {len(complete_trials)}")
    print(f"Pruned trials:      {len(df[df.state == 'PRUNED'])}")
    print(f"Failed trials:      {len(df[df.state == 'FAIL'])}")

    # --- Best Trial ---
    best_trial = study.best_trial
    print(f"\n[2] Best Trial (Score: {best_trial.value:.5f})\n" + "-"*20)
    for key, value in best_trial.params.items():
        print(f"  - {key:<30}: {value}")

    # --- Top 10 Trials ---
    print(f"\n[3] Top 10 Performing Trials\n" + "-"*20)
    # Clean up column names for display
    df_display = complete_trials.copy()
    df_display.columns = df_display.columns.str.replace('params_', '')
    
    # Sort by score (assuming higher is better)
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        df_display.sort_values(by='value', ascending=False, inplace=True)
    else:
        df_display.sort_values(by='value', ascending=True, inplace=True)

    # Select only the relevant parameter columns plus the value
    param_cols = list(study.best_params.keys())
    display_cols = ['value'] + param_cols
    print(df_display.head(10)[display_cols].to_string())

    # --- Parameter Importance ---
    print(f"\n[4] Parameter Importance\n" + "-"*20)
    print("This shows which parameters have the biggest impact on the score.")
    try:
        # Use a target to only consider completed trials for importance calculation
        param_importance = optuna.importance.get_param_importances(
            study, target=lambda t: t.state == optuna.trial.TrialState.COMPLETE
        )
        
        importance_df = pd.DataFrame.from_dict(param_importance, orient='index', columns=['Importance'])
        importance_df.sort_values(by='Importance', ascending=False, inplace=True)
        print(importance_df)
        
        top_importances = list(param_importance.items())
        top_importances.sort(key=lambda item: item[1], reverse=True)

    except Exception as e:
        print(f"Could not calculate parameter importance: {e}")
        top_importances = []


    # --- Search Space Suggestions ---
    print(f"\n[5] Search Space Suggestions\n" + "-"*20)
    print("Based on the distribution of parameters in the TOP 20% of trials.")

    if not top_importances:
        print("Cannot generate suggestions without parameter importances.")
        sys.exit(0)

    # Get the threshold for the top 20% of scores
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        top_quantile_threshold = complete_trials['value'].quantile(0.8)
        df_top_trials = complete_trials[complete_trials['value'] >= top_quantile_threshold]
    else:
        top_quantile_threshold = complete_trials['value'].quantile(0.2)
        df_top_trials = complete_trials[complete_trials['value'] <= top_quantile_threshold]

    print(f"(Analyzing {len(df_top_trials)} trials with scores >= {top_quantile_threshold:.5f})\n")

    # Analyze the top 5 most important parameters
    for param_name, _ in top_importances[:5]:
        # The column name in the DataFrame includes "params_"
        df_col_name = f"params_{param_name}"
        
        if df_col_name in df_top_trials.columns:
            stats = df_top_trials[df_col_name].describe()
            
            print(f"-> For parameter '{param_name}':")
            print(f"   The best trials have values concentrated between {stats['min']:.4f} and {stats['max']:.4f}.")
            print(f"   Mean: {stats['mean']:.4f}, Median: {df_top_trials[df_col_name].median():.4f}")
            print(f"   SUGGESTION: Consider narrowing the search range to approx. [{stats['25%']:.4f}, {stats['75%']:.4f}]\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze an Optuna study from an SQLite database.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "db_path", 
        help="Path to the Optuna sqlite .db file."
    )
    parser.add_argument(
        "--study-name", 
        required=True,
        help="Name of the study to analyze (e.g., 'm-detector-tuning-phase-5')."
    )
    
    args = parser.parse_args()
    
    analyze_study(args.db_path, args.study_name)