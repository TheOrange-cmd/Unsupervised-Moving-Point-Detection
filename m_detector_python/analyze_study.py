# simple_analyze_study.py
import optuna
import pandas as pd
import numpy as np

# --- Configuration ---
STUDY_NAME = "optuna_full_3"
DB_FILENAME = f"sqlite:///{STUDY_NAME}.db"

def analyze_parameter_importance(study):
    """Show which parameters matter most"""
    print("=" * 50)
    print("PARAMETER IMPORTANCE")
    print("=" * 50)
    
    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"{'Parameter':<20} {'Importance':<10}")
        print("-" * 30)
        for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            print(f"{param:<20} {importance:.4f}")
    except Exception as e:
        print(f"Could not calculate parameter importance: {e}")
    print()

def analyze_parameter_ranges(study):
    """Check if parameters are hitting boundaries"""
    print("=" * 50)
    print("PARAMETER RANGE ANALYSIS")
    print("=" * 50)
    
    # Get all completed trials
    trials_df = study.trials_dataframe()
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
    
    if completed_trials.empty:
        print("No completed trials found.")
        return
    
    # Get parameter columns (they start with 'params_')
    param_cols = [col for col in completed_trials.columns if col.startswith('params_')]
    
    print(f"{'Parameter':<20} {'Min Value':<12} {'Max Value':<12} {'Best 10% Range':<20}")
    print("-" * 70)
    
    # Get top 10% of trials
    n_top = max(1, len(completed_trials) // 10)
    top_trials = completed_trials.nlargest(n_top, 'value')  # assuming minimization
    
    for col in param_cols:
        param_name = col.replace('params_', '')
        values = completed_trials[col].dropna()
        top_values = top_trials[col].dropna()
        
        if len(values) > 0:
            min_val = values.min()
            max_val = values.max()
            if len(top_values) > 0:
                top_range = f"{top_values.min():.4f} - {top_values.max():.4f}"
            else:
                top_range = "N/A"
            
            print(f"{param_name:<20} {min_val:<12.4f} {max_val:<12.4f} {top_range:<20}")
    print()

def analyze_best_trials(study, n=10):
    """Show the best trials and their parameters"""
    print("=" * 50)
    print(f"TOP {n} TRIALS")
    print("=" * 50)
    
    trials_df = study.trials_dataframe()
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
    
    if completed_trials.empty:
        print("No completed trials found.")
        return
    
    top_trials = completed_trials.nlargest(n, 'value') # assuming minimization
    param_cols = [col for col in completed_trials.columns if col.startswith('params_')]
    
    for i, (idx, trial) in enumerate(top_trials.iterrows(), 1):
        print(f"Trial {i}: Score = {trial['value']:.6f}")
        for col in param_cols:
            param_name = col.replace('params_', '')
            if pd.notna(trial[col]):
                print(f"  {param_name}: {trial[col]}")
        print()

def analyze_parameter_stats(study):
    """Show basic statistics for each parameter"""
    print("=" * 50)
    print("PARAMETER STATISTICS")
    print("=" * 50)
    
    trials_df = study.trials_dataframe()
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
    
    if completed_trials.empty:
        print("No completed trials found.")
        return
    
    param_cols = [col for col in completed_trials.columns if col.startswith('params_')]
    
    print(f"{'Parameter':<20} {'Mean':<10} {'Std':<10} {'Median':<10}")
    print("-" * 50)
    
    for col in param_cols:
        param_name = col.replace('params_', '')
        values = completed_trials[col].dropna()
        
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            median_val = values.median()
            print(f"{param_name:<20} {mean_val:<10.4f} {std_val:<10.4f} {median_val:<10.4f}")
    print()

def check_convergence(study):
    """Check if the study is converging"""
    print("=" * 50)
    print("CONVERGENCE ANALYSIS")
    print("=" * 50)
    
    trials_df = study.trials_dataframe()
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE'].sort_values('number')
    
    if len(completed_trials) < 10:
        print("Not enough trials for convergence analysis.")
        return
    
    values = completed_trials['value'].values
    
    # Calculate running minimum (for minimization problems)
    running_max = np.maximum.accumulate(values)
    
    # Show last 20 trials' progress
    recent_trials = min(20, len(values))
    print(f"Last {recent_trials} trials:")
    print(f"{'Trial':<8} {'Score':<12} {'Best So Far':<12} {'Improvement':<12}")
    print("-" * 50)
    for i in range(-recent_trials, 0):
        trial_num = completed_trials.iloc[i]['number'] # Use the actual trial number
        score = values[i]
        best_so_far = running_max[i]
        improvement = "Yes" if score == best_so_far else "No"
        print(f"{trial_num:<8} {score:<12.6f} {best_so_far:<12.6f} {improvement:<12}")
    
    # Simple convergence check
    last_10_improvements = sum(1 for i in range(-10, 0) if values[i] == running_max[i])
    print(f"\nImprovements in last 10 trials: {last_10_improvements}")
    if last_10_improvements == 0:
        print("⚠️  No improvements in last 10 trials - study may have converged")
    elif last_10_improvements <= 2:
        print("⚠️  Few improvements recently - consider stopping soon")
    else:
        print("✅ Still making progress")
    print()

def main():
    print(f"Loading study '{STUDY_NAME}' from '{DB_FILENAME}'...")
    
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_FILENAME)
    except KeyError:
        print(f"Error: Study '{STUDY_NAME}' not found in the database file.")
        return
    
    print(f"Study loaded successfully with {len(study.trials)} trials.")
    print(f"Best score so far: {study.best_value:.6f}")
    print()
    
    # Run all analyses
    analyze_parameter_importance(study)
    analyze_parameter_ranges(study)
    analyze_best_trials(study)
    analyze_parameter_stats(study)
    check_convergence(study)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()