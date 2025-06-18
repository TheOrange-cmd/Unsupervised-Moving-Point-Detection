# src/visualization/metrics_plots.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc # For Area Under Curve
from typing import List, Dict, Any, Optional

def plot_roc_curve(
    fpr_list: List[float], 
    tpr_list: List[float], 
    experiment_label: str = "Experiment",
    ax: Optional[plt.Axes] = None
) -> None:
    """Plots a single ROC curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    
    roc_auc = auc(fpr_list, tpr_list) if fpr_list and tpr_list else 0.0
    ax.plot(fpr_list, tpr_list, lw=2, label=f'{experiment_label} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True)

def plot_precision_recall_curve(
    recall_list: List[float], 
    precision_list: List[float], 
    experiment_label: str = "Experiment",
    ax: Optional[plt.Axes] = None
) -> None:
    """Plots a single Precision-Recall curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    # Calculate AUC for PR curve (Average Precision)
    # Note: sklearn.metrics.auc expects x to be increasing. Recall often decreases.
    # Ensure recall is sorted if necessary, or use sklearn.metrics.average_precision_score directly
    # if you have raw scores and y_true. For pre-calculated (recall, precision) pairs:
    pr_auc = auc(np.array(recall_list), np.array(precision_list)) if recall_list and precision_list else 0.0
    
    ax.plot(recall_list, precision_list, lw=2, label=f'{experiment_label} (AP = {pr_auc:.2f})')
    ax.set_xlabel('Recall (TPR)')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left") # Or "best"
    ax.grid(True)

def plot_experiment_comparison_bar_chart(
    experiments_data: List[Dict[str, Any]], # List of experiment summary dicts
    metric_key: str = "F1", # Key for the metric to plot (e.g., "F1", "Precision", "Recall")
    id_key: str = "experiment_id", # Key for experiment identifier
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None
) -> None:
    """Plots a bar chart comparing a specific metric across multiple experiments."""
    if not experiments_data:
        print("No experiment data to plot.")
        return
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, len(experiments_data) * 0.6), 6))

    experiment_ids = [data.get(id_key, f"Exp {i}") for i, data in enumerate(experiments_data)]
    metric_values = [data.get(metric_key, 0.0) for data in experiments_data]

    bars = ax.bar(experiment_ids, metric_values, color='skyblue')
    ax.set_xlabel('Experiment ID / Configuration')
    ax.set_ylabel(metric_key)
    ax.set_title(title if title else f'{metric_key} Comparison Across Experiments')
    ax.set_ylim([0, max(1.05, np.max(metric_values) * 1.1 if metric_values else 1.05)]) # Ensure y-axis goes at least to 1.0 or slightly above max
    ax.tick_params(axis='x', rotation=45, ha='right') # Rotate x-labels if long
    
    # Add text labels on bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

    plt.tight_layout() # Adjust layout to prevent labels from overlapping