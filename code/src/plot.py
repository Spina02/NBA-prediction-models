import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Model name mapping for shorter display names
MODEL_NAME_MAP = {
    'Dummy Baseline': 'Dummy',
    'Logistic Regression': 'LR',
    'Decision Tree': 'DT',
    'K-Nearest Neighbors': 'KNN',
    'Random Forest': 'RF',
    'XGBoost': 'XGB',
    'Support Vector Machine': 'SVM'
}

# Consistent pastel color palette for models (slightly darker for visibility)
MODEL_COLORS = {
    'Dummy Baseline': '#A0A0A0',      # Medium Gray
    'Logistic Regression': '#E88A96',   # Darker Pastel Pink
    'Decision Tree': '#7BCF8B',       # Darker Pastel Green
    'K-Nearest Neighbors': '#7EC8E3',  # Darker Pastel Blue
    'Random Forest': '#C9A0DC',        # Darker Pastel Purple
    'XGBoost': '#E8C98A',             # Darker Pastel Orange/Peach
    'Support Vector Machine': '#E8E87A' # Darker Pastel Yellow
}

# Canonical model order: Dummy first, then same order everywhere
MODEL_ORDER = [
    'Dummy Baseline', 'Logistic Regression', 'Decision Tree',
    'K-Nearest Neighbors', 'Random Forest', 'XGBoost', 'Support Vector Machine'
]

# Default metrics for line plots and boxplots (must match train_models output column names)
DEFAULT_METRIC_COLS = ['Accuracy', 'F1', 'AUC', 'FPR', 'FNR']


# ------------------------------- Visualization -------------------------------
def visualize_results(
    results,
    output_dir: Path,
    metric_cols: list[str] | None = None,
    box_metrics: list[str] | None = None,
) -> None:
    """Plot and print summary of results (list of dicts or DataFrame).
    
    Args:
        results: List of metric dicts or DataFrame from train_models.
        output_dir: Directory to save plots.
        metric_cols: Metrics for the line-plot figure. If None, uses DEFAULT_METRIC_COLS.
        box_metrics: Metrics for the boxplot figure (e.g. ['Accuracy', 'AUC']). If None, uses same as metric_cols.
    """
    if isinstance(results, pd.DataFrame):
        results_df = results
        if results_df.empty:
            print("\n>>> No results were produced. Please verify that the dataset contains at least three seasons and non-empty test sets.")
            return
    else:
        if not results:
            print("\n>>> No results were produced. Please verify that the dataset contains at least three seasons and non-empty test sets.")
            return
        results_df = pd.DataFrame(results)

    metric_cols = metric_cols if metric_cols is not None else DEFAULT_METRIC_COLS
    box_metrics = box_metrics if box_metrics is not None else metric_cols
    # Restrict to columns that exist
    metric_cols = [c for c in metric_cols if c in results_df.columns]
    box_metrics = [c for c in box_metrics if c in results_df.columns]
    if not metric_cols:
        metric_cols = [c for c in DEFAULT_METRIC_COLS if c in results_df.columns]
    if not box_metrics:
        box_metrics = metric_cols

    # Create a copy for plotting with shorter names
    results_df_plot = results_df.copy()
    results_df_plot['Model_Short'] = results_df_plot['Model'].map(MODEL_NAME_MAP).fillna(results_df_plot['Model'])

    # Calculate averages
    print("\n>>> Average Metrics by Model:")
    avg_cols = metric_cols + (['Time_s'] if 'Time_s' in results_df.columns else [])
    avg_cols = [c for c in avg_cols if c in results_df.columns]
    print(results_df.groupby('Model')[avg_cols].mean().round(4))

    # Create subplots with shared x-axis (one per metric)
    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * max(1, n_metrics)), sharex=True)
    if n_metrics == 1:
        axes = [axes]
    fig.suptitle('Model Comparison: Walk-Forward Validation', fontsize=16, fontweight='bold')

    # Use canonical order (Dummy first); keep only models present in results
    unique_models = [m for m in MODEL_ORDER if m in results_df['Model'].values]
    palette = [MODEL_COLORS.get(model, '#000000') for model in unique_models]
    markers = ['o', 's', '^', 'D', 'v', 'p', 'P'][:n_metrics]

    for ax, metric, marker in zip(axes, metric_cols, markers):
        sns.lineplot(data=results_df, x='Season', y=metric, hue='Model',
                     hue_order=unique_models, palette=palette, marker=marker, ax=ax,
                     legend=(ax == axes[0]))
        ax.set_title(metric if metric not in ('F1',) else 'F1 Score (macro)', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xlabel('' if ax != axes[-1] else 'Season', fontsize=10)
        ax.grid(True, alpha=0.3)
        if metric in ('FPR', 'FNR'):
            ax.invert_yaxis()
        if ax != axes[0] and ax.get_legend() is not None:
            ax.get_legend().remove()

    # Single shared legend in one line below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    # Map full names to short names for legend
    labels_short = [MODEL_NAME_MAP.get(label, label) for label in labels]
    fig.legend(handles, labels_short, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               ncol=len(unique_models), frameon=True, fontsize=9)

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    # Save the plot before showing it (bbox_inches so legend is included)
    plot_path = output_dir / "model_comparison.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()

    # ---------------- Boxplots by model (separate figure) ----------------
    fig_box, axes_box = plt.subplots(1, len(box_metrics), figsize=(6 * len(box_metrics), 6), sharey=False)
    if len(box_metrics) == 1:
        axes_box = [axes_box]

    # Use short names for boxplots; same order as line plot (Dummy first)
    results_df_box = results_df.copy()
    results_df_box['Model'] = results_df_box['Model'].map(MODEL_NAME_MAP).fillna(results_df_box['Model'])
    unique_models_short = [MODEL_NAME_MAP[m] for m in unique_models]

    # Build palette dict for boxplot (avoids FutureWarning; seaborn expects dict when using order)
    reverse_name_map = {v: k for k, v in MODEL_NAME_MAP.items()}
    palette_dict = {
        short: MODEL_COLORS.get(reverse_name_map.get(short, short), '#000000')
        for short in unique_models_short
    }

    for ax, metric in zip(axes_box, box_metrics):
        sns.boxplot(
            data=results_df_box, x='Model', y=metric, hue='Model',
            order=unique_models_short, hue_order=unique_models_short,
            palette=palette_dict, legend=False, ax=ax,
        )
        title = f'{metric} (macro) by Model' if metric == 'F1' else f'{metric} by Model'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        if metric in ['FPR', 'FNR']:
            ax.invert_yaxis()

    plt.tight_layout()
    boxplot_path = output_dir / "model_boxplots.png"
    plt.savefig(boxplot_path, bbox_inches='tight')
    plt.show()

    # Also create a summary table visualization
    print("\n>>> Summary Statistics by Model:")
    summary_cols = [c for c in metric_cols + box_metrics if c in results_df.columns]
    summary_cols = list(dict.fromkeys(summary_cols))  # dedupe preserving order
    if summary_cols:
        summary_stats = results_df.groupby('Model')[summary_cols].agg(['mean', 'std', 'min', 'max']).round(4)
        print(summary_stats)