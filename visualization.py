"""Visualization layer — creates informative plots from pre‑computed
statistical facts and the original data.

This module NEVER computes statistics; it uses the supplied DataFrame
only for plotting raw values.  All plotting functions accept an
optional `ax` for composability and return the figure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml_toolkit.config import MLToolkitConfig, default_config
from ml_toolkit.schema import (
    CorrelationPair,
    FeatureProfile,
    NumericDistributionProfile,
    OutlierReport,
    TargetProfile,
)

# ------------------------------------------------------------------
# Plot configuration helper
# ------------------------------------------------------------------
def _apply_style(config: MLToolkitConfig) -> None:
    """Set global Seaborn style and palette from config."""
    sns.set_style(config.plot_style)
    sns.set_palette(config.color_palette)


# ------------------------------------------------------------------
# Distribution plots
# ------------------------------------------------------------------
def plot_numeric_distributions(
    df: pd.DataFrame,
    analysis_result: Dict[str, Any],
    max_cols: int = 20,
    show_outlier_lines: bool = True,
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Combined histogram + boxplot for numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame (must contain the numeric columns).
    analysis_result : dict
        Output of `EDAAnalyzer.run()`.
    max_cols : int
        Maximum number of numeric columns to plot.
    show_outlier_lines : bool
        If True, draw IQR outlier bounds as dashed lines on the histogram.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    profiles = _get_profiles(analysis_result)
    outliers = _get_outliers(analysis_result)
    numeric_profiles = [
        p for p in profiles if p.numeric_profile and not p.is_constant
    ]
    if not numeric_profiles:
        return None

    cfg = config or default_config
    _apply_style(cfg)

    profs_to_plot = numeric_profiles[:max_cols]
    n = len(profs_to_plot)
    fig, axes = plt.subplots(
        n, 2, figsize=(cfg.figure_size[0] * 1.2, 4 * n), squeeze=False
    )
    fig.suptitle("Numeric Feature Distributions", fontsize=16)

    # Build outlier lookup
    outlier_dict = {o.column: o for o in outliers}

    for i, prof in enumerate(profs_to_plot):
        col = prof.column
        num = prof.numeric_profile
        data = df[col].dropna()

        # --- Histogram + KDE ---
        ax_hist = axes[i, 0]
        sns.histplot(data, kde=True, ax=ax_hist, color="steelblue",
                     edgecolor="white")
        if num:
            ax_hist.axvline(num.mean, color="red", linestyle="--",
                            label=f"Mean={num.mean:.2f}")
            ax_hist.axvline(num.median, color="green", linestyle="-",
                            label=f"Median={num.median:.2f}")
        if show_outlier_lines and col in outlier_dict:
            o = outlier_dict[col]
            if o.lower_bound is not None:
                ax_hist.axvline(o.lower_bound, color="orange",
                                linestyle=":", label="IQR lower")
            if o.upper_bound is not None:
                ax_hist.axvline(o.upper_bound, color="orange",
                                linestyle=":", label="IQR upper")
        ax_hist.set_title(f"{col} (skew={num.skewness:.2f})")
        ax_hist.legend(loc="upper right")

        # --- Boxplot ---
        ax_box = axes[i, 1]
        sns.boxplot(x=data, ax=ax_box, color="lightblue")
        ax_box.set_title(f"{col} boxplot")
        if num:
            ax_box.set_xlabel(
                f"Min={num.min:.2f}, Max={num.max:.2f}"
            )

    plt.tight_layout()
    return fig


def plot_target_distribution(
    df: pd.DataFrame,
    analysis_result: Dict[str, Any],
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Plot the distribution of the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame (must contain the target column).
    analysis_result : dict
        Output of `EDAAnalyzer.run()`.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    target_profile = _get_target_profile(analysis_result)
    if target_profile is None:
        return None

    cfg = config or default_config
    _apply_style(cfg)

    target_col = target_profile.column
    data = df[target_col].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(cfg.figure_size[0], 5))
    # Distribution
    if target_profile.is_regression:
        sns.histplot(data, kde=True, ax=axes[0], color="teal")
        axes[0].set_title(f"Target distribution: {target_col}")
        # Boxplot
        sns.boxplot(x=data, ax=axes[1], color="lightgreen")
        axes[1].set_title(f"Target boxplot: {target_col}")
    else:
        # Classification
        value_counts = data.value_counts()
        axes[0].bar(value_counts.index.astype(str), value_counts.values,
                    color="salmon")
        axes[0].set_title(f"Target classes: {target_col}")
        axes[0].set_ylabel("Count")
        axes[1].axis("off")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Correlation plots
# ------------------------------------------------------------------
def plot_correlation_heatmap(
    df: pd.DataFrame,
    analysis_result: Dict[str, Any],
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Plot a Pearson correlation heatmap for numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    analysis_result : dict
        Output of `EDAAnalyzer.run()`.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    correlation_pairs = _get_correlations(analysis_result)
    if not correlation_pairs:
        return None

    # Get list of columns that appear in correlations
    cols_in_corr = set()
    for pair in correlation_pairs:
        cols_in_corr.add(pair.feature_a)
        cols_in_corr.add(pair.feature_b)
    # Compute full correlation matrix only for those columns
    numeric_cols = [c for c in cols_in_corr if c in df.columns]
    if len(numeric_cols) < 2:
        return None

    cfg = config or default_config
    _apply_style(cfg)

    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    fig, ax = plt.subplots(figsize=(cfg.figure_size[0], cfg.figure_size[1]))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    return fig


def plot_top_correlations_bar(
    analysis_result: Dict[str, Any],
    top_n: int = 10,
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Bar chart of the top absolute correlations.

    Parameters
    ----------
    analysis_result : dict
        Output of `EDAAnalyzer.run()`.
    top_n : int
        Number of pairs to display.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    pairs = _get_correlations(analysis_result)
    if not pairs:
        return None

    cfg = config or default_config
    _apply_style(cfg)

    # Sort by absolute coefficient descending
    sorted_pairs = sorted(pairs, key=lambda x: abs(x.coefficient), reverse=True)
    top_pairs = sorted_pairs[:top_n]

    labels = [f"{p.feature_a}\nvs {p.feature_b}" for p in top_pairs]
    values = [abs(p.coefficient) for p in top_pairs]

    fig, ax = plt.subplots(figsize=(cfg.figure_size[0], 0.5 * len(labels)))
    ax.barh(range(len(labels)), values, color="purple", edgecolor="black")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Absolute Pearson Correlation")
    ax.set_title("Top Feature Correlations")
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Missing values visualisation
# ------------------------------------------------------------------
def plot_missing_heatmap(
    df: pd.DataFrame,
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Heatmap showing missing values across columns (yellow = missing).

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    # Limit to 50 columns for readability
    cols = df.columns[:50]
    missing_data = df[cols].isnull()
    if not missing_data.any().any():
        return None

    cfg = config or default_config
    _apply_style(cfg)

    fig, ax = plt.subplots(figsize=(cfg.figure_size[0], 0.5 * len(cols)))
    # Downsample rows if > 5000
    if len(df) > 5000:
        idx = np.random.choice(len(df), 5000, replace=False)
        missing_data = missing_data.iloc[idx]
    sns.heatmap(missing_data.T, cmap=["#ffffff", "#f1c40f"],
                cbar=False, ax=ax, xticklabels=False)
    ax.set_xlabel("Rows (sample)")
    ax.set_ylabel("Columns")
    ax.set_title("Missing Values Heatmap (yellow = missing)")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Outlier summary plot
# ------------------------------------------------------------------
def plot_outlier_summary(
    analysis_result: Dict[str, Any],
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Horizontal bar chart showing outlier percentages per numeric column.

    Parameters
    ----------
    analysis_result : dict
        Output of `EDAAnalyzer.run()`.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    outliers = _get_outliers(analysis_result)
    if not outliers:
        return None

    cfg = config or default_config
    _apply_style(cfg)

    # Filter out zero outliers
    non_zero = [o for o in outliers if o.outlier_count > 0]
    if not non_zero:
        return None

    labels = [o.column for o in non_zero]
    percentages = [o.outlier_percent for o in non_zero]

    fig, ax = plt.subplots(figsize=(cfg.figure_size[0], 0.4 * len(labels)))
    ax.barh(range(len(labels)), percentages, color="coral", edgecolor="black")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Outlier Percentage (%)")
    ax.set_title("Outlier Percentage per Numeric Feature (IQR)")
    ax.invert_yaxis()
    for i, p in enumerate(percentages):
        ax.text(p + 0.5, i, f"{p:.1f}%", va="center")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Feature correlation with target (numeric)
# ------------------------------------------------------------------
def plot_target_correlations(
    df: pd.DataFrame,
    analysis_result: Dict[str, Any],
    top_n: int = 15,
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Bar chart of Pearson correlations between features and a numeric target.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    analysis_result : dict
        Output of `EDAAnalyzer.run()`.
    top_n : int
        Number of most correlated features to show.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    target_profile = _get_target_profile(analysis_result)
    if not target_profile or not target_profile.is_regression:
        return None

    target_col = target_profile.column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols or len(numeric_cols) < 2:
        return None

    cfg = config or default_config
    _apply_style(cfg)

    # Compute correlations with target (this is a simple pandas operation,
    # not a heavy statistics computation)
    corrs = df[numeric_cols].corrwith(df[target_col]).drop(target_col)
    corrs_sorted = corrs.abs().sort_values(ascending=False).head(top_n)
    # Get signed values
    signed_corrs = corrs[corrs_sorted.index]

    fig, ax = plt.subplots(figsize=(cfg.figure_size[0], 0.5 * len(signed_corrs)))
    colors = ["teal" if c >= 0 else "coral" for c in signed_corrs]
    ax.barh(range(len(signed_corrs)), signed_corrs.values, color=colors,
            edgecolor="black")
    ax.set_yticks(range(len(signed_corrs)))
    ax.set_yticklabels(signed_corrs.index)
    ax.set_xlabel("Pearson Correlation")
    ax.set_title(f"Feature Correlations with Target: {target_col}")
    ax.invert_yaxis()
    for i, v in enumerate(signed_corrs.values):
        ax.text(v + 0.01 if v >= 0 else v - 0.08, i, f"{v:.2f}", va="center")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Helper getters (repeated for clarity)
# ------------------------------------------------------------------
def _get_profiles(analysis_result: Dict[str, Any]) -> List[FeatureProfile]:
    return analysis_result.get("feature_profiles", [])


def _get_outliers(analysis_result: Dict[str, Any]) -> List[OutlierReport]:
    return analysis_result.get("outliers", [])


def _get_correlations(analysis_result: Dict[str, Any]) -> List[CorrelationPair]:
    return analysis_result.get("correlation_pairs", [])


def _get_target_profile(analysis_result: Dict[str, Any]) -> Optional[TargetProfile]:
    return analysis_result.get("target_profile")