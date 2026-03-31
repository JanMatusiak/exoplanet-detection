"""Plotting helpers extracted from notebook 02."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)


def plot_feature_histograms(
    df: pd.DataFrame,
    *,
    bins: int = 20,
    layout: tuple[int, int] = (10, 2),
    figsize: tuple[int, int] = (10, 60),
) -> None:
    df.hist(bins=bins, layout=layout, figsize=figsize)
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap(
    corr: pd.DataFrame,
    *,
    title: str = "Correlation (Spearman/Pearson)",
    mask_upper: bool = True,
    annotate: bool = False,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot a correlation heatmap using matplotlib."""
    corr_vals = corr.values.copy()

    if mask_upper:
        mask = np.triu(np.ones_like(corr_vals, dtype=bool), k=1)
        corr_plot = np.ma.array(corr_vals, mask=mask)
    else:
        corr_plot = corr_vals

    n = corr.shape[0]
    if figsize is None:
        figsize = (max(6.0, 0.5 * n), max(5.0, 0.5 * n))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_plot, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="corr")

    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(n):
            for j in range(n):
                if mask_upper and j > i:
                    continue
                value = corr_vals[i, j]
                if np.isnan(value):
                    continue
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_binary(
    y_true,
    y_pred,
    *,
    labels: tuple[int, int] = (0, 1),
    normalize: str | None = "true",
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple[float, float] = (5.2, 4.2),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a binary confusion matrix and return figure + axes."""
    fig, ax = plt.subplots(figsize=figsize)
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=list(labels),
        normalize=normalize,
        cmap=cmap,
        colorbar=False,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_roc_curve_binary(
    y_true,
    y_score,
    *,
    title: str = "ROC Curve",
    plot_chance_level: bool = True,
    figsize: tuple[float, float] = (5.2, 4.2),
) -> tuple[plt.Figure, plt.Axes, float]:
    """Plot ROC curve for binary classification and return figure + AUC."""
    roc_auc = float(roc_auc_score(y_true, y_score))
    fig, ax = plt.subplots(figsize=figsize)
    RocCurveDisplay.from_predictions(
        y_true,
        y_score,
        ax=ax,
        name=f"ROC AUC = {roc_auc:.3f}",
        plot_chance_level=plot_chance_level,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax, roc_auc


def plot_pr_curve_binary(
    y_true,
    y_score,
    *,
    title: str = "Precision-Recall Curve",
    plot_chance_level: bool = True,
    figsize: tuple[float, float] = (5.2, 4.2),
) -> tuple[plt.Figure, plt.Axes, float]:
    """Plot precision-recall curve for binary classification and return figure + AP."""
    average_precision = float(average_precision_score(y_true, y_score))
    fig, ax = plt.subplots(figsize=figsize)
    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_score,
        ax=ax,
        name=f"AP = {average_precision:.3f}",
        plot_chance_level=plot_chance_level,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax, average_precision
