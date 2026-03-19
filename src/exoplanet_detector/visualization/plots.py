"""Plotting helpers extracted from notebook 02."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
