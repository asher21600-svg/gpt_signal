"""
Plotting helpers — heatmaps for Spearman correlation (Figs 3/5/...) and
box plots for adjusted R² (Figs 4/6/...).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def correlation_heatmap(corr: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(corr)), max(5, 0.5 * len(corr))))
    sns.heatmap(corr, annot=True, fmt=".2f", center=0, cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def adj_r2_boxplot(adj_r2: pd.DataFrame, title: str, out_path: Path) -> None:
    """`adj_r2` from fama_macbeth.compare_models — date column + one per model."""
    plot_df = adj_r2.drop(columns=["date"]).melt(var_name="model", value_name="adj_r2")
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * plot_df["model"].nunique()), 5))
    sns.boxplot(data=plot_df, x="model", y="adj_r2", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Adjusted R² (cross-sectional, per date)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
