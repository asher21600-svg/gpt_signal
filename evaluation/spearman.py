"""
Spearman rank correlation between each signal and the forward return,
computed cross-sectionally at each date and then averaged across dates
(§3.2 of the paper).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def cross_sectional_spearman(panel: pd.DataFrame, signal_cols: list[str],
                             return_col: str) -> pd.DataFrame:
    """
    For each (date, signal) compute the Spearman rho between the signal's
    cross-section and the return cross-section. Returns wide DataFrame
    indexed by date.
    """
    rows = []
    for date, sub in panel.groupby("date"):
        row = {"date": date}
        for col in signal_cols:
            x, y = sub[col].values, sub[return_col].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                row[col] = np.nan
            else:
                rho, _ = spearmanr(x[mask], y[mask])
                row[col] = rho
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def mean_correlation_matrix(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Pearson correlation matrix on ranks (Spearman flavor) — averaged across
    dates. Used for Figure 3/5/etc. style heatmaps.
    """
    mats = []
    for _, sub in panel.groupby("date"):
        ranked = sub[columns].rank()
        mats.append(ranked.corr())
    return sum(mats) / len(mats)
