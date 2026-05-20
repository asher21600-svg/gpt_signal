"""
Merge fundamentals and forward returns into the long-format panel used by the
rest of the pipeline.
"""
from __future__ import annotations

import pandas as pd


def build_panel(
    fundamentals: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        date, ticker, pe, pb, roa, roe, fcf, pcf, ebitda, gm, nm, sps,
        ret_1m, ret_3m
    Rows with any NaN in signals or forward returns are dropped.
    """
    f = fundamentals.copy()
    f["date"] = pd.to_datetime(f["date"]).dt.to_period("Q").dt.end_time.dt.normalize()
    r = forward_returns.copy()
    r["date"] = pd.to_datetime(r["date"]).dt.to_period("Q").dt.end_time.dt.normalize()
    panel = f.merge(r, on=["date", "ticker"], how="inner")
    panel = panel.dropna()
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)
