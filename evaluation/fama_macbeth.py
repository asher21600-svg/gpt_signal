"""
Fama-MacBeth (1973) two-step cross-sectional regression.

Step 1 — for each company, regress that company's return time series on its
         signal time series. Get factor loadings β̂_{i,j}.
Step 2 — at each date t, regress the cross-section of returns on the
         cross-section of β̂s. Record adjusted R² at each t.

The distribution of adjusted R² values is what the box plots in the paper
show. Adding a new signal that meaningfully improves cross-sectional fit
shifts the median up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _zscore_by_date(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cross-sectional Z-score: standardize each column within each date."""
    out = panel.copy()
    for c in cols:
        out[c] = panel.groupby("date")[c].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0)
        )
    return out


def _ols_adj_r2(y: np.ndarray, X: np.ndarray) -> float:
    # Guard against empty inputs (can happen at T=1.0 when LLM formulas
    # produce all-NaN columns for some dates)
    if y is None or X is None or y.size == 0 or X.size == 0:
        return float("nan")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    try:
        Xc = sm.add_constant(X, has_constant="add")
    except Exception:
        return float("nan")
    mask = np.isfinite(y) & np.all(np.isfinite(Xc), axis=1)
    if mask.sum() <= Xc.shape[1] + 1:
        return float("nan")
    try:
        return float(sm.OLS(y[mask], Xc[mask]).fit().rsquared_adj)
    except Exception:  # noqa: BLE001
        return float("nan")


def fama_macbeth(panel: pd.DataFrame, signal_cols: list[str],
                 return_col: str) -> pd.DataFrame:
    """
    Returns a DataFrame of one row per date with columns:
        date, n_companies, adj_r2

    `signal_cols` are the regressors used in BOTH step 1 (time-series betas)
    and step 2 (cross-sectional regression of returns on betas).
    """
    panel = _zscore_by_date(panel, signal_cols)

    # Step 1: per-company time-series regression → β̂_{i,j}
    betas = {}
    for tk, sub in panel.groupby("ticker"):
        y = sub[return_col].values
        X = sub[signal_cols].values
        Xc = sm.add_constant(X, has_constant="add")
        mask = np.isfinite(y) & np.all(np.isfinite(Xc), axis=1)
        if mask.sum() <= len(signal_cols) + 1:
            continue
        try:
            res = sm.OLS(y[mask], Xc[mask]).fit()
            betas[tk] = res.params[1:]    # drop intercept
        except Exception:  # noqa: BLE001
            continue
    betas_df = pd.DataFrame(betas, index=signal_cols).T  # rows = tickers
    betas_df.index.name = "ticker"

    # Step 2: per-date cross-section regression of returns on β̂s
    rows = []
    for date, sub in panel.groupby("date"):
        merged = sub.merge(betas_df, left_on="ticker", right_index=True,
                           suffixes=("", "_beta"))
        beta_cols = [f"{c}_beta" if f"{c}_beta" in merged.columns else c
                     for c in signal_cols]
        # Statsmodels disambiguates columns; pick the beta-side ones
        beta_cols = [c if c.endswith("_beta") else c + "_beta"
                     for c in signal_cols]
        beta_cols = [c for c in beta_cols if c in merged.columns]
        if len(beta_cols) != len(signal_cols):
            beta_cols = signal_cols   # no name collision case
        X = merged[beta_cols].values
        y = merged[return_col].values
        rows.append({
            "date": date,
            "n_companies": len(merged),
            "adj_r2": _ols_adj_r2(y, X),
        })
    return pd.DataFrame(rows)


def compare_models(panel: pd.DataFrame, baseline_cols: list[str],
                   new_signal_cols: list[str], return_col: str) -> pd.DataFrame:
    """
    Run Fama-MacBeth for the baseline and for baseline+each new signal.
    Returns wide DataFrame: rows = dates, columns = adj_r2 per model.
    """
    out = {"baseline": fama_macbeth(panel, baseline_cols, return_col)
                          .set_index("date")["adj_r2"]}
    for s in new_signal_cols:
        out[s] = fama_macbeth(panel, baseline_cols + [s], return_col) \
                    .set_index("date")["adj_r2"]
    return pd.DataFrame(out).reset_index()
