"""
Pull daily adjusted closes and compute k-month forward returns at each
quarter-end date. The forward shift is what makes the eval honest — at time t
we use the signal value, and we score it against the return realized between
t and t+k.
"""
from __future__ import annotations

import os

import pandas as pd
from typing import Iterable

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


def fetch_quarterly_forward_returns(
    tickers: Iterable[str],
    start: str,
    end: str,
    horizons_months: Iterable[int] = (1, 3),
) -> pd.DataFrame:
    """Pull daily adjusted closes and compute k-month forward returns at each
    quarter-end date. Routes to FMP when FMP_API_KEY is set (covers delisted
    tickers); otherwise falls back to yfinance (free, misses delisted)."""
    # Prefer FMP if available — recovers delisted tickers that yfinance refuses
    if os.environ.get("FMP_API_KEY") and os.environ.get("DATA_SOURCE", "").lower() != "yfinance":
        from .fetch_returns_fmp import fetch_quarterly_forward_returns_fmp
        print("  (prices via FMP)")
        return fetch_quarterly_forward_returns_fmp(
            tickers, start=start, end=end, horizons_months=horizons_months,
        )

    if yf is None:
        raise ImportError("Install yfinance, or implement your own adapter.")
    px = yf.download(
        list(tickers), start=start, end=end, auto_adjust=True,
        progress=False, group_by="ticker",
    )
    if isinstance(px.columns, pd.MultiIndex):
        closes = pd.concat(
            {t: px[t]["Close"] for t in set(px.columns.get_level_values(0))},
            axis=1,
        )
    else:
        closes = px["Close"].to_frame(list(tickers)[0])

    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    qend = closes.resample("QE").last()

    rows = []
    for h in horizons_months:
        # ~21 trading days per month; we resampled quarterly so use ceil-division
        periods = max(1, h // 3)  # 1m → next-day-of-next-quarter is messy; see note
        # Cleaner: shift by h months on a daily index, then sample at quarter-ends
        fwd = closes.pct_change(periods=h * 21).shift(-h * 21)
        fwd_qend = fwd.resample("QE").last()
        long = fwd_qend.stack().reset_index()
        long.columns = ["date", "ticker", f"ret_{h}m"]
        rows.append(long.set_index(["date", "ticker"]))
    out = pd.concat(rows, axis=1).reset_index()
    return out
