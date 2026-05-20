"""
FMP price-history adapter — recovers delisted/acquired tickers that yfinance
refuses to serve. Uses /stable/historical-price-eod/light, which is on FMP
Starter and above.

Replaces fetch_returns.fetch_quarterly_forward_returns when FMP_API_KEY is in
the env. Same return schema: long-format DataFrame with date, ticker, ret_1m,
ret_3m columns.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


CACHE_DIR = Path(__file__).parent / ".fmp_cache"
BASE_URL = "https://financialmodelingprep.com/stable"


def _cache_path(symbol: str, kind: str) -> Path:
    return CACHE_DIR / f"prices_{kind}_{symbol}.json"


def _fetch_prices(symbol: str) -> pd.Series:
    """Get FMP daily closes (adjusted) for a single ticker. Cached to disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    cp = _cache_path(symbol, "eod")
    if cp.exists():
        data = json.loads(cp.read_text())
    else:
        if requests is None:
            raise ImportError("pip install requests")
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            raise RuntimeError("FMP_API_KEY not set")
        r = requests.get(
            f"{BASE_URL}/historical-price-eod/light",
            params={"symbol": symbol, "from": "2013-01-01", "to": "2025-12-31",
                    "apikey": api_key},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "Error Message" in data:
            print(f"  [{symbol}] FMP price error: {data['Error Message']}")
            return pd.Series(dtype=float)
        cp.write_text(json.dumps(data))
        time.sleep(0.15)

    if not data:
        return pd.Series(dtype=float)
    df = pd.DataFrame(data)
    if "date" not in df.columns or "price" not in df.columns:
        # /light returns {symbol, date, price}; full version returns {date, open, ...}
        # Try common alternate
        if "close" in df.columns:
            df["price"] = df["close"]
        else:
            return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.set_index("date")["price"].astype(float).sort_index()


def fetch_quarterly_forward_returns_fmp(
    tickers: Iterable[str],
    start: str,
    end: str,
    horizons_months: Iterable[int] = (1, 3),
) -> pd.DataFrame:
    """FMP equivalent of fetch_quarterly_forward_returns. Same schema."""
    rows = []
    for tk in tickers:
        prices = _fetch_prices(tk)
        if prices.empty:
            print(f"  [{tk}] no prices returned")
            continue
        # Trim to requested window
        prices = prices.loc[start:end]
        if prices.empty:
            continue
        # Build one DataFrame per horizon, then merge on (date, ticker)
        per_horizon = []
        for h in horizons_months:
            fwd = prices.pct_change(periods=h * 21).shift(-h * 21)
            fwd_qe = fwd.resample("QE").last()
            per_horizon.append(pd.DataFrame({
                "date": fwd_qe.index,
                "ticker": tk,
                f"ret_{h}m": fwd_qe.values,
            }))
        tk_df = per_horizon[0]
        for h_df in per_horizon[1:]:
            tk_df = tk_df.merge(h_df, on=["date", "ticker"], how="outer")
        rows.append(tk_df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out
