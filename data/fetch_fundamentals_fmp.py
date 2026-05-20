"""
FinancialModelingPrep adapter — drop-in alternative to yfinance for
fundamentals. Free tier gives 250 calls/day; we use 2 calls per ticker
(`ratios` + `key-metrics`), so all 93 paper tickers fit comfortably.

Endpoints used (all v3):
  /ratios/{ticker}?period=quarter        — pre-computed quarterly ratios
  /key-metrics/{ticker}?period=quarter   — pre-computed quarterly per-share metrics

We cache every HTTP response to disk so iteration doesn't hit the daily limit.
Delete `data/.fmp_cache/` to force a refresh.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


CACHE_DIR = Path(__file__).parent / ".fmp_cache"
# FMP migrated to `/stable/` endpoints in August 2025. The old /api/v3/ paths
# are now only available to legacy subscribers. New (post-Aug-2025) accounts —
# including Starter — must use /stable/. Symbol is a query param, not path.
BASE_URL = "https://financialmodelingprep.com/stable"


# Map FMP field names → our column names. Keep this in one place so when FMP
# renames a field (they sometimes do) you only edit it here.
# Field names confirmed against the live /stable/ratios + /stable/key-metrics
# endpoint responses (FMP renamed two ratios in the v3→stable migration:
# priceEarningsRatio → priceToEarningsRatio, priceCashFlowRatio →
# priceToOperatingCashFlowRatio). The rest of the names are unchanged.
_FMP_FIELD_MAP = {
    "pe":     "priceToEarningsRatio",            # /ratios — RENAMED in stable
    "pb":     "priceToBookRatio",                # /ratios
    "roa":    "returnOnAssets",                  # /key-metrics
    "roe":    "returnOnEquity",                  # /key-metrics
    "fcf":    "freeCashFlowPerShare",            # /ratios
    "pcf":    "priceToOperatingCashFlowRatio",   # /ratios — RENAMED in stable
    "ebitda": "enterpriseValueMultiple",         # /ratios (EV/EBITDA)
    "gm":     "grossProfitMargin",               # /ratios
    "nm":     "netProfitMargin",                 # /ratios
    "sps":    "revenuePerShare",                 # /ratios
}


def _cache_path(path: str, params: dict) -> Path:
    key = path.replace("/", "_") + "__" + "_".join(
        f"{k}={v}" for k, v in sorted(params.items()) if k != "apikey"
    )
    return CACHE_DIR / f"{key}.json"


def _fmp_get(path: str, **params) -> list[dict]:
    """GET an FMP endpoint, with simple on-disk caching."""
    if requests is None:
        raise ImportError("pip install requests")
    CACHE_DIR.mkdir(exist_ok=True)
    cp = _cache_path(path, params)
    if cp.exists():
        return json.loads(cp.read_text())

    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("Set FMP_API_KEY in .env to use the FMP adapter.")
    url = f"{BASE_URL}/{path}"
    r = requests.get(url, params={**params, "apikey": api_key}, timeout=30)
    r.raise_for_status()
    data = r.json()
    # FMP returns a dict with an "Error Message" key on auth/rate-limit issues
    if isinstance(data, dict) and "Error Message" in data:
        raise RuntimeError(f"FMP error for {url}: {data['Error Message']}")
    cp.write_text(json.dumps(data))
    time.sleep(0.15)   # gentle rate limit; FMP free tier ≈ 5 req/sec
    return data


def _quarterly_fmp(ticker: str, limit: int = 40) -> pd.DataFrame | None:
    """Pull quarterly ratios + key-metrics, return the 10 signal columns.

    Uses FMP's `/stable/` endpoints (post-Aug-2025 API). Symbol is a query
    parameter, not part of the URL path. Requires Starter tier or higher
    for period=quarter; free tier only allows period=annual.
    """
    try:
        ratios = _fmp_get("ratios", symbol=ticker, period="quarter", limit=limit)
        km     = _fmp_get("key-metrics", symbol=ticker, period="quarter", limit=limit)
    except Exception as exc:  # noqa: BLE001
        print(f"[{ticker}] FMP error: {exc}")
        return None
    if not ratios or not km:
        print(f"[{ticker}] FMP returned empty payload")
        return None

    r_df = pd.DataFrame(ratios)
    k_df = pd.DataFrame(km)
    if "date" not in r_df.columns or "date" not in k_df.columns:
        return None

    # Some FMP plans put dates as fiscalDateEnding strings; standardize
    r_df["date"] = pd.to_datetime(r_df["date"])
    k_df["date"] = pd.to_datetime(k_df["date"])
    merged = r_df.merge(k_df, on="date", how="inner", suffixes=("", "_km"))

    out = pd.DataFrame({"date": merged["date"]})
    for our_name, fmp_name in _FMP_FIELD_MAP.items():
        # The same field name can be in either source; prefer ratios, fall
        # back to key-metrics (the `_km` suffix added during merge).
        if fmp_name in merged.columns:
            out[our_name] = merged[fmp_name]
        elif f"{fmp_name}_km" in merged.columns:
            out[our_name] = merged[f"{fmp_name}_km"]
        else:
            out[our_name] = float("nan")
    out["ticker"] = ticker
    return out.dropna(subset=["date"]).reset_index(drop=True)


def fetch_fundamentals_fmp(tickers: Iterable[str]) -> pd.DataFrame:
    """FMP equivalent of fetch_fundamentals. Returns a long-format DataFrame."""
    frames = []
    for tk in tickers:
        f = _quarterly_fmp(tk)
        if f is not None and not f.empty:
            frames.append(f)
            print(f"[{tk}] {len(f)} quarterly rows")
    if not frames:
        return pd.DataFrame(columns=["date", "ticker"] + list(_FMP_FIELD_MAP.keys()))
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out
