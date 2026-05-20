"""
SEC EDGAR adapter — quarterly fundamentals from the U.S. SEC's XBRL API.

Free forever, no API key, no tier walls. Covers every SEC-registered company
(all 93 paper tickers qualify) back to ~1993. SEC requires you to identify
yourself in the User-Agent header, so set EDGAR_USER_AGENT in your .env:

    EDGAR_USER_AGENT=Your Name your.email@example.com

Endpoints used:
  https://www.sec.gov/files/company_tickers.json
      → ticker → CIK mapping (one fetch ever, cached)
  https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json
      → every reported XBRL concept for a company (one fetch per ticker)

Output schema matches the other adapters: long-format DataFrame with
    date, ticker, pe, pb, roa, roe, fcf, pcf, ebitda, gm, nm, sps
where income / cash-flow items are TTM (trailing 12 months) and balance
sheet items are quarter-end snapshots, both standard fundamentals practice.
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

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


CACHE_DIR = Path(__file__).parent / ".edgar_cache"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


# --- HTTP plumbing --------------------------------------------------------- #

def _user_agent() -> str:
    ua = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError(
            "SEC requires User-Agent identification. Set EDGAR_USER_AGENT in .env:\n"
            "    EDGAR_USER_AGENT=Your Name your.email@example.com"
        )
    return ua


def _sec_get(url: str, cache_name: str | None = None) -> dict:
    """GET an SEC endpoint with required headers and disk caching."""
    if requests is None:
        raise ImportError("pip install requests")
    CACHE_DIR.mkdir(exist_ok=True)
    if cache_name:
        cp = CACHE_DIR / f"{cache_name}.json"
        if cp.exists():
            return json.loads(cp.read_text())
    headers = {"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    if cache_name:
        cp.write_text(json.dumps(data))
    time.sleep(0.12)   # honor SEC's 10 req/sec soft limit
    return data


def _load_ticker_cik_map() -> dict[str, int]:
    """Return {ticker: CIK} for all SEC-registered companies."""
    raw = _sec_get(TICKER_MAP_URL, cache_name="_ticker_map")
    return {entry["ticker"]: int(entry["cik_str"]) for entry in raw.values()}


# --- XBRL concept extraction ---------------------------------------------- #

# Companies report under different XBRL tags; try each in order, take the
# first non-empty match. This is the part most likely to need tuning when
# a specific ticker fails — add a fallback name here.
CONCEPT_FALLBACKS: dict[str, list[str]] = {
    "Revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "NetIncome": ["NetIncomeLoss"],
    "Assets": ["Assets"],
    "Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "OperatingCashFlow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "CapEx": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "GrossProfit": ["GrossProfit"],
    "CostOfRevenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "LongTermDebt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "Cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "Shares": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
    "OperatingIncome": ["OperatingIncomeLoss"],
    "DepreciationAmortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
}


def _extract_concept(facts: dict, concept_names: list[str]) -> pd.DataFrame:
    """Pull a concept's raw time series from the companyfacts payload."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for name in concept_names:
        if name not in us_gaap:
            continue
        units = us_gaap[name].get("units", {})
        # Prefer USD, then shares, then any unit. Reject foreign currencies
        # by accident? — rare for US-listed S&P 500 companies.
        for unit_key in ("USD", "shares", "USD/shares"):
            if unit_key in units:
                return pd.DataFrame(units[unit_key])
        if units:
            return pd.DataFrame(next(iter(units.values())))
    return pd.DataFrame()


def _to_quarterly(df: pd.DataFrame, kind: str) -> pd.Series:
    """
    Convert a concept DataFrame to a quarter-end-indexed Series.

    kind='instantaneous' (balance sheet snapshots): take values whose `end`
        date falls on a quarter-end. Deduplicate by accession number,
        keeping the latest filing.

    kind='duration' (income / cash-flow flows): get quarterly values.
        EDGAR reports Q1/Q2/Q3 directly with `fp` in {Q1, Q2, Q3} and Q4 as
        part of the FY filing. Q4 = FY - Q1 - Q2 - Q3 for that fiscal year.
    """
    if df.empty or "end" not in df.columns:
        return pd.Series(dtype=float)

    df = df.copy()
    df["end"] = pd.to_datetime(df["end"]).dt.normalize()
    if "accn" in df.columns:
        df = df.sort_values("accn").drop_duplicates(
            subset=["end", "fp", "fy"] if "fp" in df.columns and "fy" in df.columns else ["end"],
            keep="last",
        )

    if kind == "instantaneous":
        df = df[df["end"].dt.is_quarter_end]
        return df.set_index("end")["val"].astype(float).sort_index()

    # Duration path
    if "fp" not in df.columns or "fy" not in df.columns:
        return pd.Series(dtype=float)
    quarterly = df[df["fp"].isin(["Q1", "Q2", "Q3"])][["end", "val"]].copy()

    fy_rows = df[df["fp"] == "FY"]
    q4_rows = []
    for _, fy_row in fy_rows.iterrows():
        fy = fy_row["fy"]
        q123 = df[(df["fy"] == fy) & (df["fp"].isin(["Q1", "Q2", "Q3"]))]
        if len(q123) == 3:
            q4_rows.append({
                "end": fy_row["end"],
                "val": float(fy_row["val"]) - float(q123["val"].sum()),
            })
    if q4_rows:
        quarterly = pd.concat([quarterly, pd.DataFrame(q4_rows)], ignore_index=True)

    if quarterly.empty:
        return pd.Series(dtype=float)
    quarterly["end"] = pd.to_datetime(quarterly["end"]).dt.normalize()
    quarterly = quarterly.drop_duplicates(subset=["end"], keep="last")
    return quarterly.set_index("end")["val"].astype(float).sort_index()


# --- Price source (yfinance) ---------------------------------------------- #

def _quarter_end_prices(ticker: str) -> pd.Series:
    if yf is None:
        raise ImportError("pip install yfinance")
    try:
        px = yf.Ticker(ticker).history(period="max", auto_adjust=False)["Close"]
    except Exception as exc:  # noqa: BLE001
        print(f"[{ticker}] yfinance price error: {exc}")
        return pd.Series(dtype=float)
    if px.empty:
        return pd.Series(dtype=float)
    px.index = (pd.to_datetime(px.index).tz_localize(None)
                if px.index.tz is not None else pd.to_datetime(px.index))
    return px.resample("QE").last()


# --- Main per-ticker assembly --------------------------------------------- #

def _ttm(s: pd.Series) -> pd.Series:
    return s.rolling(4, min_periods=4).sum()


def _align(s: pd.Series, qe: pd.DatetimeIndex) -> pd.Series:
    """Reindex a sparse quarterly series onto the master quarter-end grid."""
    if s.empty:
        return pd.Series(np.nan, index=qe)
    return s.reindex(qe).ffill()


def _quarterly_edgar(ticker: str, cik: int) -> pd.DataFrame | None:
    """Build the 10-signal quarterly DataFrame for one ticker."""
    try:
        facts = _sec_get(COMPANY_FACTS_URL.format(cik=cik),
                         cache_name=f"facts_{cik:010d}")
    except Exception as exc:  # noqa: BLE001
        print(f"[{ticker}] EDGAR error: {exc}")
        return None

    raw = {name: _extract_concept(facts, fallbacks)
           for name, fallbacks in CONCEPT_FALLBACKS.items()}

    rev      = _to_quarterly(raw["Revenue"],                 "duration")
    ni       = _to_quarterly(raw["NetIncome"],               "duration")
    assets   = _to_quarterly(raw["Assets"],                  "instantaneous")
    equity   = _to_quarterly(raw["Equity"],                  "instantaneous")
    ocf      = _to_quarterly(raw["OperatingCashFlow"],       "duration")
    capex    = _to_quarterly(raw["CapEx"],                   "duration").abs()
    gp       = _to_quarterly(raw["GrossProfit"],             "duration")
    cor      = _to_quarterly(raw["CostOfRevenue"],           "duration")
    debt     = _to_quarterly(raw["LongTermDebt"],            "instantaneous")
    cash     = _to_quarterly(raw["Cash"],                    "instantaneous")
    shares   = _to_quarterly(raw["Shares"],                  "instantaneous")
    op_inc   = _to_quarterly(raw["OperatingIncome"],         "duration")
    da       = _to_quarterly(raw["DepreciationAmortization"], "duration")

    # GrossProfit fallback: Revenue - CostOfRevenue
    if gp.empty and not rev.empty and not cor.empty:
        common = rev.index.intersection(cor.index)
        gp = (rev.reindex(common) - cor.reindex(common)).dropna()

    # EBITDA proxy = OperatingIncome + D&A
    if not op_inc.empty:
        common = op_inc.index.union(da.index)
        ebitda = op_inc.reindex(common).fillna(0) + da.reindex(common).fillna(0)
    else:
        ebitda = pd.Series(dtype=float)

    prices = _quarter_end_prices(ticker)

    # Build master quarter-end grid from earliest data point through last quarter end
    candidates = [s.index.min() for s in (rev, ni, assets, equity) if not s.empty]
    if not candidates:
        return None
    start = max(min(candidates), pd.Timestamp("2010-01-01"))
    end = pd.Timestamp.today().to_period("Q").end_time.normalize()
    qe = pd.date_range(start=start, end=end, freq="QE")

    rev_q     = _align(rev, qe);      ni_q      = _align(ni, qe)
    ocf_q     = _align(ocf, qe);      capex_q   = _align(capex, qe)
    gp_q      = _align(gp, qe);       shares_q  = _align(shares, qe)
    equity_q  = _align(equity, qe);   assets_q  = _align(assets, qe)
    debt_q    = _align(debt, qe);     cash_q    = _align(cash, qe)
    ebitda_q  = _align(ebitda, qe);   px_q      = _align(prices, qe)

    rev_ttm    = _ttm(rev_q)
    ni_ttm     = _ttm(ni_q)
    ocf_ttm    = _ttm(ocf_q)
    capex_ttm  = _ttm(capex_q)
    gp_ttm     = _ttm(gp_q)
    ebitda_ttm = _ttm(ebitda_q)

    eps_ttm     = ni_ttm / shares_q.replace(0, np.nan)
    bvps        = equity_q / shares_q.replace(0, np.nan)
    ocf_per_sh  = ocf_ttm / shares_q.replace(0, np.nan)
    fcf_per_sh  = (ocf_ttm - capex_ttm) / shares_q.replace(0, np.nan)
    sps_ttm     = rev_ttm / shares_q.replace(0, np.nan)
    market_cap  = px_q * shares_q
    ev          = market_cap + debt_q.fillna(0) - cash_q.fillna(0)

    out = pd.DataFrame({
        "date":   qe,
        "pe":     (px_q / eps_ttm).values,
        "pb":     (px_q / bvps).values,
        "roa":    (ni_ttm / assets_q).values,
        "roe":    (ni_ttm / equity_q).values,
        "fcf":    fcf_per_sh.values,
        "pcf":    (px_q / ocf_per_sh).values,
        "ebitda": (ev / ebitda_ttm).values,
        "gm":     (gp_ttm / rev_ttm).values,
        "nm":     (ni_ttm / rev_ttm).values,
        "sps":    sps_ttm.values,
    })
    out["ticker"] = ticker
    return out


def fetch_fundamentals_edgar(tickers: Iterable[str]) -> pd.DataFrame:
    """EDGAR equivalent of fetch_fundamentals. Returns the same schema."""
    print("Loading ticker→CIK map from SEC...")
    tcm = _load_ticker_cik_map()
    frames = []
    for tk in tickers:
        cik = tcm.get(tk)
        if cik is None:
            print(f"  [{tk}] not found in SEC ticker map — skipping")
            continue
        f = _quarterly_edgar(tk, cik)
        if f is not None and not f.empty:
            n_valid = f[["pe", "pb", "roa", "roe", "fcf", "pcf", "ebitda",
                         "gm", "nm", "sps"]].notna().any(axis=1).sum()
            frames.append(f)
            print(f"  [{tk}] CIK {cik:010d}  {n_valid} quarters with data")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out
