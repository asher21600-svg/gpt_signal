"""
Pull quarterly fundamentals for each ticker and compute the 10 baseline signals.

The paper uses FactSet (paid). Default here is yfinance (free, less complete).
Swap in your own adapter by returning a long-format DataFrame with columns:

    date, ticker, pe, pb, roa, roe, fcf, pcf, ebitda, gm, nm, sps
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from typing import Iterable

# Auto-load .env so scripts don't have to remember to call load_dotenv()
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


def _strip_tz(idx: pd.Index) -> pd.DatetimeIndex:
    """yfinance is inconsistent about tz; strip it so we can slice safely."""
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize(None) if idx.tz is not None else idx


def _quarterly_yf(ticker: str) -> pd.DataFrame | None:
    """Pull from yfinance and compute ratios. Returns None if data missing."""
    if yf is None:
        raise ImportError("Install yfinance, or implement your own adapter.")
    tk = yf.Ticker(ticker)
    try:
        income = tk.quarterly_income_stmt.T
        balance = tk.quarterly_balance_sheet.T
        cashflow = tk.quarterly_cashflow.T
        price = tk.history(period="max", interval="1d", auto_adjust=False)["Close"]
    except Exception as exc:  # noqa: BLE001
        print(f"[{ticker}] yfinance error: {exc}")
        return None
    if income.empty or balance.empty or cashflow.empty or price.empty:
        return None

    # Normalize all date indices to tz-naive before any slicing/intersection
    income.index = _strip_tz(income.index)
    balance.index = _strip_tz(balance.index)
    cashflow.index = _strip_tz(cashflow.index)
    price.index = _strip_tz(price.index)

    # Align on quarter-end timestamps
    idx = income.index.intersection(balance.index).intersection(cashflow.index)
    if len(idx) == 0:
        return None
    income, balance, cashflow = income.loc[idx], balance.loc[idx], cashflow.loc[idx]

    def col(df: pd.DataFrame, *names: str) -> pd.Series:
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series(np.nan, index=df.index)

    # Line items (yfinance column names vary by ticker/period)
    net_income = col(income, "Net Income", "NetIncome")
    revenue = col(income, "Total Revenue", "TotalRevenue")
    gross_profit = col(income, "Gross Profit", "GrossProfit")
    ebitda = col(income, "EBITDA", "Normalized EBITDA")

    total_assets = col(balance, "Total Assets", "TotalAssets")
    total_equity = col(balance, "Stockholders Equity", "Total Stockholder Equity")
    shares = col(balance, "Share Issued", "Ordinary Shares Number", "Common Stock Shares Outstanding")
    long_term_debt = col(balance, "Long Term Debt", "LongTermDebt").fillna(0)
    cash = col(balance, "Cash And Cash Equivalents", "Cash").fillna(0)

    operating_cf = col(cashflow, "Operating Cash Flow", "Total Cash From Operating Activities")
    capex = col(cashflow, "Capital Expenditure", "Capital Expenditures").fillna(0)

    # Quarter-end prices: take the close on/just before each fundamentals date
    qe_dates = idx.sort_values()
    closes = []
    for d in qe_dates:
        sub = price.loc[:d]
        closes.append(sub.iloc[-1] if not sub.empty else np.nan)
    px = pd.Series(closes, index=qe_dates)

    free_cash_flow = operating_cf - capex.abs()
    book_value = total_equity
    market_cap = px * shares
    enterprise_value = market_cap + long_term_debt - cash

    # Trailing 12-month EPS for P/E (approx — sum 4 most recent quarters)
    eps_q = (net_income / shares).sort_index()
    eps_ttm = eps_q.rolling(4).sum()

    df = pd.DataFrame({
        "date":   qe_dates,
        "pe":     (px / eps_ttm).reindex(qe_dates).values,
        "pb":     (px / (book_value / shares)).reindex(qe_dates).values,
        "roa":    (net_income.rolling(4).sum() / total_assets).reindex(qe_dates).values,
        "roe":    (net_income.rolling(4).sum() / total_equity).reindex(qe_dates).values,
        "fcf":    (free_cash_flow / shares).reindex(qe_dates).values,
        "pcf":    (px / (operating_cf / shares)).reindex(qe_dates).values,
        "ebitda": (enterprise_value / ebitda.rolling(4).sum()).reindex(qe_dates).values,
        "gm":     (gross_profit / revenue).reindex(qe_dates).values,
        "nm":     (net_income / revenue).reindex(qe_dates).values,
        "sps":    (revenue.rolling(4).sum() / shares).reindex(qe_dates).values,
    })
    df["ticker"] = ticker
    return df.dropna(subset=["date"])


def fetch_fundamentals(tickers: Iterable[str]) -> pd.DataFrame:
    """Return a long-format DataFrame of quarterly fundamentals for `tickers`.

    Auto-routes by priority (set DATA_SOURCE in .env to force one):
      1. DATA_SOURCE=fmp|edgar|yfinance → explicit override
      2. FMP_API_KEY set                → FinancialModelingPrep (pre-computed ratios)
      3. EDGAR_USER_AGENT set           → SEC EDGAR (free, computes ratios from raw)
      4. Otherwise                       → yfinance (only ~5 recent quarters, free)
    """
    forced = os.environ.get("DATA_SOURCE", "").strip().lower()
    if forced == "edgar" or (not forced and not os.environ.get("FMP_API_KEY")
                              and os.environ.get("EDGAR_USER_AGENT")):
        from .fetch_fundamentals_edgar import fetch_fundamentals_edgar
        print("Using SEC EDGAR adapter.")
        return fetch_fundamentals_edgar(tickers)

    if forced == "fmp" or (forced != "yfinance" and os.environ.get("FMP_API_KEY")):
        from .fetch_fundamentals_fmp import fetch_fundamentals_fmp
        print("Using FMP adapter (FMP_API_KEY found in env).")
        return fetch_fundamentals_fmp(tickers)

    print("Using yfinance adapter (no FMP_API_KEY set).")
    frames = []
    for tk in tickers:
        f = _quarterly_yf(tk)
        if f is not None and not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker"] + [
            "pe", "pb", "roa", "roe", "fcf", "pcf", "ebitda", "gm", "nm", "sps"
        ])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out
