"""
Realistic synthetic panel mimicking the GPT-Signal paper's data structure.

Use this when real fundamentals are unavailable (paywall, region block, time
budget). Same schema as build_panel.build_panel(); plug into the LLM and
evaluation phases unchanged.

Design choices:
  * Real ticker symbols from config.SECTORS, so the LLM prompt looks realistic.
  * Each ticker has a persistent "personality" — companies don't randomly
    change profile each quarter; values drift modestly around a per-company
    baseline.
  * Returns are driven by:
      (a) a cross-sectional market shock at each date,
      (b) a small signal-driven alpha (high ROE / low P/E predicts higher
          return, weakly — matching paper's |Spearman ρ| ≤ ~0.12), and
      (c) idiosyncratic noise.
  * No single signal dominates — the LLM has to find compound signals,
    just like in the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _ticker_personality(rng: np.random.Generator) -> dict:
    """Generate plausible per-company baseline ratio values."""
    return {
        "pe":     float(rng.lognormal(np.log(22.0), 0.35)),     # median ~22
        "pb":     float(rng.lognormal(np.log(3.5), 0.55)),      # median ~3.5
        "roa":    float(np.clip(rng.normal(0.08, 0.05), -0.10, 0.40)),
        "roe":    float(np.clip(rng.normal(0.25, 0.18), -0.30, 1.50)),
        "fcf":    float(np.clip(rng.normal(8.0, 6.0), -10.0, 40.0)),
        "pcf":    float(rng.lognormal(np.log(15.0), 0.50)),
        "ebitda": float(rng.lognormal(np.log(14.0), 0.50)),
        "gm":     float(np.clip(rng.normal(0.40, 0.15), 0.05, 0.85)),
        "nm":     float(np.clip(rng.normal(0.12, 0.08), -0.25, 0.45)),
        "sps":    float(rng.lognormal(np.log(60.0), 0.70)),
    }


def build_synthetic_panel(
    sector: str = "IT",
    n_quarters: int = 20,
    start: str = "2016-03-31",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a panel with the same schema as build_panel.build_panel():

        date, ticker, pe, pb, roa, roe, fcf, pcf, ebitda, gm, nm, sps,
        ret_1m, ret_3m

    Args:
        sector: one of config.SECTORS keys ("IT", "HealthCare", "Energy").
        n_quarters: 20 = 5 years of quarterly data, matching the paper.
        start: first quarter-end date.
        seed: reproducibility.

    Returns long-format DataFrame, len = len(tickers) * n_quarters.
    """
    rng = np.random.default_rng(seed)
    tickers = config.SECTORS[sector]
    dates = pd.date_range(start, periods=n_quarters, freq="QE")

    # Per-ticker baseline characteristics (don't change quarter to quarter)
    personalities = {tk: _ticker_personality(rng) for tk in tickers}

    rows = []
    for d in dates:
        # Cross-sectional market shock at this date — affects all companies
        market = float(rng.normal(0.005, 0.06))

        for tk in tickers:
            p = personalities[tk]

            # Build the row: personality + modest quarter-to-quarter noise
            row = {"date": d, "ticker": tk}
            for sig, baseline in p.items():
                # 8-12% quarterly fluctuation around baseline
                noise = 0.10 * rng.standard_normal()
                row[sig] = float(baseline * (1.0 + noise))

            # Cross-sectional Z-scores of "true" return-predictive signals.
            # Normalized so the signal-driven alpha contribution is ~0.5%–1%.
            z_roe = (row["roe"] - p["roe"]) / max(0.05, abs(p["roe"]) * 0.3)
            z_pe  = (row["pe"]  - p["pe"])  / max(2.0,  p["pe"]  * 0.2)
            z_pb  = (row["pb"]  - p["pb"])  / max(0.5,  p["pb"]  * 0.2)

            # Modest signal-driven alpha (matches paper's |ρ| ~ 0.1)
            alpha = 0.008 * z_roe - 0.006 * z_pe - 0.004 * z_pb

            # Quarter-ahead returns
            ret_3m = market + alpha + 0.05 * float(rng.standard_normal())
            ret_1m = (market * 0.30 + alpha * 0.30
                      + 0.025 * float(rng.standard_normal()))
            row["ret_1m"] = ret_1m
            row["ret_3m"] = ret_3m
            rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["date", "ticker", "pe", "pb", "roa", "roe", "fcf", "pcf",
            "ebitda", "gm", "nm", "sps", "ret_1m", "ret_3m"]
    return df[cols].sort_values(["date", "ticker"]).reset_index(drop=True)
