"""
The 6 signals GPT-4 generated in the paper, expressed as plain-Python
formulas. Use these as a sanity check: when your reproduction generates its
own signals, you can compare evaluation results against these hard-coded
versions to confirm the eval pipeline is wired up.

Caveats noted in REPRODUCTION_PLAN.md §4:
  - RAPS hard-codes beta=2 (the paper does too — flagged as a smell).
  - VEC is a linear combination, which the prompt forbids — kept for fidelity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PAPER_NEW_SIGNALS: dict[str, callable] = {
    "PVS":  lambda df: df["roe"] / df["pe"],
    "RAPS": lambda df: df["roe"] / (df["pe"] * 2.0),  # beta hard-coded to 2
    "EVC":  lambda df: df["roa"] * (1.0 / df["ebitda"]) * (1.0 / df["pcf"]),
    "VEC":  lambda df: (df["pe"] + df["roe"] + df["fcf"]) / 3.0,
    "PLF":  lambda df: (df["roe"] * df["gm"]) / df["pe"],
    "IQS":  lambda df: df["roe"] * (1.0 / df["pe"]) * (1.0 / df["pb"])
                       * np.log(df["sps"].clip(lower=1e-9)),
}


def apply_paper_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the 6 paper-reported new signals as new columns of `panel`."""
    out = panel.copy()
    for name, fn in PAPER_NEW_SIGNALS.items():
        out[name] = fn(out)
    return out
