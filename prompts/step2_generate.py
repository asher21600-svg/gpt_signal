"""
Step 2: ask GPT-4 to invent a new signal given definitions + a tiny tabular
sample of company data. Returns the raw text; parsing happens in
signals/parse_llm_output.py.
"""
from __future__ import annotations

import os
import random
from typing import Iterable

import pandas as pd

from .templates import STEP2_GENERATION_PROMPT
from .step1_definitions import format_definitions_block, _build_client


def _sample_block(panel: pd.DataFrame, n_companies: int, n_quarters: int,
                  return_col: str, seed: int | None = None) -> str:
    """Render a few companies' rows in the format shown in Figure 1."""
    rng = random.Random(seed)
    tickers = list(panel["ticker"].unique())
    pick = rng.sample(tickers, k=min(n_companies, len(tickers)))
    blocks = []
    sig_cols = [c for c in panel.columns
                if c not in ("date", "ticker", "ret_1m", "ret_3m")]
    for tk in pick:
        sub = panel[panel["ticker"] == tk].sort_values("date").tail(n_quarters)
        if sub.empty:
            continue
        rows = sub[["date"] + sig_cols + [return_col]].copy()
        rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")
        rows = rows.rename(columns={return_col: "Return"})
        # Truncate floats for readability — and to save tokens
        for c in sig_cols + ["Return"]:
            rows[c] = rows[c].astype(float).round(4)
        blocks.append(f"{tk}:\n{rows.to_string(index=False)}")
    return "\n\n".join(blocks)


def generate_new_signal(panel: pd.DataFrame, signal_definitions: dict[str, dict],
                        signals: list[dict], model: str, temperature: float,
                        n_companies: int = 3, n_quarters: int = 8,
                        return_col: str = "ret_3m",
                        seed: int | None = None) -> str:
    """Call GPT-4 once. Returns the raw response text."""
    definitions_block = format_definitions_block(signal_definitions, signals)
    sample_data = _sample_block(panel, n_companies, n_quarters, return_col, seed=seed)
    prompt = STEP2_GENERATION_PROMPT.format(
        signal_definitions=definitions_block,
        sample_data=sample_data,
    )

    client = _build_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def generate_n_signals(panel: pd.DataFrame, signal_definitions: dict[str, dict],
                       signals: list[dict], model: str, temperature: float,
                       n: int = 6, **kwargs) -> list[str]:
    """Run the prompt n times with different random samples → n raw responses."""
    return [
        generate_new_signal(panel, signal_definitions, signals, model,
                            temperature, seed=i, **kwargs)
        for i in range(n)
    ]
