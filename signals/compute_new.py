"""
Safely evaluate the formula string from GPT-4 against the panel DataFrame.

We restrict the eval environment to the 10 baseline signal columns plus a
small set of math functions. This is *not* a security boundary — it's a
sanity boundary to catch obviously broken formulas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Sign-safe wrappers for functions that fail (NaN) on negative arguments.
# `log(neg) = NaN` is the most common LLM-formula failure mode — particularly
# in cyclical sectors where multiplied profitability terms can flip sign. We
# wrap the argument in max(eps, abs(x)) so the log evaluates everywhere but
# the sign of the *original* product is preserved as a multiplicative factor.
_EPS = 1e-9

def _safe_log(x):
    """log of |x| but with original sign restored: sign(x) * log(max(eps, |x|))."""
    arr = np.asarray(x, dtype=float)
    return np.sign(arr) * np.log(np.maximum(_EPS, np.abs(arr)))

def _safe_sqrt(x):
    """sqrt of |x|, sign preserved."""
    arr = np.asarray(x, dtype=float)
    return np.sign(arr) * np.sqrt(np.abs(arr))


ALLOWED_FUNCS = {
    "log":   _safe_log,
    "log1p": np.log1p,
    "exp":   np.exp,
    "sqrt":  _safe_sqrt,
    "abs":   np.abs,
    "sign":  np.sign,
    "clip":  np.clip,
}

ALLOWED_SYMBOLS = {"pe", "pb", "roa", "roe", "fcf", "pcf", "ebitda", "gm", "nm", "sps"}


def compute_signal(panel: pd.DataFrame, expression: str) -> pd.Series:
    """Evaluate the expression row-wise on the panel; return a Series of values."""
    env = {k: panel[k] for k in ALLOWED_SYMBOLS if k in panel.columns}
    env.update(ALLOWED_FUNCS)
    try:
        # eval with restricted globals; pandas does the row-wise math via vector ops
        return eval(expression, {"__builtins__": {}}, env)  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to evaluate {expression!r}: {exc}") from exc


def add_signal(panel: pd.DataFrame, name: str, expression: str) -> pd.DataFrame:
    """Return a copy of `panel` with the new signal added as column `name`."""
    out = panel.copy()
    out[name] = compute_signal(out, expression)
    return out
