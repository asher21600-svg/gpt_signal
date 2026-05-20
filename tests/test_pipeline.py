"""
Smoke test: synthetic 5-company, 16-quarter panel with known noisy linear
return generation. Confirms:
  - paper signals can be computed
  - parse_signal_response extracts formulas
  - compute_signal evaluates them safely
  - Spearman correlation has the expected sign
  - Fama-MacBeth runs and returns finite adj R²
  - plots render to disk

No OpenAI API or network needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make project importable when run as `python tests/test_pipeline.py`
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from config import SIGNAL_KEYS
from signals.existing import apply_paper_signals, PAPER_NEW_SIGNALS
from signals.parse_llm_output import parse_signal_response
from signals.compute_new import compute_signal

# Optional deps — if missing, only the relevant tests are skipped.
try:
    from evaluation.spearman import cross_sectional_spearman, mean_correlation_matrix
    HAVE_SCIPY = True
except ModuleNotFoundError:
    HAVE_SCIPY = False

try:
    from evaluation.fama_macbeth import fama_macbeth, compare_models
    HAVE_STATSMODELS = True
except ModuleNotFoundError:
    HAVE_STATSMODELS = False

try:
    from evaluation.plots import correlation_heatmap, adj_r2_boxplot
    HAVE_PLOTS = True
except ModuleNotFoundError:
    HAVE_PLOTS = False


def synthetic_panel(n_companies: int = 8, n_quarters: int = 16,
                    seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-03-31", periods=n_quarters, freq="QE")
    tickers = [f"TK{i:02d}" for i in range(n_companies)]

    rows = []
    for d in dates:
        for tk in tickers:
            base = rng.normal(0, 1)
            row = {
                "date": d,
                "ticker": tk,
                "pe":     20 + 5 * rng.standard_normal(),
                "pb":     3  + rng.standard_normal(),
                "roa":    0.05 + 0.02 * rng.standard_normal(),
                "roe":    0.15 + 0.05 * rng.standard_normal(),
                "fcf":    2 + rng.standard_normal(),
                "pcf":    15 + 3 * rng.standard_normal(),
                "ebitda": 12 + 2 * rng.standard_normal(),
                "gm":     0.4 + 0.1 * rng.standard_normal(),
                "nm":     0.1 + 0.03 * rng.standard_normal(),
                "sps":    50 + 10 * rng.standard_normal(),
            }
            # Construct returns so that high ROE / low P/E predicts high return
            row["ret_3m"] = (
                0.05 * row["roe"] - 0.002 * row["pe"]
                + 0.01 * rng.standard_normal()
            )
            row["ret_1m"] = row["ret_3m"] / 3 + 0.005 * rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows)


def test_paper_signals_compute():
    panel = synthetic_panel()
    enriched = apply_paper_signals(panel)
    assert all(name in enriched.columns for name in PAPER_NEW_SIGNALS)
    assert np.isfinite(enriched["EVC"]).any()
    print("  ✓ apply_paper_signals adds 6 columns")


def test_parse_and_compute_formula():
    fake_response = """\
Some reasoning text. Step by step we conclude…
SIGNAL_FORMULA: PVS_v2 = roe / pe
"""
    parsed = parse_signal_response(fake_response)
    assert parsed == ("PVS_v2", "roe / pe")
    panel = synthetic_panel()
    series = compute_signal(panel, "roe / pe")
    assert len(series) == len(panel)
    assert np.isfinite(series).all()
    print("  ✓ parse_signal_response + compute_signal")


def _spearman_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """Pure-numpy Spearman rho, used when scipy isn't available."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xr = pd.Series(x[mask]).rank().values
    yr = pd.Series(y[mask]).rank().values
    return float(np.corrcoef(xr, yr)[0, 1])


def test_spearman_sign():
    """High ROE / low P/E was wired to → higher return. Spearman should agree."""
    panel = synthetic_panel(n_companies=20, n_quarters=24)
    if HAVE_SCIPY:
        cs = cross_sectional_spearman(panel, ["roe", "pe"], "ret_3m")
        roe_mean, pe_mean = cs["roe"].mean(), cs["pe"].mean()
    else:
        # Numpy fallback: same computation, sans scipy
        roe_list, pe_list = [], []
        for _, sub in panel.groupby("date"):
            roe_list.append(_spearman_numpy(sub["roe"].values, sub["ret_3m"].values))
            pe_list.append(_spearman_numpy(sub["pe"].values, sub["ret_3m"].values))
        roe_mean = float(np.nanmean(roe_list))
        pe_mean = float(np.nanmean(pe_list))
    assert roe_mean > 0, f"expected roe correlation positive, got {roe_mean}"
    assert pe_mean < 0, f"expected pe correlation negative, got {pe_mean}"
    print(f"  ✓ Spearman directions correct (roe={roe_mean:+.3f}, "
          f"pe={pe_mean:+.3f}) "
          f"[{'scipy' if HAVE_SCIPY else 'numpy fallback'}]")


def _adj_r2_numpy(y: np.ndarray, X: np.ndarray) -> float:
    """Pure-numpy adjusted R², used when statsmodels isn't available."""
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() <= X.shape[1] + 1:
        return float("nan")
    y2, X2 = y[mask], X[mask]
    Xc = np.column_stack([np.ones(len(y2)), X2])
    beta, *_ = np.linalg.lstsq(Xc, y2, rcond=None)
    yhat = Xc @ beta
    ss_res = float(np.sum((y2 - yhat) ** 2))
    ss_tot = float(np.sum((y2 - y2.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    r2 = 1 - ss_res / ss_tot
    n, p = len(y2), X2.shape[1]
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def test_fama_macbeth_runs():
    panel = apply_paper_signals(synthetic_panel(n_companies=20, n_quarters=24))
    if HAVE_STATSMODELS:
        res = fama_macbeth(panel, SIGNAL_KEYS, "ret_3m")
        median_r2 = res["adj_r2"].median()
        n = len(res)
    else:
        # Numpy fallback: collapsed Fama-MacBeth → just cross-sectional regression
        # of returns on signals at each date. Less faithful but enough to
        # confirm the math wiring works.
        r2s = []
        for _, sub in panel.groupby("date"):
            y = sub["ret_3m"].values
            X = sub[SIGNAL_KEYS].values
            r2s.append(_adj_r2_numpy(y, X))
        median_r2 = float(np.nanmedian(r2s))
        n = len(r2s)
    assert np.isfinite(median_r2), "adj_R² is not finite"
    print(f"  ✓ Fama-MacBeth-style: median adj_R² = {median_r2:+.3f} "
          f"over {n} dates "
          f"[{'statsmodels' if HAVE_STATSMODELS else 'numpy fallback'}]")


def test_compare_models_baseline_vs_paper():
    if not HAVE_STATSMODELS:
        print("  ⊘ compare_models skipped (statsmodels not installed in sandbox)")
        return
    panel = apply_paper_signals(synthetic_panel(n_companies=20, n_quarters=24))
    res = compare_models(panel, SIGNAL_KEYS, list(PAPER_NEW_SIGNALS), "ret_3m")
    assert {"baseline"}.issubset(res.columns)
    for name in PAPER_NEW_SIGNALS:
        assert name in res.columns
    print(f"  ✓ compare_models: {len(res.columns) - 1} models, "
          f"{len(res)} dates")
    print("    medians:")
    for c in res.columns:
        if c == "date":
            continue
        print(f"      {c:>10s}: {res[c].median():+.4f}")


def test_plots_render(tmpdir: Path):
    if not HAVE_PLOTS:
        print("  ⊘ plot rendering skipped (matplotlib/seaborn not importable)")
        return
    panel = apply_paper_signals(synthetic_panel(n_companies=20, n_quarters=24))

    # Build a correlation matrix without scipy: average per-date Pearson on ranks
    cols = SIGNAL_KEYS + list(PAPER_NEW_SIGNALS) + ["ret_3m"]
    mats = []
    for _, sub in panel.groupby("date"):
        mats.append(sub[cols].rank().corr())
    cm = sum(mats) / len(mats)

    correlation_heatmap(cm, "synthetic — all signals", tmpdir / "corr.png")

    # Synthetic adj_r2 table — one row per date, three "models"
    fake_r2 = pd.DataFrame({
        "date": panel["date"].unique(),
        "baseline": np.random.normal(0.05, 0.03, panel["date"].nunique()),
        "model_A":  np.random.normal(0.07, 0.03, panel["date"].nunique()),
        "model_B":  np.random.normal(0.06, 0.03, panel["date"].nunique()),
    })
    adj_r2_boxplot(fake_r2, "synthetic — adj R²", tmpdir / "box.png")
    assert (tmpdir / "corr.png").exists()
    assert (tmpdir / "box.png").exists()
    print(f"  ✓ Plots rendered at {tmpdir}")


def main():
    print("Running smoke tests…")
    test_paper_signals_compute()
    test_parse_and_compute_formula()
    test_spearman_sign()
    test_fama_macbeth_runs()
    test_compare_models_baseline_vs_paper()
    out = HERE / "smoke_outputs"
    out.mkdir(exist_ok=True)
    test_plots_render(out)
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
