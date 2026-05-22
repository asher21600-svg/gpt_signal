# Step-by-Step Run Guide

Companion to `REPRODUCTION_PLAN.md` (the *why*) and `REPRODUCTION_REPORT.md` (the *findings*). This file is the *what to type*, in order.

Assume you've cloned the repo and `cd`'d into `gpt_signal/`.

> **Substitutions from the paper.** This guide uses **DeepSeek-Chat** in place of GPT-4 (functionally equivalent OpenAI-compatible API, ~50× cheaper) and **FinancialModelingPrep `/stable/` Starter ($14/mo)** in place of FactSet (both expose US-listed quarterly fundamentals). Either can be swapped — see Phase 0.

---

## Phase 0 — Environment (15 min)

### 0.1 Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 0.2 Configure API keys

```bash
cp .env.example .env
```

Edit `.env` to add **one LLM key** and **one fundamentals data source**:

```
# LLM (one of these)
DEEPSEEK_API_KEY=sk-...                # DeepSeek-Chat (recommended, ¥1-2 total)
# OPENAI_API_KEY=sk-...                # GPT-4 if you have it (and change LLM_MODEL in config.py)

# Fundamentals (one of these; auto-selected by priority FMP > EDGAR > yfinance)
FMP_API_KEY=...                        # FinancialModelingPrep Starter ($14/mo, fastest)
EDGAR_USER_AGENT="Your Name your.email@example.com"   # Free fallback (SEC requires identification)
```

### 0.3 Smoke test (no network, no API spend)

```bash
python tests/test_pipeline.py
```

Expected output:

```
Running smoke tests…
  ✓ apply_paper_signals adds 6 columns
  ✓ parse_signal_response + compute_signal
  ✓ Spearman directions correct (roe=+0.16, pe=-0.57)
  ✓ Fama-MacBeth-style: median adj_R² ≈ +0.05
  ✓ compare_models: 7 models, 24 dates
  ✓ Plots rendered at tests/smoke_outputs/
All smoke tests passed.
```

If anything fails here, stop and fix it — every later step builds on these modules.

### 0.4 LLM connectivity check (~¥0.001)

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from prompts.step1_definitions import _build_client
from config import LLM_MODEL
r = _build_client().chat.completions.create(
    model=LLM_MODEL,
    messages=[{'role':'user','content':'Reply with just: OK'}],
    max_tokens=5,
)
print(r.choices[0].message.content)
"
```

Should print `OK`. If it prints an auth error, double-check your `.env`.

---

## Phase 1 — Build the data panels (10 min — 2 hr)

Two paths. **Pick one based on what you have:**

### Path A — Synthetic data (5 min, free, prototyping only)

Use this when you don't have FMP/EDGAR keys yet, or to validate the pipeline before spending on API calls.

```bash
cat > scratch_phase1_synthetic.py << 'PYEOF'
from pathlib import Path
from data.synthetic_panel import build_synthetic_panel
from evaluation.spearman import cross_sectional_spearman
from config import SIGNAL_KEYS

for sector in ["IT", "HealthCare", "Energy"]:
    panel = build_synthetic_panel(sector=sector, n_quarters=20, seed=42)
    out = Path(f"outputs/{sector}_panel.parquet")
    panel.to_parquet(out)
    print(f"{sector}: {panel.shape}  ->  {out}")

# Quick sanity: |Spearman ρ| of 0.05-0.12 confirms paper-like signal strength
panel = build_synthetic_panel(sector="IT", n_quarters=20, seed=42)
cs = cross_sectional_spearman(panel, SIGNAL_KEYS, "ret_3m")
print("\nMean cross-sectional Spearman rho (ret_3m):")
for col in SIGNAL_KEYS:
    print(f"  {col:>8s}: {cs[col].mean():+.4f}")
PYEOF
python scratch_phase1_synthetic.py
```

Synthetic data is calibrated to the paper's |ρ| ≤ 0.12 signal-return correlation range. Use it for everything in Phases 2–5; swap in real data later if you want to confirm the paper's specific numbers.

### Path B — Real S&P 500 data via FMP (5–10 min, ~1 min API time)

```bash
cat > scratch_phase1_real.py << 'PYEOF'
from pathlib import Path
import pandas as pd
from config import SECTORS
from data.fetch_fundamentals import fetch_fundamentals
from data.fetch_returns import fetch_quarterly_forward_returns
from data.build_panel import build_panel

Path("outputs").mkdir(exist_ok=True)
for sector, tickers in SECTORS.items():
    print(f"\n=== {sector} — {len(tickers)} tickers ===")
    fundamentals = fetch_fundamentals(tickers)
    returns = fetch_quarterly_forward_returns(
        tickers, start="2013-01-01", end="2025-01-01",
        horizons_months=[1, 3],
    )
    panel = build_panel(fundamentals, returns)
    out = Path(f"outputs/real_{sector}_panel.parquet")
    panel.to_parquet(out)
    print(f"  -> {panel.shape}  "
          f"Date {panel['date'].min().date()} to {panel['date'].max().date()}  "
          f"Tickers {panel['ticker'].nunique()} of {len(tickers)}  "
          f"P/E med {panel['pe'].median():.2f}  ROE med {panel['roe'].median():.3f}")
PYEOF
python scratch_phase1_real.py
```

**What you should see:**

- IT: ~1,500 rows × 14 columns, 43 of 43 tickers, P/E median 24–28
- HealthCare: ~1,050 rows, 30–31 tickers, P/E median 25–30
- Energy: ~560 rows, 16–19 tickers, P/E median 12–15

**Things to verify:**

1. **Ratio scales are right.** P/E in the 10–40 range, ROE 0.05–1.5 (fraction, not percent). If off by 100×, check `_FMP_FIELD_MAP` in `data/fetch_fundamentals_fmp.py`.
2. **All ~93 tickers present.** A few may be missing (delisted tickers PXD, MRO, HES, CTLT, ANSS, JNPR). The bundled adapter uses FMP for prices specifically to recover acquired/delisted symbols — yfinance silently drops them, which would introduce survivorship bias.
3. **No 100% NaN columns.** If a column is all NaN, the FMP field name probably changed — recheck the field map.

End of Phase 1: 3 parquet files in `outputs/`.

---

## Phase 2 — LLM defines the 10 baseline signals (10 min, ~¥0.05)

```bash
cat > scratch_phase2.py << 'PYEOF'
from prompts.step1_definitions import get_signal_definitions, format_definitions_block
from config import EXISTING_SIGNALS, LLM_MODEL, LLM_TEMP_DEFINITIONS
from dotenv import load_dotenv; load_dotenv()

defs = get_signal_definitions(EXISTING_SIGNALS, model=LLM_MODEL,
                              temperature=LLM_TEMP_DEFINITIONS, use_cache=True)
for key, content in defs.items():
    print(f"--- {key} ---")
    print(f"  definition:         {content.get('definition', '')[:150]}")
    print(f"  effect_on_returns:  {content.get('effect_on_returns', '')[:150]}")
    print(f"  preferred_tendency: {content.get('preferred_tendency', '')[:150]}")
PYEOF
python scratch_phase2.py
```

Output cached to `prompts/signal_definitions.json` — paid for once. **Read every entry** before continuing. Watch for:

- Confused valuation direction (LLM saying "higher P/E predicts higher returns" — wrong sign)
- B/P vs P/B confusion (book-to-market vs price-to-book)
- Forgetting a signal exists

If you spot anything wrong, edit `prompts/signal_definitions.json` by hand. The cached JSON is what gets baked into the Phase 3 prompt.

---

## Phase 3 — LLM generates 6 new compound signals (~5 min, ~¥0.5)

```bash
cat > scratch_phase3.py << 'PYEOF'
import json
import pandas as pd
from dotenv import load_dotenv; load_dotenv()
from prompts.step1_definitions import get_signal_definitions
from prompts.step2_generate import generate_n_signals
from signals.parse_llm_output import parse_signal_response
from signals.compute_new import add_signal
from config import EXISTING_SIGNALS, LLM_MODEL, LLM_TEMP_GENERATION

panel = pd.read_parquet("outputs/real_IT_panel.parquet")  # or "outputs/IT_panel.parquet" for synthetic
defs = get_signal_definitions(EXISTING_SIGNALS, model=LLM_MODEL, use_cache=True)

raw_responses = generate_n_signals(
    panel, defs, EXISTING_SIGNALS,
    model=LLM_MODEL, temperature=LLM_TEMP_GENERATION,
    n=6, return_col="ret_3m", n_companies=3, n_quarters=8,
)

with open("outputs/IT_new_signals_raw.jsonl", "w") as f:
    for i, r in enumerate(raw_responses):
        f.write(json.dumps({"i": i, "response": r}) + "\n")

new_signal_names = []
panel_enriched = panel.copy()
for i, raw in enumerate(raw_responses):
    parsed = parse_signal_response(raw)
    if parsed is None:
        print(f"  #{i+1}: ✗ no formula found")
        continue
    name, expr = parsed
    try:
        panel_enriched = add_signal(panel_enriched, name, expr)
        new_signal_names.append(name)
        print(f"  #{i+1}: ✓ {name} = {expr}")
    except ValueError as e:
        print(f"  #{i+1}: ✗ {name}: {e}")

panel_enriched.to_parquet("outputs/IT_panel_with_new.parquet")
print(f"\nGenerated {len(set(new_signal_names))} unique signals: {sorted(set(new_signal_names))}")
PYEOF
python scratch_phase3.py
```

**What to expect:**

- **6 of 6 parseable** in most runs; occasionally 4–5 if the LLM hallucinates `max()` or `mean()` functions our sandboxed evaluator doesn't expose.
- **Repeated names are normal** at lower temperature. Use higher temp (0.7+) for more diversity.
- **All formulas should follow the same family:** products of profitability metrics divided by valuation multiples, often wrapped in `log()`. Both GPT-4 (paper) and DeepSeek converge on this — the "Quality at a Reasonable Price" pattern.
- **`RuntimeWarning: invalid value in log`** — *harmless*, the bundled `compute_new.py` uses a sign-safe `log` wrapper (`sign(x) · log(max(eps, |x|))`) that produces a real number even on negative arguments. Without it, Energy sector returns NaN cascades.

---

## Phase 4 — Evaluate (5 min, no API spend)

### 4.1 Single sector × single horizon

```bash
cat > scratch_phase4.py << 'PYEOF'
from pathlib import Path
import pandas as pd
from config import SIGNAL_KEYS
from evaluation.spearman import mean_correlation_matrix, cross_sectional_spearman
from evaluation.fama_macbeth import compare_models
from evaluation.plots import correlation_heatmap, adj_r2_boxplot

panel = pd.read_parquet("outputs/IT_panel_with_new.parquet")
META = {"date", "ticker", "ret_1m", "ret_3m"}
new_names = [c for c in panel.columns if c not in META and c not in SIGNAL_KEYS]
figdir = Path("outputs/figures"); figdir.mkdir(parents=True, exist_ok=True)

# Heatmaps (Spearman rank correlation, averaged across dates)
for cols, label in [
    (SIGNAL_KEYS + ["ret_3m"], "existing"),
    (new_names + ["ret_3m"], "new"),
    (SIGNAL_KEYS + new_names + ["ret_3m"], "all"),
]:
    cm = mean_correlation_matrix(panel, cols)
    correlation_heatmap(cm, f"IT — {label} × ret_3m",
                        figdir / f"IT_corr_{label}.png")

# Fama-MacBeth: baseline (10 signals) vs baseline + each new signal
adj_r2 = compare_models(panel, SIGNAL_KEYS, new_names, "ret_3m")
adj_r2.to_csv("outputs/IT_adj_r2.csv", index=False)
adj_r2_boxplot(adj_r2, "IT — Adj R² (ret_3m)",
               figdir / "IT_adj_r2_boxplot.png")

# Summary table
summary = adj_r2.drop(columns=["date"]).agg(["median", "mean", "std"]).T
baseline_med = summary.loc["baseline", "median"]
summary["delta_vs_baseline"] = summary["median"] - baseline_med
summary = summary.sort_values("median", ascending=False).round(4)
print(summary)
print(f"\nModels beating baseline (median Δ > 0): "
      f"{(summary['delta_vs_baseline'] > 0).sum()} of {len(summary) - 1}")
PYEOF
python scratch_phase4.py
```

### 4.2 Cross-sector × both horizons

```bash
cat > scratch_phase4_all_sectors.py << 'PYEOF'
import json
from pathlib import Path
import pandas as pd
from config import SIGNAL_KEYS
from signals.parse_llm_output import parse_signal_response
from signals.compute_new import add_signal
from evaluation.fama_macbeth import compare_models
from evaluation.plots import adj_r2_boxplot

formulas = {}
with open("outputs/IT_new_signals_raw.jsonl") as f:
    for line in f:
        parsed = parse_signal_response(json.loads(line)["response"])
        if parsed:
            formulas[parsed[0]] = parsed[1]

figdir = Path("outputs/figures"); figdir.mkdir(parents=True, exist_ok=True)
combined = []
for sector in ["IT", "HealthCare", "Energy"]:
    for horizon in ["ret_3m", "ret_1m"]:
        panel = pd.read_parquet(f"outputs/real_{sector}_panel.parquet")
        new_names = []
        for name, expr in formulas.items():
            try:
                panel = add_signal(panel, name, expr); new_names.append(name)
            except ValueError: pass
        adj_r2 = compare_models(panel, SIGNAL_KEYS, new_names, horizon)
        adj_r2_boxplot(adj_r2, f"{sector} — Adj R² ({horizon})",
                       figdir / f"real_{sector}_adj_r2_{horizon}.png")
        s = adj_r2.drop(columns=["date"]).agg(["median"]).T
        baseline_med = s.loc["baseline", "median"]
        s["delta"] = s["median"] - baseline_med
        for name in new_names:
            combined.append({
                "sector": sector, "horizon": horizon, "model": name,
                "median": s.loc[name, "median"], "delta": s.loc[name, "delta"],
            })

df = pd.DataFrame(combined)
df.to_csv("outputs/real_cross_sector_summary.csv", index=False)
win_pct = (df["delta"] > 0).mean() * 100
print(df.pivot_table(index="model", columns=["sector","horizon"], values="delta").round(4))
print(f"\nWin rate across all cells: {(df['delta'] > 0).sum()}/{len(df)} = {win_pct:.1f}%")
PYEOF
python scratch_phase4_all_sectors.py
```

End of Phase 4: a 6-row × 5-column delta matrix, a single headline win-rate %, 6 PNG figures.

**Reference numbers (what we hit):**

- Real-data 2016–2024: win rate **26.7%**. PAVS-family is the only signal beating baseline more than 50% of the time across cells.
- Paper-window 2016–2020: win rate **76–87%** across 4 temperatures — matches the paper's 83% claim.

---

## Phase 5 — Generate the shareable report (1 min, no API spend)

```bash
python build_report.py        # outputs/reproduction_report.html (self-contained, ~1-5 MB)
python build_pdf.py           # outputs/reproduction_report.pdf  (via Chrome-headless or weasyprint)
open outputs/reproduction_report.pdf
```

What's in it:

- TL;DR hero stats (paper claim vs your in-window vs your extended-window numbers)
- All Spearman heatmaps + Fama-MacBeth box plots embedded as base64 PNGs
- Per-sector × per-horizon cell-by-cell delta table with green/red coloring
- Cross-run findings section
- Conclusion

Suitable for sharing in chat, attaching to email, or printing.

---

## Phase 6 — Extension experiments (~5 min, ~¥3)

The most informative bits beyond a vanilla reproduction.

### 6.1 Filter to the paper's exact window (2016–2020)

```bash
cat > scratch_extension_filter.py << 'PYEOF'
from pathlib import Path
import pandas as pd
for sector in ["IT", "HealthCare", "Energy"]:
    panel = pd.read_parquet(f"outputs/real_{sector}_panel.parquet")
    panel = panel[(panel["date"] >= "2016-01-01") & (panel["date"] <= "2020-12-31")].copy()
    panel.to_parquet(f"outputs/paper_{sector}_panel.parquet")
    print(f"{sector}: {len(panel)} rows  ({panel['date'].min().date()} to {panel['date'].max().date()})")
PYEOF
python scratch_extension_filter.py
```

### 6.2 Temperature sweep on the paper window

```bash
# Generate 6 signals at each of T ∈ {0.0, 0.3, 0.7, 1.0}
python scratch_extension_temp_sweep.py

# Evaluate each temp's signals across all sectors × horizons
python scratch_extension_eval.py
```

End of Phase 6: a 4-row table comparing win rates per temperature. We hit 76–87% across all four, confirming temperature isn't the load-bearing knob — the paper's headline reproduces robustly within its window regardless of temperature.

---

## Phase 7 — Validation checklist

Open `outputs/reproduction_report.pdf` and confirm:

| Check | Pass if |
|---|---|
| In-window win rate is in the 70–90% range | Yes for paper window 2016–2020 |
| Out-of-window win rate is lower | Yes for 2016–2024; ours was 26.7% (alpha decay) |
| At least one signal beats baseline at both horizons | Yes — sign-safe PAVS family |
| `EVC` or similar has top Spearman \|ρ\| | Yes in IT (matches paper §5.3) |
| Adj R² magnitudes are reasonable | Roughly −0.2 to +0.2; if you see ±0.8 something's wrong |
| Energy doesn't produce all-NaN signals | If yes, sign-safe `log` in `signals/compute_new.py` is missing |

If 4 of 6 of those check out, you've reproduced the paper.

---

## Costs and time budget (actual, with DeepSeek + FMP)

| Phase | Time | Spend |
|---|---|---|
| 0 — Env + smoke test | 15 min | ¥0 |
| 1 — Data (FMP Starter) | 5–10 min | $14/mo subscription |
| 2 — Definitions | 10 min | ~¥0.05 |
| 3 — Signal generation (×6) | 5 min | ~¥0.5 |
| 4 — Evaluation (3 sectors × 2 horizons) | 5 min | ¥0 |
| 5 — Report | 1 min | ¥0 |
| 6 — Extension (4-temp sweep) | 5 min | ~¥3 |

**Total LLM spend: under ¥5.** First-time data pull from FMP ~1 minute; subsequent runs hit disk cache and are instant.

---

## Common errors and fixes

| Error | Fix |
|---|---|
| `Premium Query Parameter: 'Special Endpoint'` | FMP free tier blocks `period=quarter`. Upgrade to Starter ($14/mo) or use the EDGAR adapter. |
| `Legacy Endpoint : Due to Legacy endpoints being no longer supported` | You're hitting the old `/api/v3/` URLs. The bundled adapter uses `/stable/` already; if you forked an older version, update the `BASE_URL`. |
| `pe is NaN for all rows` | FMP field name mismatch — `priceEarningsRatio` was renamed to `priceToEarningsRatio` in `/stable/`. See `data/fetch_fundamentals_fmp.py`. |
| `Empty panel after build_panel` | Date alignment between fundamentals (fiscal quarter-ends) and returns (calendar quarter-ends). `build_panel.py` rounds both via `dt.to_period("Q").dt.end_time` — make sure both inputs go through it. |
| `log(negative) = NaN` cascades in Energy | Use the sign-safe `_safe_log` in `signals/compute_new.py`. The wrapper is `sign(x) · log(max(eps, \|x\|))`. |
| `Authentication failed for github` on `git push` | GitHub removed password auth in August 2021. Use `gh auth login` or a Personal Access Token as your password. |
| `Repository not found` on `git push` | Create the repo first via `gh repo create <user>/gpt_signal --public --source=.` or manually at github.com/new (don't pre-populate README/LICENSE). |
