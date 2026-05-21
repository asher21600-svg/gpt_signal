# Step-by-Step Reproduction Guide

Companion to `REPRODUCTION_PLAN.md`. The plan explains the *why*; this guide
gives you the literal sequence of commands plus what to inspect at each step.

Assume you've cloned this folder and `cd`'d into `gpt_signal/`.

---

## Phase 0 — Environment (15 min)

### 0.1 Install dependencies

```bash
cd gpt_signal/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 0.2 Add your OpenAI key

```bash
cp .env.example .env
# edit .env, set OPENAI_API_KEY=sk-...
```

### 0.3 Smoke test (no network, no API)

```bash
python tests/test_pipeline.py
```

Expected output (last 6 lines):

```
  ✓ apply_paper_signals adds 6 columns
  ✓ parse_signal_response + compute_signal
  ✓ Spearman directions correct (roe=+0.16, pe=-0.57)
  ✓ Fama-MacBeth: median adj_R² ≈ +0.05–0.10
  ✓ compare_models: 7 models, 24 dates
  ✓ Plots rendered at tests/smoke_outputs/
```

If anything fails here, **stop and fix it** — every later step depends on the
modules this test exercises.

---

## Phase 1 — Build the data panel (45 min — 2 hr)

This is the slowest step. yfinance throttles and a few tickers will fail; that's
normal.

### 1.1 Pull fundamentals + returns + build the merged panel

The orchestrator does this in one shot; you can also call the steps individually
if you want to inspect intermediates.

```python
# scratch_phase1.py
from data.fetch_fundamentals import fetch_fundamentals
from data.fetch_returns import fetch_quarterly_forward_returns
from data.build_panel import build_panel
from config import SECTORS

tickers = SECTORS["IT"]
fundamentals = fetch_fundamentals(tickers)
returns = fetch_quarterly_forward_returns(tickers,
                                          start="2015-12-01",
                                          end="2021-03-31",
                                          horizons_months=[1, 3])
panel = build_panel(fundamentals, returns)
print(panel.shape)
print(panel.head())
panel.to_parquet("outputs/IT_panel.parquet")
```

```bash
mkdir -p outputs
python scratch_phase1.py
```

### 1.2 Sanity-check the panel

Things to verify:

1. **Row count ≈ tickers × quarters.** For IT: ~43 tickers × ~20 quarters
   (2016 Q1 – 2020 Q4) = ~860 rows. Expect 600–800 after dropping NaNs.
2. **No look-ahead.** The row at `date=2018-06-30` should have signal values
   computed from fundamentals filed *by* June 30 and a forward return measured
   *after* June 30. The `ret_3m` for that row is the return between 2018-06-30
   and ~2018-09-30.
3. **Ratio scales look right.** P/E typically 10–40, P/B 1–10, ROE 0.05–0.4
   (as a fraction, not percent). If anything is off by 100×, you have a units
   mismatch — check `_quarterly_yf` in `fetch_fundamentals.py`.

```python
panel.describe()
panel.groupby("date").size().tail()      # how many companies per quarter
panel[["pe","pb","roe","roa","ret_3m"]].describe()
```

### 1.3 Repeat for HealthCare and Energy

```bash
# Just change SECTORS["IT"] → SECTORS["HealthCare"] / "Energy"
```

End of Phase 1: you have 3 parquet files in `outputs/`.

---

## Phase 2 — Step-1 prompt: signal definitions (10 min, ~$0.20 in API)

### 2.1 Generate (or load cached) definitions

```python
# scratch_phase2.py
from prompts.step1_definitions import get_signal_definitions
from config import EXISTING_SIGNALS, LLM_MODEL

defs = get_signal_definitions(EXISTING_SIGNALS, model=LLM_MODEL)
for key, content in defs.items():
    print(key, "→", content["definition"][:80], "…")
```

Output is cached to `prompts/signal_definitions.json` — you only pay once. Open
that file and read each entry. If GPT-4 wrote anything weird (e.g. confusing P/B
with B/P), fix it manually before continuing.

### 2.2 Render the definitions block that goes into step 2

```python
from prompts.step1_definitions import format_definitions_block
from config import EXISTING_SIGNALS
print(format_definitions_block(defs, EXISTING_SIGNALS))
```

This is the text block that becomes the "Definition of all existing signals"
section of the step-2 prompt. Make sure it's well-formed and roughly the
length of the equivalent block in Figure 1 of the paper.

---

## Phase 3 — Step-2 prompt: generate new signals (30 min, ~$2–$10 in API)

### 3.1 Run the generation prompt 6 times with different seeds

```python
# scratch_phase3.py
import json
import pandas as pd
from prompts.step1_definitions import get_signal_definitions
from prompts.step2_generate import generate_n_signals
from signals.parse_llm_output import parse_signal_response
from signals.compute_new import add_signal
from config import EXISTING_SIGNALS, LLM_MODEL, LLM_TEMP_GENERATION

panel = pd.read_parquet("outputs/IT_panel.parquet")
defs  = get_signal_definitions(EXISTING_SIGNALS, model=LLM_MODEL)

raw_responses = generate_n_signals(
    panel, defs, EXISTING_SIGNALS,
    model=LLM_MODEL,
    temperature=LLM_TEMP_GENERATION,
    n=6,
    return_col="ret_3m",
    n_companies=3, n_quarters=8,
)

# Save the raw responses for audit trail
with open("outputs/IT_new_signals.jsonl", "w") as f:
    for r in raw_responses:
        f.write(json.dumps({"response": r}) + "\n")
```

### 3.2 Parse formulas, compute values, store the enriched panel

```python
new_signal_names = []
panel_enriched = panel.copy()

for raw in raw_responses:
    parsed = parse_signal_response(raw)
    if parsed is None:
        print("✗ no formula found in response")
        continue
    name, expr = parsed
    try:
        panel_enriched = add_signal(panel_enriched, name, expr)
        new_signal_names.append(name)
        print(f"✓ {name} = {expr}")
    except ValueError as e:
        print(f"✗ {name}: {e}")

panel_enriched.to_parquet("outputs/IT_panel_with_new.parquet")
print("Generated:", new_signal_names)
```

**Watch for:**
- GPT-4 sometimes returns formulas using undefined names (e.g. `volatility`).
  These fail parsing and are dropped. If you want fewer failures, edit
  `prompts/templates.py` to enumerate the allowed symbols even more emphatically.
- Sometimes GPT-4 produces near-duplicates (PVS twice with different names). Dedupe
  manually or by running with higher temperature.
- Names should resemble the paper's family — products/divisions of ratios with
  occasional `log()`.

### 3.3 (Optional) Use the paper's signals as a reference

If you want to validate the eval pipeline before trusting your own GPT-4 output:

```python
from signals.existing import apply_paper_signals
panel_paper = apply_paper_signals(panel)
panel_paper.to_parquet("outputs/IT_panel_paper_signals.parquet")
```

This gives you the exact 6 formulas from §5.1 of the paper. Useful as a
control — if Fama-MacBeth says **these** don't beat baseline either, your
panel is wrong, not GPT-4.

---

## Phase 4 — Evaluate (30 min)

### 4.1 Spearman rank correlation heatmaps

```python
# scratch_phase4_corr.py
import pandas as pd
from evaluation.spearman import mean_correlation_matrix
from evaluation.plots import correlation_heatmap
from config import SIGNAL_KEYS
from pathlib import Path

panel = pd.read_parquet("outputs/IT_panel_with_new.parquet")
new_names = [c for c in panel.columns if c not in (
    SIGNAL_KEYS + ["date", "ticker", "ret_1m", "ret_3m"])]

# Heatmap 1: existing signals + return  → reproduces Figure 5a
existing_cm = mean_correlation_matrix(panel, SIGNAL_KEYS + ["ret_3m"])
correlation_heatmap(existing_cm, "IT — existing signals × ret_3m",
                    Path("outputs/figures/IT_existing.png"))

# Heatmap 2: new signals + return → reproduces Figure 5b
new_cm = mean_correlation_matrix(panel, new_names + ["ret_3m"])
correlation_heatmap(new_cm, "IT — new signals × ret_3m",
                    Path("outputs/figures/IT_new.png"))

# Combined view → reproduces Figure 3
combined_cm = mean_correlation_matrix(panel, SIGNAL_KEYS + new_names + ["ret_3m"])
correlation_heatmap(combined_cm, "IT — all signals × ret_3m",
                    Path("outputs/figures/IT_all.png"))
```

**What to check:**
- The bottom row / right column of each heatmap shows the correlation of each
  signal with the return. In the paper, **|ρ| ranges roughly 0.0 to 0.12**.
  If yours is bigger by 5× or 10×, you probably have look-ahead.
- New signals' magnitudes should be similar to (often slightly larger than)
  the existing signals'.
- `EVC` should have one of the largest |ρ| (paper's claim §5.3).

### 4.2 Fama-MacBeth adjusted R² box plots

```python
# scratch_phase4_fm.py
import pandas as pd
from evaluation.fama_macbeth import compare_models
from evaluation.plots import adj_r2_boxplot
from config import SIGNAL_KEYS
from pathlib import Path

panel = pd.read_parquet("outputs/IT_panel_with_new.parquet")
new_names = [c for c in panel.columns if c not in (
    SIGNAL_KEYS + ["date", "ticker", "ret_1m", "ret_3m"])]

adj_r2 = compare_models(panel, SIGNAL_KEYS, new_names, "ret_3m")
adj_r2.to_csv("outputs/IT_adj_r2_ret_3m.csv", index=False)
adj_r2_boxplot(adj_r2, "IT — Adj R² (ret_3m)",
               Path("outputs/figures/IT_adj_r2_boxplot.png"))

# Numerical summary
summary = adj_r2.drop(columns=["date"]).agg(["median", "mean", "std"]).T
summary["delta_vs_baseline"] = summary["median"] - summary.loc["baseline", "median"]
print(summary.sort_values("median", ascending=False))
```

**The headline claim:** 5 of 6 models with a new signal show a higher median
adjusted R² than baseline. After running, look at the
`delta_vs_baseline` column — positive for at least 5/6 rows confirms the
paper's finding.

### 4.3 Repeat for `ret_1m` and the other sectors

```python
for sector in ["IT", "HealthCare", "Energy"]:
    for h in ["ret_1m", "ret_3m"]:
        # ...repeat 4.1 + 4.2 for each (sector, horizon) pair
        # to reproduce Figures 7–16 in Appendix B
```

End of Phase 4: 6 box plots and 12 heatmaps, mirroring the paper's Figures
3–16.

---

## Phase 5 — One-command reproduction

Once each phase works, you can do everything end-to-end:

```bash
python run_all.py --sector IT          --return-horizon ret_3m --out outputs/IT_3m
python run_all.py --sector IT          --return-horizon ret_1m --out outputs/IT_1m
python run_all.py --sector HealthCare  --return-horizon ret_3m --out outputs/HC_3m
python run_all.py --sector HealthCare  --return-horizon ret_1m --out outputs/HC_1m
python run_all.py --sector Energy      --return-horizon ret_3m --out outputs/EN_3m
python run_all.py --sector Energy      --return-horizon ret_1m --out outputs/EN_1m

# Or sanity-check with the paper's exact formulas (no API spend):
python run_all.py --sector IT --use-paper-signals --out outputs/IT_paper
```

---

## Phase 6 — Validation against the paper

Open each output's `summary.csv` and confirm:

| Check | What to expect |
|---|---|
| ≥5 of 6 new signals beat baseline median adj R² | Yes for IT; appendix shows similar in HC, Energy |
| EVC has top Spearman |ρ| with returns | Yes in IT (paper §5.3) |
| Adjusted R² values in range | Roughly 0.03–0.20; if you see 0.8 something is wrong |
| Heatmap visual structure matches paper figures | Similar dominant colors, similar last-column profile |

If you reproduce 3/4 of those qualitatively, you've reproduced the paper. The
remaining noise is GPT-4 nondeterminism + your data source.

---

## Phase 7 — Things to try after reproduction works

These are the natural extensions (also in REPRODUCTION_PLAN.md §12):

1. **Out-of-sample test.** Generate signals using only 2016–2018 data in the
   prompt; evaluate on 2019–2020. The paper trains-and-tests on the same
   window — this is the biggest methodological weakness.
2. **Temperature ablation.** Run `generate_n_signals` with `temperature=0`,
   `0.3`, `0.7`, `1.0`. Plot how formula complexity changes.
3. **Different LLM.** Swap `LLM_MODEL` to Claude or a local model and compare
   the generated-formula distribution.
4. **Honest baseline.** Add Fama-French 5-factor or WorldQuant Alpha 101 as a
   comparison instead of just the 10 ratios — both are well-known and free.

---

## Costs and time budget

| Phase | Time | API spend |
|---|---|---|
| 0 — Env | 15 min | $0 |
| 1 — Data | 1–2 hr | $0 (yfinance free; FMP free tier OK) |
| 2 — Definitions | 10 min | ~$0.20 |
| 3 — Signal gen | 30 min × 3 sectors | $2–$10 total |
| 4 — Evaluation | 30 min | $0 |
| 5 — One-shot all sectors | 10 min | repeats |
| 6 — Validation | 30 min | $0 |

**Total: a focused weekend.**
