# gpt_signal

> Independent reproduction of *GPT-Signal: Generative AI for Semi-automated Feature Engineering in the Alpha Research Process* (Wang, Zhao & Lawryshyn — FinNLP @ ACL 2024).

A full end-to-end reproduction of the paper's LLM-driven cross-sectional alpha pipeline, using **DeepSeek-Chat** in place of GPT-4 and **FinancialModelingPrep** in place of FactSet. Includes three independent runs, two horizons, three sectors, a temperature sweep, and a self-contained HTML/PDF report.

---

## Headline findings

| Configuration | Window | Win rate (signal × sector × horizon cells beating baseline) |
|---|---|---|
| **Paper (GPT-4 + FactSet)** | 2016–2020 | 83 % (5 of 6) |
| **This reproduction, in-window** | 2016–2020 | **76–87 % across T ∈ {0.0, 0.3, 0.7, 1.0}** |
| This reproduction, extended window | 2016–2024 | 26.7 % (8 of 30) |

**The paper reproduces faithfully within its own window. The alpha decays out-of-window** — the post-pandemic period (2021–2024) breaks the cross-sectional pattern the LLM-generated signals exploit. *Sign-safe compounds (`PAVS`-family) are the only signals that transfer across runs, sectors, and horizons.*

Full discussion in [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md).

---

## What's in this repo

| File / folder | What it is |
|---|---|
| [`REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md) | Reverse-engineering analysis of the paper, design choices, gotchas to expect |
| [`RUN_GUIDE.md`](RUN_GUIDE.md) | Step-by-step commands per phase |
| [`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md) | Final findings — in-window match, alpha decay, three novel observations |
| [`data/`](data/) | Multi-source fundamentals + price adapters (FMP `/stable/`, SEC EDGAR XBRL, yfinance, synthetic) |
| [`signals/`](signals/) | Sandboxed formula evaluator with **sign-safe `log` / `sqrt`** wrappers; LLM-output parser; the paper's 6 reference formulas |
| [`prompts/`](prompts/) | Two-step LLM prompt template (definitions → generation) with zero-shot Chain-of-Thought |
| [`evaluation/`](evaluation/) | Spearman rank correlation, Fama-MacBeth two-step regression, plotting (heatmaps + box plots) |
| [`build_report.py`](build_report.py) | Generates a self-contained HTML report with all figures embedded as base64 |
| [`build_pdf.py`](build_pdf.py) | Converts the HTML report to PDF via Chrome-headless or weasyprint |
| [`tests/`](tests/) | Smoke test with synthetic data — runs in ~30 s, no API/network needed |
| [`run_all.py`](run_all.py) | End-to-end orchestrator |
| [`quant-paper-reproduction.skill`](quant-paper-reproduction.skill) | Packaged Claude Code skill capturing this whole workflow as a reusable template for future quant-finance paper reproductions |

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env to add:
#   DEEPSEEK_API_KEY=sk-...           (any OpenAI-compatible API works)
#   FMP_API_KEY=...                   (FinancialModelingPrep Starter or higher)
#   EDGAR_USER_AGENT="Your Name your.email@example.com"   (optional fallback)

# 3. Smoke test — synthetic data, validates the math pipeline end-to-end
python tests/test_pipeline.py

# 4. Full real-data run on the paper's IT sector
python run_all.py --sector IT --start 2016-01-01 --end 2020-12-31

# 5. Build the shareable report
python build_report.py        # outputs/reproduction_report.html
python build_pdf.py           # outputs/reproduction_report.pdf
```

---

## Pipeline

```
config.py (sector tickers, signal list)
   │
   ▼
fetch_fundamentals  ─→  FMP / EDGAR / yfinance / synthetic (auto-routes via env vars)
fetch_returns       ─→  yfinance + FMP (recovers delisted tickers)
build_panel         ─→  long-format DataFrame: date, ticker, 10 signals, ret_1m, ret_3m
   │
   ▼
prompts.step1_definitions  ─→  LLM defines the 10 baseline signals (cached, ~¥0.05)
prompts.step2_generate     ─→  LLM invents 6 new compound signals
                                with zero-shot CoT ("Let's think step by step")
   │
   ▼
signals.parse_llm_output   ─→  extract SIGNAL_FORMULA from LLM response
signals.compute_new        ─→  sandboxed eval; sign-safe log/sqrt prevents
                                NaN cascades in cyclical sectors
   │
   ▼
evaluation.spearman        ─→  cross-sectional rank correlation matrix
evaluation.fama_macbeth    ─→  two-step regression, adjusted R² per date
evaluation.plots           ─→  heatmaps + box plots
   │
   ▼
build_report  +  build_pdf
```

---

## Key design choices and substitutions

These deviations from the paper are deliberate and documented:

1. **DeepSeek-Chat instead of GPT-4.** OpenAI-compatible API; same prompt mechanics; comparable instruction-following quality on this task. Confirms the method is LLM-agnostic.
2. **FMP `/stable/` instead of FactSet.** Both provide US-listed quarterly fundamentals from regulatory filings. FMP renamed `priceEarningsRatio` → `priceToEarningsRatio` and `priceCashFlowRatio` → `priceToOperatingCashFlowRatio` in the `/stable/` API; the adapter handles this. SEC EDGAR adapter is also bundled as a free alternative.
3. **Sign-safe log/sqrt evaluation.** LLM-generated formulas often use `log(product_of_signals)` where some inputs (ROE, ROA, FCF, NM) can be negative in cyclical sectors — naive evaluation produces NaN cascades. The bundled evaluator wraps `log` as `sign(x) · log(max(eps, |x|))`. **This is the most important deviation** — without it, the paper's Energy results are unreproducible.
4. **Survivorship-aware data fetch.** Acquired/delisted tickers (PXD, MRO, HES, CTLT, ANSS, JNPR) get their pre-delisting price history from FMP rather than being silently dropped by yfinance.

---

## Novel findings beyond the paper

1. **LLM nondeterminism is large.** Three independent invocations of the same prompt produce three completely different families of formulas. The paper's "5 of 6 beat baseline" is one draw from a wide distribution.
2. **Sector-dependent failure mode.** Log-heavy LLM formulas return NaN in cyclical sectors (Energy in particular) when an odd number of profitability terms is negative. A small evaluation-side wrapper (sign-safe log) fixes this without changing the prompt.
3. **Sign-safety is the survival property.** Across runs, sectors, and horizons, the signals that beat baseline most consistently are the ones with explicit sign-safety — either DeepSeek's spontaneous `(x + 1e-6)` offsets or the wrapper-applied sign-safe log.
4. **Convergence across LLMs.** Both GPT-4 (paper) and DeepSeek (here) gravitate to the same structural family: products of profitability metrics divided by valuation multiples, log-wrapped. The "Quality at a Reasonable Price" factor, rediscovered prompt-only by two different LLMs with no quant training.

---

## Citing the original paper

```
@inproceedings{wang-etal-2024-gpt-signal,
  title     = {GPT-Signal: Generative AI for Semi-automated Feature Engineering in the Alpha Research Process},
  author    = {Wang, Yining and Zhao, Jinman and Lawryshyn, Yuri},
  booktitle = {Proceedings of the Joint Workshop of the 8th Financial Technology and Natural Language Processing (FinNLP) and the 1st Agent AI for Scenario Planning (AgentScen)},
  year      = {2024},
  url       = {https://aclanthology.org/2024.finnlp-2.4/}
}
```

Author repo: <https://github.com/Yiningww/GPT-signal>

---

## License

MIT (or your preferred license — add a `LICENSE` file when ready).
