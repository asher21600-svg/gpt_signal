# gpt_signal — Reproduction scaffold

Companion code to `../REPRODUCTION_PLAN.md`. Reproduces the pipeline from
*GPT-Signal: Generative AI for Semi-automated Feature Engineering in the Alpha
Research Process* (Wang, Zhao, Lawryshyn, FinNLP 2024).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env       # add your OPENAI_API_KEY
python tests/test_pipeline.py   # smoke test on synthetic data — no API/data needed
python run_all.py --sector IT --start 2016-01-01 --end 2020-12-31
```

## Pipeline

```
config.py  ──┐
             ▼
fetch_fundamentals + fetch_returns ─→ build_panel  (long-format DataFrame)
                                         │
                                         ▼
                             step1_definitions   (GPT-4: define the 10 signals)
                                         │
                                         ▼
                             step2_generate      (GPT-4: invent a new signal)
                                         │
                                         ▼
                             parse_llm_output    (extract formula text)
                                         │
                                         ▼
                             compute_new         (apply formula to panel)
                                         │
                                         ▼
                             evaluation.spearman + evaluation.fama_macbeth
                                         │
                                         ▼
                             evaluation.plots    (heatmaps + box plots)
```

## What's stubbed vs. done

- **Done:** project structure, prompt templates, parsing, all evaluation math,
  smoke-test on synthetic data.
- **Stubbed (needs your API key / data source):** `fetch_fundamentals` defaults
  to `yfinance` which is free but incomplete. Plug in FactSet or
  FinancialModelingPrep for closer faithfulness to the paper.

## Layout

See `../REPRODUCTION_PLAN.md` section 11.
