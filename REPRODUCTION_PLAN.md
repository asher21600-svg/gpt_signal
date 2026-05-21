# Reverse-Engineering & Reproducing "GPT-Signal"

**Paper:** *GPT-Signal: Generative AI for Semi-automated Feature Engineering in the Alpha Research Process*
Yining Wang, Jinman Zhao, Yuri Lawryshyn — University of Toronto
FinNLP @ ACL 2024 (proceedings ID `2024.finnlp-2.4`)
Authors' code: `https://github.com/Yiningww/GPT-signal` (couldn't fetch from this session — GitHub is not on the allowlist. Clone it locally or enable github.com in Settings → Capabilities.)

This document is the **reverse-engineering analysis and full reproduction plan**. A runnable code scaffold lives in `gpt_signal/` next to this file.

---

## 1. What the paper actually does (one paragraph)

They feed GPT-4 a prompt containing (a) plain-English definitions of 10 well-known fundamental financial ratios, (b) a tiny tabular sample of historical values for those ratios on a handful of S&P 500 companies, and (c) the next-period stock returns. They ask GPT-4 to invent a *new* signal that is correlated with returns and is **not** a simple linear combination of the inputs. GPT-4 returns a formula plus reasoning (zero-shot Chain-of-Thought via "Let's think step by step"). They then evaluate the new signal against the originals using (1) a Spearman rank correlation matrix and (2) a Fama-MacBeth two-step cross-sectional regression, reporting adjusted R² distributions across time. They run this on three S&P 500 sectors (IT, Health Care, Energy) for 2016–2020, predicting 1-month and 3-month forward returns.

## 2. Why it works — the key design choices

Reverse-engineering the choices, not just the steps:

- **Two-step prompt, not one.** Step 1 asks GPT-4 to produce definitions/effect/preferred-tendency for the 10 signals *in its own words*. The output of step 1 becomes part of the step-2 prompt. This is doing two things: it primes the model's chain-of-thought and it strips the user of the burden of writing perfect financial definitions. You could ablate this — pre-write the definitions and skip step 1 — and probably get similar results, but the paper's prompt template is what's released.
- **Tiny tabular sample, not full data.** They include "several rows" for "some of the selected companies" in the prompt. The model never sees the whole dataset; it sees enough to ground its reasoning. This is critical for token budget and is consistent with TabLLM / CAAFE findings cited in §2.
- **"Don't provide a simple linear combination."** This explicit instruction is what produces the nonlinear formulas (logs, products, inverses) that distinguish their output from a regression. Without it, GPT-4 tends to return weighted sums.
- **Standardize after generation.** The prompt asks GPT-4 to standardize the new signal values. In practice you re-standardize in Python (Z-score) before regression — the LLM's arithmetic is not trustworthy.
- **Z-score the signals before regression.** Different ratios live on very different scales (P/E in tens, ROE as a fraction). Without normalization the OLS coefficients are meaningless.
- **Spearman rank, not Pearson.** Outliers in P/E and similar ratios are massive. Rank correlation neutralizes them.
- **Fama-MacBeth, not pooled OLS.** Cross-sectional regression at each date, then summarize across dates. This is the textbook way to evaluate factor models — pooled regression conflates time-series and cross-sectional variance.
- **Adjusted R² across time as a box plot.** "Did the new signal help?" becomes "did the distribution of cross-sectional adjusted R² shift up?" Median + IQR + outliers in one picture.

The headline result is that 5 of 6 generated signals shift the median R² above the 10-signal baseline. Effect sizes are small in absolute terms (R² in the 0.05–0.15 range) but that's typical for cross-sectional return regressions.

## 3. The 10 existing signals (inputs)

P/E, P/B, ROA, ROE, FCF (free cash flow per share), P/CF, EBITDA (i.e. EV/EBITDA), GM (gross margin), NM (net margin), SPS (sales per share). All quarterly, all from FactSet in the paper. Source list:
- MSCI GICS for sector classification
- FactSet for fundamentals
- Yahoo Finance for prices/returns

## 4. The 6 signals GPT-4 generated (outputs to expect)

| Name | Formula |
|---|---|
| PVS (Profitable Valuation Score) | ROE / (P/E) |
| RAPS (Risk-Adjusted Performance Score) | ROE / (P/E · β), β = 2 (fixed constant) |
| EVC (Efficiency Value Composite) | ROA · (1/EBITDA) · (1/P/CF) |
| VEC (Valuation Efficiency Composite) | (P/E + ROE + FCF) / 3 |
| PLF (Profitability Leverage Factor) | (ROE · GM) / (P/E) |
| IQS (Investment Quality Score) | ROE · (1/P/E) · (1/P/B) · log(SPS) |

Note `RAPS` hardcodes β=2 "for calculation convenience" — that's a smell. A real reproduction should either compute β properly (regress each stock's return on the market) or just drop that signal from the comparison. The paper still includes it.

Note also `VEC` is a *linear* combination — despite the prompt explicitly forbidding that. The LLM disobeyed and the authors kept it. That's worth flagging if you reproduce.

## 5. Companies (exact lists from Appendix A)

- **IT (43):** AAPL, AKAM, AMD, ANET, ANSS, APH, CDNS, CDW, CTSH, ENPH, EPAM, FFIV, FSLR, FTNT, GEN, GLW, IBM, INTC, IT, JNPR, KLAC, LRCX, MCHP, MPWR, MSFT, MSI, NOW, NXPI, ON, PTC, QCOM, ROP, STX, SWKS, TDY, TEL, TER, TRMB, TXN, TYL, VRSN, WDC, ZBRA
- **Health Care (31):** ABBV, ABT, ALGN, AMGN, BAX, BDX, BIO, BMY, BSX, CAH, COR, CRL, CTLT, CVS, DGX, DHR, DXCM, EW, GILD, HSIC, TMO, UHS, VRTX, VTRS, IDXX, ILMN, INCY, WST, ZTS, ISRG, JNJ
- **Energy (19):** APA, COP, CTRA, EOG, FANG, HAL, HES, KMI, MPC, MRO, OKE, OXY, PSX, PXD, SLB, TRGP, VLO, WMB, XOM

Heads-up: a few of these (PXD, ATVI-style symbols, GEN, COR, CTRA) reflect mid-2023 ticker reality. Some may have changed since 2020. Treat the list as a starting point, not a static truth.

## 6. The prompt (verbatim shape from Figure 1)

Step 2 prompt has four blocks:

```
Instructions
I will give you some financial information, including several rows of a financial
dataset of multiple companies with some signals (included in the context) and their
expected returns. I will also give you the descriptions of these signals.

Definition of all existing signals
Price/Earnings (P/E) Ratio:
  Definition: …
  Effect on predicting stock returns: …
  Preferred tendency: …
[repeated for all 10 signals — produced by step-1 prompt]

Data of different companies
AAPL:
Date         P/E      P/B     ROE     …   Return
2016-03-31   11.7542  4.4375  17.8925 …   0.1272
2016-06-30   10.9112  3.9807  16.5172 …  -0.0427
…

Query
Please create a new signal based on the provided context (existing signals), and
this new signal should be correlated to the returns, explain how you created this
signal and describe the meaning of this new signal. Note that don't provide simple
linear combination of other existing signals and focus on as many meaningful
existing signals as possible. Please also provide the calculated values of this
new signal and standardize them. Let's think step by step.
```

Step 1 prompt is simpler — for each of the 10 signals, ask GPT-4 to produce (definition, effect on predicting returns, preferred tendency). Concatenate these into the "Definition of all existing signals" block.

## 7. Reproduction phases

### Phase 0 — Environment
- Python 3.10+, `openai`, `pandas`, `numpy`, `statsmodels`, `scipy`, `yfinance`, `matplotlib`, `seaborn`, `python-dotenv`.
- `OPENAI_API_KEY` in `.env`.
- Quarterly fundamentals data source. **The paper uses FactSet, which is a paid subscription.** If you don't have FactSet, the free alternatives are:
  - `yfinance` (`Ticker.quarterly_financials`, `quarterly_balance_sheet`, `quarterly_cashflow`) — gives you raw line items; you compute ratios.
  - `financialmodelingprep.com` API (free tier, has ratios pre-computed).
  - `sec-api.io` or `python-sec-api` (direct from EDGAR).
- Expect ~30–60 min of cleanup per data source to match the paper's quarterly grid.

### Phase 1 — Data
1. For each ticker in the sector list, pull quarterly fundamentals 2015-12-31 through 2020-12-31 (need a buffer for 3-month forward returns).
2. Compute the 10 ratios. (If a source gives ratios directly, sanity-check by recomputing one or two.)
3. Pull daily adjusted close prices, resample to quarter-end, compute 1-month and 3-month forward returns (use `pct_change(periods=k).shift(-k)` semantics, ensuring no look-ahead).
4. Merge into a long-format dataframe: `(date, ticker, P/E, P/B, …, SPS, ret_1m, ret_3m)`.

### Phase 2 — Step-1 prompt (signal definitions)
1. For each of the 10 signals, ask GPT-4: *"Describe Price/Earnings (P/E). Output JSON with three keys: definition, effect_on_returns, preferred_tendency."*
2. Cache the responses to disk (`prompts/signal_definitions.json`). You don't need to regenerate these on every run.

### Phase 3 — Step-2 prompt (new signal generation)
1. Take a random subsample of companies (e.g. 3) and a window of dates (e.g. 8 quarters). Convert to the table format shown in Figure 1.
2. Concatenate: instructions + signal definitions (from step 1) + sample table + query.
3. Call GPT-4 with `temperature=0.7` or so to get variety. The paper says they "run the script multiple times, one new signal per run."
4. Parse the response: extract (a) signal name, (b) formula, (c) reasoning text. Storage: `outputs/new_signals.jsonl`.
5. **Important** — don't trust GPT-4's numerical computation of the new signal. Re-implement the formula in Python and recompute values from raw data.

### Phase 4 — Evaluation
1. **Spearman correlation matrix.** For each date *t* and each sector, compute the cross-sectional Spearman correlation between each signal (existing + new) and forward returns. Average across *t*. Plot as heatmap (Figure 3 / 5 / 7 / 9 / 11 / 13 / 15 of the paper).
2. **Fama-MacBeth two-step.**
   - *Step 1 (time series):* For each company, regress its return time series on its 10+k signal time series. Get factor exposures β̂.
   - *Step 2 (cross-section):* At each date *t*, regress the cross-section of returns on the cross-section of β̂s. Record adjusted R² for that *t*.
   - Repeat for each model variant: baseline (10 signals) and baseline+new_i for each of the 6 new signals.
   - Plot the distribution of adjusted R² values across dates as a box plot (Figures 4, 6, 8, 10, 12, 14, 16).

### Phase 5 — Comparison
- Confirm: for 5/6 new signals, median adjusted R² shifts up vs. baseline.
- Confirm: new EVC has the highest absolute Spearman correlation with returns.
- If your results disagree, the most likely causes (in order) are: (a) different data source (FactSet vs. yfinance — different point-in-time semantics), (b) different sample window inside the prompt → different signal formulas, (c) you forgot to drop look-ahead in the forward-return calculation.

## 8. Reproducibility gotchas (in order of likelihood to bite you)

1. **GPT-4 is non-deterministic.** Even at `temperature=0` the formulas will drift between runs. Cache the prompt/response pairs and version them. Plan to either (a) accept your own 6 signals and just verify the methodology works, or (b) run the prompt many times and curate the same family as the paper.
2. **Look-ahead bias.** Fundamentals are *reported* with a lag (10-Q is filed 30–45 days after quarter end). The paper says they use signals from March to predict June returns — that's safe. If you instead use signals dated 2020-03-31 to "predict" returns over 2020-Q1, you're cheating.
3. **Survivorship bias.** The company lists are S&P 500 *as of* paper-writing. Any sector membership change biases results.
4. **β=2 in RAPS.** Trivially wrong — flag it.
5. **VEC is linear.** The prompt forbids that but GPT-4 produced it anyway. Either keep it (faithful reproduction) or drop it (cleaner experiment).
6. **Standardization order.** Z-score *each signal across companies at each date* (cross-sectional), not the whole panel at once. The paper isn't explicit about this; cross-sectional is the standard interpretation for Fama-MacBeth.
7. **Companies with missing quarters.** Drop or forward-fill? The paper doesn't say. Default: drop rows with any NaN in the 10 signals or the forward return.

## 9. Validation checklist

After running the pipeline end-to-end:

- [ ] All 3 sectors have correlation heatmaps shaped like Figures 5, 9, 11, 13, 15.
- [ ] Adjusted R² box plots match the qualitative ordering: most new signals' medians > baseline median.
- [ ] EVC produces the strongest absolute Spearman correlation among new signals in IT (and similar patterns in HC, Energy).
- [ ] GPT-4 reasoning trace is preserved alongside each new signal (audit trail).
- [ ] Output a single CSV per sector summarizing median R², mean Spearman corr, and rank order — easier to compare than reading box plots.

## 10. Expected timeline

For someone with intermediate Python + light quant background:
- Phase 0–1 (env + data): 4–8 hours, mostly waiting on rate limits and cleaning quarterly mismatches.
- Phase 2–3 (prompts + GPT-4 calls): 2–4 hours; ~$5–$20 in API spend depending on how many runs.
- Phase 4 (evaluation): 4–6 hours, especially the Fama-MacBeth implementation.
- Phase 5 (matching the figures): 2–4 hours of plotting and tweaking.

Total: a focused weekend, or 1–2 weeks of evenings.

## 11. Repo layout (matches the code scaffold next door)

```
gpt_signal/
├── README.md
├── requirements.txt
├── .env.example                 # OPENAI_API_KEY=...
├── config.py                    # sector tickers, date range, signal list
├── data/
│   ├── fetch_fundamentals.py    # yfinance / FMP / FactSet adapters
│   ├── fetch_returns.py         # yfinance prices → quarterly returns
│   └── build_panel.py           # merge to long-format panel
├── prompts/
│   ├── step1_definitions.py     # ask GPT-4 to define the 10 signals
│   ├── step2_generate.py        # main signal-generation prompt
│   └── templates.py             # raw prompt strings
├── signals/
│   ├── existing.py              # the 10 baseline signals
│   ├── parse_llm_output.py      # extract formula from GPT-4 response
│   └── compute_new.py           # evaluate generated formulas on the panel
├── evaluation/
│   ├── spearman.py              # cross-sectional rank correlation
│   ├── fama_macbeth.py          # two-step regression + adjusted R²
│   └── plots.py                 # heatmaps + box plots
├── run_all.py                   # end-to-end orchestrator
└── tests/
    └── test_pipeline.py         # smoke test on synthetic data
```

## 12. Things to do *after* the basic reproduction works

- Swap GPT-4 for Claude / Llama-3 / Mixtral and see whether the generated formulas have different character (Claude tends toward more interpretable, less log-heavy expressions, anecdotally).
- Add `temperature` ablation: does temp=0 produce dull linear combinations? Does temp=1 produce gibberish?
- Out-of-sample test: generate signals on 2016–2018, evaluate on 2019–2020 only. The paper trains and evaluates on the same window — that's the biggest methodological weakness.
- Compare against a real factor library (e.g. WorldQuant Alpha 101 or Fama-French 5-factor) as an honest baseline, not just the 10 ratios.
