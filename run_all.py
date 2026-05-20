"""
End-to-end orchestrator. Run:

    python run_all.py --sector IT --start 2016-01-01 --end 2020-12-31

Outputs: panel.parquet, new_signals.jsonl, figures/*.png, summary.csv
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import config
from data.fetch_fundamentals import fetch_fundamentals
from data.fetch_returns import fetch_quarterly_forward_returns
from data.build_panel import build_panel
from prompts.step1_definitions import get_signal_definitions
from prompts.step2_generate import generate_n_signals
from signals.parse_llm_output import parse_signal_response
from signals.compute_new import add_signal
from signals.existing import apply_paper_signals
from evaluation.spearman import mean_correlation_matrix
from evaluation.fama_macbeth import compare_models
from evaluation.plots import correlation_heatmap, adj_r2_boxplot


def main(sector: str, start: str, end: str, return_horizon: str,
         n_signals: int, use_paper_signals: bool, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = config.SECTORS[sector]

    # 1. Data
    print(f"[1/6] Fundamentals for {len(tickers)} tickers in {sector}…")
    fundamentals = fetch_fundamentals(tickers)
    print(f"[2/6] Returns…")
    returns = fetch_quarterly_forward_returns(tickers, start, end,
                                              config.FORWARD_RETURN_HORIZONS)
    panel = build_panel(fundamentals, returns)
    panel.to_parquet(out_dir / "panel.parquet")
    print(f"    panel: {len(panel)} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].nunique()} dates")

    # 2. Generate or load new signals
    if use_paper_signals:
        print("[3/6] Skipping LLM — using paper's reported signals.")
        panel_with_new = apply_paper_signals(panel)
        new_signal_names = list(__import__("signals.existing", fromlist=["PAPER_NEW_SIGNALS"]).PAPER_NEW_SIGNALS.keys())
        (out_dir / "new_signals.jsonl").write_text("\n".join(
            json.dumps({"name": n, "source": "paper"}) for n in new_signal_names
        ))
    else:
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY in .env to call GPT-4")
        print("[3/6] Step 1: signal definitions…")
        defs = get_signal_definitions(config.EXISTING_SIGNALS,
                                      model=config.LLM_MODEL,
                                      temperature=config.LLM_TEMP_DEFINITIONS)
        print(f"[4/6] Step 2: generating {n_signals} new signals…")
        raw_responses = generate_n_signals(
            panel, defs, config.EXISTING_SIGNALS,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMP_GENERATION,
            n=n_signals, return_col=return_horizon,
        )
        new_signal_names = []
        panel_with_new = panel.copy()
        with (out_dir / "new_signals.jsonl").open("w") as f:
            for raw in raw_responses:
                parsed = parse_signal_response(raw)
                rec = {"raw": raw, "parsed": parsed}
                f.write(json.dumps(rec) + "\n")
                if parsed is None:
                    print("    skip (could not parse formula)")
                    continue
                name, expr = parsed
                try:
                    panel_with_new = add_signal(panel_with_new, name, expr)
                    new_signal_names.append(name)
                    print(f"    + {name} = {expr}")
                except ValueError as exc:
                    print(f"    skip ({exc})")

    # 3. Evaluate
    print("[5/6] Spearman correlations…")
    sig_cols = config.SIGNAL_KEYS + new_signal_names
    corr_existing = mean_correlation_matrix(panel_with_new,
                                            config.SIGNAL_KEYS + [return_horizon])
    corr_new = mean_correlation_matrix(panel_with_new,
                                       new_signal_names + [return_horizon])
    correlation_heatmap(corr_existing, f"{sector} — existing signals",
                        out_dir / "figures" / f"{sector}_corr_existing.png")
    correlation_heatmap(corr_new, f"{sector} — new signals",
                        out_dir / "figures" / f"{sector}_corr_new.png")

    print("[6/6] Fama-MacBeth…")
    adj_r2 = compare_models(panel_with_new, config.SIGNAL_KEYS,
                            new_signal_names, return_horizon)
    adj_r2.to_csv(out_dir / "adj_r2_by_date.csv", index=False)
    adj_r2_boxplot(adj_r2, f"{sector} — Adjusted R² by model",
                   out_dir / "figures" / f"{sector}_adj_r2_boxplot.png")

    summary = adj_r2.drop(columns=["date"]).agg(["median", "mean", "std"]).T
    summary.to_csv(out_dir / "summary.csv")
    print(summary)


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sector", choices=list(config.SECTORS.keys()), default="IT")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2020-12-31")
    p.add_argument("--return-horizon", default="ret_3m",
                   choices=["ret_1m", "ret_3m"])
    p.add_argument("--n-signals", type=int, default=6)
    p.add_argument("--use-paper-signals", action="store_true",
                   help="Skip the LLM and use the 6 signals from the paper.")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()
    main(args.sector, args.start, args.end, args.return_horizon,
         args.n_signals, args.use_paper_signals, Path(args.out))


if __name__ == "__main__":
    cli()
