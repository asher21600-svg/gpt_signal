"""
Step 1: ask GPT-4 to write definitions for each of the 10 baseline signals.
Cached to disk so we don't re-pay for the same content each run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .templates import STEP1_DEFINITION_PROMPT

# Pull LLM connection settings from config so we have one switch point.
from config import LLM_BASE_URL, LLM_API_KEY_ENV

CACHE_PATH = Path(__file__).parent / "signal_definitions.json"


def _build_client():
    """Build an OpenAI SDK client pointed at the configured endpoint."""
    from openai import OpenAI
    api_key = os.environ.get(LLM_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Set {LLM_API_KEY_ENV} in .env "
            f"(e.g. DEEPSEEK_API_KEY=sk-...)"
        )
    if LLM_BASE_URL:
        return OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
    return OpenAI(api_key=api_key)


def _call_llm(prompt: str, model: str, temperature: float) -> str:
    """Wrapped so we can stub in tests."""
    client = _build_client()
    # DeepSeek's `deepseek-chat` supports JSON mode; if it ever stops working
    # we fall back to plain text and let the caller parse.
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    return resp.choices[0].message.content or ""


def get_signal_definitions(signals: list[dict], model: str, temperature: float = 0.0,
                           use_cache: bool = True) -> dict[str, dict]:
    """Return {key: {definition, effect_on_returns, preferred_tendency}}."""
    if use_cache and CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())

    out = {}
    for s in signals:
        prompt = STEP1_DEFINITION_PROMPT.format(label=s["label"])
        raw = _call_llm(prompt, model=model, temperature=temperature)
        try:
            out[s["key"]] = json.loads(raw)
        except json.JSONDecodeError:
            # Best-effort fallback: store raw text under "definition"
            out[s["key"]] = {"definition": raw, "effect_on_returns": "", "preferred_tendency": ""}

    CACHE_PATH.write_text(json.dumps(out, indent=2))
    return out


def format_definitions_block(defs: dict[str, dict], signals: list[dict]) -> str:
    """Render the definitions block that goes into the step-2 prompt."""
    parts = []
    for s in signals:
        d = defs.get(s["key"], {})
        parts.append(
            f"{s['label']}:\n"
            f"  Definition: {d.get('definition', '')}\n"
            f"  Effect on predicting stock returns: {d.get('effect_on_returns', '')}\n"
            f"  Preferred tendency: {d.get('preferred_tendency', '')}\n"
        )
    return "\n".join(parts)
