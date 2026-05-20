"""
Extract the (name, formula) pair from a GPT-4 step-2 response.

The step-2 prompt ends with the instruction:
    SIGNAL_FORMULA: <name> = <expression>
So we just look for the last line matching that regex.
"""
from __future__ import annotations

import re

_FORMULA_RE = re.compile(
    r"^\s*SIGNAL_FORMULA:\s*(?P<name>[\w_]+)\s*=\s*(?P<expr>.+?)\s*$",
    re.MULTILINE,
)


def parse_signal_response(raw: str) -> tuple[str, str] | None:
    """Return (name, expression) or None if not parseable."""
    matches = list(_FORMULA_RE.finditer(raw))
    if not matches:
        return None
    m = matches[-1]   # use the last line — GPT-4 sometimes restates
    return m.group("name").strip(), m.group("expr").strip()
