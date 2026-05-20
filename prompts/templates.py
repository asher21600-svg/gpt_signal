"""
Raw prompt strings, lifted from Figure 1 of the paper as closely as possible.
"""
from __future__ import annotations

STEP1_DEFINITION_PROMPT = """\
You are a quantitative finance expert. For the financial signal named "{label}",
return a JSON object with exactly three keys:

  "definition": a one- to three-sentence plain-English definition.
  "effect_on_returns": how the signal is generally believed to relate to future
                       stock returns.
  "preferred_tendency": whether investors prefer higher or lower values, and why.

Respond with JSON only. Do not include any prose outside the JSON.
"""

STEP2_GENERATION_PROMPT = """\
Instructions
I will give you some financial information, including several rows of a
financial dataset of multiple companies with some signals (included in the
context) and their expected returns. I will also give you the descriptions of
these signals.

Definition of all existing signals
{signal_definitions}

Data of different companies
{sample_data}

Query
Please create a new signal based on the provided context (existing signals),
and this new signal should be correlated to the returns, explain how you
created this signal and describe the meaning of this new signal. Note that
don't provide simple linear combination of other existing signals and focus on
as many meaningful existing signals as possible. Please also provide the
calculated values of this new signal and standardize them. Let's think step by
step.

After your explanation, end your response with a single line of the exact
format:

  SIGNAL_FORMULA: <name> = <python expression using existing signal symbols>

where the symbols you may use are: pe, pb, roa, roe, fcf, pcf, ebitda, gm, nm,
sps, log, exp, sqrt, abs. Example:
  SIGNAL_FORMULA: IQS = roe * (1/pe) * (1/pb) * log(sps)
"""
