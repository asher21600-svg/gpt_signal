"""
Sector lists from Appendix A and the 10 baseline signals from §4.
Edit ticker lists here if any have changed since the paper was written.
"""
from __future__ import annotations

SECTORS = {
    "IT": [
        "AAPL", "AKAM", "AMD", "ANET", "ANSS", "APH", "CDNS", "CDW", "CTSH",
        "ENPH", "EPAM", "FFIV", "FSLR", "FTNT", "GEN", "GLW", "IBM", "INTC",
        "IT", "JNPR", "KLAC", "LRCX", "MCHP", "MPWR", "MSFT", "MSI", "NOW",
        "NXPI", "ON", "PTC", "QCOM", "ROP", "STX", "SWKS", "TDY", "TEL",
        "TER", "TRMB", "TXN", "TYL", "VRSN", "WDC", "ZBRA",
    ],
    "HealthCare": [
        "ABBV", "ABT", "ALGN", "AMGN", "BAX", "BDX", "BIO", "BMY", "BSX",
        "CAH", "COR", "CRL", "CTLT", "CVS", "DGX", "DHR", "DXCM", "EW",
        "GILD", "HSIC", "TMO", "UHS", "VRTX", "VTRS", "IDXX", "ILMN", "INCY",
        "WST", "ZTS", "ISRG", "JNJ",
    ],
    "Energy": [
        "APA", "COP", "CTRA", "EOG", "FANG", "HAL", "HES", "KMI", "MPC",
        "MRO", "OKE", "OXY", "PSX", "PXD", "SLB", "TRGP", "VLO", "WMB", "XOM",
    ],
}

# The 10 baseline signals. Names match the variables in the paper's prompt
# template. `key` is the column name used in the long-format panel; `label` is
# the human-readable name fed to GPT-4 in the prompt.
EXISTING_SIGNALS = [
    {"key": "pe",     "label": "Price/Earnings (P/E) Ratio"},
    {"key": "pb",     "label": "Price/Book Value (P/B) Ratio"},
    {"key": "roa",    "label": "Return on Assets (ROA)"},
    {"key": "roe",    "label": "Return on Equity (ROE)"},
    {"key": "fcf",    "label": "Free Cash Flow per Share (FCF)"},
    {"key": "pcf",    "label": "Price/Cash Flow (P/CF)"},
    {"key": "ebitda", "label": "Enterprise Value/EBITDA (EV/EBITDA)"},
    {"key": "gm",     "label": "Gross Margin (GM)"},
    {"key": "nm",     "label": "Net Margin (NM)"},
    {"key": "sps",    "label": "Sales per Share (SPS)"},
]

SIGNAL_KEYS = [s["key"] for s in EXISTING_SIGNALS]

# Reporting cadence
REBALANCE_FREQ = "Q"          # quarter-end
FORWARD_RETURN_HORIZONS = [1, 3]   # months

# LLM defaults. The scaffold uses the OpenAI Python SDK, but points it at
# DeepSeek's OpenAI-compatible endpoint via LLM_BASE_URL. To switch back to
# real OpenAI, set LLM_BASE_URL=None and use "gpt-4-turbo" + OPENAI_API_KEY.
LLM_MODEL = "deepseek-chat"
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_API_KEY_ENV = "DEEPSEEK_API_KEY"     # name of the env var holding the key
LLM_TEMP_DEFINITIONS = 0.0
LLM_TEMP_GENERATION = 0.7
