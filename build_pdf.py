"""
Convert outputs/reproduction_report.html → outputs/reproduction_report.pdf.

Tries three backends in order:
  1. weasyprint (Python-native, best PDF quality; needs `pip install weasyprint`)
  2. Chrome / Chromium headless (uses the browser engine that rendered the HTML)
  3. Falls back with instructions to print manually.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HTML = Path("outputs/reproduction_report.html")
PDF  = Path("outputs/reproduction_report.pdf")


def _via_weasyprint() -> bool:
    try:
        from weasyprint import HTML as WeasyHTML
    except ImportError:
        return False
    print("Backend: weasyprint")
    WeasyHTML(filename=str(HTML.absolute())).write_pdf(str(PDF.absolute()))
    return True


def _via_chrome() -> bool:
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    chrome = next((c for c in candidates if c and Path(c).exists()), None)
    if not chrome:
        return False
    print(f"Backend: Chrome headless ({chrome})")
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={PDF.absolute()}",
            f"file://{HTML.absolute()}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def main() -> None:
    if not HTML.exists():
        print(f"Missing {HTML}. Run build_report.py first.")
        sys.exit(1)

    PDF.parent.mkdir(exist_ok=True)

    if _via_weasyprint():
        pass
    elif _via_chrome():
        pass
    else:
        print(
            "\nNeither weasyprint nor Chrome/Chromium found.\n\n"
            "Easiest fix — install weasyprint into your venv:\n"
            "    .venv/bin/pip install weasyprint\n"
            "    .venv/bin/python build_pdf.py\n\n"
            "Or just open the HTML in any browser and Cmd+P → Save as PDF.\n"
        )
        sys.exit(1)

    size_mb = PDF.stat().st_size / 1024 / 1024
    print(f"  → {PDF}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
