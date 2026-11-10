"""submission/report.md -> report.html -> report.pdf with a headless Chromium browser (Edge or Chrome).

    uv run --with markdown python -m analysis.submission_pdf

Needs the `markdown` package and Edge or Chrome. Equations are written in Unicode, so no math renderer is needed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", "msedge", "google-chrome", "chromium"]
CSS = """
@page { size: Letter; margin: 16mm 14mm; }
body { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #0b0b0b; }
h1 { font-size: 17pt; margin: 0 0 6pt; } h2 { font-size: 13pt; margin: 16pt 0 4pt; border-bottom: 1px solid #ddd; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 8.5pt; page-break-inside: avoid; }
th, td { border-bottom: 1px solid #e4e3df; padding: 2pt 6pt; text-align: left; } th { background: #f3f2ee; }
blockquote { margin: 6pt 0; padding: 4pt 10pt; background: #f6f5f1; border-left: 3px solid #2a78d6; font-family: Cambria, Georgia, serif; }
.eq { font-family: Cambria, Georgia, serif; font-size: 11pt; margin: 3pt 0 3pt 18pt; white-space: pre-wrap; }
img { max-width: 100%; page-break-inside: avoid; margin: 4pt 0; } code { font-size: 8.5pt; background: #f3f2ee; padding: 0 2px; }
"""


def normalise(text: str) -> str:
    """python-markdown wants a blank line before a list and 4-space nesting; GitHub-style markdown needs neither."""
    out, prev = [], ""
    for line in text.splitlines():
        if line.startswith("  - ") or line.startswith("   ") and line.lstrip()[:2] in ("- ", "1."):
            line = "    " + line.lstrip()
        is_item = re.match(r"^(\s*)(- |\d+\. )", line) is not None
        prev_item = re.match(r"^(\s*)(- |\d+\. )", prev) is not None
        if is_item and prev.strip() and not prev_item:
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def main() -> None:
    text = normalise((SUB / "report.md").read_text(encoding="utf-8"))
    body = markdown.markdown(text, extensions=["tables", "sane_lists"])
    html = SUB / "report.html"
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Sleeve 1 report</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    exe = next(b for b in BROWSERS if Path(b).exists() or shutil.which(b))
    pdf = SUB / "report.pdf"
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, timeout=120)
    print(pdf, pdf.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
