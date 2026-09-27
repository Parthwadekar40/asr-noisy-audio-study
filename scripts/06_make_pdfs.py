#!/usr/bin/env python3
"""
06_make_pdfs.py
---------------
Renders report/technical_report.md and report/executive_summary.md to PDF via
markdown -> HTML -> WeasyPrint. Pure-Python toolchain, no LaTeX required.

Usage: python scripts/06_make_pdfs.py
"""

import os

import markdown
from weasyprint import HTML

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
REPORT = os.path.join(ROOT, "report")

CSS = """
@page {
  size: A4; margin: 1.55cm 1.5cm;
  @bottom-center { content: counter(page) " / " counter(pages);
                   font-size: 9px; color: #9ca3af; }
}
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 9.6pt;
       line-height: 1.42; color: #111827; }
h1 { font-family: "DejaVu Sans", Helvetica, sans-serif; font-size: 17pt;
     line-height: 1.2; margin: 0 0 3pt 0; }
h2 { font-family: "DejaVu Sans", Helvetica, sans-serif; font-size: 12.5pt;
     margin: 12pt 0 4pt 0; border-bottom: 1.5px solid #2563eb;
     padding-bottom: 1.5pt; }
h3 { font-family: "DejaVu Sans", Helvetica, sans-serif; font-size: 10.5pt;
     margin: 9pt 0 3pt 0; }
p { margin: 4pt 0; text-align: justify; }
li { margin: 2pt 0; }
strong { color: #111827; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 10pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0;
        font-family: "DejaVu Sans", Helvetica, sans-serif; font-size: 8pt; }
th { background: #f3f4f6; text-align: left; font-size: 7.6pt; }
td, th { border: 1px solid #d1d5db; padding: 2.5pt 4pt; vertical-align: top; }
tr:nth-child(even) td { background: #fafafa; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt;
       background: #f3f4f6; padding: 0 2pt; }
pre { background: #0f172a; color: #e2e8f0; padding: 8pt; border-radius: 4pt;
      font-size: 8pt; overflow-x: hidden; }
pre code { background: none; color: inherit; }
blockquote { border-left: 3px solid #2563eb; margin: 8pt 0; padding: 2pt 10pt;
             background: #f8fafc; color: #374151; }
h2, h3 { page-break-after: avoid; }
table, pre { page-break-inside: avoid; }
"""


def md_to_pdf(md_path: str, pdf_path: str) -> None:
    text = open(md_path).read()
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists"]
    )
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    HTML(string=html).write_pdf(pdf_path)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path)/1024:.0f} KB)")


if __name__ == "__main__":
    md_to_pdf(os.path.join(REPORT, "technical_report.md"),
              os.path.join(REPORT, "technical_report.pdf"))
    md_to_pdf(os.path.join(REPORT, "executive_summary.md"),
              os.path.join(REPORT, "executive_summary.pdf"))
