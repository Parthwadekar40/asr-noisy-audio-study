#!/usr/bin/env python3
"""
07_make_docx.py
---------------
Produces editable Word versions of the two submission documents:

  report/technical_report.docx
  report/executive_summary.docx

Route: markdown -> pandoc, styled with a generated reference.docx (serif body,
quiet headings, compact tables) so the Word files look like documents a human
formatted, not a pandoc dump.

Usage: python scripts/07_make_docx.py   (needs: pip install pypandoc-binary python-docx)
"""

import os
import subprocess
import tempfile

import pypandoc
from docx import Document
from docx.shared import Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
REPORT = os.path.join(ROOT, "report")

INK = RGBColor(0x1C, 0x18, 0x14)
SAFFRON = RGBColor(0xC2, 0x4A, 0x17)
MUT = RGBColor(0x5A, 0x53, 0x48)


def make_reference(path: str) -> None:
    """Generate pandoc's default reference doc, then restyle it."""
    subprocess.run(
        [pypandoc.get_pandoc_path(), "-o", path,
         "--print-default-data-file", "reference.docx"],
        check=True,
    )
    doc = Document(path)
    styles = doc.styles

    def tune(name, font="Georgia", size=10.5, color=INK, bold=None, space_before=None):
        try:
            st = styles[name]
        except KeyError:
            return
        st.font.name = font
        st.font.size = Pt(size)
        st.font.color.rgb = color
        if bold is not None:
            st.font.bold = bold
        pf = getattr(st, "paragraph_format", None)
        if pf is not None and space_before is not None:
            pf.space_before = Pt(space_before)

    tune("Normal", size=10.5)
    tune("Body Text", size=10.5)
    tune("First Paragraph", size=10.5)
    tune("Compact", size=10)
    tune("Title", font="Georgia", size=20, color=INK, bold=True)
    tune("Heading 1", font="Georgia", size=15, color=INK, bold=True, space_before=14)
    tune("Heading 2", font="Georgia", size=12.5, color=SAFFRON, bold=True, space_before=12)
    tune("Heading 3", font="Georgia", size=11, color=INK, bold=True, space_before=10)
    tune("Block Text", size=10)
    doc.save(path)


def convert(md_name: str, docx_name: str, ref: str) -> None:
    src = os.path.join(REPORT, md_name)
    dst = os.path.join(REPORT, docx_name)
    pypandoc.convert_file(
        src, "docx", outputfile=dst,
        extra_args=["--reference-doc", ref, "--from", "gfm+smart"],
    )
    print(f"wrote {dst} ({os.path.getsize(dst)/1024:.0f} KB)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        ref = os.path.join(td, "reference.docx")
        make_reference(ref)
        convert("technical_report.md", "technical_report.docx", ref)
        convert("executive_summary.md", "executive_summary.docx", ref)
