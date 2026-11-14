import sys

import pymupdf

from rag.rag_core.cleaning import clean_pages, normalize_unicode
from rag.rag_core.loaders import extract_csv, extract_html, extract_pdf, tesseract_available
from config import data_dir, banner, event_banner, step, prettify_in_table_display, single_line


def extract_text_from_html():
    banner("HTML Document Extraction Demo")
    html = extract_html(data_dir / "nimbu_plus_membership.html")[0].text
    print("HTML -> Markdown (menus, footer and scripts removed):\n", html[:500], "...")

# Run with:  uv run --active python demo_driver/demo_rag_concepts/demo_html_doc_extraction.py
if __name__ == "__main__":
    extract_text_from_html()