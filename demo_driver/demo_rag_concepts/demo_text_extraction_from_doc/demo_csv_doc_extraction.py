import sys

import pymupdf

from rag.rag_core.cleaning import clean_pages, normalize_unicode
from rag.rag_core.loaders import extract_csv
from config import data_dir, banner, event_banner, step, prettify_in_table_display, single_line


def extract_text_from_csv():
    banner("CSV Document Extraction Demo")
    csv_rows = extract_csv(data_dir / "product_catalog.csv", "Product Catalog")[0].text.split("\n")[2:10]
    print("\n".join(csv_rows))

# Run with:  uv run --active python demo_driver/demo_rag_concepts/demo_text_extraction_from_doc/demo_csv_doc_extraction.py
if __name__ == "__main__":
    extract_text_from_csv()