import sys

import pymupdf

from rag.rag_core.cleaning import clean_pages, normalize_unicode
from rag.rag_core.loaders import extract_csv, extract_html, extract_pdf, tesseract_available
from config import data_dir, banner, event_banner, step, prettify_in_table_display, single_line

"""
-----------------------------------------------
  Document Extraction by Native Extractor
-----------------------------------------------
file                             page  chars of text  starts with                                                   images
-------------------------------  ----  -------------  ------------------------------------------------------------  ------
travel_reimbursement_policy.pdf  1     942            Travel and Expense Reimbursement Policy (FIN-007) This polic  0     
travel_reimbursement_policy.pdf  2     675            3. Filing a claim Submit every claim in the Nimbu Expense po  0     
travel_reimbursement_policy.pdf  3     426            6. International travel International travel needs written a  0     


file                        page  chars of text  starts with  images
--------------------------  ----  -------------  -----------  ------
warranty_terms_scanned.pdf  1     0                           1     


file                         page  chars of text  starts with                                                   images
---------------------------  ----  -------------  ------------------------------------------------------------  ------
vendor_onboarding_mixed.pdf  1     365            Vendor Onboarding Guide This guide is for the Procurement te  0     
vendor_onboarding_mixed.pdf  2     193            Document checklist The checklist below is the same one print  1     



The scanned PDF has 0 characters of text: a naive RAG would silently index NOTHING from it.
"""
def extract_text_from_pdf_by_native_extractor(pdf_files:list[str]=["travel_reimbursement_policy.pdf", "warranty_terms_scanned.pdf", "vendor_onboarding_mixed.pdf"]) -> list[dict]:   
    event_banner("Document Extraction by Native Extractor")
    all_rows_list_of_col_attr_dict = []
    for name in pdf_files:
        with pymupdf.open(data_dir / name) as pdf:
            rows = [{"file": name, "page": i + 1, "chars of text": len(p.get_text()), "starts with": p.get_text()[:60],
                    "images": len(p.get_images())} for i, p in enumerate(pdf)]
        prettify_in_table_display(rows)
        all_rows_list_of_col_attr_dict.extend(rows)
    print("\nThe scanned PDF has 0 characters of text: a naive RAG would silently index NOTHING from it.")
    return all_rows_list_of_col_attr_dict



"""
-----------------------------------------------
  Document Extraction by Custom Extractor
-----------------------------------------------
file                             page  route          chars of text  starts with                                                
-------------------------------  ----  -------------  -------------  -----------------------------------------------------------
travel_reimbursement_policy.pdf  1     digital+table  982            Nimbu Mart | Internal | Finance Policy FIN-007 # Travel and
travel_reimbursement_policy.pdf  2     digital        690            Nimbu Mart | Internal | Finance Policy FIN-007 ## 3. Filing
travel_reimbursement_policy.pdf  3     digital        436            Nimbu Mart | Internal | Finance Policy FIN-007 ## 6. Intern


file                        page  route  chars of text  starts with                                               
--------------------------  ----  -----  -------------  ----------------------------------------------------------
warranty_terms_scanned.pdf  1     ocr    743            Nimbu Care Warranty Terms Every product sold by Nimbu Mart


file                         page  route    chars of text  starts with                                               
---------------------------  ----  -------  -------------  ----------------------------------------------------------
vendor_onboarding_mixed.pdf  1     digital  374            Nimbu Mart | Procurement # Vendor Onboarding Guide This gu
vendor_onboarding_mixed.pdf  2     mixed    400            Nimbu Mart | Procurement ## Document checklist The checkli
"""
def extract_text_from_pdf_by_custom_extractor(pdf_files:list[str]=["travel_reimbursement_policy.pdf", "warranty_terms_scanned.pdf", "vendor_onboarding_mixed.pdf"]) -> list[dict]:   
    event_banner("Document Extraction by Custom Extractor")
    all_rows_list_of_col_attr_dict = []
    for name in pdf_files:
        pages = extract_pdf(data_dir / name)
        rows = [{"file": name, "page": p.number, "route": p.method, "chars of text": len(p.text), "starts with": p.text[:60]} for p in pages]
        prettify_in_table_display(rows)
        all_rows_list_of_col_attr_dict.extend(rows)

    print("\nThe scanned PDF has 0 characters of text: a naive RAG would silently index NOTHING from it.")
    return all_rows_list_of_col_attr_dict

"""
----------------------------------------------------------------------------------------
  Document Extraction by Tesseract OCR. Routing every page (and every big image) to the right method
----------------------------------------------------------------------------------------

--> Tesseract OCR available: True
----------------------------------------------------------------------------------------

--> OCR output of the scanned warranty page
Nimbu Care Warranty Terms

Every product sold by Nimbu Mart carries the manufacturer warranty listed below. Keep your
invoice: it is your warranty card.

Warranty periods

Kitchen appliances (mixer grinders, induction cooktops, kettles): 24 months.
Phones and tablets: 12 months. Laptops: 12 months.

Televisions: 12 months on the product and 36 months on the display panel.
Earbuds and headphones: 6 months.

Nimbu Care+ extended warranty

Laptops can be covered for 24 extra months with Nimbu Care+ for Rs. 2,499. Nimbu Care+ must
be bought within 30 days of delivery.

How to claim

Raise a claim in the app under Orders, Warranty claim. A technician visit is booked within 3
working days. Physica
----------------------------------------------------------------------------------------

--> The checklist that was only an IMAGE inside the mixed PDF (found by block-level OCR)
Nimbu Mart | Procurement

## Document checklist

The checklist below is the same one printed on the paper form. It is included as an image because
many vendors still send scanned forms.

Mandatory documents

1. GST registration certificate (GSTIN)
2. PAN card of the business

3. Cancelled cheque for bank verification
4. Signed vendor agreement

5. FSSAI licence (only for food sellers)

Page 2 of 2
"""
def extract_image_text_from_pdf_by_tesseract_ocr(pdf_files:list[str]=["travel_reimbursement_policy.pdf", "warranty_terms_scanned.pdf", "vendor_onboarding_mixed.pdf"]):   
    event_banner("Document Extraction by Tesseract OCR. Routing every page (and every big image) to the right method")
    step(f"Tesseract OCR available: {tesseract_available()}")
    
    single_line()
    step("OCR output of the scanned warranty page")
    print(extract_pdf(data_dir / "warranty_terms_scanned.pdf")[0].text[:700] or "(empty: install Tesseract to enable OCR)")
    
    single_line()
    step("The checklist that was only an IMAGE inside the mixed PDF (found by block-level OCR)")
    mixed = extract_pdf(data_dir / "vendor_onboarding_mixed.pdf")
    print(mixed[-1].text[-400:])



def extract_tables_from_pdf(file: str = data_dir / "travel_reimbursement_policy.pdf") -> list[str]:
    event_banner("3. Post Document Extraction: Tables stay tables")

    single_line()
    step("Extracting tables from the travel reimbursement policy PDF via PyMuPDF")
    with pymupdf.open(file) as pdf:
        flat = pdf[0].get_text()
    start = flat.find("City tier")
    print("\nCompare: the same table as plain text (row/column meaning is lost):\n", " ".join(flat[start:start + 160].split()))

    single_line()
    step("Extracting tables from the travel reimbursement policy PDF via custom extractor")
    travel = extract_pdf(file)
    md_table_attr_rows_with_col_header = [l for l in travel[0].text.split("\n") if l.startswith("|")]
    print("\n".join(md_table_attr_rows_with_col_header))
    return md_table_attr_rows_with_col_header


def cleanup_pages_in_pdf(file: str = data_dir / "travel_reimbursement_policy.pdf"):
    event_banner("4. Cleaning")
    travel = extract_pdf(file)
    raw_text = travel[0].text
    cleaned_pages, report = clean_pages(travel)
    print("Removed repeated header/footer patterns:", report["removed_repeated_lines"])
    print(f"Characters: {report['chars_before']} -> {report['chars_after']}")
    for bad in ["ﬂ", "Rs. ", "Page 1 of 3"]:
        print(f"  '{bad}' in raw text: {bad in raw_text}   in cleaned text: {bad in cleaned_pages[0].text}")
    print("  NFKC example:", repr("ﬂights ５ days"), "->", repr(normalize_unicode("ﬂights ５ days")))
    return cleaned_pages



def pdf_doc_extraction():
    banner("PDF Document Extraction Demo")
    extract_text_from_pdf_by_native_extractor(pdf_files = ["travel_reimbursement_policy.pdf", "warranty_terms_scanned.pdf", "vendor_onboarding_mixed.pdf"])
    extract_text_from_pdf_by_custom_extractor(pdf_files = ["travel_reimbursement_policy.pdf", "warranty_terms_scanned.pdf", "vendor_onboarding_mixed.pdf"])
    extract_image_text_from_pdf_by_tesseract_ocr(pdf_files = ["travel_reimbursement_policy.pdf", "warranty_terms_scanned.pdf", "vendor_onboarding_mixed.pdf"])
    extract_tables_from_pdf(file = data_dir / "travel_reimbursement_policy.pdf")
    cleanup_pages_in_pdf(file = data_dir / "travel_reimbursement_policy.pdf")

# Run with:  uv run --active python demo_driver/demo_rag_concepts/demo_pdf_doc_extraction.py
if __name__ == "__main__":
    pdf_doc_extraction()