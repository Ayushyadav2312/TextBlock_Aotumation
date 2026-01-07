"""
01_pdf_extraction.py

Purpose:
- Extract layout-aware text from PDF files
- Preserve page numbers and block structure
- DO NOT apply cleaning or filtering logic here
"""

import fitz
import os
import json
from tqdm import tqdm

PDF_INPUT_DIR = "data/raw_pdfs"
OUTPUT_DIR = "data/extracted_text"

PDF_NAME = "acs_group.pdf"  # 👈 CHANGE THIS WHEN NEEDED
# PDF_NAME = None           # Uncomment this to process ALL PDFs


os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_pdf_blocks(pdf_path: str):
    doc = fitz.open(pdf_path)
    extracted_blocks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")

        for block_id, block in enumerate(blocks):
            text = block[4].strip()

            if not text:
                continue

            extracted_blocks.append({
                "file_name": os.path.basename(pdf_path),
                "page": page_num + 1,
                "block_id": block_id,
                "text": text
            })

    return extracted_blocks


def process_pdfs():
    if PDF_NAME:
        pdf_path = os.path.join(PDF_INPUT_DIR, PDF_NAME)

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"{PDF_NAME} not found in raw_pdfs")

        pdf_files = [PDF_NAME]
    else:
        pdf_files = [f for f in os.listdir(PDF_INPUT_DIR) if f.lower().endswith(".pdf")]

        if not pdf_files:
            raise FileNotFoundError("No PDF files found in data/raw_pdfs")

    for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
        pdf_path = os.path.join(PDF_INPUT_DIR, pdf_file)

        extracted_data = extract_pdf_blocks(pdf_path)

        output_path = os.path.join(
            OUTPUT_DIR,
            pdf_file.replace(".pdf", "_raw_text.json")
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved {output_path} | Blocks: {len(extracted_data)}")



if __name__ == "__main__":
    process_pdfs()
