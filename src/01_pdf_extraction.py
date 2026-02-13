"""
01_pdf_extraction.py

Purpose:
- Extract layout-aware narrative text from PDF files
- Exclude tables
- Remove headers and footers
- Preserve page numbers
"""

import fitz
import os
import json
from tqdm import tqdm
from collections import Counter

PDF_INPUT_DIR = "data/raw_pdfs"
OUTPUT_DIR = "data/extracted_text"

PDF_NAME = "acs_group.pdf"  # Set None to process all PDFs

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_table_like(text: str) -> bool:
    """
    Heuristic detection of table-like content.
    """
    lines = text.splitlines()
    if len(lines) < 2:
        return False

    char_count = len(text)
    digit_count = sum(c.isdigit() for c in text)
    digit_ratio = digit_count / max(char_count, 1)

    short_lines_ratio = sum(len(l.strip()) < 30 for l in lines) / len(lines)
    has_column_spacing = "   " in text or "\t" in text

    return digit_ratio > 0.30 and (short_lines_ratio > 0.5 or has_column_spacing)


def find_repeated_text(blocks):
    """
    Identify repeated text blocks across pages → likely headers/footers
    """
    text_counter = Counter([block["text"] for block in blocks])
    repeated_texts = set([text for text, count in text_counter.items() if count > 2])
    return repeated_texts


def extract_pdf_blocks(pdf_path: str):
    """
    Extract text blocks from PDF, excluding tables and headers/footers.
    """
    doc = fitz.open(pdf_path)
    all_blocks = []

    # First pass: collect all blocks for repeated text detection
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        for block_id, block in enumerate(blocks):
            text = block[4].strip()
            if text:
                all_blocks.append({"page": page_num + 1, "text": text})

    # Detect headers/footers
    repeated_texts = find_repeated_text(all_blocks)

    # Second pass: save only narrative blocks
    extracted_blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        for block_id, block in enumerate(blocks):
            text = block[4].strip()
            if not text or text in repeated_texts or is_table_like(text):
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
        pdf_files = [PDF_NAME]
    else:
        pdf_files = [f for f in os.listdir(PDF_INPUT_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError("No PDF files found")

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
