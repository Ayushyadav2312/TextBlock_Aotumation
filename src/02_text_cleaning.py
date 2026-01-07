"""
02_text_cleaning.py

Purpose:
- Clean raw extracted PDF text
- Merge fragmented sentences into coherent paragraphs
- Remove repeated headers / footers
- Remove obvious gibberish and OCR noise
- Preserve page traceability

Input:
- data/extracted_text/*_raw_text.json

Output:
- data/processed_chunks/*_cleaned_text.json
"""

import os
import json
import re
from collections import Counter
from tqdm import tqdm

# ---------------- CONFIG ---------------- #

INPUT_DIR = "data/extracted_text"
OUTPUT_DIR = "data/processed_chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thresholds
REPEAT_THRESHOLD = 0.6     # Appears on >60% of pages → header/footer
MIN_CHAR_LENGTH = 40       # Ignore very small fragments after cleaning

# ---------------------------------------- #


def normalize_text(text: str) -> str:
    """Basic normalization without destroying semantics."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[•■◦▪●]", " ", text)
    return text.strip()


def is_gibberish(text: str) -> bool:
    """
    Detect obvious garbage text.
    Conservative by design.
    """
    if len(text) < MIN_CHAR_LENGTH:
        return True

    # Too many non-alphabetic characters
    non_alpha_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
    if non_alpha_ratio > 0.35:
        return True

    return False


def merge_fragments(blocks):
    """
    Merge fragmented blocks into paragraph-level text.
    Uses punctuation and casing heuristics.
    """
    merged = []
    buffer_text = ""
    buffer_page = None

    for block in blocks:
        text = normalize_text(block["text"])

        if not text:
            continue

        if not buffer_text:
            buffer_text = text
            buffer_page = block["page"]
            continue

        # Heuristic: merge if sentence likely continues
        if (
            not buffer_text.endswith((".", "?", "!", ":"))
            and text[0].islower()
        ):
            buffer_text += " " + text
        else:
            merged.append({
                "page": buffer_page,
                "text": buffer_text
            })
            buffer_text = text
            buffer_page = block["page"]

    if buffer_text:
        merged.append({
            "page": buffer_page,
            "text": buffer_text
        })

    return merged


def detect_repeated_text(blocks):
    """
    Detect text repeated across many pages (headers/footers).
    """
    page_map = {}
    for b in blocks:
        page_map.setdefault(b["page"], set()).add(b["text"])

    total_pages = len(page_map)
    freq_counter = Counter()

    for texts in page_map.values():
        freq_counter.update(texts)

    repeated = {
        text for text, count in freq_counter.items()
        if count / total_pages >= REPEAT_THRESHOLD
    }

    return repeated


def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_blocks = json.load(f)

    if not raw_blocks:
        return

    # Sort blocks by page, then appearance order
    raw_blocks.sort(key=lambda x: (x["page"], x["block_id"]))

    repeated_texts = detect_repeated_text(raw_blocks)

    filtered_blocks = [
        b for b in raw_blocks
        if b["text"] not in repeated_texts
    ]

    merged_blocks = merge_fragments(filtered_blocks)

    cleaned_output = []
    for idx, block in enumerate(merged_blocks):
        text = block["text"]

        if is_gibberish(text):
            continue

        cleaned_output.append({
            "chunk_id": f"P{block['page']}_{idx}",
            "page": block["page"],
            "text": text
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_output, f, indent=2, ensure_ascii=False)


def process_all_files():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_raw_text.json")]

    if not files:
        raise FileNotFoundError("No extracted text files found.")

    for file in tqdm(files, desc="Cleaning text"):
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(
            OUTPUT_DIR,
            file.replace("_raw_text.json", "_cleaned_text.json")
        )

        process_file(input_path, output_path)

        print(f"Cleaned: {output_path}")


if __name__ == "__main__":
    process_all_files()
