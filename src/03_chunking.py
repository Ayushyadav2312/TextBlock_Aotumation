"""
03_chunking.py

Purpose:
- Create chunks strictly page-wise
- One chunk belongs to ONE page only
- Prevent cross-page text merging
"""

import os
import json
from tqdm import tqdm

# ---------------- CONFIG ---------------- #

INPUT_DIR = "data/processed_chunks"
OUTPUT_DIR = "data/processed_chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_CHARS = 900
MIN_CHARS = 250

# ---------------------------------------- #


def chunk_paragraphs(paragraphs):
    """
    Create chunks STRICTLY within the same page.
    """
    chunks = []
    buffer_text = ""
    current_page = None

    for para in paragraphs:
        text = para["text"]
        page = para["page"]

        # New page detected → flush buffer
        if current_page is not None and page != current_page:
            if len(buffer_text) >= MIN_CHARS:
                chunks.append({
                    "page": current_page,
                    "text": buffer_text
                })
            buffer_text = text
            current_page = page
            continue

        # First paragraph
        if current_page is None:
            buffer_text = text
            current_page = page
            continue

        # Same page → append if size allows
        if len(buffer_text) + len(text) <= MAX_CHARS:
            buffer_text += "\n\n" + text
        else:
            if len(buffer_text) >= MIN_CHARS:
                chunks.append({
                    "page": current_page,
                    "text": buffer_text
                })
            buffer_text = text

    # Final flush
    if buffer_text and len(buffer_text) >= MIN_CHARS:
        chunks.append({
            "page": current_page,
            "text": buffer_text
        })

    return chunks


def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        paragraphs = json.load(f)

    if not paragraphs:
        return

    # Ensure page order
    paragraphs.sort(key=lambda x: x["page"])

    chunks = chunk_paragraphs(paragraphs)

    final_chunks = []
    for idx, chunk in enumerate(chunks):
        final_chunks.append({
            "chunk_id": f"CH_{idx}",
            "page": chunk["page"],          # ✅ single page
            "text": chunk["text"],    # ✅ consistent naming
            "char_length": len(chunk["text"])
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2, ensure_ascii=False)


def process_all_files():
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.endswith("_cleaned_text.json")
    ]

    if not files:
        raise FileNotFoundError("No cleaned text files found.")

    for file in tqdm(files, desc="Chunking text"):
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(
            OUTPUT_DIR,
            file.replace("_cleaned_text.json", "_final_chunks.json")
        )

        process_file(input_path, output_path)
        print(f"Chunked: {output_path}")


if __name__ == "__main__":
    process_all_files()
