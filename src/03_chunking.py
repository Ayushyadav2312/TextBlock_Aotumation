"""
03_chunking.py

Purpose:
- Create semantically meaningful chunks from cleaned paragraphs
- Control chunk size for embeddings / LLMs
- Preserve page traceability
- Avoid breaking logical paragraphs unnecessarily

Input:
- data/processed_chunks/*_cleaned_text.json

Output:
- data/processed_chunks/*_final_chunks.json
"""

import os
import json
from tqdm import tqdm

# ---------------- CONFIG ---------------- #

INPUT_DIR = "data/processed_chunks"
OUTPUT_DIR = "data/processed_chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chunk sizing (embedding friendly)
MAX_CHARS = 900     # safe for most embedding models
MIN_CHARS = 250     # avoid tiny chunks

# ---------------------------------------- #


def chunk_paragraphs(paragraphs):
    """
    Combine adjacent paragraphs into semantic chunks.
    """
    chunks = []
    buffer_text = ""
    buffer_pages = set()

    for para in paragraphs:
        text = para["text"]
        page = para["page"]

        if not buffer_text:
            buffer_text = text
            buffer_pages.add(page)
            continue

        # If adding paragraph stays within limit → merge
        if len(buffer_text) + len(text) <= MAX_CHARS:
            buffer_text += "\n\n" + text
            buffer_pages.add(page)
        else:
            # Commit current chunk
            if len(buffer_text) >= MIN_CHARS:
                chunks.append({
                    "pages": sorted(buffer_pages),
                    "text": buffer_text
                })

            buffer_text = text
            buffer_pages = {page}

    # Add final chunk
    if buffer_text and len(buffer_text) >= MIN_CHARS:
        chunks.append({
            "pages": sorted(buffer_pages),
            "text": buffer_text
        })

    return chunks


def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        paragraphs = json.load(f)

    if not paragraphs:
        return

    # Sort by page order
    paragraphs.sort(key=lambda x: x["page"])

    chunks = chunk_paragraphs(paragraphs)

    final_chunks = []
    for idx, chunk in enumerate(chunks):
        final_chunks.append({
            "chunk_id": f"CH_{idx}",
            "pages": chunk["pages"],
            "text": chunk["text"],
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
