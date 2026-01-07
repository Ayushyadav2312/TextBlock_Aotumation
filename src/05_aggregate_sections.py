# 05_aggregate_sections.py
"""
Purpose:
- Take classified chunk JSON
- Flatten chunks page-wise
- Create Excel: one row per text block per page
- Output: output/{pdf_name}_output.xlsx
"""

import os
import json
import pandas as pd
from tqdm import tqdm

# -------------------------------
# 1️⃣ Configuration
# -------------------------------
INPUT_DIR = "data/processed_chunks"
OUTPUT_FOLDER = "output"
PDF_BASE_NAME = "acs_group"  # change for other PDFs

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

INPUT_FILE = os.path.join(INPUT_DIR, f"{PDF_BASE_NAME}_classified_chunks.json")
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, f"{PDF_BASE_NAME}_output.xlsx")

# -------------------------------
# 2️⃣ Load JSON
# -------------------------------
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Classified chunks file not found: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -------------------------------
# 3️⃣ Flatten page-wise
# -------------------------------
rows = []

for chunk in chunks:
    text = chunk.get("text", "").strip()
    section = chunk.get("section", "Other")
    pages = chunk.get("pages", [])

    if not text or not pages:
        continue

    for page in pages:
        rows.append({
            "Textblock": section,
            "file_page_number": page,
            "text": text
        })

# -------------------------------
# 4️⃣ Create DataFrame & sort
# -------------------------------
df = pd.DataFrame(rows)
df.sort_values(by=["file_page_number"], inplace=True)
df.reset_index(drop=True, inplace=True)

# -------------------------------
# 5️⃣ Save to Excel
# -------------------------------
df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
print(f"[OK] Excel saved: {OUTPUT_FILE}")
print(f"[OK] Total rows: {len(df)}")
