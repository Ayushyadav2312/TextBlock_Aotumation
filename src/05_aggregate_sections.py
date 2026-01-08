"""
05_aggregate_sections.py

Purpose:
- Take classified chunk JSON
- Flatten data page-wise
- One row per text block
- Repeat page number where needed
- Output Excel
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
PDF_BASE_NAME = "acs_group"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

INPUT_FILE = os.path.join(INPUT_DIR, f"{PDF_BASE_NAME}_classified_chunks.json")
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, f"{PDF_BASE_NAME}_output.xlsx")

# -------------------------------
# 2️⃣ Load JSON
# -------------------------------
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"File not found: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -------------------------------
# 3️⃣ Flatten data (SAFE MODE)
# -------------------------------
rows = []

for chunk in tqdm(chunks, desc="Aggregating sections"):
    text = chunk.get("text", "").strip()
    section = chunk.get("section", "Other")

    # Handle different page formats safely
    page = (
        chunk.get("page")
        or chunk.get("file_page_number")
    )

    if not text or page is None:
        continue

    rows.append({
        "Section": section,
        "Page_Number": page,
        "Text": text
    })

# -------------------------------
# 4️⃣ Create DataFrame
# -------------------------------
df = pd.DataFrame(rows)

if df.empty:
    raise ValueError("No valid rows created. Check classified JSON structure.")

df.sort_values(by="Page_Number", inplace=True)
df.reset_index(drop=True, inplace=True)

# -------------------------------
# 5️⃣ Save to Excel
# -------------------------------
df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")

print(f"[OK] Excel saved: {OUTPUT_FILE}")
print(f"[OK] Total rows: {len(df)}")
