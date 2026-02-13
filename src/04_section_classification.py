"""
04_section_classification.py (Base FinBERT - No Fine-Tuning)

Purpose:
- Classify each semantic chunk using base FinBERT (sentiment)
- Preserve page traceability
- Output structured JSON
"""

import os
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------
# CONFIG
# -------------------------------
INPUT_DIR = "data/processed_chunks"
OUTPUT_DIR = "data/processed_chunks"

PDF_BASE_NAME = "acs_group"

CHUNKS_FILE = f"{INPUT_DIR}/{PDF_BASE_NAME}_final_chunks.json"
OUTPUT_FILE = f"{OUTPUT_DIR}/{PDF_BASE_NAME}_classified_chunks.json"

MODEL_PATH = "models/ProsusAIfinbert"   # ← Fix this path

MAX_LENGTH = 256
BATCH_SIZE = 16

# -------------------------------
# LOAD MODEL
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"FinBERT model not found at {MODEL_PATH}")

print("Loading base FinBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Get built-in label mapping from model
id2label = model.config.id2label

# -------------------------------
# LOAD CHUNKS
# -------------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

df = pd.DataFrame(chunks)

if "text" not in df.columns:
    raise ValueError("Expected 'text' field not found in chunk JSON")

# -------------------------------
# BATCH CLASSIFICATION
# -------------------------------
def classify_batch(texts):

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    results = []

    for i in range(len(texts)):
        label = id2label[preds[i].item()]
        results.append(label)

    return results

# -------------------------------
# RUN CLASSIFICATION
# -------------------------------
print("Starting FinBERT classification...")

sections = []

for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch_texts = df["text"].iloc[i:i+BATCH_SIZE].tolist()
    batch_preds = classify_batch(batch_texts)
    sections.extend(batch_preds)

df["section"] = sections

# -------------------------------
# SAVE OUTPUT
# -------------------------------
df.to_json(
    OUTPUT_FILE,
    orient="records",
    indent=2,
    force_ascii=False
)

print(f"[OK] Classified chunks saved to: {OUTPUT_FILE}")
