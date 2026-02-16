"""
04_section_classification.py (Fine-Tuned FinBERT)

Purpose:
- Classify each semantic chunk into predefined report sections
- Use fine-tuned FinBERT model
- Preserve page traceability
"""

import os
import json
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

MODEL_PATH = "models/finbert_section_model"

MAX_LENGTH = 256
BATCH_SIZE = 16

# -------------------------------
# LOAD MODEL
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run train_from_excel.py first.")

print("Loading fine-tuned FinBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load label mapping
with open(os.path.join(MODEL_PATH, "label_mapping.json"), "r") as f:
    label_map = json.load(f)

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

    preds = torch.argmax(outputs.logits, dim=1)

    results = []

    for i in range(len(texts)):
        label = label_map[str(preds[i].item())]
        results.append(label)

    return results

# -------------------------------
# RUN CLASSIFICATION
# -------------------------------
print("Starting section classification...")

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
