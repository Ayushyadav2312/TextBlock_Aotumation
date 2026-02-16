"""
train_from_excel.py

Purpose:
- Fine-tune FinBERT on labeled Excel data
- Save trained model for use in 04_section_classification.py
"""

import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "data/labeled_data.xlsx"
BASE_MODEL = "models/ProsusAIfinbert"
OUTPUT_DIR = "models/finbert_section_model"

MAX_LENGTH = 256
EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 2e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load Excel Data
# -------------------------
print("Loading labeled Excel data...")
df = pd.read_excel(DATA_PATH)

if "text" not in df.columns or "section" not in df.columns:
    raise ValueError("Excel must contain 'text' and 'section' columns")

df = df.dropna(subset=["text", "section"])
df["section"] = df["section"].str.strip().str.lower()
df = df[df["text"].str.len() > 20]

print("Total usable samples:", len(df))

# -------------------------
# Encode Labels
# -------------------------
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["section"])

num_labels = len(label_encoder.classes_)
print("Detected sections:", list(label_encoder.classes_))

# Save label mapping
label_map = {str(i): label for i, label in enumerate(label_encoder.classes_)}

with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w") as f:
    json.dump(label_map, f, indent=2)

# -------------------------
# Train / Validation Split
# -------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_ds = Dataset.from_dict({
    "text": train_texts.tolist(),
    "label": train_labels.tolist()
})

val_ds = Dataset.from_dict({
    "text": val_texts.tolist(),
    "label": val_labels.tolist()
})

# -------------------------
# Tokenization
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds.set_format("torch")
val_ds.set_format("torch")

# -------------------------
# Load Model
# -------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=num_labels
)

# -------------------------
# Metrics
# -------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# -------------------------
# Training Arguments
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_dir="logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# -------------------------
# Train Model
# -------------------------
print("Starting training...")
trainer.train()

# -------------------------
# Save Model
# -------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuned model saved at:", OUTPUT_DIR)
