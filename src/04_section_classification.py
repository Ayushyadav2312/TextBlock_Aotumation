"""
04_section_classification.py

Purpose:
- Classify each semantic chunk into predefined report sections
- Prevent NOTES TO ACCOUNTS from being classified as MANAGEMENT
- Allow notes for other sections if relevant
- Use open-source LLM via HuggingFace (LangChain LCEL)
- Preserve page traceability
- Output structured JSON for aggregation (Step 05)
"""

import os
import json
import re
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# -------------------------------
# LangChain imports
# -------------------------------
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser

# -------------------------------
# CONFIG
# -------------------------------
INPUT_DIR = "data/processed_chunks"
OUTPUT_DIR = "data/processed_chunks"

PDF_BASE_NAME = "acs_group"   # 🔴 change per PDF

CHUNKS_FILE = f"{INPUT_DIR}/{PDF_BASE_NAME}_final_chunks.json"
OUTPUT_FILE = f"{OUTPUT_DIR}/{PDF_BASE_NAME}_classified_chunks.json"

SECTION_DEFINITION_FILE = "config/section_definitions.json"

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env")

# -------------------------------
# Load section definitions
# -------------------------------
with open(SECTION_DEFINITION_FILE, "r", encoding="utf-8") as f:
    SECTION_DEFINITIONS = json.load(f)

SECTION_NAMES = list(SECTION_DEFINITIONS.keys())
DEFINITIONS_TEXT = "\n".join(
    [f"{k}: {v}" for k, v in SECTION_DEFINITIONS.items()]
)

# -------------------------------
# NOTES TO ACCOUNTS DETECTION
# -------------------------------
NOTES_KEYWORDS = [
    "notes to the financial statements",
    "notes to the accounts",
    "accounting policies",
    "basis of preparation",
    "significant accounting policies",
    "accounting standards",
    "ind as",
    "ifrs",
    "gaap",
    "recognition and measurement",
    "financial instruments",
    "fair value measurement",
    "related party disclosures",
    "contingent liabilities",
    "commitments",
    "critical accounting estimates"
]

def is_notes_content(text: str) -> bool:
    text_lower = text.lower()

    for kw in NOTES_KEYWORDS:
        if kw in text_lower:
            return True

    # Structural patterns: "Note 1", "Note 12(a)"
    if re.search(r"\bnote\s+\d+", text_lower):
        return True

    return False

# -------------------------------
# Load HuggingFace LLM
# -------------------------------
raw_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.1,
    max_new_tokens=50
)

llm = ChatHuggingFace(llm=raw_llm)

# -------------------------------
# Prompt Template
# -------------------------------
prompt = PromptTemplate.from_template("""
You are an expert financial analyst classifying annual report text.

Assign the text to EXACTLY ONE section name
from the list below.

--------------------
SECTION DEFINITIONS
--------------------
{definitions}

--------------------
STRICT RULES
--------------------
1. Classify ONLY based on explicit information.
2. DO NOT infer missing context.
3. IGNORE tables, numeric-only text, headers, footers.
4. If unclear or mixed → return "Other".
5. Choose the MOST dominant section.

--------------------
MANAGEMENT RULE
--------------------
• management:
  - ONLY historical financial and operational results
  - Includes revenue, profit, margins, cash flow, KPIs
  - MUST NOT include accounting policies or notes

--------------------
TEXT
--------------------
{text}

--------------------
OUTPUT
--------------------
Return ONLY ONE section name.
""")

# -------------------------------
# LCEL Chain
# -------------------------------
chain = prompt | llm | StrOutputParser()

# -------------------------------
# Load chunks
# -------------------------------
if not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(f"Chunk file not found: {CHUNKS_FILE}")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

df = pd.DataFrame(chunks)

if "text" not in df.columns:
    raise ValueError("Expected 'text' field not found in chunk JSON")

# -------------------------------
# Classification logic
# -------------------------------
def classify_text(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) < 30:
        return "Other"

    try:
        result = chain.invoke({
            "text": text[:1200],
            "definitions": DEFINITIONS_TEXT
        }).strip()

        predicted_section = (
            result.split("\n")[0]
            .replace('"', "")
            .strip()
        )

        if predicted_section not in SECTION_NAMES:
            return "Other"

        # 🔒 FINAL SAFETY: management MUST NOT include notes
        if predicted_section == "management" and is_notes_content(text):
            return "Other"

        return predicted_section

    except Exception as e:
        print(f"Classification error: {e}")
        return "Other"

# -------------------------------
# Run classification
# -------------------------------
tqdm.pandas()
print("Starting section classification...")

df["section"] = df["text"].progress_apply(classify_text)

# -------------------------------
# Save output
# -------------------------------
df.to_json(
    OUTPUT_FILE,
    orient="records",
    indent=2,
    force_ascii=False
)

print(f"[OK] Classified chunks saved to: {OUTPUT_FILE}")
