"""
04_section_classification.py

Purpose:
- Classify each semantic chunk into predefined report sections
- Use open-source LLM via HuggingFace (LangChain LCEL)
- Preserve page traceability
- Output structured JSON for aggregation (Step 05)

Input:
- data/processed_chunks/*_final_chunks.json

Output:
- data/processed_chunks/*_classified_chunks.json
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# -------------------------------
# Modern LangChain imports
# -------------------------------
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser

# -------------------------------
# CONFIG
# -------------------------------
INPUT_DIR = "data/processed_chunks"
OUTPUT_DIR = "data/processed_chunks"

PDF_BASE_NAME = "acs_group"  # 🔴 change when processing another PDF

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
# Load HuggingFace LLM
# -------------------------------
raw_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.1,
    max_new_tokens=50,
    # task="text-generation" # Explicitly avoid text-generation if using featherless
)

# 2. Wrap it in ChatHuggingFace to handle the "conversational" task 🔴
llm = ChatHuggingFace(llm=raw_llm)

# -------------------------------
# Prompt Template
# -------------------------------
prompt = PromptTemplate.from_template("""
You are a financial report analysis expert.

Below are section names and their definitions:
{definitions}

Classify the following text into ONE most appropriate section name
from the list above.

Text:
{text}

Return ONLY the section name exactly as given.
""")

# -------------------------------
# LCEL Chain (modern LangChain)
# -------------------------------
chain = prompt | llm | StrOutputParser()

# -------------------------------
# Load chunked data
# -------------------------------
if not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(f"Chunk file not found: {CHUNKS_FILE}")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

df = pd.DataFrame(chunks)

# -------------------------------
# Classification function
# -------------------------------
def classify_text(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) < 5:
        return "Other"

    try:
        result = chain.invoke({
            "text": text[:1500],  # safety truncation
            "definitions": DEFINITIONS_TEXT
        }).strip()

        clean_result = (
            result.split("\n")[0]
            .replace('"', "")
            .strip()
        )

        if clean_result not in SECTION_NAMES:
            return "Other"

        return clean_result

    except Exception as e:
        print(f"Classification error: {e}")
        return "Error"

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
