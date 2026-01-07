"""
run_pipeline.py

Purpose:
- Run the complete PDF processing pipeline (01 → 05) in order
- Single command execution
"""

import subprocess
import sys

PIPELINE_STEPS = [
    "src/01_pdf_extraction.py",
    "src/02_text_cleaning.py",
    "src/03_chunking.py",
    "src/04_section_classification.py",
    "src/05_aggregate_sections.py",
]


def run_step(step_path):
    print(f"\n🚀 Running: {step_path}")
    result = subprocess.run(
        [sys.executable, step_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Error in {step_path}")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)
    print(f"✅ Completed: {step_path}")


def run_pipeline():
    print("\n📊 Starting Full PDF Processing Pipeline\n")

    for step in PIPELINE_STEPS:
        run_step(step)

    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print("📁 Check final output in: data/output/\n")


if __name__ == "__main__":
    run_pipeline()
