"""
ValidationV5_AutoFixer.py

Reads validation_report_v4.csv, finds all MISMATCH questions,
and uses an LLM to:
  - Identify the TRUE correct answer
  - Rewrite distractors (better incorrect but plausible options)
  - Output a CSV of suggested fixes

Run:
    python ValidationV5_AutoFixer.py
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from openai import OpenAI

# ==============================
# CONFIG
# ==============================

VALIDATION_FILE = "validation_report_v4.csv"
QUESTION_FILE = "java_questions_adaptive.csv"
OUTPUT_FIXES = "validation_suggested_fixes.csv"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
MODEL = "gpt-4.1-mini"      # cheap but great
TEMPERATURE = 0.2           # low = accuracy > creativity

client = OpenAI(api_key=OPENAI_API_KEY)


# ==============================
# PROMPT GENERATOR
# ==============================

def build_fix_prompt(question_row: Dict[str, Any]) -> str:
    stem = question_row["question_stem"]
    A = question_row["option_a"]
    B = question_row["option_b"]
    C = question_row["option_c"]
    D = question_row["option_d"]

    prompt = f"""
You are an expert Java instructor and assessment designer.
You are given a Java multiple-choice question which FAILED validation.
Your task:

1. Re-evaluate the question carefully.
2. Determine the TRUE correct answer (A, B, C, or D).
3. If the provided distractors are weak, ambiguous, or incorrect,
   REWRITE improved distractors.
4. Ensure the corrected answer follows Java specification strictly.
5. Provide a JSON-only output in the format below.

QUESTION STEM:
{stem}

OPTIONS:
A) {A}
B) {B}
C) {C}
D) {D}

OUTPUT STRICTLY AS JSON (NO extra text):
{{
  "true_answer": "A"|"B"|"C"|"D",
  "improved_option_a": "...",
  "improved_option_b": "...",
  "improved_option_c": "...",
  "improved_option_d": "...",
  "explanation": "short explanation for correctness and distractor fixes"
}}
"""
    return prompt.strip()


def parse_json(raw: str) -> Dict[str, Any]:
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except:
        return {
            "true_answer": None,
            "improved_option_a": None,
            "improved_option_b": None,
            "improved_option_c": None,
            "improved_option_d": None,
            "explanation": f"PARSE_ERROR: {raw[:400]}"
        }


def llm_fix_question(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_fix_prompt(row)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": "You are an expert Java exam content auditor."},
                {"role": "user", "content": prompt},
            ]
        )
        raw = response.choices[0].message.content
        data = parse_json(raw)
        data["raw_response"] = raw
        return data
    except Exception as e:
        return {
            "true_answer": None,
            "improved_option_a": None,
            "improved_option_b": None,
            "improved_option_c": None,
            "improved_option_d": None,
            "explanation": f"LLM_ERROR: {e}",
            "raw_response": ""
        }


# ==============================
# MAIN SCRIPT
# ==============================

def main():
    print(f"📥 Loading validation file: {VALIDATION_FILE}")
    df_val = pd.read_csv(VALIDATION_FILE)

    print(f"📥 Loading question bank: {QUESTION_FILE}")
    df_q = pd.read_csv(QUESTION_FILE)

    # Convert question_id to str to avoid merge issues
    if "question_id" in df_q.columns:
        df_q["question_id"] = df_q["question_id"].astype(str)

    # The validation report doesn't include full text—must join using hashes or question text
    # Here, we re-merge by question stem snippet / hash
    # (Your ValidationV4 already saved "question_hash")
    if "question_hash" not in df_val.columns:
        raise ValueError("validation_report_v4.csv is missing 'question_hash' column")

    # Compute hashes for question file
    def hash_question(row):
        parts = [
            str(row.get("topic", "")),
            str(row.get("bloom_level", "")),
            str(row.get("question_stem", "")),
            str(row.get("option_a", "")),
            str(row.get("option_b", "")),
            str(row.get("option_c", "")),
            str(row.get("option_d", "")),
            str(row.get("correct_answer", "")),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    import hashlib
    df_q["question_hash"] = df_q.apply(hash_question, axis=1)

    # Merge original full questions onto validation results
    merged = df_val.merge(df_q, on="question_hash", how="left")

    # Filter mismatches
    mismatches = merged[merged["final_status"] == "MISMATCH"]
    print(f"⚠ Found {len(mismatches)} mismatched questions.")

    if mismatches.empty:
        print("🎉 No mismatches to fix. All answers validated successfully.")
        return

    fixes_out = []

    for idx, row in mismatches.iterrows():
        print(f"🔧 Fixing Q-hash: {row['question_hash'][:12]}...")

        question_data = {
            "question_stem": row["question_stem"],
            "option_a": row["option_a"],
            "option_b": row["option_b"],
            "option_c": row["option_c"],
            "option_d": row["option_d"],
        }

        fix = llm_fix_question(question_data)

        fixes_out.append({
            "question_hash": row["question_hash"],
            "topic": row.get("topic", ""),
            "old_correct": row.get("correct_answer", ""),
            "llm_true_answer": fix["true_answer"],
            "improved_option_a": fix["improved_option_a"],
            "improved_option_b": fix["improved_option_b"],
            "improved_option_c": fix["improved_option_c"],
            "improved_option_d": fix["improved_option_d"],
            "explanation": fix["explanation"],
            "raw_response": fix.get("raw_response", ""),
        })

    df_fixes = pd.DataFrame(fixes_out)
    df_fixes.to_csv(OUTPUT_FIXES, index=False, encoding="utf-8")

    print(f"\n✅ Suggested fixes saved to: {OUTPUT_FIXES}")
    print("Review these, then update your question bank with the corrected keys/distractors.")


if __name__ == "__main__":
    main()
