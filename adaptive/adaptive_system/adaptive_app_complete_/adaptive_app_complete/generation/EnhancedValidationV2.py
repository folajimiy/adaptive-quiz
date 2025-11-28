# ---------------------------------------------------------------
# EnhancedValidationV2.py
# Multi-Model Java MCQ Validator
# ---------------------------------------------------------------

import pandas as pd
import hashlib
import json
import time
from openai import OpenAI

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------

API_KEY = "YOUR_OPENAI_API_KEY"
PRIMARY_MODEL = "gpt-4.1-mini"     # Fast, cheap, reliable
SECONDARY_MODEL = "gpt-4.1"         # Medium cost, higher accuracy
PREMIUM_MODEL = None                # e.g. "o1-mini" or "o1-preview" (optional)

INPUT_FILE = "java_questions_adaptive.csv"
OUTPUT_FILE = "validation_report_v2.csv"

client = OpenAI(api_key=API_KEY)


# ===============================================================
# UTILITIES
# ===============================================================

def hash_question(row):
    """
    Hash question content so unchanged items don't need revalidation.
    """
    s = (
        row["question_stem"]
        + row["option_a"]
        + row["option_b"]
        + row["option_c"]
        + row["option_d"]
        + row["correct_answer"]
    )
    return hashlib.sha256(s.encode()).hexdigest()


def clean_json(s):
    """
    Remove markdown fences and ensure valid JSON.
    """
    return (
        s.replace("```json", "")
         .replace("```", "")
         .strip()
    )


# ===============================================================
# PROMPTS
# ===============================================================

def build_primary_prompt(row):
    """
    Fast model: correctness only + low-cost reasoning.
    """
    return f"""
You are a *precise Java code evaluator*.

TASK:
1. Solve the Java MCQ exactly.
2. Determine the correct option (A/B/C/D).
3. Indicate your confidence (high/medium/low).
4. Compare with the PROVIDED answer: '{row['correct_answer']}'.

QUESTION:
{row['question_stem']}

OPTIONS:
A) {row['option_a']}
B) {row['option_b']}
C) {row['option_c']}
D) {row['option_d']}

OUTPUT JSON ONLY:
{{
  "ai_answer": "A/B/C/D",
  "status": "MATCH | MISMATCH | UNSURE",
  "confidence": "high | medium | low"
}}
"""


def build_secondary_prompt(row):
    """
    Mid-tier model: deeper validation.
    Includes Bloom-level, distractor quality, ambiguity detection.
    """
    return f"""
You are an *expert Java examiner*.

Steps:
1. Solve the MCQ fully.
2. State the correct option.
3. Check if the question has multiple valid answers.
4. Evaluate Bloom level: Remember / Understand / Apply / Analyze / Evaluate.
5. Rate distractor quality (0–10).
6. Compare with PROVIDED KEY: '{row['correct_answer']}'.

QUESTION:
{row['question_stem']}

OPTIONS:
A) {row['option_a']}
B) {row['option_b']}
C) {row['option_c']}
D) {row['option_d']}

OUTPUT JSON ONLY:
{{
  "ai_answer": "A/B/C/D",
  "agreement": "MATCH | MISMATCH | MULTIPLE_CORRECT | UNCLEAR",
  "bloom_level": "Remember/Understand/Apply/Analyze/Evaluate",
  "distractor_quality": 0-10,
  "needs_human_review": true/false
}}
"""


def build_premium_prompt(row):
    """
    Premium expert model for final arbitration.
    """
    return f"""
You are a *master-level Java validator*.

Resolve all disputes. Provide:
- Correct answer
- Explanation
- Whether the provided answer key is correct

QUESTION:
{row['question_stem']}

OPTIONS:
A) {row['option_a']}
B) {row['option_b']}
C) {row['option_c']}
D) {row['option_d']}

OUTPUT JSON ONLY:
{{
  "final_answer": "A/B/C/D",
  "provided_key_correct": true/false,
  "explanation": "short reason"
}}
"""


# ===============================================================
# LLM CALL HELPERS
# ===============================================================

def call_model(model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return clean_json(response.choices[0].message.content)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ===============================================================
# VALIDATION PIPELINE
# ===============================================================

def validate_question(row):
    result = {
        "question_hash": hash_question(row),
        "primary": {},
        "secondary": {},
        "premium": {},
    }

    # --------------------------------
    # PASS 1: Primary Cheap Model
    # --------------------------------
    raw1 = call_model(PRIMARY_MODEL, build_primary_prompt(row))
    try:
        res1 = json.loads(raw1)
    except:
        res1 = {"status": "PARSE_ERROR", "raw": raw1}

    result["primary"] = res1

    # If High Confidence MATCH → accept
    if res1.get("status") == "MATCH" and res1.get("confidence") == "high":
        result["final_status"] = "MATCH_CONFIDENT"
        return result

    # --------------------------------
    # PASS 2: Secondary Mid-Cost Model
    # --------------------------------
    raw2 = call_model(SECONDARY_MODEL, build_secondary_prompt(row))
    try:
        res2 = json.loads(raw2)
    except:
        res2 = {"agreement": "PARSE_ERROR", "raw": raw2}

    result["secondary"] = res2

    # If primary + secondary both agree → accept
    if (
        res2.get("agreement") == "MATCH"
        and res1.get("ai_answer") == res2.get("ai_answer")
    ):
        result["final_status"] = "MATCH_REVIEWED"
        return result

    # Secondary says unclear? → escalate
    if res2.get("needs_human_review"):
        result["final_status"] = "NEEDS_HUMAN_REVIEW"
        return result

    # --------------------------------
    # PASS 3: Optional Premium Escalation
    # --------------------------------
    if PREMIUM_MODEL:
        raw3 = call_model(PREMIUM_MODEL, build_premium_prompt(row))
        try:
            res3 = json.loads(raw3)
        except:
            res3 = {"raw": raw3, "final_error": "parse_failed"}

        result["premium"] = res3

        result["final_status"] = (
            "PREMIUM_VERIFIED"
            if res3.get("provided_key_correct")
            else "PREMIUM_DISAGREEMENT"
        )
    else:
        result["final_status"] = "UNRESOLVED_DISCREPANCY"

    return result


# ===============================================================
# MAIN EXECUTION
# ===============================================================

def main():
    print(f"Loading {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    results = []

    for i, row in df.iterrows():
        print(f"Validating Q{i+1}/{len(df)}...", end="", flush=True)
        out = validate_question(row)

        results.append({
            "index": i + 1,
            "question_snippet": row["question_stem"][:60] + "...",
            "provided_key": row["correct_answer"],
            "primary_ai": out["primary"],
            "secondary_ai": out["secondary"],
            "premium_ai": out["premium"],
            "final_status": out["final_status"]
        })

        print(f" [{out['final_status']}]")
        time.sleep(0.4)     # avoid rate limits

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_FILE, index=False)

    print("\nDone!")
    print(f"Validation saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
