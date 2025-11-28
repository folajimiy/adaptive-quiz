# ---------------------------------------------------------------
# EnhancedValidationV2_Hybrid.py
# Hybrid OpenAI + Claude Java MCQ Validator
# ---------------------------------------------------------------

import os
import time
import json
import hashlib
import pandas as pd

from openai import OpenAI
from anthropic import Anthropic

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------

# You can also read from environment variables:
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY")

# OpenAI models
PRIMARY_MODEL = "gpt-4.1-mini"     # fast, cheaper
PREMIUM_MODEL = None               # e.g., "gpt-4.1" or "o1-mini" if you want a 3rd pass

# Claude model
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # adjust to your available Claude model

INPUT_FILE = "java_questions_adaptive.csv"
OUTPUT_FILE = "validation_report_hybrid.csv"

# Instantiate clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)


# ===============================================================
# UTILITIES
# ===============================================================

def hash_question(row) -> str:
    """
    Hash question content so unchanged items don't need revalidation.
    You can later cache based on this hash if you want.
    """
    parts = [
        str(row.get("question_stem", "")),
        str(row.get("option_a", "")),
        str(row.get("option_b", "")),
        str(row.get("option_c", "")),
        str(row.get("option_d", "")),
        str(row.get("correct_answer", "")),
    ]
    s = "||".join(parts)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def clean_json_block(s: str) -> str:
    """
    Remove markdown fences and whitespace so we can json.loads safely.
    """
    return (
        s.replace("```json", "")
         .replace("```", "")
         .strip()
    )


def safe_json_load(s: str, fallback: dict) -> dict:
    try:
        return json.loads(clean_json_block(s))
    except Exception:
        fallback["raw"] = s
        return fallback


# ===============================================================
# PROMPTS
# ===============================================================

def build_primary_prompt(row: pd.Series) -> str:
    """
    OpenAI primary pass: correctness, simple status, confidence.
    """
    question = row.get("question_stem", "")
    options = (
        f"A) {row.get('option_a', '')}\n"
        f"B) {row.get('option_b', '')}\n"
        f"C) {row.get('option_c', '')}\n"
        f"D) {row.get('option_d', '')}"
    )
    provided = row.get("correct_answer", "")

    return f"""
You are a precise Java MCQ solver.

TASK:
1. Solve the Java question exactly (simulate execution if needed).
2. Determine which single option (A, B, C, D) is correct.
3. Compare your solution with the PROVIDED KEY: '{provided}'.
4. Estimate your confidence.

QUESTION:
{question}

OPTIONS:
{options}

Return ONLY **one line of JSON**, no extra text:

{{
  "ai_answer": "A" | "B" | "C" | "D",
  "status": "MATCH" | "MISMATCH" | "UNSURE",
  "confidence": "high" | "medium" | "low"
}}
""".strip()


def build_claude_prompt(row: pd.Series) -> str:
    """
    Claude deeper pass: correctness, ambiguity, Bloom, distractor quality, human review flag.
    """
    question = row.get("question_stem", "")
    options = (
        f"A) {row.get('option_a', '')}\n"
        f"B) {row.get('option_b', '')}\n"
        f"C) {row.get('option_c', '')}\n"
        f"D) {row.get('option_d', '')}"
    )
    provided = row.get("correct_answer", "")
    bloom = row.get("bloom_level", "Unknown")

    return f"""
You are an expert Java exam validator.

Do the following:

1. Solve the MCQ correctly and state which option (A/B/C/D) is truly correct.
2. Compare your result with PROVIDED KEY: '{provided}'.
3. Check for ambiguity:
   - Is there more than one reasonably correct option?
   - Is the stem unclear or underspecified?
4. Evaluate Bloom level:
   - Based on the cognitive demand, choose one:
     Remember, Understand, Apply, Analyze, Evaluate
5. Evaluate distractors:
   - How plausible are the wrong options?
   - Rate as integer from 0 to 10 (10 = excellent distractors).
6. Decide if this item needs human review:
   - True if ambiguous, poorly worded, multiple-correct, or key seems wrong.

QUESTION:
{question}

OPTIONS:
{options}

Claimed Bloom Level in dataset: {bloom}

Return ONLY a single JSON object (no commentary):

{{
  "ai_answer": "A" | "B" | "C" | "D",
  "agreement": "MATCH" | "MISMATCH" | "MULTIPLE_CORRECT" | "UNCLEAR",
  "bloom_level": "Remember" | "Understand" | "Apply" | "Analyze" | "Evaluate",
  "distractor_quality": 0,
  "needs_human_review": true | false
}}
""".strip()


def build_premium_prompt(row: pd.Series) -> str:
    """
    Optional third-pass final arbiter (you can hook a stronger model here).
    """
    question = row.get("question_stem", "")
    options = (
        f"A) {row.get('option_a', '')}\n"
        f"B) {row.get('option_b', '')}\n"
        f"C) {row.get('option_c', '')}\n"
        f"D) {row.get('option_d', '')}"
    )
    provided = row.get("correct_answer", "")

    return f"""
You are the final arbiter for this Java multiple-choice question.

1. Determine the single best correct option (A/B/C/D).
2. Decide if the provided answer key '{provided}' is correct or not.
3. Give a short explanation.

QUESTION:
{question}

OPTIONS:
{options}

Return ONLY JSON:

{{
  "final_answer": "A" | "B" | "C" | "D",
  "provided_key_correct": true | false,
  "explanation": "short explanation here"
}}
""".strip()


# ===============================================================
# LLM WRAPPERS
# ===============================================================

def call_openai(model: str, prompt: str) -> str:
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return json.dumps({"error": f"OpenAI error: {str(e)}"})


def call_claude(model: str, prompt: str) -> str:
    try:
        resp = claude_client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns a list of content blocks, we concat text ones
        texts = []
        for block in resp.content:
            if block.type == "text":
                texts.append(block.text)
        return "\n".join(texts).strip()
    except Exception as e:
        return json.dumps({"error": f"Claude error: {str(e)}"})


# ===============================================================
# CORE VALIDATION PIPELINE
# ===============================================================

def validate_question(row: pd.Series) -> dict:
    """
    Run the hybrid pipeline on a single question row.

    Steps:
    1. Primary: OpenAI gpt-4.1-mini → fast correctness + confidence
    2. If not high-confidence MATCH → Claude Sonnet deeper analysis
    3. Optional: premium 3rd model if configured
    """
    q_hash = hash_question(row)

    result = {
        "question_id": str(row.get("question_id", "")),
        "question_hash": q_hash,
        "primary": {},
        "secondary": {},
        "premium": {},
        "final_status": None,
    }

    # ---------------------------
    # Primary (OpenAI)
    # ---------------------------
    p_prompt = build_primary_prompt(row)
    raw_p = call_openai(PRIMARY_MODEL, p_prompt)
    primary = safe_json_load(raw_p, {
        "status": "PARSE_ERROR",
        "ai_answer": None,
        "confidence": "low",
    })
    result["primary"] = primary

    # Quick exit: high-confidence MATCH
    if primary.get("status") == "MATCH" and primary.get("confidence") == "high":
        result["final_status"] = "MATCH_CONFIDENT_PRIMARY"
        return result

    # ---------------------------
    # Secondary (Claude)
    # ---------------------------
    c_prompt = build_claude_prompt(row)
    raw_c = call_claude(CLAUDE_MODEL, c_prompt)
    secondary = safe_json_load(raw_c, {
        "agreement": "PARSE_ERROR",
        "ai_answer": None,
        "bloom_level": None,
        "distractor_quality": None,
        "needs_human_review": True,
    })
    result["secondary"] = secondary

    # Both primary and secondary agree on answer and MATCH?
    if (
        secondary.get("agreement") == "MATCH"
        and primary.get("ai_answer") is not None
        and primary.get("ai_answer") == secondary.get("ai_answer")
        and not secondary.get("needs_human_review", False)
    ):
        result["final_status"] = "MATCH_AGREED"
        return result

    # Claude thinks it's ambiguous, multiple-correct, or unclear
    if secondary.get("agreement") in ["MULTIPLE_CORRECT", "UNCLEAR"]:
        result["final_status"] = "AMBIGUOUS_OR_MULTI_CORRECT"
        return result

    # Claude explicitly says needs human review
    if secondary.get("needs_human_review", False):
        result["final_status"] = "NEEDS_HUMAN_REVIEW"
        return result

    # ---------------------------
    # Optional Premium Pass
    # ---------------------------
    if PREMIUM_MODEL:
        prmpt = build_premium_prompt(row)
        raw_prem = call_openai(PREMIUM_MODEL, prmpt)
        premium = safe_json_load(raw_prem, {
            "final_answer": None,
            "provided_key_correct": None,
            "explanation": "",
        })
        result["premium"] = premium

        if premium.get("provided_key_correct") is True:
            result["final_status"] = "PREMIUM_CONFIRMED_KEY"
        elif premium.get("provided_key_correct") is False:
            result["final_status"] = "PREMIUM_FLAGGED_KEY"
        else:
            result["final_status"] = "PREMIUM_UNCLEAR"
    else:
        # No premium—just mark as unresolved discrepancy
        result["final_status"] = "UNRESOLVED_DISCREPANCY"

    return result


# ===============================================================
# MAIN BATCH DRIVER
# ===============================================================

def main():
    print(f"📥 Loading questions from {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)

    results = []

    total = len(df)
    print(f"Found {total} questions. Starting hybrid validation...\n")

    for idx, row in df.iterrows():
        print(f"Q{idx+1}/{total} ...", end="", flush=True)
        res = validate_question(row)

        # Flatten some useful fields for CSV
        primary = res.get("primary", {})
        secondary = res.get("secondary", {})
        premium = res.get("premium", {})

        results.append({
            "row_index": idx + 1,
            "question_id": res.get("question_id", ""),
            "topic": row.get("topic", ""),
            "bloom_level_dataset": row.get("bloom_level", ""),
            "question_snippet": str(row.get("question_stem", ""))[:80] + "...",
            "provided_key": row.get("correct_answer", ""),
            "primary_ai_answer": primary.get("ai_answer"),
            "primary_status": primary.get("status"),
            "primary_confidence": primary.get("confidence"),
            "secondary_ai_answer": secondary.get("ai_answer"),
            "secondary_agreement": secondary.get("agreement"),
            "secondary_bloom_level": secondary.get("bloom_level"),
            "secondary_distractor_quality": secondary.get("distractor_quality"),
            "secondary_needs_human_review": secondary.get("needs_human_review"),
            "premium_final_answer": premium.get("final_answer"),
            "premium_provided_key_correct": premium.get("provided_key_correct"),
            "final_status": res.get("final_status"),
        })

        print(f" {res.get('final_status')}")

        # gentle sleep to reduce rate-limit risk
        time.sleep(0.4)

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n✅ Done! Hybrid validation report saved to: {OUTPUT_FILE}")

    # quick summary
    summary = out_df["final_status"].value_counts()
    print("\n📊 Final Status Summary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
