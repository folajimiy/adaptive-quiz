# ================================================================
# ValidationV5_Cache.py
# Hybrid Multi-LLM Validator + Change Detection + Performance Cache
# ================================================================

import os
import json
import time
import hashlib
import pandas as pd
from datetime import datetime

# -------------------------------
# CONFIGURATION
# -------------------------------
INPUT_FILE = "java_questions_adaptive.csv"
OUTPUT_FILE = "validation_output.csv"
CACHE_FILE = "validation_cache.json"
FIX_FILE = "validation_suggested_fixes.csv"

ENABLE_OPENAI = True
ENABLE_CLAUDE = True
ENABLE_GEMINI = False
ENABLE_DEEPSEEK = False

SAVE_EVERY = 25   # Save partial progress every N questions


# ==========================================================
# SAFE CSV LOADING
# ==========================================================
def load_csv_safely(path):
    encodings = ["utf-8", "latin1", "cp1252"]
    for enc in encodings:
        try:
            print(f"Trying encoding {enc} ...")
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as e:
            print(f"[WARN] Failed with {enc}: {e}")

    print("[WARN] Final fallback using latin1 + ignore errors")
    return pd.read_csv(path, encoding="latin1", engine="python", on_bad_lines="skip", errors="ignore")


# ==========================================================
# HASH FUNCTION (Detect question changes)
# ==========================================================
def compute_question_hash(row):
    """
    Hashes all important fields to detect changes.
    If ANY content is edited, the hash changes => revalidate.
    """
    fields = [
        str(row.get("question_stem", "")),
        str(row.get("option_a", "")),
        str(row.get("option_b", "")),
        str(row.get("option_c", "")),
        str(row.get("option_d", "")),
        str(row.get("correct_answer", "")),
        str(row.get("topic", "")),
        str(row.get("bloom_level", "")),
    ]
    joined = "||".join(fields).encode("utf-8", errors="ignore")
    return hashlib.sha256(joined).hexdigest()


# ==========================================================
# LOAD/WRITE CACHE
# ==========================================================
def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


# ==========================================================
# LLM CALLS (PLACEHOLDERS — you plug in your providers)
# ==========================================================
def call_openai_validator(prompt):
    return {
        "provider": "openai",
        "ai_answer": "A",
        "status": "MATCH",
        "explanation": "placeholder"
    }


def call_claude_validator(prompt):
    return {
        "provider": "claude",
        "ai_answer": "A",
        "status": "MATCH",
        "explanation": "placeholder"
    }


def call_gemini_validator(prompt):
    return {
        "provider": "gemini",
        "ai_answer": "A",
        "status": "MATCH",
        "explanation": "placeholder"
    }


def call_deepseek_validator(prompt):
    return {
        "provider": "deepseek",
        "ai_answer": "A",
        "status": "MATCH",
        "explanation": "placeholder"
    }


# ==========================================================
# BUILD PROMPT
# ==========================================================
def build_prompt(row):
    question = row["question_stem"]
    options = (
        f"A) {row['option_a']}\n"
        f"B) {row['option_b']}\n"
        f"C) {row['option_c']}\n"
        f"D) {row['option_d']}"
    )
    key = row["correct_answer"]

    return f"""
You are a Java correctness validator.
Solve step-by-step, determine the correct option, and compare to the provided key.

QUESTION:
{question}

OPTIONS:
{options}

Provided Correct Answer: {key}

Return JSON only:
{{
 "status": "MATCH or MISMATCH",
 "ai_calculated_answer": "A/B/C/D",
 "explanation": "short reason"
}}
"""


# ==========================================================
# VALIDATION LOGIC
# ==========================================================
def run_validators(prompt):
    results = []

    if ENABLE_OPENAI:
        results.append(call_openai_validator(prompt))
    if ENABLE_CLAUDE:
        results.append(call_claude_validator(prompt))
    if ENABLE_GEMINI:
        results.append(call_gemini_validator(prompt))
    if ENABLE_DEEPSEEK:
        results.append(call_deepseek_validator(prompt))

    return results


def pick_final_result(results, provided_key):
    """
    Majority voting among providers.
    """
    if not results:
        return {
            "status": "ERROR",
            "ai_answer": None,
            "explanation": "No validators enabled"
        }

    # Count votes
    votes = {}
    for r in results:
        ans = r["ai_answer"]
        if ans not in votes:
            votes[ans] = 0
        votes[ans] += 1

    # Majority answer
    ai_answer = max(votes, key=votes.get)
    status = "MATCH" if ai_answer == provided_key else "MISMATCH"

    # Pick explanation from any model that gave that answer
    explanation = ""
    for r in results:
        if r["ai_answer"] == ai_answer:
            explanation = r["explanation"]
            break

    return {
        "status": status,
        "ai_answer": ai_answer,
        "explanation": explanation,
        "votes": votes,
        "raw": results,
    }


# ==========================================================
# MAIN VALIDATION LOOP
# ==========================================================
def validate_question(row, cache):
    qid = str(row.get("id", row.name))
    qhash = compute_question_hash(row)

    # --------------------------
    # Check cache hit
    # --------------------------
    if qid in cache and cache[qid]["hash"] == qhash:
        return cache[qid]  # unchanged → no need to recompute

    # --------------------------
    # LLM Validation
    # --------------------------
    prompt = build_prompt(row)
    results = run_validators(prompt)
    final_res = pick_final_result(results, row["correct_answer"])

    # Store in cache
    cache[qid] = {
        "hash": qhash,
        "status": final_res["status"],
        "ai_answer": final_res["ai_answer"],
        "explanation": final_res["explanation"],
        "votes": final_res.get("votes", {}),
        "timestamp": datetime.now().isoformat(),
    }
    return cache[qid]


# ==========================================================
# POST-PROCESSING: Suggested Fixes
# ==========================================================
def write_suggested_fix_csv(df, cache):
    rows = []
    for idx, row in df.iterrows():
        qid = str(row.get("id", idx))
        res = cache.get(qid)

        if res and res["status"] == "MISMATCH":
            rows.append({
                "question_id": qid,
                "question_stem": row["question_stem"],
                "provided_answer": row["correct_answer"],
                "suggested_correct_answer": res["ai_answer"],
                "ai_explanation": res["explanation"]
            })

    pd.DataFrame(rows).to_csv(FIX_FILE, index=False)
    print(f"💡 Suggested fixes saved to {FIX_FILE}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print(f"📥 Loading questions from {INPUT_FILE} ...")
    df = load_csv_safely(INPUT_FILE)

    print(f"Found {len(df)} questions.")
    print(f"Providers enabled: OpenAI={ENABLE_OPENAI}, Claude={ENABLE_CLAUDE}, Gemini={ENABLE_GEMINI}, DeepSeek={ENABLE_DEEPSEEK}")

    cache = load_cache()
    print(f"Loaded cache entries: {len(cache)}")

    results = []

    for idx, row in df.iterrows():
        qid = str(row.get("id", idx))

        print(f"\nValidating Q{idx+1}/{len(df)} (id={qid}) ...", end=" ")

        # Validate with caching
        entry = validate_question(row, cache)
        print(f"[{entry['status']}]")

        results.append({
            "question_id": qid,
            "status": entry["status"],
            "ai_answer": entry["ai_answer"],
            "explanation": entry["explanation"],
        })

        # Save partial cache
        if (idx + 1) % SAVE_EVERY == 0:
            save_cache(cache)
            print("💾 Partial cache saved.")

        time.sleep(0.25)

    # Final save
    save_cache(cache)
    print("💾 Final cache saved.")

    # Save output
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"📄 Validation results saved to {OUTPUT_FILE}")

    # Suggested fixes
    write_suggested_fix_csv(df, cache)


if __name__ == "__main__":
    main()
