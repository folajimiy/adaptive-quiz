"""
ValidationV6_JVMHybrid.py

Hybrid validator for Java MCQs:
- Uses a real JVM to execute code-output questions when possible
- Uses an LLM (OpenAI GPT model) to reason about correctness
- Caches results per question hash to avoid re-validating unchanged items
- Produces:
    - validation_v6_results.csv
    - validation_v6_suggested_fixes.csv
    - validation_cache_v6.json (for incremental runs)

Run:
    OPENAI_API_KEY=your_key_here python ValidationV6_JVMHybrid.py
"""

import os
import json
import time
import hashlib
import tempfile
import subprocess
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import pandas as pd
from openai import OpenAI

# ==========================================================
# CONFIG
# ==========================================================

INPUT_FILE = "java_questions_adaptive.csv"
RESULTS_FILE = "validation_v6_results.csv"
FIXES_FILE = "validation_v6_suggested_fixes.csv"
CACHE_FILE = "validation_cache_v6.json"

# How often to save cache during long runs
SAVE_EVERY = 25

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
PRIMARY_MODEL = "gpt-4.1"   # good balance of cost / reasoning
TEMPERATURE = 0.0                # deterministic validation

# Basic heuristic: treat questions containing code-like patterns as code-output
CODE_KEYWORDS = [
    "System.out", "public static void main", "class ", "int ", "double ", "String ",
    "```java", "```"
]

# ==========================================================
# INITIALIZE OPENAI CLIENT
# ==========================================================

client = None
if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_KEY_HERE":
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("[WARN] OPENAI_API_KEY not set or placeholder. LLM validation will be skipped.")


# ==========================================================
# SAFE CSV LOADING
# ==========================================================

def load_csv_safely(path: str) -> pd.DataFrame:
    """
    Loads a CSV using multiple encodings and the python engine with on_bad_lines='skip'.
    This avoids most Unicode / malformed line issues.
    """
    encodings = ["utf-8", "latin1", "cp1252"]
    for enc in encodings:
        try:
            print(f"Trying encoding {enc} ...")
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as e:
            print(f"[WARN] Failed with {enc}: {e}")

    print("[WARN] Final fallback using latin1 + errors='ignore'")
    return pd.read_csv(path, encoding="latin1", engine="python", on_bad_lines="skip", errors="ignore")


# ==========================================================
# QUESTION HASHING (change detection)
# ==========================================================

def compute_question_hash(row: pd.Series) -> str:
    """
    Hashes all key fields that affect semantic correctness.
    If any of these change, the hash changes and we revalidate.
    """
    fields = [
        str(row.get("topic", "")),
        str(row.get("bloom_level", "")),
        str(row.get("question_stem", "")),
        str(row.get("option_a", "")),
        str(row.get("option_b", "")),
        str(row.get("option_c", "")),
        str(row.get("option_d", "")),
        str(row.get("correct_answer", "")),
    ]
    joined = "||".join(fields).encode("utf-8", errors="ignore")
    return hashlib.sha256(joined).hexdigest()


# ==========================================================
# CACHE UTILITIES
# ==========================================================

def load_cache() -> Dict[str, Any]:
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache: Dict[str, Any]) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


# ==========================================================
# CODE QUESTION DETECTION & EXTRACTION
# ==========================================================

def is_probably_code_question(row: pd.Series) -> bool:
    stem = str(row.get("question_stem", "") or "").lower()
    for kw in CODE_KEYWORDS:
        if kw.lower() in stem:
            return True
    return False


def extract_java_code(row: pd.Series) -> Optional[str]:
    """
    Tries to extract a Java snippet from question_stem.
    Supports fenced ```java blocks; falls back to entire stem if code-like.
    """
    text = str(row.get("question_stem", "") or "")

    # Prefer fenced ```java blocks
    if "```java" in text:
        try:
            after = text.split("```java", 1)[1]
            snippet = after.split("```", 1)[0]
            return snippet.strip()
        except Exception:
            pass
    elif "```" in text:
        # Any fenced code
        try:
            after = text.split("```", 1)[1]
            snippet = after.split("```", 1)[0]
            return snippet.strip()
        except Exception:
            pass

    # Fallback: if it looks like code, use the whole thing
    if is_probably_code_question(row):
        return text.strip()

    return None


# ==========================================================
# JVM EXECUTION
# ==========================================================

def run_java_snippet(snippet: str, timeout: int = 3) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempts to compile and run a Java snippet.
    Returns (stdout, error_message_or_None).
    If javac/java not found or compilation fails, returns (None, error_message).
    """
    tmpdir = tempfile.mkdtemp(prefix="java_check_")
    classname = "Main"

    try:
        # Heuristic: if snippet contains 'class ' and 'main(' assume it's a full program
        if "class " in snippet and "main(" in snippet:
            code = snippet
        else:
            # Wrap snippet into a simple main
            code = f"""
public class {classname} {{
    public static void main(String[] args) throws Exception {{
{snippet}
    }}
}}
""".strip()

        java_path = os.path.join(tmpdir, f"{classname}.java")
        with open(java_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Compile
        try:
            proc = subprocess.run(
                ["javac", f"{classname}.java"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return None, "javac_not_found"

        if proc.returncode != 0:
            return None, f"compile_error: {proc.stderr.strip()}"

        # Run
        proc = subprocess.run(
            ["java", classname],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if proc.returncode != 0:
            return None, f"runtime_error: {proc.stderr.strip()}"

        stdout = proc.stdout
        return stdout, None

    except Exception as e:
        return None, f"exception: {e}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def normalize_output(s: str) -> str:
    """
    Normalizes output for comparison with answer choices:
    - strip
    - collapse whitespace
    - remove trailing newlines
    """
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    # collapse multiple spaces
    parts = s.split()
    return " ".join(parts)


def map_output_to_option(stdout: str, row: pd.Series) -> Optional[str]:
    """
    Given JVM stdout, see which option matches.
    If exactly one matches, return its letter (A/B/C/D).
    """
    out_norm = normalize_output(stdout)
    if not out_norm:
        return None

    candidates = []
    for letter, col in [("A", "option_a"), ("B", "option_b"),
                        ("C", "option_c"), ("D", "option_d")]:
        text = str(row.get(col, "") or "")
        if normalize_output(text) == out_norm:
            candidates.append(letter)

    if len(candidates) == 1:
        return candidates[0]
    return None


# ==========================================================
# LLM UTILITIES
# ==========================================================

def parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    Parses a JSON response from the LLM, stripping ```json fences if present.
    """
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {
            "status": "PARSE_ERROR",
            "ai_calculated_answer": None,
            "explanation": raw[:400],
        }


def call_openai_validator(row: pd.Series) -> Dict[str, Any]:
    """
    Calls the OpenAI model to validate the question.
    Returns dict with keys: status, ai_calculated_answer, explanation.
    """
    if client is None:
        return {
            "status": "LLM_SKIPPED",
            "ai_calculated_answer": None,
            "explanation": "No OPENAI_API_KEY set; skipped LLM validation.",
        }

    question = str(row.get("question_stem", "") or "")
    options = (
        f"A) {row.get('option_a', '')}\n"
        f"B) {row.get('option_b', '')}\n"
        f"C) {row.get('option_c', '')}\n"
        f"D) {row.get('option_d', '')}\n"
    )
    provided = str(row.get("correct_answer", "") or "").strip().upper()

    prompt = f"""
You are an expert Java technical reviewer.

Task:
1. Carefully reason about the following multiple-choice question.
2. Determine which option (A, B, C, or D) is correct.
3. Compare your answer with the provided key: "{provided}".
4. Return a SINGLE LINE of JSON only.

QUESTION:
{question}

OPTIONS:
{options}

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "MATCH" or "MISMATCH",
  "ai_calculated_answer": "A"|"B"|"C"|"D",
  "explanation": "brief explanation of reasoning"
}}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=PRIMARY_MODEL,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": "You are a strict Java correctness validator."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        data = parse_llm_json(raw)

        # Normalize fields
        ai_ans = data.get("ai_calculated_answer")
        if isinstance(ai_ans, str):
            ai_ans = ai_ans.strip().upper()
        else:
            ai_ans = None

        status = data.get("status", "UNKNOWN")
        explanation = data.get("explanation", "")

        return {
            "status": status,
            "ai_calculated_answer": ai_ans,
            "explanation": explanation,
            "raw_response": raw,
        }

    except Exception as e:
        return {
            "status": "LLM_ERROR",
            "ai_calculated_answer": None,
            "explanation": f"LLM error: {e}",
        }


# ==========================================================
# HYBRID DECISION LOGIC (JVM + LLM)
# ==========================================================

def validate_question_hybrid(row: pd.Series, cache: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a single question with:
    - JVM execution (when applicable)
    - LLM reasoning
    - Hybrid decision logic
    Uses cache keyed by question_id + hash.
    """
    qid = str(row.get("id", row.name))
    qhash = compute_question_hash(row)
    provided_key = str(row.get("correct_answer", "") or "").strip().upper()

    # ---------------------
    # CACHE CHECK
    # ---------------------
    if qid in cache and cache[qid].get("hash") == qhash:
        return cache[qid]

    # ---------------------
    # 1) JVM PATH
    # ---------------------
    java_answer = None
    java_stdout = None
    java_error = None
    java_status = "NOT_RUN"

    code_like = is_probably_code_question(row)
    if code_like:
        snippet = extract_java_code(row)
        if snippet:
            java_stdout, java_error = run_java_snippet(snippet)
            if java_stdout is not None:
                java_answer = map_output_to_option(java_stdout, row)
                if java_answer is not None and provided_key:
                    java_status = "MATCH" if java_answer == provided_key else "MISMATCH"
                else:
                    java_status = "NO_UNIQUE_MAPPING"
            else:
                java_status = f"JVM_ERROR: {java_error}"

    # ---------------------
    # 2) LLM PATH
    # ---------------------
    llm_res = call_openai_validator(row)
    llm_answer = llm_res.get("ai_calculated_answer")
    llm_status = llm_res.get("status", "UNKNOWN")

    # Normalize LLM status vs provided key if possible
    if llm_answer is not None and provided_key and llm_status not in ["LLM_ERROR", "LLM_SKIPPED", "PARSE_ERROR"]:
        llm_status = "MATCH" if llm_answer == provided_key else "MISMATCH"

    # ---------------------
    # 3) HYBRID DECISION
    # ---------------------
    final_status = None
    final_source = None
    final_answer = None

    if java_answer is not None and llm_answer is not None:
        if java_answer == llm_answer:
            # Strong signal: both agree
            final_answer = java_answer
            final_source = "JVM+LLM_AGREE"
            final_status = "MATCH" if final_answer == provided_key else "MISMATCH_JVM_LLM"
        else:
            # JVM and LLM disagree
            final_answer = java_answer  # or choose to be conservative; JVM has priority for code
            final_source = "CONFLICT_JVM_vs_LLM"
            final_status = "CONFLICT"
    elif java_answer is not None:
        # We only trust JVM for code-output
        final_answer = java_answer
        final_source = "JVM_ONLY"
        final_status = "MATCH" if final_answer == provided_key else "MISMATCH_JVM"
    elif llm_answer is not None:
        final_answer = llm_answer
        final_source = "LLM_ONLY"
        final_status = "MATCH" if final_answer == provided_key else "MISMATCH_LLM"
    else:
        final_answer = None
        final_source = "NONE"
        final_status = "UNDETERMINED"

    # ---------------------
    # BUILD RESULT ENTRY
    # ---------------------
    entry = {
        "hash": qhash,
        "question_id": qid,
        "topic": str(row.get("topic", "")),
        "bloom_level": str(row.get("bloom_level", "")),
        "provided_key": provided_key,
        "final_status": final_status,
        "final_source": final_source,
        "final_answer": final_answer,
        # JVM details
        "code_like": bool(code_like),
        "java_status": java_status,
        "java_answer": java_answer,
        "java_stdout": java_stdout,
        "java_error": java_error,
        # LLM details
        "llm_status": llm_status,
        "llm_answer": llm_answer,
        "llm_explanation": llm_res.get("explanation", ""),
        # metadata
        "validated_at": datetime.now().isoformat(),
    }

    cache[qid] = entry
    return entry


# ==========================================================
# SUGGESTED FIXES CSV
# ==========================================================

def write_suggested_fixes(df: pd.DataFrame, cache: Dict[str, Any]) -> None:
    """
    Writes a CSV of only questions where final_status indicates a problem.
    """
    rows = []
    for idx, row in df.iterrows():
        qid = str(row.get("id", idx))
        entry = cache.get(qid)
        if not entry:
            continue

        if entry["final_status"] in ["MISMATCH_JVM", "MISMATCH_LLM", "MISMATCH_JVM_LLM", "CONFLICT"]:
            rows.append({
                "question_id": qid,
                "topic": row.get("topic", ""),
                "bloom_level": row.get("bloom_level", ""),
                "question_stem": row.get("question_stem", ""),
                "option_a": row.get("option_a", ""),
                "option_b": row.get("option_b", ""),
                "option_c": row.get("option_c", ""),
                "option_d": row.get("option_d", ""),
                "provided_key": row.get("correct_answer", ""),
                "final_answer": entry.get("final_answer", ""),
                "final_status": entry.get("final_status", ""),
                "final_source": entry.get("final_source", ""),
                "java_answer": entry.get("java_answer", ""),
                "java_status": entry.get("java_status", ""),
                "java_stdout": entry.get("java_stdout", ""),
                "java_error": entry.get("java_error", ""),
                "llm_answer": entry.get("llm_answer", ""),
                "llm_status": entry.get("llm_status", ""),
                "llm_explanation": entry.get("llm_explanation", ""),
            })

    if not rows:
        print("✅ No mismatches/conflicts to record in suggested fixes.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(FIXES_FILE, index=False, encoding="utf-8")
    print(f"💡 Suggested fixes written to {FIXES_FILE}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    print(f"📥 Loading questions from {INPUT_FILE} ...")
    df = load_csv_safely(INPUT_FILE)
    print(f"Found {len(df)} questions.")

    cache = load_cache()
    print(f"Loaded {len(cache)} cached entries from {CACHE_FILE}.")

    results_rows = []

    for idx, row in df.iterrows():
        qid = str(row.get("id", idx))
        print(f"\nQ{idx+1}/{len(df)} (id={qid}) ...", end=" ")

        entry = validate_question_hybrid(row, cache)
        print(f"[{entry['final_status']} via {entry['final_source']}]")

        results_rows.append(entry)

        if (idx + 1) % SAVE_EVERY == 0:
            save_cache(cache)
            print("💾 Partial cache saved.")

        time.sleep(0.2)  # be kind to the API

    # Final cache + results write
    save_cache(cache)
    print(f"💾 Final cache saved to {CACHE_FILE}")

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(RESULTS_FILE, index=False, encoding="utf-8")
    print(f"📄 Full validation results saved to {RESULTS_FILE}")

    write_suggested_fixes(df, cache)
    print("✅ ValidationV6_JVMHybrid complete.")


if __name__ == "__main__":
    main()
