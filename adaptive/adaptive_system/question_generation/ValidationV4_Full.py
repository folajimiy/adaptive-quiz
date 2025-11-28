"""
ValidationV4_Full.py

Hybrid multi-LLM question validator with:
- Caching (only revalidate changed questions)
- Optional JVM execution hook for code-output questions
- Majority vote aggregation + mismatch reporting

Run with:
    python ValidationV4_Full.py

Make sure you:
- Set your API keys in environment variables or directly below.
- Have java_questions_adaptive.csv in the same folder (or update INPUT_FILE).
"""

import os
import time
import json
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

# ==============================
# CONFIGURATION
# ==============================

# --- Core files ---
INPUT_FILE = "java_questions_adaptive.csv"
OUTPUT_FILE = "validation_report_v4.csv"

# --- Cache ---
CACHE_DIR = Path(".cache")
CACHE_FILE = CACHE_DIR / "validation_cache_v4.json"

# --- Provider toggles (turn on/off as needed) ---
USE_OPENAI   = True
USE_ANTHROPIC = False   # set True if you configure Anthropic
USE_GEMINI   = False    # set True if you configure Gemini
USE_DEEPSEEK = False    # set True if you configure DeepSeek

# --- OpenAI config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
OPENAI_MODEL   = "gpt-4.1-mini"   # good balance of cost/quality

# --- Anthropic (Claude) config (optional) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_KEY_HERE")
ANTHROPIC_MODEL   = "claude-3-5-sonnet-20241022"

# --- Gemini config (optional) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")
GEMINI_MODEL   = "gemini-1.5-pro"

# --- DeepSeek config (optional) ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_KEY_HERE")
DEEPSEEK_MODEL   = "deepseek-chat"

# --- Rate limiting / speed ---
SLEEP_BETWEEN_QUESTIONS = 0.4   # seconds between questions
SLEEP_BETWEEN_CALLS     = 0.1   # seconds between provider API calls
MAX_QUESTIONS_PER_RUN   = None  # set to int to test subset, e.g. 100

# ==============================
# GLOBAL CLIENTS (lazy import)
# ==============================

openai_client = None
anthropic_client = None
gemini_client = None
deepseek_client = None

if USE_OPENAI:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"[WARN] Could not init OpenAI client: {e}")
        USE_OPENAI = False

if USE_ANTHROPIC:
    try:
        import anthropic
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception as e:
        print(f"[WARN] Could not init Anthropic client: {e}")
        USE_ANTHROPIC = False

if USE_GEMINI:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        print(f"[WARN] Could not init Gemini client: {e}")
        USE_GEMINI = False

if USE_DEEPSEEK:
    try:
        import openai as deepseek_openai
        deepseek_openai.api_key = DEEPSEEK_API_KEY
        deepseek_openai.base_url = "https://api.deepseek.com"
        deepseek_client = deepseek_openai
    except Exception as e:
        print(f"[WARN] Could not init DeepSeek client: {e}")
        USE_DEEPSEEK = False


# ==============================
# CACHE HELPERS
# ==============================


if CACHE_FILE.exists():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            VALIDATION_CACHE: Dict[str, Any] = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read existing cache ({e}), starting fresh.")
        VALIDATION_CACHE = {}
else:
    VALIDATION_CACHE = {}


def save_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(VALIDATION_CACHE, f, indent=2)


def load_cached_result(q_hash: str) -> Optional[Dict[str, Any]]:
    return VALIDATION_CACHE.get(q_hash)


def save_cached_result(q_hash: str, result: Dict[str, Any]):
    VALIDATION_CACHE[q_hash] = result
    save_cache()





def load_csv_safely(path):
    """
    Loads a CSV using multiple fallbacks:
      - UTF-8
      - latin-1
      - windows-1252
      - tolerant python engine
    """
    encodings = ["utf-8", "latin1", "cp1252"]

    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                engine="python",
                on_bad_lines="skip"
            )
        except Exception as e:
            print(f"[WARN] Failed reading with encoding {enc}: {e}")

    # Final fallback
    print("[WARN] Using final fallback: encoding errors ignored")
    return pd.read_csv(
        path,
        encoding="latin1",
        engine="python",
        on_bad_lines="skip",
        errors="ignore"
    )








# ==============================
# QUESTION HASHING
# ==============================

def hash_question(row: pd.Series) -> str:
    """
    Compute a stable hash based on the "semantics" of the question:
    topic, stem, options, correct answer.
    If any of these change, the hash changes, and we revalidate.
    """
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
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ==============================
# JAVA CODE EXTRACTION + JVM HOOK
# ==============================

def extract_java_code(row: pd.Series) -> Optional[str]:
    """
    Attempts to extract a Java code snippet from a question row.
    Returns:
        - The extracted code snippet string
        - OR None if the question has no runnable snippet.
    """
    stem = str(row.get("question_stem", ""))

    # 1. Fenced code block ```java ... ```
    fenced = re.search(r"```(?:java)?(.*?)```", stem, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    # 2. Block inside braces { ... }
    brace = re.search(r"\{([^{}]+;[^{}]*)\}", stem, flags=re.DOTALL)
    if brace:
        return brace.group(1).strip()

    # 3. Heuristic: contains Java-ish tokens
    if any(k in stem for k in ["class ", "int ", "String ", "System.out"]):
        return stem

    return None



def execute_java_snippet(snippet: str, timeout_sec: float = 2.0) -> Dict[str, Any]:
    """
    Optional JVM ground-truth executor.
    Right now, this is a SAFE STUB that does NOT actually run Java.

    If you want real execution:
    - Install JDK (javac/java)
    - Replace the STUB return with real subprocess logic
    """
    # ---- REAL IMPLEMENTATION (commented out for safety) ----
    # template = (
    #     "public class Wrapper {\n"
    #     "  public static void main(String[] args) throws Exception {\n"
    #     "    %s\n"
    #     "  }\n"
    #     "}\n"
    # )
    #
    # java_code = template % snippet
    #
    # with tempfile.TemporaryDirectory() as tmp:
    #     src = Path(tmp) / "Wrapper.java"
    #     src.write_text(java_code, encoding="utf-8")
    #
    #     # compile
    #     c = subprocess.run(
    #         ["javac", "Wrapper.java"],
    #         cwd=tmp,
    #         capture_output=True,
    #         text=True
    #     )
    #     if c.returncode != 0:
#             return {"success": False, "stdout": "", "stderr": c.stderr}
    #
    #     # run
    #     try:
    #         r = subprocess.run(
    #             ["java", "Wrapper"],
    #             cwd=tmp,
    #             capture_output=True,
    #             text=True,
    #             timeout=timeout_sec
    #         )
    #         return {
    #             "success": r.returncode == 0,
    #             "stdout": r.stdout,
    #             "stderr": r.stderr
    #         }
    #     except subprocess.TimeoutExpired:
    #         return {"success": False, "stdout": "", "stderr": "Timeout"}

    # ---- SAFE STUB (no JVM) ----
    return {
        "success": False,
        "stdout": "",
        "stderr": "JVM execution not implemented (stub)."
    }


# ==============================
# LLM HELPERS
# ==============================

def build_validation_prompt(row: pd.Series) -> str:
    """
    Creates a common prompt for all LLMs.
    """
    question = str(row.get("question_stem", ""))
    A = str(row.get("option_a", ""))
    B = str(row.get("option_b", ""))
    C = str(row.get("option_c", ""))
    D = str(row.get("option_d", ""))
    provided_key = str(row.get("correct_answer", "")).strip()

    prompt = f"""
You are an expert Java instructor and exam validator.

Your job:
1. Carefully analyze the Java multiple-choice question.
2. Determine which option (A, B, C, or D) is CORRECT.
3. Compare your answer to the PROVIDED ANSWER KEY.
4. Return a single JSON line only.

QUESTION STEM:
{question}

OPTIONS:
A) {A}
B) {B}
C) {C}
D) {D}

PROVIDED ANSWER KEY: "{provided_key}"

Return ONLY valid JSON in this exact schema (no extra text):

{{
  "ai_answer": "A" | "B" | "C" | "D",
  "status": "MATCH" | "MISMATCH" | "UNCERTAIN",
  "explanation": "short explanation of reasoning and any discrepancy"
}}
"""
    return prompt.strip()


def parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    Robust JSON extractor: strips code fences and tries to parse.
    """
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {
            "ai_answer": None,
            "status": "PARSE_ERROR",
            "explanation": raw[:500]
        }


def call_openai_model(row: pd.Series) -> Dict[str, Any]:
    if not USE_OPENAI or openai_client is None:
        return {"model": "openai", "answer": None, "status": "DISABLED", "reason": ""}

    prompt = build_validation_prompt(row)
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict Java MCQ validator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = resp.choices[0].message.content
        data = parse_llm_json(content)
        return {
            "model": f"openai:{OPENAI_MODEL}",
            "answer": (data.get("ai_answer") or "").strip(),
            "status": data.get("status", "UNKNOWN"),
            "reason": data.get("explanation", "")
        }
    except Exception as e:
        return {
            "model": f"openai:{OPENAI_MODEL}",
            "answer": None,
            "status": "ERROR",
            "reason": str(e)
        }


def call_claude_model(row: pd.Series) -> Dict[str, Any]:
    if not USE_ANTHROPIC or anthropic_client is None:
        return {"model": "claude", "answer": None, "status": "DISABLED", "reason": ""}

    prompt = build_validation_prompt(row)
    try:
        resp = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        # anthropic returns content pieces; get the first text
        text = ""
        for block in resp.content:
            if block.type == "text":
                text += block.text
        data = parse_llm_json(text)
        return {
            "model": f"anthropic:{ANTHROPIC_MODEL}",
            "answer": (data.get("ai_answer") or "").strip(),
            "status": data.get("status", "UNKNOWN"),
            "reason": data.get("explanation", "")
        }
    except Exception as e:
        return {
            "model": f"anthropic:{ANTHROPIC_MODEL}",
            "answer": None,
            "status": "ERROR",
            "reason": str(e)
        }


def call_gemini_model(row: pd.Series) -> Dict[str, Any]:
    if not USE_GEMINI or gemini_client is None:
        return {"model": "gemini", "answer": None, "status": "DISABLED", "reason": ""}

    prompt = build_validation_prompt(row)
    try:
        resp = gemini_client.generate_content(prompt)
        text = resp.text.strip()
        data = parse_llm_json(text)
        return {
            "model": f"gemini:{GEMINI_MODEL}",
            "answer": (data.get("ai_answer") or "").strip(),
            "status": data.get("status", "UNKNOWN"),
            "reason": data.get("explanation", "")
        }
    except Exception as e:
        return {
            "model": f"gemini:{GEMINI_MODEL}",
            "answer": None,
            "status": "ERROR",
            "reason": str(e)
        }


def call_deepseek_model(row: pd.Series) -> Dict[str, Any]:
    if not USE_DEEPSEEK or deepseek_client is None:
        return {"model": "deepseek", "answer": None, "status": "DISABLED", "reason": ""}

    prompt = build_validation_prompt(row)
    try:
        resp = deepseek_client.ChatCompletion.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict Java MCQ validator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        text = resp["choices"][0]["message"]["content"]
        data = parse_llm_json(text)
        return {
            "model": f"deepseek:{DEEPSEEK_MODEL}",
            "answer": (data.get("ai_answer") or "").strip(),
            "status": data.get("status", "UNKNOWN"),
            "reason": data.get("explanation", "")
        }
    except Exception as e:
        return {
            "model": f"deepseek:{DEEPSEEK_MODEL}",
            "answer": None,
            "status": "ERROR",
            "reason": str(e)
        }


# Which providers to actually call (in order)
PROVIDER_FUNCS = [
    call_openai_model,
    call_claude_model,
    call_gemini_model,
    call_deepseek_model,
]


# ==============================
# AGGREGATION
# ==============================

def aggregate_votes(
    row: pd.Series,
    votes: List[Dict[str, Any]],
    dataset_bloom: str,
    provided_key: str,
    jvm_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Combine:
    - JVM output (if available and successful)
    - LLM panel votes

    Returns a dict suitable for CSV export & caching.
    """
    # --- 1. JVM ground truth, if available & working ---
    jvm_status = None
    jvm_stdout = None
    jvm_stderr = None
    if jvm_result is not None:
        jvm_status = "OK" if jvm_result.get("success") else "FAIL"
        jvm_stdout = (jvm_result.get("stdout") or "").strip()
        jvm_stderr = (jvm_result.get("stderr") or "").strip()

    final_status = "UNKNOWN"
    majority_answer = None

    # If we had working JVM output, you could try matching it to options here.
    # For now, we just record it; we won't override the key purely on JVM stub.
    # You can extend this later.

    # --- 2. Majority vote among LLM answers ---
    answers = [v["answer"] for v in votes if v.get("answer")]
    if answers:
        # simple majority
        majority_answer = max(set(answers), key=answers.count)
        final_status = "MATCH" if majority_answer == provided_key else "MISMATCH"
    else:
        final_status = "NO_VALID_VOTES"

    # Count how many models actually responded with a letter
    n_models = len([a for a in answers])

    return {
        "final_status": final_status,
        "majority_answer": majority_answer,
        "provided_key": provided_key,
        "dataset_bloom": dataset_bloom,
        "n_models_with_answer": n_models,
        "votes": votes,
        "jvm_status": jvm_status,
        "jvm_stdout": jvm_stdout,
        "jvm_stderr": jvm_stderr,
    }


# ==============================
# CORE VALIDATION FUNCTION
# ==============================

def validate_question(row: pd.Series) -> Dict[str, Any]:
    provided_key = str(row.get("correct_answer", "")).strip()
    dataset_bloom = str(row.get("bloom_level", ""))
    q_hash = hash_question(row)

    # ---- 1. Cache check ----
    cached = load_cached_result(q_hash)
    if cached:
        cached["cached"] = True
        return cached

    # ---- 2. JVM execution (optional) ----
    java_snippet = extract_java_code(row)
    jvm_result = None
    if java_snippet:
        try:
            jvm_result = execute_java_snippet(java_snippet)
        except Exception as e:
            jvm_result = {"success": False, "stdout": "", "stderr": str(e)}

    # ---- 3. Hybrid LLM panel ----
    votes: List[Dict[str, Any]] = []
    for func in PROVIDER_FUNCS:
        # Skip disabled providers quickly
        name = func.__name__
        try:
            result = func(row)
            votes.append(result)
        except Exception as e:
            votes.append({
                "model": name,
                "answer": None,
                "status": "ERROR",
                "reason": str(e),
            })
        time.sleep(SLEEP_BETWEEN_CALLS)

    # ---- 4. Aggregate evidence ----
    final_output = aggregate_votes(
        row=row,
        votes=votes,
        dataset_bloom=dataset_bloom,
        provided_key=provided_key,
        jvm_result=jvm_result,
    )

    # ---- 5. Attach debug question snippet ----
    q_snippet = str(row.get("question_stem", ""))[:120].replace("\n", " ")
    final_output.update({
        "question_hash": q_hash,
        "cached": False,
        "topic": str(row.get("topic", "")),
        "question_snippet": q_snippet,
    })

    # ---- 6. Cache it ----
    save_cached_result(q_hash, final_output)

    return final_output


# ==============================
# MAIN
# ==============================

def main():

    print(f"📥 Loading questions from {INPUT_FILE} ...")
    df = load_csv_safely(INPUT_FILE)

    total = len(df)
    print(f"Found {total} questions.")

    enabled_providers = [
        name for name, flag in [
            ("OpenAI", USE_OPENAI),
            ("Claude", USE_ANTHROPIC),
            ("Gemini", USE_GEMINI),
            ("DeepSeek", USE_DEEPSEEK),
        ] if flag
    ]
    print(f"Providers enabled: {', '.join(enabled_providers) if enabled_providers else 'NONE'}")

    results = []
    start_time = time.time()

    max_q = MIN = 0
    max_q = total if MAX_QUESTIONS_PER_RUN is None else min(MAX_QUESTIONS_PER_RUN, total)

    for idx in range(max_q):
        row = df.iloc[idx]
        q_hash = hash_question(row)

        # If already validated and cached, skip API calls
        cached = load_cached_result(q_hash)
        if cached:
            # You can still collect them into results for the CSV, if desired
            results.append(cached)
            print(f"Q{idx+1}/{max_q}: [CACHED] {cached.get('final_status','?')}")
            continue

        print(f"Q{idx+1}/{max_q} ...", end="", flush=True)
        try:
            res = validate_question(row)
            results.append(res)
            print(f" {res.get('final_status','?')} (maj={res.get('majority_answer','?')})")
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user. Saving partial results...")
            break
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "final_status": "EXCEPTION",
                "majority_answer": None,
                "provided_key": str(row.get("correct_answer", "")),
                "dataset_bloom": str(row.get("bloom_level", "")),
                "n_models_with_answer": 0,
                "votes": [],
                "jvm_status": None,
                "jvm_stdout": None,
                "jvm_stderr": str(e),
                "question_hash": hash_question(row),
                "cached": False,
                "topic": str(row.get("topic", "")),
                "question_snippet": str(row.get("question_stem", ""))[:120],
            })

        time.sleep(SLEEP_BETWEEN_QUESTIONS)

    # Save CSV report (flatten essential fields)
    if results:
        flat_rows = []
        for r in results:
            flat_rows.append({
                "topic": r.get("topic", ""),
                "question_hash": r.get("question_hash", ""),
                "question_snippet": r.get("question_snippet", ""),
                "final_status": r.get("final_status", ""),
                "provided_key": r.get("provided_key", ""),
                "majority_answer": r.get("majority_answer", ""),
                "n_models_with_answer": r.get("n_models_with_answer", 0),
                "jvm_status": r.get("jvm_status", ""),
                "jvm_stdout": (r.get("jvm_stdout") or "")[:120],
                "cached": r.get("cached", False),
            })
        out_df = pd.DataFrame(flat_rows)
        out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"\n✅ Validation report saved to {OUTPUT_FILE}")
    else:
        print("\nNo results to save (no questions processed).")

    # Summary of mismatches
    if results:
        mismatches = [r for r in results if r.get("final_status") == "MISMATCH"]
        print(f"\n--- SUMMARY ---")
        print(f"Total questions processed: {len(results)}")
        print(f"Mismatches: {len(mismatches)}")
        if mismatches:
            print("Example mismatches:")
            for r in mismatches[:10]:
                print(
                    f"  • topic={r.get('topic','')}, "
                    f"key={r.get('provided_key')}, "
                    f"maj={r.get('majority_answer')}, "
                    f"status={r.get('final_status')}"
                )

    elapsed = time.time() - start_time
    print(f"\n⏱ Done in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
