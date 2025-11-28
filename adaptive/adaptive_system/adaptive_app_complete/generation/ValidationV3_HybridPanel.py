"""
ValidationV3_HybridPanel.py

Research-grade hybrid validator for Java MCQs.

- Multi-model "AI committee":
    - DeepSeek (optional, OpenAI-compatible)
    - OpenAI (gpt-4.1-mini or similar)
    - Claude Sonnet (Anthropic)
    - Gemini (Google Generative AI)

- Each model produces a JSON verdict.
- We normalize all answers into a common schema.
- Then we run:
    - Majority / consensus voting on correctness.
    - Cross-check of provided key vs model answers.
    - Bloom-level consistency check (dataset vs models).
    - Distractor quality scoring.
    - Ambiguity / multiple-correct signal.
    - Final "needs_human_review" flag.

Output: CSV with one row per question summarizing everything.
"""

import os
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

# -----------------------------
# Optional provider imports
# -----------------------------
try:
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "java_questions_adaptive.csv"
OUTPUT_FILE = "validation_report_v3_hybrid.csv"

# --- API Keys (prefer environment variables) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # optional

# --- Model names ---
OPENAI_MODEL_PRIMARY = "gpt-4.1-mini"      # cheap, good
OPENAI_MODEL_PREMIUM = None               # e.g. "gpt-4.1" or "o1-mini" if you want a 3rd pass
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # update if needed
GEMINI_MODEL = "gemini-2.0-pro-exp"       # or any Gemini model you have
DEEPSEEK_MODEL = "deepseek-reasoner"      # example; adjust to your account

# --- General settings ---
SLEEP_BETWEEN_CALLS = 0.4     # to reduce rate-limit issues
ENABLE_DEEPSEEK = bool(DEEPSEEK_API_KEY)
ENABLE_OPENAI = bool(OPENAI_API_KEY) and OpenAIClient is not None
ENABLE_CLAUDE = bool(ANTHROPIC_API_KEY) and Anthropic is not None
ENABLE_GEMINI = bool(GEMINI_API_KEY) and genai is not None


# ============================================================
# CLIENTS
# ============================================================

openai_client: Optional[OpenAIClient] = None
if ENABLE_OPENAI:
    openai_client = OpenAIClient(api_key=OPENAI_API_KEY)

claude_client: Optional[Anthropic] = None
if ENABLE_CLAUDE:
    claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)

if ENABLE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
else:
    gemini_model = None

deepseek_client: Optional[OpenAIClient] = None
if ENABLE_DEEPSEEK and OpenAIClient is not None:
    # DeepSeek has an OpenAI-compatible API; adjust base_url if needed.
    deepseek_client = OpenAIClient(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )


# ============================================================
# UTILITIES
# ============================================================

def hash_question(row: pd.Series) -> str:
    """
    Hash question content so unchanged items don't need revalidation (if you cache later).
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
    """Remove markdown fences and whitespace so json.loads has a cleaner shot."""
    return (
        s.replace("```json", "")
         .replace("```", "")
         .replace("```", "")
         .strip()
    )


def safe_json_load(s: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(clean_json_block(s))
    except Exception:
        fb = dict(fallback)
        fb["raw"] = s
        return fb


def normalize_letter(val: Any) -> Optional[str]:
    """Normalize 'A', 'a', 'A.' etc. into 'A', or None if not valid."""
    if not isinstance(val, str):
        return None
    v = val.strip().upper().replace(".", "")
    return v if v in {"A", "B", "C", "D"} else None


def confidence_to_weight(conf: Optional[str]) -> float:
    if not isinstance(conf, str):
        return 0.5
    c = conf.lower()
    if c == "high":
        return 1.0
    if c == "medium":
        return 0.7
    if c == "low":
        return 0.4
    return 0.5


# ============================================================
# PROMPTS
# ============================================================

def build_question_block(row: pd.Series) -> str:
    return (
        f"QUESTION:\n{row.get('question_stem', '')}\n\n"
        f"OPTIONS:\n"
        f"A) {row.get('option_a', '')}\n"
        f"B) {row.get('option_b', '')}\n"
        f"C) {row.get('option_c', '')}\n"
        f"D) {row.get('option_d', '')}\n"
    )


def build_core_prompt(row: pd.Series, provider: str) -> str:
    """
    Shared structure; we may tune wording slightly by provider if desired.
    """
    provided = row.get("correct_answer", "")
    bloom = row.get("bloom_level", "Unknown")

    return f"""
You are an expert Java MCQ validator.

Your tasks:

1. Solve the question exactly and determine which single option (A/B/C/D) is most correct.
2. Compare your answer with the PROVIDED KEY: '{provided}'.
3. Identify if the item is ambiguous:
   - Multiple options could be reasonably correct.
   - The stem is underspecified or unclear.
4. Evaluate Bloom level (cognitive demand):
   - Choose one: Remember, Understand, Apply, Analyze, Evaluate.
5. Rate distractor quality 0-10 (how plausible the wrong options are).
6. Decide if this question needs human review:
   - True if ambiguous, multiple correct answers, tricky edge cases, or key appears wrong.

{build_question_block(row)}
Claimed Bloom Level in dataset: {bloom}

Return ONLY a single JSON object (no prose, no explanation outside JSON):

{{
  "ai_answer": "A" | "B" | "C" | "D",
  "thinks_key_correct": true | false,
  "agreement": "MATCH" | "MISMATCH" | "MULTIPLE_CORRECT" | "UNCLEAR",
  "bloom_level": "Remember" | "Understand" | "Apply" | "Analyze" | "Evaluate",
  "distractor_quality": 0,
  "confidence": "high" | "medium" | "low",
  "needs_human_review": true | false
}}
""".strip()


def build_simple_openai_prompt(row: pd.Series) -> str:
    """
    Cheaper primary OpenAI prompt: correctness + basic confidence.
    """
    provided = row.get("correct_answer", "")
    return f"""
You are a precise Java MCQ solver.

1. Solve the question correctly.
2. Choose which option (A/B/C/D) is correct.
3. Compare with provided key '{provided}'.
4. Estimate your confidence.

{build_question_block(row)}

Return ONLY JSON:

{{
  "ai_answer": "A" | "B" | "C" | "D",
  "status": "MATCH" | "MISMATCH" | "UNSURE",
  "confidence": "high" | "medium" | "low"
}}
""".strip()


# ============================================================
# PROVIDER CALLS
# ============================================================

def call_openai_simple(row: pd.Series) -> Dict[str, Any]:
    if not ENABLE_OPENAI or openai_client is None:
        return {"error": "openai_disabled"}
    prompt = build_simple_openai_prompt(row)
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL_PRIMARY,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        txt = resp.choices[0].message.content.strip()
        base = {"ai_answer": None, "status": "PARSE_ERROR", "confidence": "low"}
        data = safe_json_load(txt, base)
        return data
    except Exception as e:
        return {"error": f"openai_primary_error: {e}"}


def call_openai_deep(row: pd.Series) -> Dict[str, Any]:
    if not ENABLE_OPENAI or openai_client is None:
        return {"error": "openai_disabled"}
    prompt = build_core_prompt(row, "openai")
    model = OPENAI_MODEL_PREMIUM or OPENAI_MODEL_PRIMARY
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        txt = resp.choices[0].message.content.strip()
        base = {
            "ai_answer": None,
            "thinks_key_correct": None,
            "agreement": "PARSE_ERROR",
            "bloom_level": None,
            "distractor_quality": None,
            "confidence": "medium",
            "needs_human_review": True,
        }
        data = safe_json_load(txt, base)
        return data
    except Exception as e:
        return {"error": f"openai_deep_error: {e}"}


def call_claude(row: pd.Series) -> Dict[str, Any]:
    if not ENABLE_CLAUDE or claude_client is None:
        return {"error": "claude_disabled"}
    prompt = build_core_prompt(row, "claude")
    try:
        resp = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        texts = []
        for block in resp.content:
            if block.type == "text":
                texts.append(block.text)
        txt = "\n".join(texts).strip()
        base = {
            "ai_answer": None,
            "thinks_key_correct": None,
            "agreement": "PARSE_ERROR",
            "bloom_level": None,
            "distractor_quality": None,
            "confidence": "medium",
            "needs_human_review": True,
        }
        data = safe_json_load(txt, base)
        return data
    except Exception as e:
        return {"error": f"claude_error: {e}"}


def call_gemini(row: pd.Series) -> Dict[str, Any]:
    if not ENABLE_GEMINI or gemini_model is None:
        return {"error": "gemini_disabled"}
    prompt = build_core_prompt(row, "gemini")
    try:
        resp = gemini_model.generate_content(prompt)
        txt = resp.text.strip()
        base = {
            "ai_answer": None,
            "thinks_key_correct": None,
            "agreement": "PARSE_ERROR",
            "bloom_level": None,
            "distractor_quality": None,
            "confidence": "medium",
            "needs_human_review": True,
        }
        data = safe_json_load(txt, base)
        return data
    except Exception as e:
        return {"error": f"gemini_error: {e}"}


def call_deepseek(row: pd.Series) -> Dict[str, Any]:
    if not ENABLE_DEEPSEEK or deepseek_client is None:
        return {"error": "deepseek_disabled"}
    prompt = build_core_prompt(row, "deepseek")
    try:
        resp = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        txt = resp.choices[0].message.content.strip()
        base = {
            "ai_answer": None,
            "thinks_key_correct": None,
            "agreement": "PARSE_ERROR",
            "bloom_level": None,
            "distractor_quality": None,
            "confidence": "medium",
            "needs_human_review": True,
        }
        data = safe_json_load(txt, base)
        return data
    except Exception as e:
        return {"error": f"deepseek_error: {e}"}


# ============================================================
# NORMALIZATION & VOTING
# ============================================================

def normalize_full_vote(
    provider: str,
    raw: Dict[str, Any],
    provided_key: str
) -> Optional[Dict[str, Any]]:
    """
    Normalize each provider output into a common "vote" object.

    vote = {
        "source": "claude",
        "ai_answer": "A",
        "thinks_key_correct": True/False/None,
        "agreement": "MATCH/MISMATCH/...",
        "bloom_level": "Apply",
        "distractor_quality": float or None,
        "confidence": "high/medium/low",
        "needs_human_review": bool,
    }
    """
    if "error" in raw:
        return None

    ai_answer = normalize_letter(raw.get("ai_answer"))
    pk = normalize_letter(provided_key)
    confidence = raw.get("confidence", "medium")

    # "status" is only used by simple OpenAI pass
    status = raw.get("status")

    agreement = raw.get("agreement")
    thinks_key_correct = raw.get("thinks_key_correct")

    if agreement is None and status is not None and pk and ai_answer:
        # derive agreement from simple status
        if status == "MATCH" and ai_answer == pk:
            agreement = "MATCH"
            thinks_key_correct = True
        elif status == "MISMATCH" and ai_answer != pk:
            agreement = "MISMATCH"
            thinks_key_correct = False
        else:
            agreement = "UNCLEAR"
            thinks_key_correct = None

    # Fallback if still None
    if agreement is None:
        if ai_answer and pk:
            agreement = "MATCH" if ai_answer == pk else "MISMATCH"
        else:
            agreement = "UNCLEAR"

    needs_review = bool(raw.get("needs_human_review", False))
    bloom_level = raw.get("bloom_level")
    dq = raw.get("distractor_quality")

    # attempt to cast distractor quality to float
    try:
        dq = float(dq) if dq is not None else None
    except (TypeError, ValueError):
        dq = None

    vote = {
        "source": provider,
        "ai_answer": ai_answer,
        "thinks_key_correct": thinks_key_correct,
        "agreement": agreement,
        "bloom_level": bloom_level,
        "distractor_quality": dq,
        "confidence": confidence,
        "weight": confidence_to_weight(confidence),
        "needs_human_review": needs_review,
    }
    return vote


def aggregate_votes(
    votes: List[Dict[str, Any]],
    dataset_bloom: Optional[str],
    provided_key: str
) -> Dict[str, Any]:
    """
    Aggregate all model votes into a final consensus verdict.
    """
    provided = normalize_letter(provided_key)
    n_models = len(votes)

    # ---- Majority answer ----
    answer_weights: Dict[str, float] = {}
    key_agree_weight = 0.0
    key_disagree_weight = 0.0
    any_ambiguous = False
    any_needs_review = False

    bloom_votes: Dict[str, float] = {}
    distractor_scores: List[float] = []

    for v in votes:
        a = v["ai_answer"]
        w = v["weight"]
        if a:
            answer_weights[a] = answer_weights.get(a, 0.0) + w
            if provided and a == provided:
                key_agree_weight += w
            elif provided and a != provided:
                key_disagree_weight += w

        if v["agreement"] in ["MULTIPLE_CORRECT", "UNCLEAR"]:
            any_ambiguous = True
        if v["needs_human_review"]:
            any_needs_review = True

        b = v["bloom_level"]
        if isinstance(b, str) and b.strip():
            bloom_votes[b] = bloom_votes.get(b, 0.0) + w

        if v["distractor_quality"] is not None:
            distractor_scores.append(v["distractor_quality"])

    # Determine majority answer (argmax on weighted votes)
    majority_answer = None
    majority_weight = 0.0
    for ans, w in answer_weights.items():
        if w > majority_weight:
            majority_answer = ans
            majority_weight = w

    # Bloom consensus
    consensus_bloom = None
    bloom_weight = 0.0
    for b, w in bloom_votes.items():
        if w > bloom_weight:
            consensus_bloom = b
            bloom_weight = w

    # Distractor quality
    avg_dq = sum(distractor_scores) / len(distractor_scores) if distractor_scores else None

    # Final status
    if n_models == 0:
        final_status = "NO_MODELS_USED"
    else:
        # strong consensus that key is correct
        if key_agree_weight >= 1.5 and key_agree_weight > key_disagree_weight and not any_ambiguous:
            final_status = "CONSENSUS_KEY_CORRECT"
        # strong consensus that key is wrong
        elif key_disagree_weight >= 1.5 and key_disagree_weight > key_agree_weight:
            final_status = "CONSENSUS_KEY_INCORRECT"
        # ambiguous or flagged
        elif any_ambiguous or any_needs_review:
            final_status = "AMBIGUOUS_OR_NEEDS_REVIEW"
        # conflicting but mild
        else:
            final_status = "MODEL_DISAGREEMENT"

    bloom_mismatch = None
    if consensus_bloom and dataset_bloom:
        dataset_norm = dataset_bloom.strip().capitalize()
        if dataset_norm != consensus_bloom:
            bloom_mismatch = True
        else:
            bloom_mismatch = False

    return {
        "majority_answer": majority_answer,
        "majority_weight": majority_weight,
        "key_agree_weight": key_agree_weight,
        "key_disagree_weight": key_disagree_weight,
        "any_ambiguous_or_review": any_ambiguous or any_needs_review,
        "consensus_bloom": consensus_bloom,
        "dataset_bloom": dataset_bloom,
        "bloom_mismatch": bloom_mismatch,
        "avg_distractor_quality": avg_dq,
        "final_status": final_status,
        "n_models": n_models,
    }


# ============================================================
# SINGLE QUESTION VALIDATION
# ============================================================

def validate_question(row: pd.Series) -> Dict[str, Any]:
    """
    Full ValidationV3 pipeline for a single question.
    """
    provided_key = row.get("correct_answer", "")
    dataset_bloom = row.get("bloom_level", "")

    q_hash = hash_question(row)

    all_votes: List[Dict[str, Any]] = []
    raw_by_model: Dict[str, Any] = {}

    # --- 1. Optional DeepSeek ---
    if ENABLE_DEEPSEEK:
        raw_deep = call_deepseek(row)
        raw_by_model["deepseek"] = raw_deep
        v_deep = normalize_full_vote("deepseek", raw_deep, provided_key)
        if v_deep:
            all_votes.append(v_deep)
        time.sleep(SLEEP_BETWEEN_CALLS)

    # --- 2. OpenAI Primary (simple) ---
    if ENABLE_OPENAI:
        raw_o1 = call_openai_simple(row)
        raw_by_model["openai_simple"] = raw_o1
        v_o1 = normalize_full_vote("openai_simple", raw_o1, provided_key)
        if v_o1:
            all_votes.append(v_o1)
        time.sleep(SLEEP_BETWEEN_CALLS)

        # OpenAI deep (optional but nice)
        raw_o2 = call_openai_deep(row)
        raw_by_model["openai_deep"] = raw_o2
        v_o2 = normalize_full_vote("openai_deep", raw_o2, provided_key)
        if v_o2:
            all_votes.append(v_o2)
        time.sleep(SLEEP_BETWEEN_CALLS)

    # --- 3. Claude ---
    if ENABLE_CLAUDE:
        raw_c = call_claude(row)
        raw_by_model["claude"] = raw_c
        v_c = normalize_full_vote("claude", raw_c, provided_key)
        if v_c:
            all_votes.append(v_c)
        time.sleep(SLEEP_BETWEEN_CALLS)

    # --- 4. Gemini ---
    if ENABLE_GEMINI:
        raw_g = call_gemini(row)
        raw_by_model["gemini"] = raw_g
        v_g = normalize_full_vote("gemini", raw_g, provided_key)
        if v_g:
            all_votes.append(v_g)
        time.sleep(SLEEP_BETWEEN_CALLS)

    # Aggregate
    agg = aggregate_votes(all_votes, dataset_bloom, provided_key)

    # Flatten for CSV row
    out = {
        "question_id": row.get("question_id", ""),
        "topic": row.get("topic", ""),
        "dataset_bloom": dataset_bloom,
        "question_snippet": str(row.get("question_stem", ""))[:80] + "...",
        "provided_key": provided_key,
        "question_hash": q_hash,

        "final_status": agg["final_status"],
        "majority_answer": agg["majority_answer"],
        "majority_weight": agg["majority_weight"],
        "key_agree_weight": agg["key_agree_weight"],
        "key_disagree_weight": agg["key_disagree_weight"],
        "any_ambiguous_or_review": agg["any_ambiguous_or_review"],
        "consensus_bloom": agg["consensus_bloom"],
        "bloom_mismatch": agg["bloom_mismatch"],
        "avg_distractor_quality": agg["avg_distractor_quality"],
        "n_models_used": agg["n_models"],
    }

    # Optionally include raw JSON (as strings) for later deep analysis
    for name, raw in raw_by_model.items():
        out[f"{name}_raw"] = json.dumps(raw, ensure_ascii=False)

    return out


# ============================================================
# MAIN DRIVER
# ============================================================

def main():
    print(f"📥 Loading questions from {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, encoding="latin1")

    results = []

    total = len(df)
    print(f"Found {total} questions.")
    print(f"Providers enabled: "
          f"OpenAI={ENABLE_OPENAI}, Claude={ENABLE_CLAUDE}, "
          f"Gemini={ENABLE_GEMINI}, DeepSeek={ENABLE_DEEPSEEK}\n")

    for idx, row in df.iterrows():
        print(f"Q{idx+1}/{total} ...", end="", flush=True)
        res = validate_question(row)
        results.append(res)
        print(f" {res['final_status']}")
        # additional sleep is already handled inside validate_question

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n✅ Validation complete. Report saved to: {OUTPUT_FILE}")

    # Quick summary
    print("\n📊 Final Status Summary:")
    print(out_df["final_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
