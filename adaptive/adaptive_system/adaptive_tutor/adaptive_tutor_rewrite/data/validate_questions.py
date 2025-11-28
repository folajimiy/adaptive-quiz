import sys
import pandas as pd
import json
import ast

# Columns that must exist and be non-empty
REQUIRED_NONEMPTY = [
    "id", "timestamp", "topic", "subtopic", "subject", "bloom_level",
    "question_stem",
    "option_a", "option_b", "option_c", "option_d",
    "correct_answer",
    "main_explanation",
]

# Columns that must exist but can be empty (optional metadata)
REQUIRED_OPTIONAL = [
    "cognitive_process", "kc_tags",
    "a_explanation", "b_explanation", "c_explanation", "d_explanation",
    "item_type", "predicted_difficulty_level", "predicted_difficulty_label",
    "reasoning_depth", "linguistic_complexity", "estimated_time_seconds",
    "distractor_analysis", "variant_group_id", "irt_difficulty_b",
    "irt_discrimination_a", "irt_guessing_c",
    "remediation_reference", "safety_notes", "bias_notes",
    "raw_model_response", "retrieved_slide_files",
    "retrieved_chunk_indices", "generation_seed_prompt",
    "eval_relevance", "eval_bloom_alignment", "eval_accuracy",
    "eval_explainability", "eval_justification",
    "misconception_tags_per_option",
]

VALID_BLOOM = ["Remember", "Understand", "Apply", "Analyze", "Evaluate"]
VALID_ANSWERS = ["A", "B", "C", "D"]


def parse_json_like(value, field_name, warnings, idx):
    """Try to parse JSON-like fields. We only enforce this for fields we rely on."""
    if pd.isna(value) or value == "":
        return None
    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            warnings.append(f"Row {idx}: Field '{field_name}' is not valid JSON (this is a warning, not fatal).")
            return None


def validate_row(row, idx, errors, warnings):
    # 1) Check required non-empty columns
    for col in REQUIRED_NONEMPTY:
        if col not in row or pd.isna(row[col]) or str(row[col]).strip() == "":
            errors.append(f"Row {idx}: REQUIRED field '{col}' is missing or empty")

    # 2) Ensure all required optional columns exist (but they may be empty)
    for col in REQUIRED_OPTIONAL:
        if col not in row:
            errors.append(f"Row {idx}: Missing expected column '{col}'")

    # 3) Basic logical checks
    bloom = row.get("bloom_level")
    if bloom not in VALID_BLOOM:
        errors.append(f"Row {idx}: Invalid bloom level '{bloom}'")

    ca = row.get("correct_answer")
    if ca not in VALID_ANSWERS:
        errors.append(f"Row {idx}: Correct answer invalid -> {ca}")

    # 4) Ensure options are unique (soft requirement but helpful)
    opts = [
        str(row.get("option_a", "")).strip(),
        str(row.get("option_b", "")).strip(),
        str(row.get("option_c", "")).strip(),
        str(row.get("option_d", "")).strip(),
    ]
    if "" in opts:
        errors.append(f"Row {idx}: One or more options are empty")
    elif len(set(opts)) < 4:
        warnings.append(f"Row {idx}: Options are not unique")

    # 5) Parse only the JSON we actually need for the adaptive system
    # misconception_tags_per_option: important
    parse_json_like(row.get("misconception_tags_per_option"), "misconception_tags_per_option", errors, idx)

    # retrieved_chunk_indices: nice to have, treat parsing failures as warning
    parse_json_like(row.get("retrieved_chunk_indices"), "retrieved_chunk_indices", warnings, idx)

    # We DO NOT enforce raw_model_response to be JSON—it's free text.

    # 6) Numeric fields (if present) should be numeric
    numeric_fields = [
        "predicted_difficulty_level",
        "reasoning_depth",
        "estimated_time_seconds",
        "irt_difficulty_b",
        "irt_discrimination_a",
        "irt_guessing_c",
    ]

    for nf in numeric_fields:
        if nf in row and not pd.isna(row[nf]):
            try:
                float(row[nf])
            except Exception:
                errors.append(f"Row {idx}: Field '{nf}' should be numeric")

    return errors, warnings


def validate_csv(path):
    df = pd.read_csv(path)

    errors = []
    warnings = []

    for idx, row in df.iterrows():
        validate_row(row, idx, errors, warnings)

    print("====================================")
    print("Validation Report")
    print("====================================")
    if not errors:
        print("✅ No critical errors found.")
    else:
        print(f"❌ Found {len(errors)} critical issues:")
        for err in errors:
            print(" -", err)

    if warnings:
        print("\n⚠️ Warnings (non-fatal):")
        for w in warnings[:50]:
            print(" -", w)
        if len(warnings) > 50:
            print(f"   ... and {len(warnings) - 50} more warnings.")

    print("\nTotal rows:", len(df))
    return errors, warnings


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "java_questions_adaptive.csv"

    validate_csv(csv_path)
