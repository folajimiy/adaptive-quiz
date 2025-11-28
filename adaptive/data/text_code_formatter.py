import pandas as pd
import re

# Load dataset
df = pd.read_csv("comp_1050_fixed.csv")

# Columns to process
TEXT_COLUMNS = [
    "question_stem",
    "option_a", "option_b", "option_c", "option_d",
    "a_explanation", "b_explanation",
    "c_explanation", "d_explanation",
    "main_explanation"
]

# Heuristics to detect Java-like code
CODE_HINTS = [
    r";",                   # code statements
    r"{", r"}",
    r"\bclass\b",
    r"\bpublic\b", r"\bprivate\b", r"\bprotected\b",
    r"\bstatic\b",
    r"\bvoid\b",
    r"\bint\b", r"\bdouble\b", r"\bboolean\b", r"\bString\b",
    r"\bfor\s*\(", r"\bwhile\s*\(",
    r"\bSystem\.out\.println\b"
]

def clean_and_block_code(text):
    if pd.isna(text):
        return text

    original = text.strip()

    # Skip if block already exists
    if "```" in original:
        return original

    # Split into lines for easier classification
    words = original.split()
    code_tokens = []
    text_tokens = []

    # Simple heuristic: look for any token that resembles code
    for token in words:
        if any(re.search(p, token) for p in CODE_HINTS):
            code_tokens.append(token)
        else:
            text_tokens.append(token)

    if not code_tokens:
        return original  # no code detected

    # Break code tokens into nice line-separated block
    code_block = "\n".join(code_tokens)

    # Rebuild cell
    cleaned = (
        " ".join(text_tokens).strip()
        + "\n\n```\n"
        + code_block
        + "\n```"
    )

    return cleaned.strip()


# Apply transformation across all columns
for col in TEXT_COLUMNS:
    if col in df.columns:
        df[col] = df[col].astype(str).apply(clean_and_block_code)

# Save output
df.to_csv("comp_1050_code_cleaned_simple.csv", index=False, encoding="utf-8")

print("🔥 Done! Saved as comp_1050_code_cleaned_simple.csv")
