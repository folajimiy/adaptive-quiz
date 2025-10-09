import pandas as pd
import re

# Load your existing dataset
INPUT_PATH = "data/java_question_bank_with_topics_cleaned.csv"
OUTPUT_PATH = "data/java_question_bank_with_topics_cleaned_formatted.csv"

df = pd.read_csv(INPUT_PATH)

# Keywords to help detect Java syntax
java_keywords = r"(int|String|boolean|char|float|double|System\.out|class|public|private|static|void|new)"

def format_java_code_block(text):
    """Wraps Java code in markdown block if it matches Java-like syntax."""
    if pd.isna(text):
        return text
    lines = text.split(';')
    code_lines = [line.strip() + ';' for line in lines if re.search(java_keywords, line)]
    if code_lines:
        return f"```java\n" + "\n".join(code_lines) + "\n```"
    return text.strip()

# def format_question_stem(text):
#     """Detects Java code in the stem and formats it."""
#     if pd.isna(text):
#         return text
#     parts = text.split(':', 1)
#     if len(parts) == 2:
#         intro, rest = parts
#         formatted_code = format_java_code_block(rest)
#         return f"{intro.strip()}:\n\n{formatted_code}"
#     else:
#         return format_java_code_block(text)


def smart_format_question_stem(text):
    """Preserves non-code text and wraps only Java code blocks in markdown."""
    if pd.isna(text):
        return text

    java_keywords = r"\b(int|String|System\.out|boolean|class|public|private|static|void|new|float|double|char)\b"
    
    # Split text based on semicolons while preserving other parts
    parts = text.split(';')
    code_lines = []
    rest_text = []

    for part in parts:
        if re.search(java_keywords, part):
            code_lines.append(part.strip() + ';')
        else:
            rest_text.append(part.strip())

    formatted = ""

    if code_lines:
        formatted += "```java\n" + "\n".join(code_lines) + "\n```"

    # Include non-code text before or after
    if rest_text:
        if text.strip().startswith(tuple(rest_text)):  # text starts with instruction
            formatted = " ".join(rest_text) + "\n\n" + formatted
        else:
            formatted += "\n\n" + " ".join(rest_text)

    return formatted.strip()



# Apply formatting
df["question_stem"] = df["question_stem"].apply(smart_format_question_stem)
df["option_a"] = df["option_a"].apply(format_java_code_block)
df["option_b"] = df["option_b"].apply(format_java_code_block)
df["option_c"] = df["option_c"].apply(format_java_code_block)
df["option_d"] = df["option_d"].apply(format_java_code_block)

# Save the new file
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Cleaned and saved to {OUTPUT_PATH}")

