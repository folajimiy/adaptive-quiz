import pandas as pd

INPUT_CSV = "java_questions_adaptive.csv"
OUTPUT_CSV = "java_questions_adaptive_clean.csv"

df = pd.read_csv(INPUT_CSV)

# Rows where any option is NaN or empty
bad_mask = df[["option_a", "option_b", "option_c", "option_d"]].isna().any(axis=1) | \
           (df[["option_a", "option_b", "option_c", "option_d"]].astype(str).apply(lambda s: s.str.strip() == "").any(axis=1))

bad_rows = df[bad_mask]

print(f"Found {len(bad_rows)} rows with missing/empty options.")
if len(bad_rows) > 0:
    print("Their IDs are:")
    print(bad_rows["id"].tolist())

df_clean = df[~bad_mask].copy()
df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Cleaned dataset written to: {OUTPUT_CSV}")
