# core/data_access.py

import os
import pandas as pd
from typing import Optional

from core.models import Student
from config import STUDENT_CSV, QUESTION_CSV, LOGS_DIR


# ============================================================
# STUDENT DATA
# ============================================================

def load_students() -> pd.DataFrame:
    """Load student CSV; create it if missing."""
    if not os.path.exists(STUDENT_CSV):
        df = pd.DataFrame(
            columns=[
                "student_id",
                "name",
                "role",             # 👈 new
                "level",
                "current_bloom",
                "unlocked_topics"
            ]
        )
        df.to_csv(STUDENT_CSV, index=False)
        return df

    df = pd.read_csv(STUDENT_CSV, dtype=str)

    # ensure new column exists
    if "role" not in df.columns:
        df["role"] = "student"

    if "unlocked_topics" not in df.columns:
        df["unlocked_topics"] = ""

    return df


def save_students(df: pd.DataFrame):
    """Save updated student CSV."""
    df.to_csv(STUDENT_CSV, index=False)


def get_student(student_id: str):
    """Retrieve a Student object from CSV."""
    df = load_students()
    row = df[df["student_id"] == student_id]

    if row.empty:
        return None

    r = row.iloc[0]

    # Parse unlocked topics
    raw = r.get("unlocked_topics", "")
    if isinstance(raw, float) and pd.isna(raw):
        unlocked = []
    elif isinstance(raw, str) and raw.strip():
        unlocked = [t.strip() for t in raw.split("|") if t.strip()]
    else:
        unlocked = []

    return Student(
        student_id=r["student_id"],
        name=r.get("name", ""),
        role=r.get("role", "student"),       # 👈 now included
        level=r.get("level", "Beginner"),
        current_bloom=r.get("current_bloom", "Remember"),
        unlocked_topics=unlocked
    )


def upsert_student(student: Student):
    """Insert or update a student in the CSV."""
    df = load_students()
    mask = df["student_id"] == str(student.student_id)

    unlocked_str = "|".join(student.unlocked_topics or [])

    if mask.any():
        # Update existing student
        df.loc[mask, "name"] = student.name
        df.loc[mask, "role"] = student.role          # 👈 new
        df.loc[mask, "level"] = student.level
        df.loc[mask, "current_bloom"] = student.current_bloom
        df.loc[mask, "unlocked_topics"] = unlocked_str
    else:
        # Insert new student
        new_row = {
            "student_id": student.student_id,
            "name": student.name,
            "role": student.role,
            "level": student.level,
            "current_bloom": student.current_bloom,
            "unlocked_topics": unlocked_str
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    save_students(df)


def create_default_student(student_id, name=""):
    """Convenience helper for creating a new student."""
    return Student(
        student_id=student_id,
        name=name,
        role="student",
        level="Beginner",
        current_bloom="Remember",
        unlocked_topics=[]
    )


# ============================================================
# QUESTION BANK
# ============================================================

def load_questions() -> pd.DataFrame:
    """Load the question bank with full metadata."""
    if not os.path.exists(QUESTION_CSV):
        raise FileNotFoundError(f"Question CSV not found at: {QUESTION_CSV}")

    # Load questions safely
    df = pd.read_csv(QUESTION_CSV, encoding="latin1")

    # 🔥 Ensure question_id stays string-based UUID-like ID
    if "question_id" in df.columns:
        df["question_id"] = df["question_id"].astype(str)

    # ----------------------------------------
    # Bloom Level: ordered category (required!)
    # ----------------------------------------
    if "bloom_level" in df.columns:
        # Ensure no NaN — replace with "Understand" as neutral fallback
        df["bloom_level"] = df["bloom_level"].fillna("Understand")

        df["bloom_level"] = (
            df["bloom_level"]
            .astype(str)
            .str.strip()
            .str.capitalize()  # normalize case
        )

        # Set allowed categories
        bloom_order = ["Remember", "Understand", "Apply", "Analyze", "Evaluate"]

        df["bloom_level"] = pd.Categorical(
            df["bloom_level"],
            categories=bloom_order,
            ordered=True
        )

    return df


# ============================================================
# LOGS
# ============================================================
def load_student_logs(student_id):
    path = os.path.join(LOGS_DIR, f"student_{student_id}_responses.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(
        path,
        on_bad_lines='skip',     # skip malformed rows
        engine="python"          # more tolerant CSV parser
    )

def load_student_logs(student_id: str) -> pd.DataFrame:
    """
    Load a single student's logs safely.
    Handles malformed lines caused by commas in explanations,
    JSON strings, reinforcement text, or accidental line breaks.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    path = os.path.join(LOGS_DIR, f"student_{student_id}.csv")

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        # SAFER: tolerant CSV parser, skips bad lines
        df = pd.read_csv(
            path,
            engine="python",       # flexible parsing
            on_bad_lines="skip"    # skip corrupted/misaligned rows
        )
    except Exception as e:
        # If something catastrophic happens, fail gracefully
        print(f"Error reading log file for {student_id}: {e}")
        return pd.DataFrame()

    # Convert timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df



def append_log(student_id: str, log_row: dict):
    """Append a single row to a student's log CSV."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    path = os.path.join(LOGS_DIR, f"student_{student_id}.csv")

    new_df = pd.DataFrame([log_row])
    new_df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)




def load_all_logs() -> pd.DataFrame:
    """
    Load & merge all student log CSVs safely.

    - Ensures question_id is string
    - Skips corrupted lines (bad rows)
    - Normalizes missing columns
    - Never crashes the teacher dashboard
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    all_rows = []
    expected_cols = [
        "timestamp", "student_id", "session_id",
        "topic", "bloom_level", "question_id",
        "selected", "correct_option", "is_correct",
        "confidence", "mode", "response_time_sec",
        "reinforcement_reason", "misconception_tags"
    ]

    for file in os.listdir(LOGS_DIR):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(LOGS_DIR, file)

        try:
            df = pd.read_csv(path, on_bad_lines="skip")   # ← avoids parser errors
        except Exception as e:
            print(f"[WARN] Could not read {file}: {e}")
            continue

        if df.empty:
            continue

        # normalize columns
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        # enforce types
        df["question_id"] = df["question_id"].astype(str)
        df["student_id"] = df["student_id"].astype(str)

        all_rows.append(df[expected_cols])

    if not all_rows:
        return pd.DataFrame(columns=expected_cols)

    full_df = pd.concat(all_rows, ignore_index=True)

    # parse timestamp safely
    if "timestamp" in full_df.columns:
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], errors="coerce")

    return full_df


