import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st

# ==========================
# CONFIG
# ==========================

ITEMS_CSV_PATH = os.path.join("data", "java_questions_adaptive_clean.csv")
ROSTER_CSV_PATH = "student_roster.csv"
DB_PATH = "adaptive_tutor.db"

ADMIN_PASSWORD_ENV_VAR = "ADMIN_PASSWORD"  # set this in your .env or OS env


# ==========================
# DATA LOADING
# ==========================

@st.cache_data
def load_item_bank(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # minimal sanity: required columns
    required = [
        "id", "topic", "subtopic", "bloom_level",
        "question_stem", "option_a", "option_b", "option_c", "option_d",
        "correct_answer", "main_explanation"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Item bank is missing required columns: {missing}")
    return df


@st.cache_data
def load_roster(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        # create a tiny example
        example = pd.DataFrame([
            {"student_id": "stu001", "name": "Ada"},
            {"student_id": "stu002", "name": "Grace"},
        ])
        example.to_csv(path, index=False)
    return pd.read_csv(path).astype({"student_id": str})


# ==========================
# DATABASE HELPERS
# ==========================

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize all required tables, including mastery + test mode tables."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Core tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            question_id TEXT,
            topic TEXT,
            subtopic TEXT,
            bloom_level TEXT,
            chosen_option TEXT,
            correct_answer TEXT,
            is_correct INTEGER,
            response_time REAL,
            timestamp TEXT
        )
    """)

    # Mastery learning
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mastery_progress (
            student_id TEXT,
            topic TEXT,
            subtopic TEXT,
            mastery_score REAL,
            last_updated TIMESTAMP,
            PRIMARY KEY (student_id, topic, subtopic)
        )
    """)

    # Test mode
    cur.execute("""
        CREATE TABLE IF NOT EXISTS test_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            score REAL,
            total_items INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS test_responses (
            session_id INTEGER,
            question_id TEXT,
            student_answer TEXT,
            correct_answer TEXT,
            is_correct INTEGER,
            response_time REAL
        )
    """)

    # Student settings (mode, future prefs)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS student_settings (
            student_id TEXT PRIMARY KEY,
            mode TEXT
        )
    """)

    conn.commit()
    conn.close()


def ensure_students_from_roster(roster_df: pd.DataFrame):
    conn = get_db_connection()
    cur = conn.cursor()
    for _, row in roster_df.iterrows():
        cur.execute(
            "INSERT OR IGNORE INTO students (student_id, name) VALUES (?, ?)",
            (str(row.get("student_id")), row.get("name", None))
        )
    conn.commit()
    conn.close()


# ==========================
# LOGGING HELPERS
# ==========================

def log_response(student_id: str,
                 question_row: pd.Series,
                 chosen_option: str,
                 response_time: float):
    """Log a practice-mode response into the main responses table."""
    conn = get_db_connection()
    cur = conn.cursor()
    is_correct = int(chosen_option == question_row["correct_answer"])
    cur.execute(
        """
        INSERT INTO responses (
            student_id, question_id, topic, subtopic,
            bloom_level, chosen_option, correct_answer,
            is_correct, response_time, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            student_id,
            str(question_row["id"]),
            question_row["topic"],
            question_row["subtopic"],
            question_row["bloom_level"],
            chosen_option,
            question_row["correct_answer"],
            is_correct,
            response_time,
            datetime.now().isoformat()
        )
    )
    conn.commit()
    conn.close()

    # Also update mastery
    update_mastery(student_id, question_row["topic"], question_row["subtopic"], is_correct)


def fetch_student_history(student_id: str) -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.read_sql_query(
        "SELECT * FROM responses WHERE student_id = ? ORDER BY id ASC",
        conn,
        params=(student_id,)
    )
    conn.close()
    return df


def fetch_all_responses() -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.read_sql_query(
        "SELECT * FROM responses ORDER BY timestamp ASC",
        conn
    )
    conn.close()
    return df


def fetch_students() -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM students ORDER BY student_id ASC", conn)
    conn.close()
    return df


# ==========================
# MASTERY MODEL
# ==========================

def update_mastery(student_id: str, topic: str, subtopic: str, is_correct: int) -> float:
    """
    Exponentially Weighted Moving Average (EWMA) mastery update.

    new_mastery = alpha * correctness + (1 - alpha) * old_mastery
    """
    alpha = 0.35  # learning rate

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT mastery_score FROM mastery_progress
        WHERE student_id=? AND topic=? AND subtopic=?
    """, (student_id, topic, subtopic))
    row = cur.fetchone()

    old_mastery = row["mastery_score"] if row else 0.5  # neutral prior

    new_mastery = alpha * is_correct + (1 - alpha) * old_mastery

    cur.execute("""
        INSERT INTO mastery_progress (student_id, topic, subtopic, mastery_score, last_updated)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(student_id, topic, subtopic)
        DO UPDATE SET mastery_score=excluded.mastery_score,
                      last_updated=CURRENT_TIMESTAMP
    """, (student_id, topic, subtopic, new_mastery))

    conn.commit()
    conn.close()

    return new_mastery


def mastery_status(mastery_score: float) -> str:
    if mastery_score >= 0.85:
        return "Mastered"
    elif mastery_score >= 0.70:
        return "Approaching"
    else:
        return "Not Mastered"


def fetch_mastery_for_student(student_id: str) -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.read_sql_query(
        "SELECT * FROM mastery_progress WHERE student_id = ? ORDER BY topic, subtopic",
        conn,
        params=(student_id,)
    )
    conn.close()
    return df


def fetch_all_mastery() -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.read_sql_query(
        "SELECT * FROM mastery_progress",
        conn
    )
    conn.close()
    return df


# ==========================
# STUDENT MODE SETTINGS
# ==========================

def set_student_mode(student_id: str, mode: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO student_settings (student_id, mode)
        VALUES (?, ?)
        ON CONFLICT(student_id) DO UPDATE SET mode=excluded.mode
    """, (student_id, mode))
    conn.commit()
    conn.close()


def get_student_mode(student_id: str) -> str:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT mode FROM student_settings WHERE student_id=?", (student_id,))
    row = cur.fetchone()
    conn.close()
    if row and row["mode"] in ("practice", "test"):
        return row["mode"]
    return "practice"


# ==========================
# TEST MODE HELPERS
# ==========================

def start_test_session(student_id: str) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO test_sessions (student_id, start_time)
        VALUES (?, CURRENT_TIMESTAMP)
    """, (student_id,))
    session_id = cur.lastrowid
    conn.commit()
    conn.close()
    return session_id


def log_test_response(session_id: int,
                      question_id: str,
                      student_answer: str,
                      correct_answer: str,
                      is_correct: int,
                      response_time: float):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO test_responses (session_id, question_id, student_answer,
                                    correct_answer, is_correct, response_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, question_id, student_answer, correct_answer, is_correct, response_time))
    conn.commit()
    conn.close()

    # Optionally also update mastery based on test performance
    # Here we fetch minimal info from item bank if needed outside.


def finalize_test_session(session_id: int) -> float:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*), SUM(is_correct) FROM test_responses
        WHERE session_id=?
    """, (session_id,))
    total, correct = cur.fetchone()
    total = total or 0
    correct = correct or 0
    score = (correct / total) if total > 0 else 0.0

    cur.execute("""
        UPDATE test_sessions
        SET end_time=CURRENT_TIMESTAMP, score=?, total_items=?
        WHERE session_id=?
    """, (score, total, session_id))

    conn.commit()
    conn.close()
    return score


def fetch_test_sessions() -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.read_sql_query(
        "SELECT * FROM test_sessions ORDER BY start_time DESC",
        conn
    )
    conn.close()
    return df


def fetch_test_sessions_for_student(student_id: str) -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.read_sql_query(
        "SELECT * FROM test_sessions WHERE student_id=? ORDER BY start_time DESC",
        conn,
        params=(student_id,)
    )
    conn.close()
    return df


def fetch_test_responses_for_session(session_id: int) -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.read_sql_query(
        "SELECT * FROM test_responses WHERE session_id=?",
        conn,
        params=(session_id,)
    )
    conn.close()
    return df


def select_test_question_ids(items_df: pd.DataFrame, num_items: int = 10) -> list:
    if len(items_df) <= num_items:
        return items_df["id"].astype(str).tolist()
    return items_df.sample(num_items)["id"].astype(str).tolist()


# ==========================
# ADAPTIVE LOGIC (PRACTICE MODE)
# ==========================

def choose_next_question(student_id: str, items_df: pd.DataFrame):
    """
    Practice-mode adaptive strategy:
      1. Get student's history from DB.
      2. Find weakest topic/subtopic by accuracy.
      3. Among unseen questions in that topic/subtopic, pick a medium difficulty if available.
      4. Fallback: unseen questions overall.
      5. Fallback: any question.
    """
    history = fetch_student_history(student_id)
    if history.empty:
        # No history: pick a medium-difficulty random item
        if "predicted_difficulty_level" in items_df.columns:
            mids = items_df.copy()
            mids["diff_gap"] = (mids["predicted_difficulty_level"].fillna(3) - 3).abs()
            mids = mids.sort_values("diff_gap")
            return mids.iloc[0]
        else:
            return items_df.sample(1).iloc[0]

    # Compute accuracy per topic/subtopic
    agg = (
        history.groupby(["topic", "subtopic"])["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "accuracy"})
    )
    weakest = agg.sort_values("accuracy", ascending=True).iloc[0]
    weak_topic = weakest["topic"]
    weak_subtopic = weakest["subtopic"]

    seen_ids = set(history["question_id"].astype(str).tolist())
    candidates = items_df[
        (items_df["topic"] == weak_topic)
        & (items_df["subtopic"] == weak_subtopic)
        & (~items_df["id"].astype(str).isin(seen_ids))
    ]

    # Prefer medium difficulty
    if not candidates.empty and "predicted_difficulty_level" in candidates.columns:
        candidates = candidates.copy()
        candidates["diff_gap"] = (candidates["predicted_difficulty_level"].fillna(3) - 3).abs()
        candidates = candidates.sort_values("diff_gap")
        return candidates.iloc[0]

    # Fallback: unseen questions overall
    unseen = items_df[~items_df["id"].astype(str).isin(seen_ids)]
    if not unseen.empty:
        return unseen.sample(1).iloc[0]

    # absolute fallback: allow repeats
    return items_df.sample(1).iloc[0]


# ==========================
# UI HELPERS
# ==========================

def reset_student_session_state():
    st.session_state.current_question = None
    st.session_state.chosen_option = None
    st.session_state.question_start_time = None
    st.session_state.last_feedback = None

    # Test-mode session state
    st.session_state.test_session_id = None
    st.session_state.test_question_ids = None
    st.session_state.test_current_index = 0
    st.session_state.test_chosen_option = None
    st.session_state.test_question_start_time = None
    st.session_state.test_finished = False


def student_login(roster_df: pd.DataFrame):
    st.subheader("Student Login")

    student_id = st.text_input("Enter your Student ID")

    if st.button("Start Session"):
        if student_id.strip() == "":
            st.error("Please enter your Student ID.")
            return

        valid_ids = set(roster_df["student_id"].astype(str))
        if student_id not in valid_ids:
            st.error("This Student ID is not in the class roster. Please check with your instructor.")
            return

        st.session_state.role = "student"
        st.session_state.student_id = student_id
        reset_student_session_state()
        # Ensure they have a default mode
        set_student_mode(student_id, get_student_mode(student_id))
        st.success(f"Welcome, {student_id}! Select a mode and begin.")


def admin_login():
    st.subheader("Instructor Login")
    passwd = st.text_input("Enter admin password", type="password")
    if st.button("Log in as Instructor"):
        env_pw = os.environ.get(ADMIN_PASSWORD_ENV_VAR)
        if env_pw is None:
            st.error(f"ADMIN password not set. Define {ADMIN_PASSWORD_ENV_VAR} in your environment.")
            return
        if passwd == env_pw:
            st.session_state.role = "admin"
            st.session_state.admin_authenticated = True
            st.success("Instructor login successful.")
        else:
            st.error("Incorrect password.")


# ==========================
# STUDENT VIEW — PRACTICE MODE
# ==========================

def practice_mode_view(items_df: pd.DataFrame):
    sid = st.session_state.student_id

    col_q, col_hist = st.columns([2, 1])

    with col_q:
        st.subheader("Practice Questions")

        if st.session_state.current_question is None:
            if st.button("➡️ Next Question"):
                q_row = choose_next_question(sid, items_df)
                st.session_state.current_question = q_row.to_dict()
                st.session_state.chosen_option = None
                st.session_state.question_start_time = datetime.now()
                st.session_state.last_feedback = None
            else:
                st.info("Click 'Next Question' to get started.")
        else:
            q = st.session_state.current_question
            st.markdown(f"**Topic:** {q['topic']} **Subtopic:** {q['subtopic']}")
            st.markdown(f"### {q['question_stem']}")

            options = {
                "A": q["option_a"],
                "B": q["option_b"],
                "C": q["option_c"],
                "D": q["option_d"],
            }

            st.session_state.chosen_option = st.radio(
                "Choose an answer:",
                options=list(options.keys()),
                format_func=lambda k: f"{k}) {options[k]}",
                index=0 if st.session_state.chosen_option is None else ["A", "B", "C", "D"].index(st.session_state.chosen_option)
            )

            if st.button("✅ Submit Answer"):
                if st.session_state.chosen_option is None:
                    st.warning("Please select an option.")
                else:
                    start_time = st.session_state.question_start_time or datetime.now()
                    rt = (datetime.now() - start_time).total_seconds()

                    q_row = pd.Series(q)
                    log_response(
                        student_id=sid,
                        question_row=q_row,
                        chosen_option=st.session_state.chosen_option,
                        response_time=rt
                    )

                    is_correct = (st.session_state.chosen_option == q["correct_answer"])
                    st.session_state.last_feedback = {
                        "is_correct": is_correct,
                        "correct_answer": q["correct_answer"],
                        "main_explanation": q["main_explanation"],
                        "response_time": rt
                    }

            if st.session_state.last_feedback:
                fb = st.session_state.last_feedback
                st.markdown("---")
                if fb["is_correct"]:
                    st.success(f"✅ Correct! (answered in {fb['response_time']:.1f}s)")
                else:
                    st.error(f"❌ Not quite. Correct answer: {fb['correct_answer']} (answered in {fb['response_time']:.1f}s)")
                st.markdown(f"**Explanation:** {fb['main_explanation']}")

                if st.button("➡️ Next Question", key="next_after_feedback"):
                    st.session_state.current_question = None
                    st.session_state.chosen_option = None
                    st.session_state.question_start_time = None
                    st.session_state.last_feedback = None

    with col_hist:
        st.subheader("Your Progress")
        hist = fetch_student_history(sid)
        if hist.empty:
            st.write("No answers recorded yet.")
        else:
            total = len(hist)
            correct = int(hist["is_correct"].sum())
            st.metric("Questions Answered", total)
            st.metric("Correct Answers", correct)
            st.metric("Accuracy", f"{100*correct/total:.1f}%")

            # Quick topic breakdown
            topic_acc = (
                hist.groupby("topic")["is_correct"]
                .mean()
                .reset_index()
                .rename(columns={"is_correct": "accuracy"})
            )
            st.write("By Topic (accuracy):")
            st.dataframe(topic_acc, use_container_width=True)

        # Mastery map
        st.markdown("---")
        st.subheader("Mastery Map")
        mastery_df = fetch_mastery_for_student(sid)
        if mastery_df.empty:
            st.write("No mastery data yet. Answer more questions to build your mastery profile.")
        else:
            mastery_df = mastery_df.copy()
            mastery_df["status"] = mastery_df["mastery_score"].apply(mastery_status)
            mastery_df["mastery_pct"] = (mastery_df["mastery_score"] * 100).round(1)
            st.dataframe(
                mastery_df[["topic", "subtopic", "mastery_pct", "status"]],
                use_container_width=True
            )


# ==========================
# STUDENT VIEW — TEST MODE
# ==========================

def test_mode_view(items_df: pd.DataFrame):
    sid = st.session_state.student_id
    st.subheader("Test Mode (Summative Assessment)")

    NUM_TEST_ITEMS = 10

    # Initialize test if needed
    if st.session_state.test_session_id is None and not st.session_state.test_finished:
        if st.button("🧪 Start New Test"):
            # Select questions
            q_ids = select_test_question_ids(items_df, NUM_TEST_ITEMS)
            session_id = start_test_session(sid)

            st.session_state.test_session_id = session_id
            st.session_state.test_question_ids = q_ids
            st.session_state.test_current_index = 0
            st.session_state.test_chosen_option = None
            st.session_state.test_question_start_time = datetime.now()
            st.session_state.test_finished = False
        else:
            st.info("Click 'Start New Test' to begin a fixed-length assessment.")
            # Also show previous test results
            show_previous_tests_for_student(sid)
            return

    # Active test in progress
    if st.session_state.test_session_id is not None and not st.session_state.test_finished:
        q_ids = st.session_state.test_question_ids or []
        idx = st.session_state.test_current_index

        if idx >= len(q_ids):
            # Safety: finalize
            score = finalize_test_session(st.session_state.test_session_id)
            st.session_state.test_finished = True
            st.success(f"Test finished! Score: {score*100:.1f}%")
            show_previous_tests_for_student(sid)
            return

        current_qid = q_ids[idx]
        q_row = items_df[items_df["id"].astype(str) == str(current_qid)]
        if q_row.empty:
            st.error(f"Question with id {current_qid} not found in item bank.")
            return
        q = q_row.iloc[0].to_dict()

        st.markdown(f"**Question {idx+1} of {len(q_ids)}**")
        st.markdown(f"**Topic:** {q['topic']} **Subtopic:** {q['subtopic']}")
        st.markdown(f"### {q['question_stem']}")

        options = {
            "A": q["option_a"],
            "B": q["option_b"],
            "C": q["option_c"],
            "D": q["option_d"],
        }

        st.session_state.test_chosen_option = st.radio(
            "Choose an answer:",
            options=list(options.keys()),
            format_func=lambda k: f"{k}) {options[k]}",
            index=0 if st.session_state.test_chosen_option is None else ["A", "B", "C", "D"].index(st.session_state.test_chosen_option),
            key=f"test_radio_{idx}"
        )

        if st.button("✅ Submit Answer (No immediate feedback)", key=f"submit_test_{idx}"):
            if st.session_state.test_chosen_option is None:
                st.warning("Please select an option.")
            else:
                start_time = st.session_state.test_question_start_time or datetime.now()
                rt = (datetime.now() - start_time).total_seconds()

                is_correct = int(st.session_state.test_chosen_option == q["correct_answer"])
                log_test_response(
                    session_id=st.session_state.test_session_id,
                    question_id=str(q["id"]),
                    student_answer=st.session_state.test_chosen_option,
                    correct_answer=q["correct_answer"],
                    is_correct=is_correct,
                    response_time=rt
                )

                # advance
                st.session_state.test_current_index += 1
                st.session_state.test_chosen_option = None
                st.session_state.test_question_start_time = datetime.now()
                st.experimental_rerun()

    # Test finished
    if st.session_state.test_finished:
        score = finalize_test_session(st.session_state.test_session_id)
        st.success(f"Test finished! Score: {score*100:.1f}%")
        show_previous_tests_for_student(sid)


def show_previous_tests_for_student(student_id: str):
    st.markdown("---")
    st.subheader("Previous Test Results")
    tests_df = fetch_test_sessions_for_student(student_id)
    if tests_df.empty:
        st.write("No prior tests recorded.")
        return
    tests_df = tests_df.copy()
    tests_df["score_pct"] = (tests_df["score"] * 100).round(1)
    st.dataframe(
        tests_df[["session_id", "start_time", "end_time", "score_pct", "total_items"]],
        use_container_width=True
    )


# ==========================
# STUDENT VIEW (MODE ROUTER)
# ==========================

def student_view(items_df: pd.DataFrame):
    st.header("Student Interface")

    if "student_id" not in st.session_state:
        st.error("No student logged in. Please go back to the main page and log in.")
        return

    sid = st.session_state.student_id
    st.markdown(f"**Student ID:** `{sid}`")

    # Mode selection
    current_mode = get_student_mode(sid)
    st.markdown("### Select Mode")
    mode = st.radio(
        "Choose how you want to work today:",
        options=["practice", "test"],
        format_func=lambda m: "Practice Mode (feedback + explanations)" if m == "practice" else "Test Mode (no immediate feedback)",
        index=0 if current_mode == "practice" else 1
    )

    if mode != current_mode:
        if st.button("Apply Mode Change"):
            set_student_mode(sid, mode)
            reset_student_session_state()
            st.experimental_rerun()

    st.markdown("---")

    # Show appropriate view
    effective_mode = get_student_mode(sid)
    if effective_mode == "practice":
        practice_mode_view(items_df)
    else:
        test_mode_view(items_df)


# ==========================
# ADMIN VIEW
# ==========================

def admin_view(items_df: pd.DataFrame):
    st.header("Instructor Dashboard")

    if not st.session_state.get("admin_authenticated", False):
        st.error("You are not logged in as an instructor.")
        return

    tabs = st.tabs(["Class Overview", "Student Detail", "Mastery Overview", "Test Sessions", "Exports"])

    # --- Class Overview ---
    with tabs[0]:
        st.subheader("Class Overview")
        students_df = fetch_students()
        st.write("Enrolled Students:")
        st.dataframe(students_df, use_container_width=True)

        all_hist = fetch_all_responses()
        if all_hist.empty:
            st.info("No responses yet.")
        else:
            total = len(all_hist)
            correct = int(all_hist["is_correct"].sum())
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Responses", total)
            col2.metric("Total Correct", correct)
            col3.metric("Class Accuracy", f"{100*correct/total:.1f}%")

            topic_acc = (
                all_hist.groupby("topic")["is_correct"]
                .mean()
                .reset_index()
                .rename(columns={"is_correct": "accuracy"})
            )
            st.subheader("Accuracy by Topic")
            st.dataframe(topic_acc, use_container_width=True)
            try:
                st.bar_chart(topic_acc.set_index("topic")["accuracy"])
            except Exception:
                pass

            # Difficulty distribution if available
            if "predicted_difficulty_level" in items_df.columns:
                st.subheader("Item Difficulty Distribution (Item Bank)")
                diff_counts = items_df["predicted_difficulty_level"].value_counts().sort_index()
                st.bar_chart(diff_counts)

    # --- Student Detail ---
    with tabs[1]:
        st.subheader("Student Detail View")
        students_df = fetch_students()
        if students_df.empty:
            st.info("No students in DB yet.")
        else:
            sid = st.selectbox(
                "Select a student",
                students_df["student_id"].tolist()
            )
            hist = fetch_student_history(sid)
            if hist.empty:
                st.info("No responses yet for this student.")
            else:
                st.write(f"History for **{sid}**:")
                st.dataframe(hist, use_container_width=True)

                total = len(hist)
                correct = int(hist["is_correct"].sum())
                st.metric("Questions Answered", total)
                st.metric("Correct Answers", correct)
                st.metric("Accuracy", f"{100*correct/total:.1f}%")

                topic_acc = (
                    hist.groupby("topic")["is_correct"]
                    .mean()
                    .reset_index()
                    .rename(columns={"is_correct": "accuracy"})
                )
                st.write("Accuracy by Topic:")
                st.dataframe(topic_acc, use_container_width=True)

                try:
                    st.bar_chart(topic_acc.set_index("topic")["accuracy"])
                except Exception:
                    pass

                # Individual mastery
                st.markdown("---")
                st.subheader("Mastery for this Student")
                mastery_df = fetch_mastery_for_student(sid)
                if mastery_df.empty:
                    st.write("No mastery data yet for this student.")
                else:
                    mastery_df = mastery_df.copy()
                    mastery_df["status"] = mastery_df["mastery_score"].apply(mastery_status)
                    mastery_df["mastery_pct"] = (mastery_df["mastery_score"] * 100).round(1)
                    st.dataframe(
                        mastery_df[["topic", "subtopic", "mastery_pct", "status"]],
                        use_container_width=True
                    )

    # --- Mastery Overview ---
    with tabs[2]:
        st.subheader("Mastery Overview (All Students)")
        mastery_df = fetch_all_mastery()
        if mastery_df.empty:
            st.info("No mastery records yet.")
        else:
            mastery_df = mastery_df.copy()
            mastery_df["mastery_pct"] = (mastery_df["mastery_score"] * 100).round(1)
            mastery_df["status"] = mastery_df["mastery_score"].apply(mastery_status)
            st.dataframe(
                mastery_df[["student_id", "topic", "subtopic", "mastery_pct", "status", "last_updated"]],
                use_container_width=True
            )

            # Aggregate by topic
            topic_mastery = (
                mastery_df.groupby("topic")["mastery_score"]
                .mean()
                .reset_index()
                .rename(columns={"mastery_score": "avg_mastery"})
            )
            topic_mastery["avg_mastery_pct"] = (topic_mastery["avg_mastery"] * 100).round(1)
            st.subheader("Average Mastery by Topic")
            st.dataframe(topic_mastery, use_container_width=True)
            try:
                st.bar_chart(topic_mastery.set_index("topic")["avg_mastery"])
            except Exception:
                pass

    # --- Test Sessions ---
    with tabs[3]:
        st.subheader("Test Sessions")
        tests_df = fetch_test_sessions()
        if tests_df.empty:
            st.info("No test sessions recorded yet.")
        else:
            tests_df = tests_df.copy()
            tests_df["score_pct"] = (tests_df["score"] * 100).round(1)
            st.dataframe(
                tests_df[["session_id", "student_id", "start_time", "end_time", "score_pct", "total_items"]],
                use_container_width=True
            )

            session_ids = tests_df["session_id"].tolist()
            selected_session = st.selectbox("Select a session to inspect", session_ids)
            if selected_session is not None:
                resp_df = fetch_test_responses_for_session(int(selected_session))
                if resp_df.empty:
                    st.write("No responses for this session (unexpected).")
                else:
                    st.subheader(f"Responses for Session {selected_session}")
                    st.dataframe(resp_df, use_container_width=True)

    # --- Exports ---
    with tabs[4]:
        st.subheader("Exports")
        all_hist = fetch_all_responses()
        if all_hist.empty:
            st.info("No responses to export yet.")
        else:
            import io
            csv_buffer = io.StringIO()
            all_hist.to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇️ Download All Responses (CSV)",
                data=csv_buffer.getvalue(),
                file_name="all_responses.csv",
                mime="text/csv"
            )

            students_df = fetch_students()
            sid = st.selectbox(
                "Select student for per-student export",
                students_df["student_id"].tolist(),
                key="export_student_select"
            )
            hist = fetch_student_history(sid)
            if not hist.empty:
                csv_buffer2 = io.StringIO()
                hist.to_csv(csv_buffer2, index=False)
                st.download_button(
                    f"⬇️ Download {sid}'s Responses (CSV)",
                    data=csv_buffer2.getvalue(),
                    file_name=f"{sid}_responses.csv",
                    mime="text/csv"
                )
            else:
                st.info("No responses yet for this student.")


# ==========================
# MAIN APP
# ==========================

def main():
    st.set_page_config(page_title="Adaptive Java Tutor", layout="wide")

    st.title("Adaptive Java Tutor (Mastery + Practice/Test Modes)")

    # Initialize DB, roster, and item bank
    init_db()
    roster_df = load_roster(ROSTER_CSV_PATH)
    ensure_students_from_roster(roster_df)

    if not os.path.exists(ITEMS_CSV_PATH):
        st.error(f"Item bank not found at {ITEMS_CSV_PATH}. Please place your java_questions_adaptive.csv there.")
        return

    items_df = load_item_bank(ITEMS_CSV_PATH)

    # Role routing
    if "role" not in st.session_state:
        st.session_state.role = None

    if st.session_state.role is None:
        # Landing page: choose student or admin
        st.markdown("### Who are you?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎓 I am a Student", use_container_width=True):
                st.session_state.role = "login_student"
        with col2:
            if st.button("🧑‍🏫 I am an Instructor", use_container_width=True):
                st.session_state.role = "login_admin"

    # --- Student login flow ---
    if st.session_state.role == "login_student":
        if st.button("⬅️ Back to role selection"):
            st.session_state.role = None
            return
        student_login(roster_df)

    # --- Admin login flow ---
    if st.session_state.role == "login_admin":
        if st.button("⬅️ Back to role selection", key="back_admin"):
            st.session_state.role = None
            return
        admin_login()

    # --- Active student session ---
    if st.session_state.role == "student":
        if st.button("⬅️ Logout", key="student_logout"):
            st.session_state.clear()
            st.experimental_rerun()
        student_view(items_df)

    # --- Active admin session ---
    if st.session_state.role == "admin":
        if st.button("⬅️ Logout", key="admin_logout"):
            st.session_state.clear()
            st.experimental_rerun()
        admin_view(items_df)


if __name__ == "__main__":
    main()
