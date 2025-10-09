
import streamlit as st
import pandas as pd
import random
from collections import defaultdict
from datetime import datetime

st.set_page_config(page_title="Adaptive Learning Dashboard", layout="wide")

# --- SESSION STATE ---
if "page" not in st.session_state:
    st.session_state.page = "Login"
if "retry_queue" not in st.session_state:
    st.session_state.retry_queue = []
if "mastery" not in st.session_state:
    st.session_state.mastery = defaultdict(float)
if "attempts" not in st.session_state:
    st.session_state.attempts = defaultdict(list)
if "student_name" not in st.session_state:
    st.session_state.student_name = None
if "loaded_data" not in st.session_state:
    st.session_state.loaded_data = None
if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now()

# --- LOGIN + DATA UPLOAD ---
with st.sidebar:
    view = st.radio("Select View", ["Student", "Teacher"])
    uploaded = st.file_uploader("Upload Question Bank CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        st.session_state.loaded_data = df.copy()
        st.success("✅ Question bank loaded")

# --- STUDENT VIEW ---
if view == "Student":
    st.title("🎓 Adaptive Learning - Student Mode")

    if not st.session_state.student_name:
        name = st.text_input("Enter your name to begin:")
        if name:
            st.session_state.student_name = name
            st.session_state.page = "Quiz"

    if st.session_state.page == "Quiz":
        df = st.session_state.loaded_data
        if df is not None and not df.empty:
            if st.session_state.retry_queue:
                q = st.session_state.retry_queue.pop(0)
                repeat = True
            else:
                q = df.sample(1).iloc[0].to_dict()
                repeat = False

            st.subheader(f"📝 Question: {q['question']}")
            st.caption(f"Bloom Level: **{q['bloom_level']}** | ID: {q['question_id']}")

            options = eval(q['options']) if isinstance(q['options'], str) else ["True", "False"]
            correct_answer = str(q['correct_answer']).strip()

            selected = st.radio("Choose your answer:", options, key=f"opt_{q['question_id']}")
            submitted = st.button("Submit Answer", key=f"submit_{q['question_id']}")

            if submitted:
                correct = (selected == correct_answer)
                qid = str(q['question_id'])
                bloom = q['bloom_level']

                # Log attempt
                st.session_state.attempts[qid].append({
                    "timestamp": datetime.now(),
                    "answer": selected,
                    "correct": correct,
                    "bloom": bloom
                })

                # Update mastery
                if correct:
                    st.success("✅ Correct!")
                    st.session_state.mastery[qid] = min(1.0, st.session_state.mastery[qid] + 0.2)
                else:
                    st.error("❌ Incorrect.")
                    st.session_state.retry_queue.append(q)
                    st.session_state.mastery[qid] = max(0.0, st.session_state.mastery[qid] - 0.1)

                if not correct and "explanation" in q:
                    st.info(f"💡 Explanation: {q['explanation']}")

# --- TEACHER VIEW ---
elif view == "Teacher":
    st.title("👩‍🏫 Teacher Dashboard")

    # Mastery summary
    if st.session_state.mastery:
        mastery_df = pd.DataFrame([
            {"question_id": k, "mastery": v}
            for k, v in st.session_state.mastery.items()
        ])

        st.subheader("📊 Mastery per Question")
        st.dataframe(mastery_df)

        # Bloom-level mastery
        bloom_mastery = defaultdict(list)
        for qid, logs in st.session_state.attempts.items():
            for entry in logs:
                if entry["correct"]:
                    bloom_mastery[entry["bloom"]].append(1)
                else:
                    bloom_mastery[entry["bloom"]].append(0)
        bloom_summary = pd.DataFrame([
            {"bloom_level": b, "accuracy": sum(vals)/len(vals) if vals else 0.0}
            for b, vals in bloom_mastery.items()
        ])
        st.subheader("🌱 Bloom-Level Mastery Summary")
        st.dataframe(bloom_summary)

        # Export mastery logs
        export_data = []
        for qid, logs in st.session_state.attempts.items():
            for entry in logs:
                export_data.append({
                    "student": st.session_state.student_name,
                    "question_id": qid,
                    "bloom": entry["bloom"],
                    "answer": entry["answer"],
                    "correct": entry["correct"],
                    "timestamp": entry["timestamp"]
                })
        if export_data:
            df_exp = pd.DataFrame(export_data)
            st.download_button(
                "📥 Download Attempt Log as CSV",
                df_exp.to_csv(index=False).encode("utf-8"),
                "student_attempt_log.csv",
                "text/csv"
            )

        # Session time summary
        st.subheader("⏱️ Session Duration")
        elapsed = datetime.now() - st.session_state.start_time
        st.write(f"Total Time Spent: {elapsed}")
    else:
        st.info("ℹ️ No student activity yet.")
