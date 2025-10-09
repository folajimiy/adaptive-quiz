import streamlit as st
import pandas as pd
import random
import time
import os

# -------------------------------
# 1. Load question bank
# -------------------------------
DATA_PATH = "data/java_question_bank_with_topics_cleaned.csv"
df = pd.read_csv(DATA_PATH, encoding="latin1")

# Define custom topic order (by simplicity)
topic_order = [
    "Java Syntax", "Variables and Data Types", "Operators", "Control Flow", "Loops",
    "Arrays", "Methods", "Classes and Objects", "Constructors", "Inheritance",
    "Polymorphism", "Abstraction", "Encapsulation", "Interfaces", "Exception Handling",
    "File Handling", "Collections", "Generics", "Multithreading"
]
available_topics = df["topic"].dropna().unique().tolist()
topics = [t for t in topic_order if t in available_topics]

# -------------------------------
# 2. Init session state (FIXED)
# -------------------------------
if "student_mode" not in st.session_state:
    st.session_state.student_mode = True
if "topic" not in st.session_state:
    st.session_state.topic = topics[0]
if "answered_questions" not in st.session_state:
    st.session_state.answered_questions = []
if "score" not in st.session_state:
    st.session_state.score = {}
if "answers_log" not in st.session_state or not isinstance(st.session_state.answers_log, list):
    st.session_state.answers_log = []
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "question_counter" not in st.session_state:
    st.session_state.question_counter = 0

# -------------------------------
# 3. Sidebar login & topic
# -------------------------------
st.sidebar.title("User Panel")
role = st.sidebar.radio("Choose your view", ["Student", "Teacher"])
st.session_state.student_mode = (role == "Student")

if st.session_state.student_mode:
    selected_topic = st.sidebar.radio("Select Topic:", topics)
    st.session_state.topic = selected_topic

# -------------------------------
# 4. Student View
# -------------------------------
if st.session_state.student_mode:
    st.title("🎓 Adaptive Java Learning")

    # Filter by topic
    topic_df = df[df["topic"] == st.session_state.topic].copy()
    topic_df = topic_df[~topic_df["question_id"].isin(st.session_state.answered_questions)]

    if st.session_state.question_counter < 10 and not topic_df.empty:
        # Randomize based on Bloom level (mixing levels is optional here)
        question_row = topic_df.sample(1).iloc[0]

        st.markdown(f"**🧠 Bloom Level:** {question_row['bloom_level']}")
        st.markdown(f"**📝 Question {st.session_state.question_counter + 1}/10**")
        st.write(question_row["question_stem"])

        options = ["a", "b", "c", "d"]
        labels = [
            f"A. {question_row['option_a']}",
            f"B. {question_row['option_b']}",
            f"C. {question_row['option_c']}",
            f"D. {question_row['option_d']}"
        ]
        selected = st.radio("Choose your answer:", options,
                            format_func=lambda x: labels[options.index(x)],
                            key=f"q{question_row['question_id']}")

        if st.button("Submit Answer"):
            correct = question_row["correct_option"].strip().lower()
            is_correct = (selected == correct)

            # Feedback
            if is_correct:
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Incorrect. Correct answer is **{correct.upper()}**")

            # Update tracking
            st.session_state.answered_questions.append(question_row["question_id"])
            bloom = question_row["bloom_level"]
            topic = question_row["topic"]
            st.session_state.score.setdefault(topic, {}).setdefault(bloom, {"correct": 0, "total": 0})
            st.session_state.score[topic][bloom]["total"] += 1
            if is_correct:
                st.session_state.score[topic][bloom]["correct"] += 1

            st.session_state.answers_log.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "question_id": question_row["question_id"],
                "question": question_row["question_stem"],
                "topic": topic,
                "bloom_level": bloom,
                "selected": selected,
                "correct": correct,
                "is_correct": is_correct
            })

            st.session_state.question_counter += 1
            st.rerun()

    elif st.session_state.question_counter >= 10:
        st.success("🎉 You've completed this session (10 questions)!")
        if st.button("Start New Session"):
            for key in ["answered_questions", "score", "answers_log"]:
                st.session_state[key] = [] if key == "answered_questions" else {}
            st.session_state.question_counter = 0
            st.rerun()
    else:
        st.warning("⚠️ No more questions available for this topic.")

    # Show Bloom-level Mastery
    if st.session_state.score:
        st.markdown("### 📈 Topic-wise Bloom Mastery")
        for topic, bloom_data in st.session_state.score.items():
            st.markdown(f"#### 📘 {topic}")
            for level, counts in bloom_data.items():
                st.write(f"- {level}: **{counts['correct']} / {counts['total']}**")

# -------------------------------
# 5. Teacher View
# -------------------------------
else:
    st.title("📊 Teacher Dashboard")
    st.markdown("### 🧠 Topic-Bloom Distribution")
    topic_bloom = df.groupby(["topic", "bloom_level"]).size().unstack().fillna(0)
    st.bar_chart(topic_bloom)

    if st.session_state.answers_log:
        logs_df = pd.DataFrame(st.session_state.answers_log)
        st.markdown("### 📝 Recent Student Session")
        st.dataframe(logs_df)

        csv = logs_df.to_csv(index=False)
        st.download_button("📥 Download Log", csv, file_name="student_log.csv")
    else:
        st.info("No student logs yet.")
