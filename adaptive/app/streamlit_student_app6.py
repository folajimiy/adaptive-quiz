
# adaptive_java_learning_app.py

import streamlit as st
import pandas as pd
import time
import os

# Load question bank
df = pd.read_csv("data/java_question_bank_with_topics_cleaned_gpt.csv", encoding="latin1")

topic_order = [
    "Basic Syntax", "Data Types", "Variables", "Operators", "Control Flow",
    "Loops", "Methods", "Arrays", "Object-Oriented Programming",
    "Inheritance", "Polymorphism", "Abstraction", "Encapsulation",
    "Exception Handling", "File I/O", "Multithreading", "Collections", "Generics"
]
df['topic'] = pd.Categorical(df['topic'], categories=topic_order, ordered=True)
df['bloom_level'] = pd.Categorical(df['bloom_level'], ordered=True)
df = df.sort_values(['topic', 'bloom_level'])

st.set_page_config("Adaptive Java Learning", layout="wide")

# Session state setup
for key, val in {
    "started": False,
    "log": [],
    "wrong_qs": [],
    "bookmarked": set(),
    "score": {},
    "topic_mastery": {},
    "asked_qs": set(),
    "mode": "Normal",
    "submitted": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Sidebar
st.sidebar.title("📚 Adaptive Java Learning")
role = st.sidebar.radio("Select Role", ["Student", "Teacher"])
topics = df["topic"].dropna().unique().tolist()
selected_topic = st.sidebar.selectbox("📘 Choose a Topic", topics)
st.sidebar.radio("Mode", ["Normal", "Retry Bookmarked", "Retry Missed"], key="mode")

if st.sidebar.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Teacher View
if role == "Teacher":
    st.title("👩‍🏫 Teacher Dashboard")
    st.bar_chart(df.groupby(["topic", "bloom_level"]).size().unstack().fillna(0))

    if st.session_state.log:
        log_df = pd.DataFrame(st.session_state.log)
        st.dataframe(log_df)
        st.download_button("📥 Download Log", log_df.to_csv(index=False), "student_log.csv")
    else:
        st.info("No logs yet.")

# Student View
else:
    st.title("🎓 Java Learning – Student Mode")
    st.subheader(f"📘 Topic: {selected_topic}")

    if not st.session_state.started:
        st.markdown("👋 Welcome! Master Java by progressing through Bloom levels.")
        if st.button("🚀 Start Learning"):
            st.session_state.started = True
            st.rerun()
        st.stop()

    # Initialize topic state
    if selected_topic not in st.session_state.topic_mastery:
        st.session_state.topic_mastery[selected_topic] = {b: 0 for b in df['bloom_level'].cat.categories}
    if selected_topic not in st.session_state.score:
        st.session_state.score[selected_topic] = {b: {"correct": 0, "total": 0} for b in df['bloom_level'].cat.categories}

    topic_df = df[df["topic"] == selected_topic]

    # Determine current Bloom level
    for level in df['bloom_level'].cat.categories:
        if st.session_state.topic_mastery[selected_topic][level] < 2:
            target_level = level
            break
    else:
        target_level = None

    # Question pool based on mode
    if st.session_state.mode == "Retry Bookmarked":
        pool = topic_df[topic_df["question_id"].isin(st.session_state.bookmarked)]
    elif st.session_state.mode == "Retry Missed":
        pool = topic_df[topic_df["question_id"].isin(st.session_state.wrong_qs)]
    elif target_level:
        pool = topic_df[topic_df["bloom_level"] == target_level]
        pool = pool[~pool["question_id"].isin(st.session_state.asked_qs)]
    else:
        pool = pd.DataFrame()

    # Load question
    if not pool.empty:
        q = pool.sample(1).iloc[0]
        st.session_state.current_q = q.to_dict()
        st.session_state.asked_qs.add(q["question_id"])
        st.session_state.submitted = False
    elif st.session_state.mode == "Normal" and st.session_state.wrong_qs:
        qid = st.session_state.wrong_qs.pop(0)
        q = topic_df[topic_df["question_id"] == qid].iloc[0]
        st.session_state.current_q = q.to_dict()
        st.session_state.submitted = False
    else:
        q = None

    if q is not None:
        q = st.session_state.current_q
        st.info(f"🧠 Bloom: {q['bloom_level']} | ID: {q['question_id']}")
        st.markdown(f"**📝 Q:** {q['question_stem']}")

        choice = st.radio("Choose:", ["a", "b", "c", "d"], index=None, format_func=lambda x: f"{x.upper()}. {q[f'option_{x}']}", key=f"choice_{q['question_id']}")

        col1, col2 = st.columns([1, 1])
        with col1:
            submit = st.button("✅ Submit Answer")
        with col2:
            if st.button("🔖 Bookmark"):
                st.session_state.bookmarked.add(q["question_id"])
                st.success("Bookmarked!")

        if submit and not st.session_state.submitted:
            if not choice:
                st.warning("Select an answer first!")
                st.stop()
            correct = q["correct_option"].strip().lower()
            is_correct = choice == correct

            if is_correct:
                st.success("✅ Correct!")
                st.session_state.topic_mastery[selected_topic][q["bloom_level"]] += 1
            else:
                st.error(f"❌ Incorrect. Correct answer: {correct.upper()}")
                st.session_state.wrong_qs.append(q["question_id"])

            st.session_state.score[selected_topic][q["bloom_level"]]["total"] += 1
            if is_correct:
                st.session_state.score[selected_topic][q["bloom_level"]]["correct"] += 1

            st.session_state.log.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "topic": q["topic"],
                "bloom_level": q["bloom_level"],
                "question_id": q["question_id"],
                "question": q["question_stem"],
                "selected": choice,
                "correct_option": correct,
                "is_correct": is_correct
            })

            # Auto save to CSV
            pd.DataFrame(st.session_state.log).to_csv("student_log.csv", index=False)
            st.session_state.submitted = True

        if st.session_state.submitted:
            st.button("➡️ Next Question", on_click=lambda: st.rerun())

    else:
        st.success("🎉 Done with this topic!")
        mastery = pd.DataFrame(st.session_state.topic_mastery[selected_topic], index=["Mastery"]).T
        st.bar_chart(mastery)
        st.download_button("📥 Download Log", pd.DataFrame(st.session_state.log).to_csv(index=False), "student_log.csv")
        if st.session_state.bookmarked:
            st.markdown("### 🔖 Bookmarked Questions")
            for b_id in st.session_state.bookmarked:
                bq = topic_df[topic_df["question_id"] == b_id].iloc[0]
                st.markdown(f"**ID {b_id}:** {bq['question_stem']}")
