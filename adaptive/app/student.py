# app/student.py
import streamlit as st
import pandas as pd
import random
import time
from utils import load_data, get_next_bloom_level, get_topic_order, log_response

def run_student_mode():
    df = load_data()
    topics_ordered = get_topic_order(df)

    # Sidebar topic selection
    topic = st.sidebar.selectbox("📘 Choose Topic", topics_ordered)
    topic_df = df[df["topic"] == topic]

    # Track session state
    if "answers_log" not in st.session_state:
        st.session_state.answers_log = []
    if "question_attempts" not in st.session_state:
        st.session_state.question_attempts = {}
    if "completed_questions" not in st.session_state:
        st.session_state.completed_questions = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = None

    # Determine next question Bloom level based on mastery
    bloom_to_ask = get_next_bloom_level(topic, st.session_state.question_attempts)
    question_pool = topic_df[(topic_df["bloom_level"] == bloom_to_ask) & (~topic_df["question_id"].isin(st.session_state.completed_questions))]

    if question_pool.empty:
        st.warning("✅ No more questions available at this level. Try another topic.")
        return

    # Randomly select a question
    question = question_pool.sample(1).iloc[0]
    st.session_state.current_question = question
    qid = question["question_id"]
    bloom = question["bloom_level"]

    st.markdown(f"**🧠 Bloom Level:** `{bloom}` — *(Selected for your current mastery needs)*")
    st.markdown(f"**📝 Question:** {question['question_stem']}")

    # Display MCQ options
    options = ["a", "b", "c", "d"]
    labels = [f"A. {question['option_a']}", f"B. {question['option_b']}", f"C. {question['option_c']}", f"D. {question['option_d']}"]
    selected = st.radio("Choose your answer:", options, format_func=lambda x: labels[options.index(x)], key=qid)

    if st.button("✅ Submit Answer"):
        correct = question["correct_option"].strip().lower()
        is_correct = selected == correct

        # Feedback
        st.success("Correct!" if is_correct else f"Incorrect. The correct answer is **{correct.upper()}**")

        # Update performance
        log_response(qid, topic, bloom, selected, correct, is_correct)

        # Track attempts
        st.session_state.question_attempts.setdefault(topic, {}).setdefault(bloom, []).append(is_correct)
        st.session_state.completed_questions.append(qid)

        st.rerun()
