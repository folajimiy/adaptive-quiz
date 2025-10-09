# app/utils.py
import pandas as pd
import streamlit as st
import os
import time


DATA_PATH = "data/java_question_bank_with_topics_cleaned.csv"
LOG_PATH = "logs/student_performance_log.csv"

bloom_order = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate']

def load_data():
    return pd.read_csv(DATA_PATH)

def get_topic_order(df):
    # Manual order for logical simplicity
    ordered = ['Basic Syntax', 'Variables', 'Operators', 'Control Flow', 'OOP Basics', 'Inheritance', 'Interfaces', 'Exception Handling']
    return [t for t in ordered if t in df['topic'].unique()] + sorted(set(df['topic']) - set(ordered))

def get_next_bloom_level(topic, attempts_dict):
    attempts = attempts_dict.get(topic, {})
    for level in bloom_order:
        results = attempts.get(level, [])
        if len(results) < 3 or sum(results[-3:]) < 2:
            return level
    return bloom_order[-1]  # Highest if all mastered


def log_response(qid, topic, bloom, selected, correct, is_correct):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_row = {
        "timestamp": timestamp,
        "question_id": qid,
        "topic": topic,
        "bloom_level": bloom,
        "selected": selected,
        "correct_answer": correct,
        "correct": is_correct
    }

    # ✅ Ensure the 'logs' directory exists
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    df = pd.DataFrame([log_row])
    if os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)








def load_logs():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return None
