# core/progression_engine.py

from typing import List
from core.data_access import get_student, upsert_student, load_questions
from core.mastery_engine import is_topic_mastered
import streamlit as st


# ------------------------------------------------------------
# TOPIC ORDER (You may edit for your course)
# ------------------------------------------------------------
TOPIC_SEQUENCE = [
    "Java Fundamentals",
    "Objects and Classes",
    "Methods",
    "Control Flow",
    "Loops",
    "Arrays",
    "Strings",
    "Encapsulation",
    "Inheritance",
    "Polymorphism",
    "Abstract Classes",
    "Interfaces",
    "Generics",
    "Collections",
    "Event-Driven Programming",
    "JavaFX"
]


# ------------------------------------------------------------
# UNLOCKED TOPICS MANAGER
# ------------------------------------------------------------

def initialize_student_topic_state(student_id: str):
    """Ensure student has unlocked_topics initialized."""
    student = get_student(student_id)

    if not hasattr(student, "unlocked_topics") or not student.unlocked_topics:
        # Unlock the first topic only
        student.unlocked_topics = [TOPIC_SEQUENCE[0]]
        upsert_student(student)

    return student.unlocked_topics


def is_topic_unlocked(student_id: str, topic: str) -> bool:
    student = get_student(student_id)
    return topic in student.unlocked_topics


def unlock_next_topic(student_id: str, mastered_topic: str):
    """Unlock the next topic after student masters the given one."""
    student = get_student(student_id)

    # If topic is the last one, nothing to unlock
    if mastered_topic not in TOPIC_SEQUENCE:
        return

    idx = TOPIC_SEQUENCE.index(mastered_topic)
    if idx == len(TOPIC_SEQUENCE) - 1:
        return  # last topic, nothing to unlock

    next_topic = TOPIC_SEQUENCE[idx + 1]

    if next_topic not in student.unlocked_topics:
        student.unlocked_topics.append(next_topic)
        upsert_student(student)


# ------------------------------------------------------------
# MASTER CHECK FOR TOPIC
# ------------------------------------------------------------

def evaluate_topic_mastery(student_id: str, topic: str, log_df):
    """Check if a topic is mastered, and unlock next if so."""
    topic_log = log_df[log_df["topic"] == topic]

    if is_topic_mastered(topic_log.to_dict(orient="records")):
        unlock_next_topic(student_id, topic)
        return True

    return False
