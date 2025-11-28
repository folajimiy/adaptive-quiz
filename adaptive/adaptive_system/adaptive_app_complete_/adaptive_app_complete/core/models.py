# core/models.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import datetime as dt




# ============================================================
# QUESTION OBJECT
# ============================================================

@dataclass
class Question:
    question_id: str
    topic: str
    bloom_level: str
    question_stem: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_option: str
    main_explanation: str
    sub_concept: Optional[str] = None
    predicted_difficulty_level: Optional[int] = None
    ambiguity_score: Optional[float] = None
    novelty_score: Optional[float] = None



# ============================================================
# STUDENT OBJECT
# ============================================================

class Student:
    def __init__(self, student_id, name="", role="student",
                 level="Beginner", current_bloom="Remember",
                 unlocked_topics=None):
        self.student_id = student_id
        self.name = name
        self.role = role
        self.level = level
        self.current_bloom = current_bloom
        self.unlocked_topics = unlocked_topics or []



# ============================================================
# SESSION LOG OBJECT
# ============================================================

@dataclass
class SessionLog:
    timestamp: dt.datetime
    student_id: str
    session_id: str
    topic: str
    bloom_level: str
    question_id: str
    selected: str
    correct_option: str
    is_correct: bool
    confidence: int
    mode: str                         # Practice or Test
    response_time_sec: Optional[float] = None
    reinforcement_reason: str = ""
    extra: Optional[Dict[str, Any]] = None


