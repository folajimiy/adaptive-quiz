# config.py
# config.py
import os

# -----------------------------
# Base Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# -----------------------------
# CSV Files
# -----------------------------
STUDENT_CSV = os.path.join(DATA_DIR, "student_list.csv")
QUESTION_CSV = os.path.join(DATA_DIR, "java_questions_adaptive.csv")

# -----------------------------
# Mastery Settings
# -----------------------------
MIN_ITEMS_FOR_MASTERY = 8
MASTERY_THRESHOLD = 0.80

# -----------------------------
# Session Settings
# -----------------------------
PRACTICE_QUESTION_LIMIT = 10
TEST_QUESTION_LIMIT = 15

# -----------------------------
# Skill Graph System
# -----------------------------
SKILL_GRAPH_JSON = os.path.join(DATA_DIR, "skill_graph.json")
SKILL_MIN_ATTEMPTS = 3
SKILL_MASTERY_THRESHOLD = 0.75

# -----------------------------
# Security
# -----------------------------
TEACHER_PASSWORD = "admin123"   # Change this in production!
