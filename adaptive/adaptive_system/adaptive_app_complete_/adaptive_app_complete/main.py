# main.py

import streamlit as st
import pandas as pd
import os

from ui.student import run_student_mode
from ui.teacher import run_teacher_dashboard
from core.data_access import load_students, upsert_student, Student
from config import DATA_DIR, STUDENT_CSV



st.set_page_config(page_title="Adaptive Java Tutor", layout="wide")

# ---------------------------
# Session State Init
# ---------------------------
for key, default in {
    "role": None,
    "user_id": "",
    "name": "",
    "intro_done": False,
    "awaiting_id": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(STUDENT_CSV):
    pd.DataFrame(columns=["student_id", "name", "level", "current_bloom"]).to_csv(STUDENT_CSV, index=False)

df_students = load_students()


def save_student_info(student_id, name):
    student = Student(student_id=student_id, name=name, level="Beginner", current_bloom="Remember")
    upsert_student(student)


# =============================
# INTRO / ROLE SELECTION SCREEN
# =============================
# =====================
# HARD INTRO CHECK
# =====================

if "awaiting_id" not in st.session_state:
    st.session_state.awaiting_id = False

# ---------- SHOW INTRO UNTIL DONE ----------
if not st.session_state.intro_done:

    # 1. ROLE SELECTION (only if NOT awaiting ID)
    if not st.session_state.awaiting_id:
        st.markdown(
            """
            <h1 style='text-align: center; color: #4F8BF9;'>👋 Welcome to 
            <span style='color:#F97C4F;'>Adaptive Java Tutor</span>!</h1>
            <p style='text-align: center; font-size: 20px;'>Choose your role to begin.</p>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👨‍🎓 I am a Student")
            if st.button("Start as Student"):
                st.session_state.role = "Student"
                st.session_state.awaiting_id = True
                st.rerun()

        with col2:
            st.subheader("👩‍🏫 I am a Teacher")
            if st.button("Start as Teacher"):
                st.session_state.role = "Teacher"
                st.session_state.awaiting_id = True
                st.rerun()

        st.stop()   # IMPORTANT: Prevents rest of app from rendering


    # 2. ID ENTRY SCREEN (only if awaiting ID)
    st.markdown(f"### Enter your {st.session_state.role} ID:")

    user_id = st.text_input("ID:")

    if st.button("Continue"):
        if not user_id.strip():
            st.warning("Please enter your ID.")
        else:
            # Save ID
            st.session_state.user_id = user_id.strip()
            st.session_state.intro_done = True
            st.session_state.awaiting_id = False
            st.rerun()

    st.stop()  # Prevents dashboard from appearing until intro is done



# ---------------------------
# MAIN ROUTING
# ---------------------------

# Only run dashboards AFTER intro is completed
if st.session_state.intro_done:

    if st.session_state.role == "Student":
        run_student_mode()

    elif st.session_state.role == "Teacher":
        run_teacher_dashboard()

    else:
        st.warning("Please select a role to continue.")
