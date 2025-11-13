import streamlit as st
import pandas as pd
import os
from streamlit_student_app5 import run_student_mode
from teacher import run_teacher_mode

# --- Page setup ---
st.set_page_config(page_title="Adaptive Java Tutor", layout="wide")

# --- Persistent session state ---
for key in ["role", "user_id", "name", "level", "intro_done", "awaiting_id", "selected_mode"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "intro_done" and key != "awaiting_id" else False

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "student_list.csv")

# --- Load existing CSV ---
if os.path.exists(CSV_PATH):
    df_students = pd.read_csv(CSV_PATH, dtype=str)
else:
    st.error(f"Student list not found at {CSV_PATH}.")
    st.stop()

# Map experience level → starting Bloom level
STARTING_BLOOM = {
    "Beginner": "Remember",
    "Intermediate": "Apply",
    "Advanced": "Evaluate"
}

def save_student_info(student_id, name, level=None):
    df_students = pd.read_csv(CSV_PATH, dtype=str)
    if (df_students["student_id"] == student_id).any():
        # Update name if provided
        if name.strip():
            df_students.loc[df_students["student_id"] == student_id, "name"] = name.strip()
        # Update level and initialize current_bloom if provided
        if level:
            df_students.loc[df_students["student_id"] == student_id, "level"] = level
            df_students.loc[df_students["student_id"] == student_id, "current_bloom"] = STARTING_BLOOM.get(level, "Remember")
    else:
        # New student
        data = {
            "student_id": [student_id],
            "name": [name.strip()],
            "level": [level if level else "Beginner"],
            "current_bloom": [STARTING_BLOOM.get(level, "Remember")]
        }
        df_students = pd.concat([df_students, pd.DataFrame(data)], ignore_index=True)

    df_students.to_csv(CSV_PATH, index=False)



# --- Intro screen ---
if not st.session_state.intro_done:
    st.markdown("""
        <h1 style='text-align: center; color: #4F8BF9;'>👋 Welcome to 
        <span style='color:#F97C4F;'>Adaptive Java Tutor</span>!</h1>
        <p style='text-align: center; font-size: 20px;'>Empowering 
        <b>Students</b> and <b>Teachers</b> with AI-driven quizzes.</p>
        """, unsafe_allow_html=True)
    st.divider()

    # --- Role selection ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👨‍🎓 Student")
        st.write("• Take adaptive quizzes\n• Track your progress\n• Get instant feedback")
        if st.button("🚀 Start as Student", key="student_btn"):
            st.session_state.role = "Student"
            st.session_state.awaiting_id = True
    with col2:
        st.subheader("👩‍🏫 Teacher")
        st.write("• Upload question banks\n• Monitor student mastery\n• Export results")
        if st.button("🛠️ Start as Teacher", key="teacher_btn"):
            st.session_state.role = "Teacher"
            st.session_state.awaiting_id = True

# --- ID, Name & Level Form ---
if st.session_state.awaiting_id:
    st.divider()
    id_exists = False
    existing_name = ""
    existing_level = ""
    show_name_input = False
    show_level_input = False

    with st.form(key="login_form", clear_on_submit=False):
        user_id = st.text_input(f"Enter your {st.session_state.role} ID:")

        id_exists = user_id.strip() in df_students["student_id"].values
        if id_exists:
            existing_name = df_students.loc[df_students["student_id"] == user_id, "name"].values[0]
            existing_name = "" if pd.isna(existing_name) else str(existing_name)
            existing_level = df_students.loc[df_students["student_id"] == user_id, "level"].values[0] if "level" in df_students.columns else ""
            existing_level = "" if pd.isna(existing_level) else str(existing_level)
            show_name_input = existing_name.strip() == ""
            show_level_input = existing_level.strip() == ""
        elif user_id.strip():
            show_name_input = True
            show_level_input = True

        name = st.text_input("Enter your name:") if show_name_input else existing_name
        level = st.radio("Select your experience level:", ("Beginner", "Intermediate", "Advanced")) if show_level_input else existing_level

        submitted = st.form_submit_button("Continue")

    if submitted:
        login_message = ""
        if not user_id.strip():
            login_message = "Please enter your ID to continue."
        elif not id_exists and st.session_state.role == "Student":
            login_message = "❌ Invalid Student ID."
        elif show_name_input and not name.strip():
            login_message = "Please enter your name for first-time login."
        elif show_level_input and not level:
            login_message = "Please select your experience level."
        else:
            if show_name_input or show_level_input:
                save_student_info(user_id, name.strip(), level)
            st.session_state.user_id = user_id.strip()
            st.session_state.name = name.strip()
            st.session_state.level = level
            st.session_state.intro_done = True
            st.session_state.awaiting_id = False
            st.rerun()

        if login_message:
            st.warning(login_message)
    st.stop()

# --- Initialize session state for main menu ---
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = None

# --- Show main menu only after login ---
if st.session_state.intro_done and st.session_state.selected_mode is None:
    st.divider()
    st.markdown(f"### 👋 Welcome, {st.session_state.name}!")
    st.markdown(f"**Level:** {st.session_state.level} | **ID:** {st.session_state.user_id}")
    st.markdown("<br>", unsafe_allow_html=True)

    # --- CSS for bigger buttons ---
    st.markdown("""
    <style>
    div.stButton > button {
        height: 80px;
        width: 300px;
        font-size: 22px;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Center buttons ---
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Practice Mode", key="practice_btn"):
            st.session_state.selected_mode = "practice"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔖 Review Mode", key="review_btn"):
            st.session_state.selected_mode = "review"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📊 View Progress", key="progress_btn"):
            st.session_state.selected_mode = "progress"
            st.rerun()

# --- After selection, run the chosen mode ---
if st.session_state.selected_mode:
    mode = st.session_state.selected_mode
    if mode == "practice":
        run_student_mode()
    elif mode == "review":
        st.info("Review mode not implemented yet.")
    elif mode == "progress":
        st.info("Progress view not implemented yet.")




