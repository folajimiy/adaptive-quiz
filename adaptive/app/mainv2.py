import streamlit as st
import pandas as pd
import os
from streamlit_student_app6 import run_student_mode
# from teacher import run_teacher_mode

# --- Page setup ---
st.set_page_config(page_title="Adaptive Java Tutor", layout="wide")

# --- Persistent session state ---
if "role" not in st.session_state:
    st.session_state.role = None
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "name" not in st.session_state:
    st.session_state.name = ""
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False
if "awaiting_id" not in st.session_state:
    st.session_state.awaiting_id = False

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

# --- Function to save/update student info ---
def save_student_info(student_id, name):
    df_students = pd.read_csv(CSV_PATH, dtype=str)
    if (df_students["student_id"] == student_id).any():
        if name.strip():
            df_students.loc[df_students["student_id"] == student_id, "name"] = name.strip()
    else:
        df_students = pd.concat([df_students, pd.DataFrame({"student_id": [student_id], "name": [name.strip()]})])
    df_students.to_csv(CSV_PATH, index=False)

# --- Intro screen ---
if not st.session_state.intro_done:
    st.markdown(
        """
        <h1 style='text-align: center; color: #4F8BF9;'>👋 Welcome to 
        <span style='color:#F97C4F;'>Adaptive Java Tutor</span>!</h1>
        <p style='text-align: center; font-size: 20px;'>Empowering 
        <b>Students</b> and <b>Teachers</b> with AI-driven quizzes.</p>
        """,
        unsafe_allow_html=True
    )
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
            # st.session_state.awaiting_id = True

    # --- ID input ---
if st.session_state.awaiting_id:
    st.divider()
    user_id = st.text_input(f"Enter your {st.session_state.role} ID:", key="user_id_input")

    # Check if ID exists in CSV
    id_exists = user_id in df_students["student_id"].values
    existing_name = ""
    if id_exists:
        existing_name = df_students.loc[df_students["student_id"] == user_id, "name"].values[0]
        if pd.isna(existing_name):  # convert NaN to empty string
            existing_name = ""

    # Only show name input if ID is valid but name is blank
    show_name_input = id_exists and existing_name.strip() == ""

    if show_name_input:
        name = st.text_input("Enter your name:", key="name_input")
    else:
        name = existing_name  # use existing name or leave blank if ID invalid

    # Validate
    valid_id = id_exists
    if user_id.strip() and not valid_id:
        st.warning("❌ Invalid Student ID. Please enter a valid ID from the list.")

    if st.button("Continue", key="continue_btn"):
        if not user_id.strip():
            st.warning("Please enter your ID to continue.")
        elif show_name_input and not name.strip():
            st.warning("Please enter your name for the first-time login.")
        elif not valid_id:
            st.warning("Invalid ID. Cannot continue.")
        else:
            st.session_state.user_id = user_id.strip()
            st.session_state.name = name.strip()
            if st.session_state.role == "Student" and show_name_input:
                save_student_info(st.session_state.user_id, st.session_state.name)
            st.session_state.intro_done = True
            st.session_state.awaiting_id = False
            st.rerun()


    st.stop()  # stops only intro

# --- Main app logic (after intro) ---
st.divider()

if st.session_state.role == "Student":
    st.success(f"👋 Welcome, {st.session_state.name} (ID: {st.session_state.user_id})!")
    run_student_mode()
elif st.session_state.role == "Teacher":
    st.info("👩‍🏫 Teacher mode not implemented yet.")
    # run_teacher_mode()
else:
    st.warning("Please select a role to continue.")



