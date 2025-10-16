import streamlit as st
from streamlit_student_app5 import run_student_mode
from teacher import run_teacher_mode

# --- Page setup ---
st.set_page_config(page_title="Adaptive Java Tutor", layout="wide")

# --- Initialize session state ---
if "role" not in st.session_state:
    st.session_state.role = None
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False
if "awaiting_id" not in st.session_state:
    st.session_state.awaiting_id = False

# --- Intro screen ---
if not st.session_state.intro_done:
    # Header only shows before continuing
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
            st.session_state.awaiting_id = True

    # --- ID input (same run, no rerun issue) ---
    if st.session_state.awaiting_id:
        st.divider()
        user_id = st.text_input(f"Enter your {st.session_state.role} ID:", key="user_id_input")
        continue_clicked = st.button("Continue", key="continue_btn")

        if continue_clicked and user_id.strip():
            st.session_state.user_id = user_id.strip()
            st.session_state.intro_done = True
            st.session_state.awaiting_id = False
            st.rerun()  # ✅ triggers next screen immediately
        elif continue_clicked:
            st.warning("Please enter your ID to continue.")
    st.stop()

# --- Main app logic ---
if st.session_state.role == "Student":
    run_student_mode()
elif st.session_state.role == "Teacher":
    st.info("👩‍🏫 Teacher mode not implemented yet.")
    run_teacher_mode()
else:
    st.warning("Please select a role to continue.")


