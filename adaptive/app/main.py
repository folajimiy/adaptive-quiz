# app/main.py
import streamlit as st
from student import run_student_mode
from teacher import run_teacher_mode

st.set_page_config(page_title="Adaptive Java Tutor", layout="wide")
st.title("📚 AI for Inclusive Education")

# Sidebar role selector
mode = st.sidebar.radio("Select Mode", ["Student", "Teacher"])
if mode == "Student":
    run_student_mode()
else:
    run_teacher_mode()
