# ui/student.py
import streamlit as st
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from io import BytesIO

from typing import List
import numpy as np
from io import BytesIO



from core.progression_engine import evaluate_topic_mastery

from core.topic_unlock_engine import generate_topic_path
from core.data_access import load_student_logs

# from core.skill_engine import SkillGraph

from core.skill_engine import get_skill_graph



from core.progression_engine import (
    initialize_student_topic_state,
    is_topic_unlocked,
)




from config import (
    PRACTICE_QUESTION_LIMIT,
    TEST_QUESTION_LIMIT,
    MIN_ITEMS_FOR_MASTERY,
    MASTERY_THRESHOLD,
)
from core.data_access import (
    load_questions,
    load_student_logs,
    append_log,
    get_student,
    upsert_student,
)
from core.adaptive_engine import get_next_question
from core.mastery_engine import (
    compute_session_accuracy,
    bloom_breakdown,
    is_topic_mastered,
    suggest_level_change,
)

SESSION_KEEP_KEYS = ["user_id", "name", "role", "intro_done", "awaiting_id"]


# ---------------------------
# SESSION INITIALIZATION
# ---------------------------
def initialize_session_state():
    defaults = {
        "started": False,
        "log": [],
        "bookmarked": set(),
        "score": {},
        "wrong_qs": [],
        "confidence_record": {},
        "topic_mastery": {},
        "topic_mastery_status": {},
        "asked_qs": set(),
        "submitted": False,
        "current_question": None,
        "current_reason": "",
        "review_mode": False,
        "remediation_queue": [],
        "session_id": str(int(time.time())),
        "selected_mode": None,
        "session_done": False,
        "question_count": 0,
        "question_start_time": None,
        "student_view_mode": "Dashboard",
        "ability_score": {},
    }
    
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v



# ---------------------------
# CONFIDENCE MAPPING
# ---------------------------
CONFIDENCE_MAP = {
    "😬 Guessing": 1,
    "🤔 Unsure": 2,
    "😐 Okay": 3,
    "🙂 Confident": 4,
    "😎 Very Confident": 5
}


# ---------------------------
# REPORT CARD PDF
# ---------------------------
def generate_report_card_pdf(student_id: str, student_name: str, df_logs: pd.DataFrame) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, height - 72, "Adaptive Java Tutor - Report Card")

    c.setFont("Helvetica", 12)
    c.drawString(72, height - 100, f"Student ID: {student_id}")
    c.drawString(72, height - 116, f"Name: {student_name or 'N/A'}")

    y = height - 150

    if df_logs.empty:
        c.drawString(72, y, "No activity recorded yet.")
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer.getvalue()

    # Overall accuracy
    total = len(df_logs)
    correct = df_logs["is_correct"].sum()
    overall_acc = (correct / total) * 100

    c.drawString(72, y, f"Total Questions: {total}")
    y -= 16
    c.drawString(72, y, f"Overall Accuracy: {overall_acc:.1f}%")
    y -= 24

    # Per-topic accuracy
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, "Topic Performance:")
    y -= 18
    c.setFont("Helvetica", 11)

    topic_acc = df_logs.groupby("topic")["is_correct"].mean() * 100

    for topic, acc in topic_acc.items():
        if y < 72:
            c.showPage()
            y = height - 72
        c.drawString(90, y, f"- {topic}: {acc:.1f}%")
        y -= 14

    # Bloom performance
    y -= 10
    if y < 72:
        c.showPage()
        y = height - 72
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, "Bloom-Level Breakdown:")
    y -= 18
    c.setFont("Helvetica", 11)

    bloom_acc = df_logs.groupby("bloom_level")["is_correct"].mean() * 100

    for b, acc in bloom_acc.items():
        if y < 72:
            c.showPage()
            y = height - 72
        c.drawString(90, y, f"- {b}: {acc:.1f}%")
        y -= 14

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def generate_advanced_report_card(student_id: str, student_name: str, df_logs: pd.DataFrame) -> bytes:
    import matplotlib.pyplot as plt
    import numpy as np
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from io import BytesIO
    import pandas as pd

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # ============================
    # COVER PAGE
    # ============================
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, height - 72, "Adaptive Java Tutor – Performance Report")

    c.setFont("Helvetica", 12)
    c.drawString(72, height - 110, f"Student ID: {student_id}")
    c.drawString(72, height - 126, f"Name: {student_name}")
    c.drawString(72, height - 142, f"Total Questions Answered: {len(df_logs)}")

    if not df_logs.empty:
        overall_acc = df_logs["is_correct"].mean() * 100
        c.drawString(72, height - 158, f"Overall Accuracy: {overall_acc:.1f}%")

    c.drawString(72, height - 190, "This report provides insights into your mastery, ability growth,")
    c.drawString(72, height - 205, "Bloom-level progression, misconceptions, and learning recommendations.")

    c.showPage()

    # # ============================
    # # 1️⃣ MASTERY RADAR CHART
    # # ============================
    # if not df_logs.empty:
    #     topics = df_logs["topic"].unique()

    #     topic_acc = (
    #         df_logs.groupby("topic")["is_correct"].mean().reindex(topics).fillna(0)
    #     )

    #     labels = topics
    #     values = topic_acc.values
    #     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    #     values = np.concatenate((values, [values[0]]))
    #     angles = np.concatenate((angles, [angles[0]]))

    #     fig = plt.figure(figsize=(6, 6))
    #     ax = fig.add_subplot(111, polar=True)
    #     ax.plot(angles, values, linewidth=2)
    #     ax.fill(angles, values, alpha=0.25)
    #     # ax.set_thetagrids(angles * 180/np.pi, labels)
    #     # Only label the real points, not the duplicated last point
    #     ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)

    #     ax.set_title("Mastery by Topic (Radar Chart)", fontsize=14)

    #     img = BytesIO()
    #     plt.savefig(img, format='png', bbox_inches='tight')
    #     img.seek(0)
    #     plt.close()

    #     c.drawImage(ImageReader(img), 50, 200, width=500, height=500, preserveAspectRatio=True)

    #     c.showPage()


    # ============================
    # 1️⃣ MASTERY RADAR CHART (Robust)
    # ============================
    if not df_logs.empty:
        topics = df_logs["topic"].unique()

        topic_acc = (
            df_logs.groupby("topic")["is_correct"].mean().reindex(topics).fillna(0)
        )

        labels = list(topics)
        N = len(labels)

        if N >= 3:
            # --- Radar chart only when >= 3 axes ---
            values = topic_acc.values.tolist()
            values += [values[0]]  # close loop

            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += [angles[0]]

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, polar=True)

            ax.plot(angles, values, linewidth=2)
            ax.fill(angles, values, alpha=0.25)

            # Label only the real N points
            ax.set_thetagrids([a * 180/np.pi for a in angles[:-1]], labels)
            ax.set_title("Mastery by Topic (Radar Chart)", fontsize=14)

            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plt.close()

            c.drawImage(ImageReader(img), 50, 200, width=500, height=420, preserveAspectRatio=True)
            c.showPage()

    else:
        # --- Fallback: Bar chart ---
        fig, ax = plt.subplots(figsize=(7, 4))
        topic_acc.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Mastery by Topic")
        ax.set_ylabel("Accuracy (%)")

        img = BytesIO()
        plt.savefig(img, format="png", bbox_inches="tight")
        img.seek(0)
        plt.close()

        c.drawImage(ImageReader(img), 40, 250, width=520, height=300)
        c.showPage()


    # ============================
    # 2️⃣ ABILITY SCORE TRAJECTORY
    # ============================
    if "ability_score_history" in st.session_state and st.session_state.ability_score_history:
        history = st.session_state.ability_score_history

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history, marker="o")
        ax.set_title("Ability Score Progression")
        ax.set_xlabel("Session Step")
        ax.set_ylabel("Estimated Ability (θ)")

        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        c.drawImage(ImageReader(img), 40, 200, width=520, height=350)
        c.showPage()

    # ============================
    # 3️⃣ BLOOM PERFORMANCE
    # ============================
    if not df_logs.empty:
        bloom_acc = df_logs.groupby("bloom_level")["is_correct"].mean() * 100

        fig, ax = plt.subplots(figsize=(7, 4))
        bloom_acc.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Accuracy by Bloom Level")
        ax.set_ylabel("Accuracy (%)")

        img = BytesIO()
        plt.savefig(img, format="png", bbox_inches="tight")
        img.seek(0)
        plt.close()

        c.drawImage(ImageReader(img), 40, 250, width=520, height=300)
        c.showPage()

    # ============================
    # 4️⃣ MISCONCEPTIONS ANALYSIS
    # ============================
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Misconception Analysis")

    c.setFont("Helvetica", 11)

    if "misconception_tags" in df_logs.columns:
        tags = df_logs["misconception_tags"].dropna()

        if not tags.empty:
            tag_counts = (
                tags.astype(str)
                .str.split(",")
                .explode()
                .str.strip()
                .value_counts()
                .head(10)
            )

            y = height - 110
            for tag, count in tag_counts.items():
                c.drawString(72, y, f"- {tag}: {count} occurrences")
                y -= 16
        else:
            c.drawString(72, height - 110, "No misconceptions found.")
    else:
        c.drawString(72, height - 110, "No misconception data available.")

    c.showPage()

    # ============================
    # 5️⃣ RECOMMENDATIONS
    # ============================
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Personalized Recommendations")

    c.setFont("Helvetica", 11)

    def write(line, y):
        c.drawString(72, y, line)
        return y - 16

    y = height - 110

    if overall_acc < 60:
        y = write("• Focus on foundational concepts. Your accuracy suggests unstable schema formation.", y)
    if "Analyze" in bloom_acc and bloom_acc["Analyze"] < 50:
        y = write("• Your analysis-level thinking needs reinforcement. Try multi-step tracing problems.", y)
    if "Apply" in bloom_acc and bloom_acc["Apply"] < 50:
        y = write("• Apply-level performance is weak. Practice scenario-based questions.", y)

    y = write("• Review topics with low radar scores—they indicate structural knowledge gaps.", y)
    y = write("• Use Guided Practice to reinforce weak subconcepts.", y)

    c.showPage()

    # FINALIZE PDF
    c.save()
    buffer.seek(0)
    return buffer.getvalue()



# ---------------------------
# SIDEBAR
# ---------------------------
def render_sidebar(topics):
    st.sidebar.title("📚 Adaptive Java Tutor")

    # Topic selector ONLY for topic-based modes
    if st.session_state.selected_mode in ["Guided Practice", "Topic Test"]:
        selected_topic = st.sidebar.selectbox("📘 Choose a Topic", topics)
    else:
        selected_topic = None

    # 🔖 ADD BOOKMARK REVIEW BUTTON HERE
    if st.sidebar.button("🔖 Review Bookmarked"):
        st.session_state.selected_mode = "Bookmark Review"
        st.session_state.started = False
        st.rerun()

    # Navigation Buttons
    if st.sidebar.button("⬅️ Back to Modes"):
        st.session_state.selected_mode = None
        st.session_state.started = False
        st.session_state.current_question = None
        st.rerun()

    if st.sidebar.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            if key not in ["user_id", "name", "role"]:
                del st.session_state[key]
        st.rerun()

    return selected_topic



# ---------------------------
# STUDENT DASHBOARD
# ---------------------------
def render_student_dashboard(df):
    student_id = st.session_state.get("user_id", "unknown")
    student_name = st.session_state.get("name", "")

    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
          <h1 style="margin-bottom: 0.2rem;">🏠 Student Dashboard</h1>
          <p style="color: #555; font-size: 0.95rem;">
            View your progress, mastery levels, and download a report card.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    df_logs = load_student_logs(student_id)

    sg = get_skill_graph()
    skill_stats = sg.stats(df_logs)

    st.markdown("### 🧠 Skill Readiness")
    for skill, s in skill_stats.items():
        st.write(f"**{skill}:** {s['readiness']:.2f} readiness, {s['accuracy']:.1f}% accuracy")



    c1, c2, c3 = st.columns(3)
    if df_logs.empty:
        c1.metric("Total Questions", 0)
        c2.metric("Overall Accuracy", "0.0%")
        c3.metric("Topics Practiced", 0)
        st.info("No activity yet. Start a Practice or Test session.")
        return
    else:
        total = len(df_logs)
        correct = df_logs["is_correct"].sum()
        overall_acc = (correct / total) * 100 if total else 0
        topics = df_logs["topic"].nunique()

        c1.metric("Total Questions", total)
        c2.metric("Overall Accuracy", f"{overall_acc:.1f}%")
        c3.metric("Topics Practiced", topics)

    st.markdown("### 📈 Mastery by Topic")
    topic_acc = df_logs.groupby("topic")["is_correct"].mean() * 100
    st.bar_chart(topic_acc.to_frame("Accuracy (%)"))

    # Progress rings
    st.markdown("### 🎯 Focus Topics")
    top_topics = topic_acc.sort_values(ascending=False).head(3)
    cols = st.columns(len(top_topics) if len(top_topics) else 1)

    for (topic, acc), col in zip(top_topics.items(), cols):
        with col:
            fig, ax = plt.subplots(figsize=(2.5, 2.5))
            ax.pie([acc, 100-acc], startangle=90, counterclock=False, wedgeprops=dict(width=0.3))
            ax.set(aspect="equal")
            ax.set_title(f"{topic}\n{acc:.0f}%")
            st.pyplot(fig)

    # Bloom stats
    st.markdown("### 🧠 Bloom-Level Performance")
    bloom_acc = df_logs.groupby("bloom_level")["is_correct"].mean() * 100
    st.dataframe(bloom_acc.to_frame("Accuracy (%)"))

    # Report card
    st.markdown("### 📝 Download Report Card")
    pdf_bytes = generate_report_card_pdf(student_id, student_name, df_logs)
    st.download_button(
        "📥 Download PDF Report Card",
        pdf_bytes,
        file_name=f"java_tutor_report_{student_id}.pdf",
        mime="application/pdf"
    )





# ---------------------------
# DISPLAY QUESTION (FULL REFACTOR)
# ---------------------------
def display_question(q: pd.Series, topic: str, mode_label: str):
    """
    Clean, robust question rendering for all 4 modes:
    - Guided Practice (adaptive, mastery-focused)
    - Free Practice (exploratory all-topics)
    - Topic Test (summative, topic-restricted)
    - Exam Simulator (summative, mixed topics)
    """

    # ---------------------------
    # Header: Reason for selection
    # ---------------------------
    reason = st.session_state.get("current_reason", "")
    if reason:
        st.info(reason)

    # ---------------------------
    # Question Metadata
    # ---------------------------
    # st.markdown(
    #     f"""
    #     **Bloom:** `{q['bloom_level']}`  
    #     **ID:** `{q['question_id']}`  
    #     **Topic:** `{q['topic']}`  
    #     **Subtopic:** `{q.get('subtopic', 'General')}`
    #     """
    # )

        st.markdown(
        f"""
        **Bloom:** `{q['bloom_level']}`  
        **Topic:** `{q['topic']}`  
        """
    )


    # Main question stem
    st.code(q["question_stem"], language="java")

    # ---------------------------
    # CHOICE SELECTION
    # ---------------------------
    choice_key = f"choice_{q['question_id']}"
    conf_key = f"conf_{q['question_id']}"

    col1, col2 = st.columns(2)

    with col1:
        choice = st.radio(
            "Choose an answer:",
            ["a", "b", "c", "d"],
            index=None,
            format_func=lambda opt: f"{opt.upper()}. {q[f'option_{opt}']}",
            key=choice_key
        )

    # ---------------------------
    # CONFIDENCE SELECTION
    # ---------------------------
    with col2:
        confidence = st.radio(
            "Confidence:",
            list(CONFIDENCE_MAP.keys()),
            index=None,
            key=conf_key
        )

    # ---------------------------
    # BOOKMARK BUTTON (Practice only)
    # ---------------------------
    if mode_label in ["Guided Practice", "Free Practice"]:
        if st.button("🔖 Bookmark", key=f"bm_{q['question_id']}"):
            st.session_state.bookmarked.add(q["question_id"])
            st.success("Bookmarked!")

    # ---------------------------
    # SUBMIT BUTTON
    # ---------------------------
    submit_key = f"submit_{q['question_id']}"
    if st.button("✅ Submit", key=submit_key, use_container_width=True) and not st.session_state.submitted:
        # Validation
        if choice is None:
            st.warning("Please select an answer.")
            return
        if confidence is None:
            st.warning("Please select your confidence level.")
            return

        # Convert confidence → numeric
        numeric_conf = CONFIDENCE_MAP[confidence]

        # Submit & grade
        handle_submission(
            q,
            choice,
            numeric_conf,
            topic,
            mode_label  # <-- full mode name always passed
        )

        st.session_state.submitted = True

    # ---------------------------
    # EXPLANATION & NEXT BUTTON
    # ---------------------------
    if st.session_state.submitted:

        # Explanation shown only in Practice modes
        if mode_label in ["Guided Practice", "Free Practice"]:
            with st.expander("📘 View Explanation"):
                st.markdown(q.get("main_explanation", "_No explanation provided._"))

            with st.expander("📘 More Explanation on incorrect options"):    
                st.markdown(f"**Option A:** {q.get("a_explanation", " ")}")
                st.markdown(f"**Option B:** {q.get("b_explanation", " ")}")
                st.markdown(f"**Option C:** {q.get("c_explanation", " ")}")
                st.markdown(f"**Option D:** {q.get("d_explanation", " ")}")


        # Next question
        if st.button("➡️ Next Question", key=f"next_{q['question_id']}", use_container_width=True):
            st.session_state.submitted = False
            st.session_state.current_question = None
            st.session_state.question_start_time = None
            st.rerun()







# ---------------------------
# HANDLE SUBMISSION
# ---------------------------
def handle_submission(q, choice, confidence, topic, mode_label):
    # Use correct column name from your CSV
    correct = q["correct_answer"].strip().lower()
    is_correct = (choice == correct)
    bloom = q["bloom_level"]

    # Determine behavior type
    practice_like = mode_label in ["Guided Practice", "Free Practice"]
    
    mastery_mode = mode_label == "Guided Practice"  # only this counts for mastery

    if is_correct:
        if practice_like:
            st.success("Correct! 🎉")
        else:
            st.success("Recorded as correct!")
    else:
        if practice_like:
            st.error(f"Incorrect. Correct answer is **{correct.upper()}**.")
        else:
            st.error("Recorded as incorrect!")

    # # Show explanation only for Practice-like modes
    # if practice_like:
    #     with st.expander("📘 View Explanation"):
    #         st.markdown(q.get("main_explanation", "_No explanation provided._"))



    # --- Ability Update (ELO-Like Logistic Update) ---
    ability = st.session_state.ability_score.get(topic, 0.0)
    difficulty = q["predicted_difficulty_level"]

    # normalize predicted difficulty to range approx -2 to +2
    normalized_diff = (difficulty - 3) / 2  

    expected = 1 / (1 + np.exp(-(ability - normalized_diff)))  # logistic prediction
    k = 0.45  # learning rate, can tune later
    outcome = 1 if is_correct else 0

    new_ability = ability + k * (outcome - expected)
    new_ability = max(-2.0, min(2.0, new_ability))  # avoid runaway drift

    st.session_state.ability_score[topic] = new_ability


   

    # Mastery & per-topic tracking ONLY in Guided Practice
    if mastery_mode:
        points = 0.5 + (confidence / 10.0) if is_correct else 0

        st.session_state.topic_mastery.setdefault(topic, {})
        st.session_state.topic_mastery[topic].setdefault(bloom, 0)
        st.session_state.topic_mastery[topic][bloom] += points

        st.session_state.score.setdefault(topic, {})
        st.session_state.score[topic].setdefault(bloom, {"correct": 0, "total": 0})
        st.session_state.score[topic][bloom]["total"] += 1
        st.session_state.score[topic][bloom]["correct"] += int(is_correct)

        st.session_state.confidence_record.setdefault(topic, {})
        st.session_state.confidence_record[topic].setdefault(bloom, [])
        st.session_state.confidence_record[topic][bloom].append({
            "question_id": q["question_id"],
            "confidence": confidence,
            "correct": is_correct
        })

    # # Explanation: only in practice-like modes (Guided + Free)
    # if practice_like:
    #     with st.expander("📘 View Explanation"):
    #         st.markdown(q["main_explanation"])


    # Time
    start = st.session_state.question_start_time
    response_time = time.time() - start if start else None

    # Logging: always log, but record the mode_label
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "student_id": st.session_state.user_id,
        "session_id": st.session_state.session_id,
        "topic": topic,
        "bloom_level": bloom,
        "question_id": q["question_id"],
        "selected": choice,
        "correct_option": correct,
        "is_correct": is_correct,
        "confidence": confidence,
        "mode": mode_label,           # 👈 full mode name goes here
        "response_time_sec": response_time,
        "reinforcement_reason": st.session_state.get("current_reason", ""),
        "misconception_tags": q.get("misconception_tags_per_option", "")
    }
    
    append_log(st.session_state.user_id, log_entry)

    # Keep an in-memory session log for summaries
    if "log" not in st.session_state or st.session_state.log is None:
        st.session_state.log = []
    st.session_state.log.append(log_entry)



# ---------------------------
# SESSION SUMMARY
# ---------------------------
def show_session_summary(topic, mode):
    session_log = [
        l for l in st.session_state.log
        if l["topic"] == topic and l["session_id"] == st.session_state.session_id
    ]

    if not session_log:
        st.info("No questions answered.")
        return

    accuracy = compute_session_accuracy(session_log)

    if mode == "Test":
        st.header("🧪 Test Summary")
        st.metric("Score", f"{accuracy:.1f}%")
    else:
        st.header("📘 Practice Summary")
        st.metric("Accuracy", f"{accuracy:.1f}%")

    bloom_stats = bloom_breakdown(session_log)
    st.subheader("🧠 Bloom Breakdown")

    cols = st.columns(len(bloom_stats))
    for i, (b, stats) in enumerate(bloom_stats.items()):
        pct = (stats["correct"] / stats["total"]) * 100
        with cols[i]:
            st.markdown(f"**{b}**")
            st.markdown(f"{stats['correct']} / {stats['total']} correct")
            st.progress(int(pct))

    incorrect_qs = [l["question_id"] for l in session_log if not l["is_correct"]]

    if mode == "Practice" and incorrect_qs:
        if st.button("🔁 Review Incorrect Questions"):
            st.session_state.review_mode = True
            st.session_state.incorrect_review_queue = incorrect_qs.copy()
            st.session_state.current_question = None
            st.session_state.submitted = False
            st.session_state.session_done = False
            st.session_state.question_count = 0
            st.rerun()

    if is_topic_mastered(session_log):
        st.success(f"🎉 You mastered **{topic}**!")
        st.balloons()
        st.session_state.topic_mastery_status[topic] = "Mastered"
    else:
        st.session_state.topic_mastery_status[topic] = "In progress"

    # Level movement
    if mode == "Practice":
        student = get_student(st.session_state.user_id)
        new_level = suggest_level_change(student.level, accuracy)
        if new_level != student.level:
            st.info(f"Your level has been adjusted to **{new_level}**.")
            student.level = new_level

        upsert_student(student)



    # Unlock next topic if mastered
    df_logs = load_student_logs(st.session_state.user_id)
    mastered = evaluate_topic_mastery(st.session_state.user_id, topic, df_logs)

    if mastered:
        st.success(f"🎉 You mastered {topic}! A new topic is now unlocked.")
        st.balloons()


    if st.button("🔄 Start New Session"):
        reset_session_for_topic(topic)
        st.rerun()




# ---------------------------
# RESET
# ---------------------------
def reset_session_for_topic(topic):
    st.session_state.started = False
    st.session_state.selected_mode = None
    st.session_state.current_question = None
    st.session_state.submitted = False
    st.session_state.review_mode = False
    st.session_state.session_done = False
    st.session_state.question_count = 0
    st.session_state.asked_qs = set()
    st.session_state.current_reason = ""



# --------------------------------------------------
# MODE SELECTION HUB (NEW CLEAN DESIGN)
# --------------------------------------------------
def render_mode_hub():
    st.markdown("""
    <h1>🎓 Choose How You Want to Practice</h1>
    <p style="color:#555; font-size:1.1rem;">
        Select a mode to get started. You can return here anytime using
        <b>"Back to Modes"</b>.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5 = st.columns(1)[0]

    with col1:
        if st.button("🧭 Guided Practice", use_container_width=True):
            st.session_state.selected_mode = "Guided Practice"
            st.session_state.started = False
            st.rerun()

    with col2:
        if st.button("🧪 Topic Test", use_container_width=True):
            st.session_state.selected_mode = "Topic Test"
            st.session_state.started = False
            st.rerun()

    with col3:
        if st.button("🎲 Free Practice (Mixed)", use_container_width=True):
            st.session_state.selected_mode = "Free Practice"
            st.session_state.started = False
            st.rerun()

    with col4:
        if st.button("📝 Exam Simulator", use_container_width=True):
            st.session_state.selected_mode = "Exam Simulator"
            st.session_state.started = False
            st.rerun()

    with col5:
        if st.button("📊 Performance Dashboard", use_container_width=True):
            st.session_state.selected_mode = "Performance Dashboard"
            st.session_state.started = False
            st.rerun()


# # ---------------------------
# # MODE HUB (CHOOSE LEARNING MODE)
# # ---------------------------
# def render_mode_hub():
#     st.markdown(
#         """
#         <h1>🎓 Choose How You Want to Practice</h1>
#         <p style="color:#555;">
#         Select a mode to get started. You can always return here using "Back to Main Menu".
#         </p>
#         """,
#         unsafe_allow_html=True
#     )

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("🧭 Guided Practice")
#         st.write(
#             "- Topic-focused\n"
#             "- Adaptive by Bloom & difficulty\n"
#             "- Contributes to mastery & readiness"
#         )
#         if st.button("Start Guided Practice", key="mode_guided"):
#             _select_mode("Guided Practice")

#     with col2:
#         st.subheader("🧪 Topic Test")
#         st.write(
#             "- Topic-specific test\n"
#             "- Limited feedback during test\n"
#             "- Great for checking mastery"
#         )
#         if st.button("Start Topic Test", key="mode_test"):
#             _select_mode("Topic Test")

#     col3, col4 = st.columns(2)

#     with col3:
#         st.subheader("🎲 Free Practice (Mixed)")
#         st.write(
#             "- Any topic, any time\n"
#             "- Random questions at your level\n"
#             "- Does NOT affect mastery unlocking"
#         )
#         if st.button("Start Free Practice", key="mode_free"):
#             _select_mode("Free Practice")

#     with col4:
#         st.subheader("📝 Exam Simulator")
#         st.write(
#             "- Mixed topics\n"
#             "- Test-like conditions\n"
#             "- No explanations until the end"
#         )
#         if st.button("Start Exam Simulator", key="mode_exam"):
#             _select_mode("Exam Simulator")


# --------------------------------------------------
# PERFORMANCE DASHBOARD (NEW)
# --------------------------------------------------
def render_performance_dashboard(df):
    student_id = st.session_state.get("user_id", "")
    student_name = st.session_state.get("name", "")

    df_logs = load_student_logs(student_id)




    st.markdown("""
    <h1>📊 Performance Dashboard</h1>
    <p style="color:#555; font-size:1.1rem;">
        View your mastery, readiness score, misconceptions, and learning path.
    </p>
    """, unsafe_allow_html=True)

    if df_logs.empty:
        st.info("No activity yet. Complete some practice to unlock insights.")
        return

    # ---- METRICS --------------------------------
    total = len(df_logs)
    correct = df_logs["is_correct"].sum()
    overall_acc = (correct / total) * 100 if total else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Questions", total)
    c2.metric("Overall Accuracy", f"{overall_acc:.1f}%")
    c3.metric("Topics Practiced", df_logs["topic"].nunique())

    # ---- TOPIC MASTERY ---------------------------
    st.markdown("### 🎯 Topic Mastery")
    topic_acc = df_logs.groupby("topic")["is_correct"].mean() * 100
    st.bar_chart(topic_acc.to_frame("Accuracy (%)"))

    # ---- BLOOM PERFORMANCE -----------------------
    st.markdown("### 🧠 Bloom-Level Mastery")
    bloom_acc = df_logs.groupby("bloom_level")["is_correct"].mean() * 100
    st.bar_chart(bloom_acc.to_frame("Accuracy (%)"))

    # ---- MISCONCEPTIONS --------------------------
    st.markdown("### ❗ Common Misconceptions")
    wrong_df = df_logs[df_logs["is_correct"] == False]
    if wrong_df.empty:
        st.success("No major misconceptions identified — great work!")
    else:
        st.write(
            wrong_df.groupby("topic")["question_id"].count().to_frame("Incorrect Items")
        )

    # ---- READINESS SCORE -------------------------
    st.markdown("### 🔥 Readiness Score (Prototype)")
    readiness = overall_acc * 0.6 + bloom_acc.mean() * 0.4
    st.metric("Readiness Score", f"{readiness:.1f}")

    # ---- LEARNING PATH / NEXT TOPIC --------------
    st.markdown("### 🧭 Recommended Next Topics")

    weakest_topics = topic_acc.sort_values().head(3)
    for topic, acc in weakest_topics.items():
        st.write(f"- **{topic}** ({acc:.1f}% accuracy)")

    pdf_bytes = generate_report_card_pdf(student_id, student_name, df_logs)

    st.download_button(
        "📥 Download Report Card (PDF)",
        data=pdf_bytes,
        file_name=f"report_card_{student_id}.pdf",   # ✅ correct Streamlit parameter
        mime="application/pdf"
    )


    pdf_bytes = generate_advanced_report_card(student_id, student_name, df_logs)

    st.download_button(
        "📥 Download Advanced Report Card (PDF)",
        pdf_bytes,
        file_name=f"report_card{student_id}.pdf",
        mime="application/pdf"
    )



    # ---- RETURN BUTTON ---------------------------
    if st.button("⬅️ Back to Modes", use_container_width=True):
        st.session_state.selected_mode = None
        st.rerun()




def _select_mode(mode_label: str):
    # Reset session for a fresh run in new mode
    for key in list(st.session_state.keys()):
        if key not in SESSION_KEEP_KEYS:
            del st.session_state[key]
    st.session_state.selected_mode = mode_label
    st.session_state.started = False
    st.session_state.session_done = False
    st.session_state.question_count = 0
    st.session_state.student_view_mode = "Practice/Test"
    st.rerun()


def reset_question_state():
    st.session_state.current_question = None
    st.session_state.submitted = False
    st.session_state.review_mode = False
    st.session_state.session_done = False
    st.session_state.question_count = 0
    st.session_state.current_reason = ""
    st.session_state.asked_qs = set()
    st.session_state.incorrect_review_queue = []
    # ✅ new session id
    st.session_state.session_id = str(int(time.time()))
    # optional: clear in-memory session log just for fresh summaries
    st.session_state.log = []


def show_topic_mastery_snapshot(topic):
    st.markdown("#### 📈 Mastery Snapshot")

    topic_score = st.session_state.score.get(topic, {})
    if not topic_score:
        st.progress(0)
        return

    total_attempted = sum(v["total"] for v in topic_score.values())
    total_correct = sum(v["correct"] for v in topic_score.values())
    pct = int((total_correct / total_attempted * 100) if total_attempted else 0)
    st.progress(pct)


# ---------------------------
# MAIN STUDENT VIEW (FULL REFACTOR)
# ---------------------------
def render_student_view(df):
    from core.skill_engine import get_skill_graph

    # -----------------------------------------------------
    # 0. Load student + basic session state
    # -----------------------------------------------------
    student = get_student(st.session_state.user_id)
    if not student:
        st.error("Student not found.")
        return

    st.session_state.student_level = student.level
    st.session_state.current_bloom = student.current_bloom

    # ---------------------------------------
    # 📘 SKILL GRAPH ACCESS (Correct Location)
    # ---------------------------------------
    from core.skill_engine import get_skill_graph

    sg = get_skill_graph()
    df_logs = load_student_logs(st.session_state.user_id)
    skill_stats = sg.stats(df_logs)     # readiness + mastery per skill
    st.session_state.skill_stats = skill_stats


    # -----------------------------------------------------
    # 1. Topics in CSV (sorted, unique)
    # -----------------------------------------------------
    topics = sorted(df["topic"].dropna().unique().tolist())

    # Sidebar returns a single selected_topic
    selected_topic = render_sidebar(topics)

    # Ensure ability score dictionary exists
    if "ability_score" not in st.session_state:
        st.session_state.ability_score = {}

    # Init ability for this topic
    if selected_topic not in st.session_state.ability_score:
        st.session_state.ability_score[selected_topic] = 0.0

    # -----------------------------------------------------
    # 2. MODE ROUTING LOGIC
    # -----------------------------------------------------
    mode = st.session_state.get("selected_mode", None)

    # If user hasn't picked a mode yet → show hub
    if mode is None:
        render_mode_hub()
        return

    # ---- Mode: Performance Dashboard ---------------------------------------
    if mode == "Performance Dashboard":
        render_performance_dashboard(df)
        return

    # ---- Mode: Bookmark Review ---------------------------------------------
    if mode == "Bookmark Review":
        render_bookmark_review(df)
        return

    # -----------------------------------------------------
    # 3. Mode labels + behavior type (practice/test)
    # -----------------------------------------------------
    if mode == "Guided Practice":
        mode_label = "Guided Practice"
        behavior_type = "Practice"
        limit = PRACTICE_QUESTION_LIMIT

    elif mode == "Topic Test":
        mode_label = "Topic Test"
        behavior_type = "Test"
        limit = TEST_QUESTION_LIMIT

    elif mode == "Free Practice":
        mode_label = "Free Practice"
        behavior_type = "Practice"
        limit = PRACTICE_QUESTION_LIMIT

    else:  # Exam Simulator
        mode_label = "Exam Simulator"
        behavior_type = "Test"
        limit = TEST_QUESTION_LIMIT

    # -----------------------------------------------------
    # 4. Display header
    # -----------------------------------------------------
    st.header(f"🎓 {mode_label}")
    st.write("Practice Java concepts with adaptive questions and mastery.")

    # Topic display logic
    if mode in ["Guided Practice", "Topic Test"]:
        st.markdown(f"### 📘 Topic: {selected_topic}")
    else:
        st.markdown("### 📘 Topic: Mixed (All Topics)")

    # -----------------------------------------------------
    # 5. Mastery Snapshot (topic-only modes)
    # -----------------------------------------------------
    if mode in ["Guided Practice", "Topic Test"]:
        st.markdown("#### 📈 Mastery Snapshot")
        topic_score = st.session_state.score.get(selected_topic, {})
        total_attempted = sum(v["total"] for v in topic_score.values()) if topic_score else 0
        total_correct = sum(v["correct"] for v in topic_score.values()) if topic_score else 0
        pct = (total_correct / total_attempted * 100) if total_attempted else 0
        st.progress(int(pct))

    # -----------------------------------------------------
    # 6. SESSION START LOGIC
    # -----------------------------------------------------
    if not st.session_state.started:
        if st.button(
            {
                "Guided Practice": "🚀 Start Guided Practice",
                "Topic Test": "🧪 Start Topic Test",
                "Free Practice": "🎲 Start Free Practice",
                "Exam Simulator": "📝 Start Exam Simulator",
            }[mode_label]
        ):
            st.session_state.started = True
            st.session_state.session_done = False
            st.session_state.current_question = None
            st.session_state.submitted = False
            st.session_state.question_count = 0
            st.session_state.session_id = str(int(time.time()))
            st.rerun()
        return

    # -----------------------------------------------------
    # 7. SESSION DONE SCREEN
    # -----------------------------------------------------
    if st.session_state.session_done:
        logical_topic = selected_topic if mode in ["Guided Practice", "Topic Test"] else "Mixed"
        show_session_summary(logical_topic, behavior_type)
        return

    # -----------------------------------------------------
    # 8. SELECT QUESTION POOL
    # -----------------------------------------------------
    if mode in ["Guided Practice", "Topic Test"]:
        topic_df = df[df["topic"] == selected_topic]
    else:
        topic_df = df  # Free Practice & Exam Simulator → all topics

    # -----------------------------------------------------
    # 9. QUESTION SELECTION
    # -----------------------------------------------------
    if st.session_state.current_question is None and not st.session_state.submitted:

        q = None
        reason = ""

        # ---- Review incorrect questions (Practice only)
        if (
            st.session_state.review_mode
            and st.session_state.get("incorrect_review_queue")
        ):
            next_qid = st.session_state.incorrect_review_queue.pop(0)
            row = df[df["question_id"] == next_qid]
            if not row.empty:
                q = row.iloc[0].to_dict()
                reason = "🔄 Reviewing incorrect question"

        else:
            # ---- Normal adaptive selection
            q_series, reason = get_next_question(topic_df, selected_topic)
            if q_series is None:
                st.session_state.session_done = True
                st.rerun()

            q = q_series.to_dict()

        # ---- Ensure question has topic assigned
        if "topic" not in q or pd.isna(q["topic"]):
            if mode in ["Free Practice", "Exam Simulator"]:
                # get real topic from DF
                found = df[df["question_id"] == q["question_id"]]
                if not found.empty:
                    q["topic"] = found.iloc[0]["topic"]
            else:
                q["topic"] = selected_topic

        # Store question in session
        st.session_state.current_question = q
        st.session_state.current_reason = reason
        st.session_state.current_bloom = q["bloom_level"]
        st.session_state.asked_qs.add(q["question_id"])
        st.session_state.question_start_time = time.time()

    # -----------------------------------------------------
    # 10. DISPLAY QUESTION
    # -----------------------------------------------------
    question_series = pd.Series(st.session_state.current_question)
    logical_topic = (
        selected_topic if mode in ["Guided Practice", "Topic Test"] else question_series["topic"]
    )

    display_question(
        question_series,
        logical_topic,
        mode_label
    )

    # -----------------------------------------------------
    # 11. NEXT QUESTION LOGIC
    # -----------------------------------------------------
    if st.session_state.submitted:
        st.session_state.question_count += 1

        if st.session_state.question_count >= limit and not st.session_state.review_mode:
            st.session_state.session_done = True
            st.rerun()



def mode_allows_bookmarking(mode: str) -> bool:
    return mode in ["Guided Practice", "Free Practice"]

# ---------------------------
# PUBLIC ENTRY POINT
# ---------------------------
def run_student_mode():
    initialize_session_state()

    df = load_questions()
    if df.empty:
        st.error("No questions available.")
        return

    # 🔹 If no mode is chosen yet, show the mode hub
    if st.session_state.selected_mode is None:
        render_mode_hub()
        return

    render_student_view(df)


# ===================================================
# ADVANCED BOOKMARK REVIEW SYSTEM (NEW)
# ===================================================
def render_bookmark_review(df):
    st.markdown("""
    <h1>🔖 Bookmarked Questions</h1>
    <p style="color:#555; font-size:1.05rem;">
        Review tricky or important questions you've saved.  
        Use filters to focus on specific topics, Bloom levels, or difficulty bands.
    </p>
    """, unsafe_allow_html=True)

    student_id = st.session_state.user_id
    df_logs = load_student_logs(student_id)

    # No bookmarks? Stop early.
    if not st.session_state.bookmarked:
        st.info("You haven't bookmarked any questions yet.")
        return

    # Retrieve bookmarked questions
    bm_df = df[df["question_id"].isin(st.session_state.bookmarked)].copy()
    if bm_df.empty:
        st.warning("Bookmarks exist, but these questions aren't in the dataset.")
        return

    # ------------------------------
    # FILTER PANEL
    # ------------------------------
    st.markdown("### 🎛 Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        topic_filter = st.selectbox(
            "Filter by Topic",
            ["All"] + sorted(bm_df["topic"].unique().tolist())
        )

    with col2:
        bloom_filter = st.selectbox(
            "Filter by Bloom Level",
            ["All"] + sorted(bm_df["bloom_level"].unique().tolist())
        )

    with col3:
        diff_filter = st.selectbox(
            "Predicted Difficulty",
            ["All", "Easy", "Medium", "Hard"]
        )

    # Apply filters
    if topic_filter != "All":
        bm_df = bm_df[bm_df["topic"] == topic_filter]

    if bloom_filter != "All":
        bm_df = bm_df[bm_df["bloom_level"] == bloom_filter]

    if diff_filter != "All":
        label_map = {"Easy": 1, "Medium": 2, "Hard": 3}
        target = label_map[diff_filter]
        bm_df = bm_df[bm_df["predicted_difficulty_label"] == target]

    st.markdown("---")

    # ------------------------------
    # BOOKMARK SUMMARY METRICS
    # ------------------------------
    st.markdown("### 📊 Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Bookmarked", len(bm_df))
    c2.metric("Unique Topics", bm_df["topic"].nunique())
    c3.metric("Avg Difficulty", f"{bm_df['predicted_difficulty_level'].mean():.1f}")

    st.markdown("---")

    # ------------------------------
    # RENDER EACH BOOKMARK AS A CARD
    # ------------------------------
    for _, row in bm_df.iterrows():
        qid = row["question_id"]
        topic = row["topic"]
        bloom = row["bloom_level"]
        stem = row["question_stem"]
        diff = row["predicted_difficulty_label"]

        # Pull past attempts on this question
        q_logs = df_logs[df_logs["question_id"] == qid]

        with st.container():
            st.markdown(f"""
            <div style="padding:1rem; border:1px solid #DDD; border-radius:10px; margin-bottom:1rem;">
                <h3>📝 {row['topic']} — <span style='font-size:0.9rem;'>Bloom: {bloom}</span></h3>
                <pre style="white-space:pre-wrap; background:white; padding:0.7rem;">{stem}</pre>
            """, unsafe_allow_html=True)

            # Past attempt summary
            if not q_logs.empty:
                latest = q_logs.iloc[-1]
                correctness = "✔ Correct" if latest["is_correct"] else "✖ Incorrect"
                color = "green" if latest["is_correct"] else "red"
                st.markdown(f"**Last Attempt:** <span style='color:{color};'>{correctness}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence History:** {list(q_logs['confidence'])}")
            else:
                st.markdown("*No attempts yet for this bookmarked question.*")

            st.markdown(f"**Predicted Difficulty:** {diff}")

            # ------- ACTION BUTTONS -------
            colA, colB, colC = st.columns(3)

            with colA:
                if st.button(f"🧠 Practice Similar", key=f"similar_btn_{qid}"):
                    # You may later replace with AI similarity matching
                    st.session_state.remediation_queue = df[
                        (df["topic"] == topic) &
                        (df["bloom_level"] == bloom) &
                        (df["question_id"] != qid)
                    ].sample(min(3, len(df))).to_dict("records")
                    st.success("Added similar questions to your practice queue.")
            
            with colB:
                if st.button(f"🗑 Remove ({qid})"):
                    st.session_state.bookmarked.remove(qid)
                    st.rerun()

            with colC:
                if st.button(f"📘 Show Explanation ({qid})"):
                    st.info(row["main_explanation"])

            st.markdown("</div>", unsafe_allow_html=True)
