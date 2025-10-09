import streamlit as st
import pandas as pd
import random
import datetime
import plotly.express as px

st.set_page_config(page_title="AI Adaptive Learning", layout="wide")
st.title("🎓 AI for Inclusive Education: Adaptive Learning with Bloom's Taxonomy")

# ---- INIT SESSION ----
if "view" not in st.session_state:
    st.session_state.view = "Student"

# ---- SIDEBAR SWITCH ----
with st.sidebar:
    st.markdown("## 🔁 Choose Mode")
    view = st.radio("Select View", ["Student", "Teacher"])
    st.session_state.view = view

# ---- DATA ----
@st.cache_data
def load_questions():
    df = pd.read_csv("data/java_question_bank_utf8.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

df_q = load_questions()
bloom_levels = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
bloom_to_num = {b: i + 1 for i, b in enumerate(bloom_levels)}

def get_reward(bloom_level, correct):
    base = 0.3 + bloom_to_num.get(bloom_level.lower(), 3) * 0.05
    return base + 0.5 if correct else base - 0.2

# ---- STUDENT VIEW ----
if st.session_state.view == "Student":
    st.header("👩‍🎓 Student Mode")

    if "student_id" not in st.session_state:
        st.session_state.student_id = st.text_input("Enter your Student ID to begin:", key="student_id_input")
        st.stop()

    if "proficiency" not in st.session_state:
        st.session_state.proficiency = {level: 0.5 for level in df_q["bloom_level"].unique()}
        st.session_state.history = []
        st.session_state.current_q = None
        st.session_state.submitted = False

    
    with st.sidebar:
        st.markdown("## 🧠 Your Progress")
        avg_mastery = int(100 * sum(st.session_state.proficiency.values()) / len(st.session_state.proficiency))
        st.progress(avg_mastery)

        if st.session_state.history:
            df_hist = pd.DataFrame(st.session_state.history)
            st.download_button("📥 Download Session Log", df_hist.to_csv(index=False),
                               file_name=f"{st.session_state.student_id}_log.csv")

        mastered = [lvl for lvl, score in st.session_state.proficiency.items() if score >= 0.75]
        st.markdown(f"🎯 Mastered Levels: **{len(mastered)} / {len(bloom_levels)}**")
        if len(mastered) == len(bloom_levels):
            st.balloons()
            st.success("🏆 You’ve mastered all levels!")

    def select_question():
        weakest = min(st.session_state.proficiency, key=st.session_state.proficiency.get)
        candidates = df_q[df_q["bloom_level"].str.lower() == weakest.lower()]
        return candidates.sample(1).iloc[0] if not candidates.empty else df_q.sample(1).iloc[0]

    if st.session_state.current_q is None or st.session_state.submitted:
        st.session_state.current_q = select_question()
        st.session_state.submitted = False

    q = st.session_state.current_q
    st.markdown(f"### 📝 {q['question_stem']}")
    st.caption(f"Bloom Level: `{q['bloom_level']}` | ID: `{q['question_id']}`")

    option_labels = []
    for letter in ['a', 'b', 'c', 'd']:
        opt_col = f"option_{letter}"
        if pd.notnull(q.get(opt_col)):
            option_labels.append((letter.upper(), q[opt_col]))

    selected_option = st.radio("Choose your answer:",
                               [f"{opt[0]}. {opt[1]}" for opt in option_labels], key="mcq")

    if st.button("✅ Submit Answer"):
        selected_letter = selected_option.split(".")[0].strip()
        correct_letter = str(q["correct_option"]).strip().upper()
        bloom = q["bloom_level"].strip()
        correct = int(selected_letter == correct_letter)
        reward = get_reward(bloom.lower(), correct)

        # Update proficiency
        st.session_state.proficiency[bloom] += (reward - 0.5) * 0.2
        st.session_state.proficiency[bloom] = max(0.0, min(1.0, st.session_state.proficiency[bloom]))

        # Log performance
        st.session_state.history.append({
            "student_id": st.session_state.student_id,
            "timestamp": datetime.datetime.now(),
            "question_id": q["question_id"],
            "bloom_level": bloom,
            "student_answer": selected_letter,
            "correct_answer": correct_letter,
            "correct": correct,
            "reward": round(reward, 2)
        })

        if correct:
            st.success(f"✅ Correct! You selected **{selected_letter}**.")
        else:
            st.error(f"❌ Incorrect. You chose **{selected_letter}**, correct is **{correct_letter}**.")
            if "explanation" in q and pd.notnull(q["explanation"]):
                st.info(f"💡 Explanation: {q['explanation']}")

        st.session_state.submitted = True
        st.rerun()

    if st.session_state.history:
        st.subheader("📋 Recent Activity")
        df_hist = pd.DataFrame(st.session_state.history[-5:])
        st.dataframe(df_hist)

        st.subheader("📊 Bloom Level Mastery")
        st.bar_chart(pd.Series(st.session_state.proficiency))

# ---- TEACHER VIEW ----
else:
    st.header("👨‍🏫 Teacher Dashboard")

    uploaded = st.file_uploader("📤 Upload Student Log (.csv)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip().str.lower()

        if 'student_id' in df.columns:
            student = st.selectbox("🎓 Choose Student", df["student_id"].unique())
            s_df = df[df["student_id"] == student]

            st.metric("Total Questions", len(s_df))
            st.metric("Accuracy", f"{100 * s_df['correct'].mean():.1f}%")

            bloom_perf = s_df.groupby("bloom_level")["correct"].mean().reset_index()
            fig = px.bar(bloom_perf, x="bloom_level", y="correct", title="Accuracy by Bloom Level",
                         labels={"correct": "Accuracy"}, color="bloom_level")
            st.plotly_chart(fig, use_container_width=True)

            if "timestamp" in s_df.columns:
                s_df["timestamp"] = pd.to_datetime(s_df["timestamp"])
                timeline = s_df.groupby("timestamp")["correct"].mean().reset_index()
                fig2 = px.line(timeline, x="timestamp", y="correct", title="Accuracy Over Time")
                st.plotly_chart(fig2, use_container_width=True)

            with st.expander("📄 View Raw Log"):
                st.dataframe(s_df)
        else:
            st.warning("Missing `student_id` column in uploaded file.")
    else:
        st.info("Please upload a session log from a student.")
