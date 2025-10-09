import streamlit as st
import pandas as pd
import plotly.express as px
import random
import os

# Setup
st.set_page_config(page_title="Bloom RL Simulator", layout="wide")
st.title("📚 Adaptive RL Simulation with Bloom’s Taxonomy")

# Bloom levels mapping
bloom_levels = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
bloom_to_num = {b: i+1 for i, b in enumerate(bloom_levels)}
num_to_bloom = {v: k for k, v in bloom_to_num.items()}

# Sidebar - Upload + Controls
st.sidebar.header("📁 Upload Java Question Bank")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.header("🎛️ Simulation Controls")
start_level = st.sidebar.selectbox("Start Proficiency", bloom_levels, index=1)
num_rounds = st.sidebar.slider("Rounds", min_value=10, max_value=50, value=30)

simulate_button = st.sidebar.button("🚀 Run Adaptive Simulation")

# Adaptive RL logic
def run_adaptive_rl_simulation(df, start_level="Understand", total_rounds=30):
    df["bloom_numeric"] = df["bloom_level"].map(bloom_to_num)
    history = []
    proficiency = bloom_to_num[start_level]
    correct_streak = 0
    fail_streak = 0

    for round_num in range(1, total_rounds + 1):
        pool = df[df["bloom_numeric"].between(proficiency - 1, proficiency + 1)]
        if pool.empty:
            pool = df

        q = pool.sample(1).iloc[0]
        chosen_level = q["bloom_numeric"]
        chosen_bloom = q["bloom_level"]
        qid = q["question_id"]
        qstem = q.get("question_stem", "")[:120]

        success_prob = 0.9 if chosen_level < proficiency else 0.7 if chosen_level == proficiency else 0.3
        correct = random.random() < success_prob

        reward = 10 if correct else -5
        delta = 0

        if correct:
            correct_streak += 1
            fail_streak = 0
            if chosen_level > proficiency:
                reward += 5
            if correct_streak >= 3:
                delta = 2
        else:
            correct_streak = 0
            fail_streak += 1
            if chosen_level < proficiency:
                reward -= 5
            if fail_streak >= 2:
                delta = -2

        proficiency = max(1, min(6, proficiency + (1 if delta > 0 else -1 if delta < 0 else 0)))

        history.append({
            "Round": round_num,
            "Question_ID": qid,
            "Bloom_Level": chosen_bloom,
            "Correct": int(correct),
            "Reward": reward,
            "Updated_Proficiency": proficiency,
            "Updated_Bloom": num_to_bloom[proficiency],
            "Question_Snippet": qstem,
            "Confusion": chosen_level - proficiency
        })

    return pd.DataFrame(history)

# Run simulation if file uploaded

if uploaded_file is not None:
    try:
        # Auto-clean column headers
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Validation: Required columns
        if 'question_id' not in df.columns or 'bloom_level' not in df.columns:
            st.error("❌ CSV must contain `question_id` and `bloom_level` columns.")
            st.stop()

        st.success("✅ File loaded and cleaned successfully.")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()

# if uploaded_file:
#     # df = pd.read_csv(uploaded_file)
#     try:
#         df = pd.read_csv(uploaded_file, encoding="utf-8")
#     except UnicodeDecodeError:
#         df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")  # fallback


    required_cols = {"question_id", "bloom_level"}
    if not required_cols.issubset(df.columns):
        st.error("CSV must contain `question_id`, `bloom_level` columns.")
    else:
        if simulate_button:
            sim_df = run_adaptive_rl_simulation(df, start_level, num_rounds)
            st.session_state.sim_df = sim_df

# Display if simulation ran
if "sim_df" in st.session_state:
    sim_df = st.session_state.sim_df

    st.success("✅ Simulation complete!")

    # Show raw results
    st.subheader("📋 Simulation Log")
    st.dataframe(sim_df)

    # Question snippets
    st.subheader("📌 Sample Question Snippets")
    for _, row in sim_df.head(10).iterrows():
        st.markdown(f"**Q{int(row['Question_ID'])}** ({row['Bloom_Level']}): {row['Question_Snippet']}")

    # Line chart: Bloom proficiency
    st.subheader("📈 Bloom Proficiency Progression")
    fig1 = px.line(sim_df, x="Round", y="Updated_Proficiency", markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    # Reward chart
    st.subheader("💰 Reward per Round")
    fig2 = px.bar(sim_df, x="Round", y="Reward", color="Correct", title="Reward Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    # Accuracy by Bloom Level
    st.subheader("🎯 Accuracy by Bloom Level")
    acc_df = sim_df.groupby("Bloom_Level")["Correct"].mean().reset_index()
    fig3 = px.bar(acc_df, x="Bloom_Level", y="Correct", title="Correct Answer Rate")
    st.plotly_chart(fig3, use_container_width=True)

    # 🔥 Confusion Heatmap
    st.subheader("🔥 Bloom Confusion Heatmap")
    sim_df["Proficiency_Bloom"] = sim_df["Updated_Bloom"]
    heatmap_df = sim_df.groupby(["Proficiency_Bloom", "Bloom_Level"]).size().reset_index(name="count")
    fig4 = px.density_heatmap(
        heatmap_df, x="Bloom_Level", y="Proficiency_Bloom", z="count",
        color_continuous_scale="Reds", title="Misalignment Between Chosen vs Actual Bloom Level"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # 📊 Question difficulty histogram
    st.subheader("📊 Question Difficulty Distribution")
    q_dist = df["bloom_level"].value_counts().reset_index()
    q_dist.columns = ["Bloom_Level", "Count"]
    fig5 = px.bar(q_dist, x="Bloom_Level", y="Count", title="Available Questions per Bloom Level")
    st.plotly_chart(fig5, use_container_width=True)

    # Export
    st.download_button("📥 Download Simulation CSV", sim_df.to_csv(index=False), "RL_simulation_output.csv", "text/csv")

else:
    st.info("📤 Upload a valid CSV and click 'Run Simulation' to begin.")
