
# ui/teacher.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from core.data_access import load_all_logs

from io import BytesIO

from core.data_access import load_questions, load_all_logs

from config import SKILL_GRAPH_JSON
from core.skill_engine import get_skill_graph

from core.data_access import get_student




# ---------------------------
# HELPERS
# ---------------------------



def require_teacher():
    user = get_student(st.session_state.user_id)

    if not user:
        st.error("Not logged in.")
        st.stop()

    if user.role != "teacher":
        st.error("⛔ Access denied: Teacher account required.")
        st.stop()


def _export_csv_button(df: pd.DataFrame, file_name: str, label: str):
    if df.empty:
        st.info("No data to export.")
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label,
        data=csv,
        file_name=file_name,   # ✅ new Streamlit arg
        mime="text/csv",
    )


def _estimate_ability_from_logs(df_logs: pd.DataFrame, df_q: pd.DataFrame) -> pd.DataFrame:
    """
    Offline ability estimation per student/topic using the same
    logistic update idea you use in the student app.
    """
    if df_logs.empty:
        return pd.DataFrame(columns=["student_id", "topic", "theta"])

    # Merge difficulty into logs
    df = df_logs.merge(
        df_q[["question_id", "predicted_difficulty_level"]],
        on="question_id",
        how="left"
    )

    abilities = []

    for (student_id, topic), group in df.groupby(["student_id", "topic"]):
        # sort by time to simulate chronological learning
        group = group.sort_values("timestamp")

        ability = 0.0
        for _, row in group.iterrows():
            diff = row.get("predicted_difficulty_level", 3)
            if pd.isna(diff):
                diff = 3
            normalized_diff = (diff - 3) / 2.0

            is_correct = bool(row.get("is_correct", False))
            outcome = 1 if is_correct else 0

            # logistic expected probability
            expected = 1 / (1 + np.exp(-(ability - normalized_diff)))
            k = 0.45
            ability = ability + k * (outcome - expected)

            # clamp
            ability = max(-2.0, min(2.0, ability))

        abilities.append(
            {"student_id": student_id, "topic": topic, "theta": ability}
        )

    return pd.DataFrame(abilities)


def _plot_heatmap_topic_bloom(df_logs: pd.DataFrame):
    """
    Class-level heatmap: topic × Bloom level → accuracy.
    """
    if df_logs.empty:
        st.info("No logs available for heatmap.")
        return

    pivot = (
        df_logs
        .groupby(["topic", "bloom_level"])["is_correct"]
        .mean()
        .unstack(fill_value=0.0)
    )

    st.markdown("#### 🔥 Class Mastery Heatmap (Topic × Bloom)")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j] * 100
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color="white" if val < 50 else "black")

    ax.set_xlabel("Bloom Level")
    ax.set_ylabel("Topic")
    fig.colorbar(im, ax=ax, label="Accuracy")

    st.pyplot(fig)


def _show_mode_metrics(df: pd.DataFrame, mode_label: str):
    st.subheader(f"📊 {mode_label} Analytics")

    if df.empty:
        st.info(f"No data yet for **{mode_label}**.")
        return

    total = len(df)
    acc = df["is_correct"].mean() * 100
    avg_conf = df["confidence"].mean() if "confidence" in df.columns else np.nan
    avg_time = df["response_time_sec"].mean() if "response_time_sec" in df.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Responses", total)
    c2.metric("Accuracy", f"{acc:.1f}%")
    c3.metric("Avg Confidence", f"{avg_conf:.2f}" if not np.isnan(avg_conf) else "N/A")
    c4.metric("Avg Time (s)", f"{avg_time:.1f}" if not np.isnan(avg_time) else "N/A")

    # Topic accuracy
    st.markdown("**Accuracy by Topic (%)**")
    topic_acc = df.groupby("topic")["is_correct"].mean().sort_values(ascending=False) * 100
    st.bar_chart(topic_acc)

    # Bloom accuracy
    st.markdown("**Accuracy by Bloom Level (%)**")
    bloom_acc = df.groupby("bloom_level")["is_correct"].mean().sort_values(ascending=False) * 100
    st.bar_chart(bloom_acc)


def _show_misconception_clusters(df_logs: pd.DataFrame):
    st.subheader("🧠 Misconception Clusters")

    if "misconception_tags" not in df_logs.columns:
        st.info("No misconception tag data available.")
        return

    tags = df_logs["misconception_tags"].dropna()
    if tags.empty:
        st.info("No misconceptions logged yet.")
        return

    exploded = (
        tags.astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )
    exploded = exploded[exploded != ""]

    if exploded.empty:
        st.info("No valid misconception tags.")
        return

    counts = exploded.value_counts().head(20)

    st.markdown("**Top Misconceptions (All Modes)**")
    st.dataframe(counts.rename("count").to_frame())

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.sort_values(ascending=True).plot(kind="barh", ax=ax)
    ax.set_xlabel("Occurrences")
    ax.set_title("Top Misconception Tags")
    st.pyplot(fig)


def _show_skill_readiness_view(df_logs: pd.DataFrame, df_q: pd.DataFrame):
    st.subheader("🧭 Skill Readiness (via Skill Graph)")

    try:
        graph = get_skill_graph.from_json(SKILL_GRAPH_JSON)
    except Exception as e:
        st.warning(f"Skill graph not available or invalid: {e}")
        return

    if df_logs.empty:
        st.info("No logs available yet to compute skill readiness.")
        return

    skill_stats = compute_skill_stats_from_logs(df_logs, df_q, graph)

    if not skill_stats:
        st.info("No skill-level mappings found (check kc_tags in questions).")
        return

    # Aggregate readiness per skill
    rows = []
    for sid, node in graph.nodes.items():
        r = readiness_score(sid, skill_stats, graph)
        m = mastery_score_for_skill(sid, skill_stats)
        stat = skill_stats.get(sid, {"attempts": 0, "accuracy": 0.0})
        rows.append({
            "skill_id": sid,
            "skill_name": node.name,
            "topic": node.topic,
            "attempts": stat.get("attempts", 0),
            "accuracy": stat.get("accuracy", 0.0),
            "mastery_score": m,
            "readiness": r,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("readiness")

    st.markdown("**Skills Sorted by Readiness (low → high)**")
    st.dataframe(df.style.format({
        "accuracy": "{:.2f}",
        "mastery_score": "{:.2f}",
        "readiness": "{:.2f}"
    }))

    st.markdown("**Recommended Skills to Target Next**")
    recs = recommend_next_skills(skill_stats, graph, max_suggestions=7)
    if not recs:
        st.info("No clear weak-but-ready skills found.")
    else:
        for sid, r in recs:
            node = graph.get_node(sid)
            st.markdown(
                f"- **{node.name}** (topic: `{node.topic}`) — readiness: `{r:.2f}`"
            )


def _show_ability_distribution(df_logs: pd.DataFrame, df_q: pd.DataFrame):
    st.subheader("📈 Ability (θ) Distributions")

    df_theta = _estimate_ability_from_logs(df_logs, df_q)

    if df_theta.empty:
        st.info("No data to estimate ability yet.")
        return

    # Overall distribution
    st.markdown("**Global Ability Distribution**")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df_theta["theta"], bins=15)
    ax.set_xlabel("Ability (θ)")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Estimated Ability (All Topics)")
    st.pyplot(fig)

    # Topic-specific ability
    topic_choice = st.selectbox(
        "View ability distribution for a specific topic:",
        ["All"] + sorted(df_theta["topic"].unique().tolist())
    )

    if topic_choice != "All":
        df_t = df_theta[df_theta["topic"] == topic_choice]
    else:
        df_t = df_theta

    if not df_t.empty:
        st.markdown(f"**Ability Distribution for Topic: {topic_choice}**")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist(df_t["theta"], bins=15)
        ax2.set_xlabel("Ability (θ)")
        ax2.set_ylabel("Number of Students")
        st.pyplot(fig2)
    else:
        st.info("No ability data for this topic.")


# ---------------------------
# MAIN TEACHER DASHBOARD
# ---------------------------

def run_teacher_dashboard():
    require_teacher()
    st.title("👩‍🏫 Teacher Dashboard")
    st.markdown(
        "View mastery trends, mode-separated analytics, misconceptions, "
        "Bloom performance, skill readiness, ability distributions, and "
        "exportable class reports."
    )

    df_questions = load_questions()
    df_logs = load_all_logs()
    
    skill_graph = get_skill_graph()


    if df_logs.empty:
        st.info("No student responses logged yet.")
        return

    # Basic normalization
    if "timestamp" in df_logs.columns:
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors="coerce")

    # Filters
    topics = ["All"] + sorted(df_logs["topic"].dropna().unique().tolist())
    topic_filter = st.selectbox("📘 Filter by Topic", topics)

    if topic_filter != "All":
        df_logs = df_logs[df_logs["topic"] == topic_filter]

    # Mode filters
    mode_values = df_logs["mode"].dropna().unique().tolist()
    # canonical names
    guided_df = df_logs[df_logs["mode"] == "Guided Practice"]
    free_df = df_logs[df_logs["mode"] == "Free Practice"]
    topic_test_df = df_logs[df_logs["mode"] == "Topic Test"]
    exam_df = df_logs[df_logs["mode"] == "Exam Simulator"]

    # Overview metrics
    st.markdown("### 📊 Class Overview")

    total_responses = len(df_logs)
    overall_acc = df_logs["is_correct"].mean() * 100
    n_students = df_logs["student_id"].nunique()
    n_topics = df_logs["topic"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Responses", total_responses)
    c2.metric("Overall Accuracy", f"{overall_acc:.1f}%")
    c3.metric("Students Active", n_students)
    c4.metric("Topics Covered", n_topics)

    # Tabs for deeper analytics
    tab_overview, tab_modes, tab_skills, tab_miscon, tab_ability, tab_heatmap, tab_export = st.tabs(
        [
            "Overview",
            "Mode Analytics",
            "Skill Readiness",
            "Misconceptions",
            "Ability & θ",
            "Mastery Heatmap",
            "Export",
        ]
    )

    # ---------------------------
    # Overview tab
    # ---------------------------
    with tab_overview:
        st.markdown("#### Accuracy by Topic")
        topic_acc = df_logs.groupby("topic")["is_correct"].mean().sort_values(ascending=False) * 100
        st.bar_chart(topic_acc)

        st.markdown("#### Accuracy by Bloom Level")
        bloom_acc = df_logs.groupby("bloom_level")["is_correct"].mean().sort_values(ascending=False) * 100
        st.bar_chart(bloom_acc)

        render_skill_unlock_map(df_logs, skill_graph)

        render_class_readiness(df_logs, skill_graph)

        render_misconceptions(df_logs, skill_graph)

        render_mode_analytics(df_logs)






    # ---------------------------
    # Mode Analytics tab
    # ---------------------------
    with tab_modes:
        st.markdown("#### Mode-Separated Analytics")

        _show_mode_metrics(guided_df, "Guided Practice")
        st.divider()
        _show_mode_metrics(free_df, "Free Practice")
        st.divider()
        _show_mode_metrics(topic_test_df, "Topic Test")
        st.divider()
        _show_mode_metrics(exam_df, "Exam Simulator")

    # ---------------------------
    # Skill Readiness tab (Skill Graph)
    # ---------------------------
    with tab_skills:
        _show_skill_readiness_view(df_logs, df_questions)

    # ---------------------------
    # Misconceptions tab
    # ---------------------------
    with tab_miscon:
        _show_misconception_clusters(df_logs)

    # ---------------------------
    # Ability & θ tab
    # ---------------------------
    with tab_ability:
        _show_ability_distribution(df_logs, df_questions)

    # ---------------------------
    # Mastery Heatmap tab
    # ---------------------------
    with tab_heatmap:
        _plot_heatmap_topic_bloom(df_logs)
        render_skill_mastery_heatmap(df_logs, skill_graph)

    # ---------------------------
    # Export tab
    # ---------------------------
    with tab_export:
        st.markdown("#### 📥 Export Logs")
        _export_csv_button(df_logs, "all_logs_filtered.csv", "Download Filtered Logs CSV")



def render_skill_mastery_heatmap(df_logs, skill_graph):
    st.subheader("🔥 Skill Mastery Heatmap")

    stats = skill_graph.stats(df_logs)
    skills = list(stats.keys())

    mastery_scores = [
        (stats[s]["accuracy"] if stats[s]["accuracy"] is not None else np.nan)
        for s in skills
    ]

    df = pd.DataFrame({"Skill": skills, "Accuracy": mastery_scores})
    df = df.pivot_table(values="Accuracy", index="Skill", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(8, 12))
    sns.heatmap(df, annot=True, cmap="Blues", vmin=0, vmax=1, ax=ax)
    st.pyplot(fig)



def render_skill_unlock_map(df_logs, skill_graph):
    st.subheader("🔐 Skill Unlock Map")

    stats = skill_graph.stats(df_logs)
    unlocks = skill_graph.unlocks(df_logs)

    for skill in skill_graph.skills:
        prereqs = skill_graph.get_prereqs(skill)
        unlocked = unlocks[skill]

        color = "🟢" if unlocked else "🔴"

        st.write(f"{color} **{skill}**")

        if prereqs:
            st.write(" Prerequisites:")
            for p in prereqs:
                m = "✔" if stats[p]["mastered"] else "✘"
                st.write(f"  {m} {p}")
        st.write("---")




def render_class_readiness(df_logs, skill_graph):
    st.subheader("📊 Class Readiness Distribution")

    readiness_data = {}

    stats = skill_graph.stats(df_logs)
    for skill in skill_graph.skills:
        readiness_data[skill] = skill_graph.readiness(df_logs, skill)

    df = pd.DataFrame.from_dict(readiness_data, orient="index", columns=["Readiness"])

    st.bar_chart(df)



def render_misconceptions(df_logs, skill_graph):
    st.subheader("⚠ Top Misconceptions")

    clusters = skill_graph.misconceptions(df_logs)
    if not clusters:
        st.info("No misconception tags available.")
        return

    df = pd.DataFrame(clusters.items(), columns=["Misconception", "Count"])
    st.dataframe(df)


def render_mode_analytics(df_logs):
    st.subheader("🎛 Mode-Specific Analytics")

    modes = ["Guided Practice", "Free Practice", "Topic Test", "Exam Simulator"]

    for mode in modes:
        subset = df_logs[df_logs["mode"] == mode]

        st.markdown(f"### {mode}")
        if subset.empty:
            st.write(" No data available.")
            continue

        acc = subset["is_correct"].mean() * 100
        rt = subset["response_time_sec"].mean()

        st.metric("Accuracy", f"{acc:.1f}%")
        st.metric("Avg Response Time", f"{rt:.1f} sec")
        st.write("---")





# # ui/teacher.py

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from core.data_access import load_all_logs
# from io import BytesIO


# from core.data_access import load_students, upsert_student
# from core.progression_engine import TOPIC_SEQUENCE

# def teacher_unlock_panel():
#     st.subheader("🔧 Topic Unlock Overrides")

#     df = load_students()
#     student_ids = df["student_id"].tolist()
#     sid = st.selectbox("Select student:", student_ids)

#     student = get_student(sid)

#     st.write("Current unlocked topics:")
#     st.write(student.unlocked_topics)

#     new_unlock = st.multiselect("Unlock additional topics:", TOPIC_SEQUENCE)

#     if st.button("Apply Unlock Changes"):
#         student.unlocked_topics = new_unlock
#         upsert_student(student)
#         st.success("Updated!")

# # ============================================
# # Helper Functions
# # ============================================

# def _plot_bar(title: str, data: pd.Series):
#     """Utility: clean bar chart for a Series."""
#     st.markdown(f"### {title}")
#     fig, ax = plt.subplots(figsize=(6, 3))
#     data.plot(kind="bar", ax=ax)
#     ax.set_ylabel("Value")
#     ax.set_title(title)
#     st.pyplot(fig)


# def _plot_heatmap(df: pd.DataFrame, title: str):
#     """Simple green heatmap using DataFrame.style."""
#     st.markdown(f"### {title}")
#     if df.empty:
#         st.info("No data available.")
#         return
#     st.dataframe(df.style.background_gradient(cmap="YlGn"))


# def _export_csv_button(df: pd.DataFrame, filename: str, label: str):
#     csv = df.to_csv(index=False).encode("utf-8")
#     st.download_button(
#         label=label,
#         data=csv,
#         file_name=filename,   # <-- correct parameter
#         mime="text/csv"
#     )



# # ============================================
# # Teacher Dashboard
# # ============================================

# def run_teacher_dashboard():
#     st.title("👩‍🏫 Teacher Dashboard")
#     st.markdown(
#         """
#         <p style="font-size: 17px; color: #555;">
#         View mastery trends, mode-separated analytics, misconceptions, Bloom performance,
#         and exportable class reports.
#         </p>
#         """,
#         unsafe_allow_html=True,
#     )

#     df_logs = load_all_logs()

#     if df_logs.empty:
#         st.info("No student activity yet. Once students practice or test, logs will appear here.")
#         return

#     # Make sure timestamp is parsed if present
#     if "timestamp" in df_logs.columns:
#         df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors="coerce")

#     # Ensure mode exists
#     if "mode" not in df_logs.columns:
#         df_logs["mode"] = "Practice"

#     st.divider()

#     # ============================================
#     # Top-Level Filters
#     # ============================================
#     topics = sorted(df_logs["topic"].dropna().unique())
#     selected_topic = st.selectbox("📘 Filter by Topic", ["All"] + topics)

#     if selected_topic != "All":
#         df_logs = df_logs[df_logs["topic"] == selected_topic]

#     mode_view = st.radio(
#         "View Type",
#         ["Overview", "Practice Analytics", "Test Analytics", "Student Detail"],
#         horizontal=True,
#     )

#     st.divider()

#     # ============================================
#     # OVERVIEW
#     # ============================================
#     if mode_view == "Overview":
#         st.header("📊 Class Overview")

#         total = len(df_logs)
#         accuracy = df_logs["is_correct"].mean() * 100 if total else 0
#         total_students = df_logs["student_id"].nunique()

#         c1, c2, c3 = st.columns(3)
#         c1.metric("Total Responses", total)
#         c2.metric("Overall Accuracy", f"{accuracy:.1f}%")
#         c3.metric("Active Students", total_students)

#         # ---- PER TOPIC ACCURACY ----
#         if selected_topic == "All":
#             topic_acc = df_logs.groupby("topic")["is_correct"].mean() * 100
#             _plot_bar("Per-Topic Accuracy (%)", topic_acc)

#         # ---- BLOOM HEATMAP ----
#         bloom_matrix = (
#             df_logs.groupby(["topic", "bloom_level"])["is_correct"]
#             .mean()
#             .unstack()
#             .fillna(0) * 100
#         )
#         _plot_heatmap(bloom_matrix, "Bloom-Level Accuracy by Topic")

#         # ---- MISCONCEPTIONS ----
#         st.header("⚠️ Common Misconceptions")

#         wrong = df_logs[df_logs["is_correct"] == 0]
#         common_wrong = wrong["question_id"].value_counts().head(10)

#         if common_wrong.empty:
#             st.info("No misconceptions yet.")
#         else:
#             _plot_bar("Most Frequently Missed Questions (Top 10)", common_wrong)

#             st.markdown("#### Download full misconceptions list:")
#             _export_csv_button(wrong, "misconceptions.csv", "📥 Export Misconceptions CSV")


#     # ============================================
#     # PRACTICE ANALYTICS
#     # ============================================
#     elif mode_view == "Practice Analytics":
#         st.header("🧠 Practice Mode Analytics")

#         practice_df = df_logs[df_logs["mode"] == "Practice"]

#         if practice_df.empty:
#             st.info("No practice data available.")
#             return

#         # Accuracy by topic
#         topic_acc = practice_df.groupby("topic")["is_correct"].mean() * 100
#         _plot_bar("Practice Accuracy by Topic (%)", topic_acc)

#         # Bloom breakdown
#         bloom_acc = practice_df.groupby("bloom_level")["is_correct"].mean() * 100
#         _plot_bar("Bloom Accuracy in Practice (%)", bloom_acc)

#         # Time on task
#         if "timestamp" in practice_df.columns:
#             st.subheader("⏱️ Average Response Time (s)")
#             tt = practice_df.groupby("topic")["response_time_sec"].mean()
#             _plot_bar("Response Time by Topic (Practice)", tt)

#         # Export
#         st.markdown("### 📥 Export Practice Mode Data")
#         _export_csv_button(practice_df, "practice_data.csv", "Download Practice CSV")


#     # ============================================
#     # TEST ANALYTICS
#     # ============================================
#     elif mode_view == "Test Analytics":
#         st.header("🧪 Test Mode Analytics")

#         test_df = df_logs[df_logs["mode"] == "Test"]

#         if test_df.empty:
#             st.info("No test data available.")
#             return

#         # Test scores per session
#         scores = (
#             test_df.groupby(["student_id", "session_id"])["is_correct"].mean() * 100
#         )
#         st.markdown("### 📊 Test Scores by Student")
#         st.dataframe(scores.to_frame("Score (%)"))

#         # Distribution chart
#         st.markdown("#### Score Distribution")
#         fig, ax = plt.subplots(figsize=(6, 3))
#         scores.plot(kind="hist", bins=10, ax=ax)
#         ax.set_xlabel("Score (%)")
#         st.pyplot(fig)

#         # Export
#         _export_csv_button(test_df, "test_data.csv", "📥 Download Test CSV")


#     # ============================================
#     # STUDENT DETAIL VIEW
#     # ============================================
#     elif mode_view == "Student Detail":
#         st.header("👤 Student Detail View")

#         students = sorted(df_logs["student_id"].unique())
#         target_student = st.selectbox("Select Student", students)

#         s_df = df_logs[df_logs["student_id"] == target_student]

#         if s_df.empty:
#             st.info("No data for this student.")
#             return

#         st.subheader("Performance Summary")

#         total = len(s_df)
#         accuracy = s_df["is_correct"].mean() * 100
#         topics = s_df["topic"].nunique()

#         c1, c2, c3 = st.columns(3)
#         c1.metric("Total Responses", total)
#         c2.metric("Accuracy", f"{accuracy:.1f}%")
#         c3.metric("Topics Attempted", topics)

#         # Topic accuracy
#         st.markdown("### 📘 Accuracy by Topic")
#         topic_acc = s_df.groupby("topic")["is_correct"].mean() * 100
#         _plot_bar("Student's Accuracy by Topic", topic_acc)

#         # Bloom accuracy
#         st.markdown("### 🧠 Bloom-Level Breakdown")
#         bloom_acc = s_df.groupby("bloom_level")["is_correct"].mean() * 100
#         _plot_bar("Bloom-Level Accuracy", bloom_acc)

#         # Timeline graph
#         if "timestamp" in s_df.columns:
#             st.markdown("### 📅 Performance Over Time")
#             timeline = (
#                 s_df.sort_values("timestamp")
#                 .set_index("timestamp")["is_correct"]
#                 .rolling(window=5, min_periods=1)
#                 .mean()
#                 * 100
#             )
#             fig, ax = plt.subplots(figsize=(6, 3))
#             timeline.plot(ax=ax)
#             ax.set_ylabel("Accuracy (%)")
#             st.pyplot(fig)

#         # Export
#         st.markdown("### 📥 Export Student Data")
#         _export_csv_button(s_df, f"student_{target_student}.csv", "Download Student CSV")
