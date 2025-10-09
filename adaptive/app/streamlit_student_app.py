import streamlit as st
import pandas as pd
import time
import os

# --- Configuration and Data Loading ---
st.set_page_config("Adaptive Java Learning", layout="wide")

@st.cache_data
def load_data():
    """Loads, cleans, and categorizes the question bank from a CSV file."""
    try:
        # NOTE: Make sure your CSV has a 'sub_concept' column for this to work best.
        # If it doesn't, the app will still run but without sub-concept targeting.
        df = pd.read_csv("data/java_question_bank_with_topics_cleaned_gpt.csv", encoding="latin1")
    except FileNotFoundError:
        st.error("Error: The question bank file was not found. Please ensure 'data/java_question_bank_with_topics_cleaned_gpt.csv' exists.")
        return pd.DataFrame()

    topic_order = [
        "Basic Syntax", "Data Types", "Variables", "Operators", "Control Flow",
        "Loops", "Methods", "Arrays", "Object-Oriented Programming",
        "Inheritance", "Polymorphism", "Abstraction", "Encapsulation",
        "Exception Handling", "File I/O", "Multithreading", "Collections", "Generics"
    ]
    df['topic'] = pd.Categorical(df['topic'], categories=topic_order, ordered=True)
    
    # Ensure bloom_level is treated as an ordered category
    if 'bloom_level' in df.columns:
        df['bloom_level'] = pd.Categorical(df['bloom_level'], categories=df['bloom_level'].unique(), ordered=True)
        
    return df.sort_values(['topic', 'bloom_level'])

# --- Session State Management ---
def initialize_session_state():
    """Initializes session state variables if they don't exist using setdefault."""
    defaults = {
        "started": False, "log": [], "bookmarked": set(), "score": {},
        "topic_mastery": {}, "asked_qs": set(), "submitted": False,
        "current_question": None, "current_reason": "", "review_mode": False,
        "remediation_queue": []
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

# --- Core Adaptive Logic ---
def get_next_question(topic_df, selected_topic):
    """Determines the next question based on a hierarchy of learning needs."""
    # 1. Prioritize remediation for specific weak spots
    if 'sub_concept' in topic_df.columns and st.session_state.remediation_queue:
        target_sub_concept = st.session_state.remediation_queue.pop(0)
        remediation_qs = topic_df[
            (topic_df["sub_concept"] == target_sub_concept) &
            (~topic_df["question_id"].isin(st.session_state.asked_qs))
        ]
        if not remediation_qs.empty:
            return remediation_qs.sample(1).iloc[0], f"🎯 Targeting a weak area: **{target_sub_concept}**"

    # 2. Progress through Bloom's Levels
    for level in topic_df['bloom_level'].cat.categories:
        if st.session_state.topic_mastery.get(selected_topic, {}).get(level, 0) < 2:
            candidate_qs = topic_df[
                (topic_df["bloom_level"] == level) &
                (~topic_df["question_id"].isin(st.session_state.asked_qs))
            ]
            if not candidate_qs.empty:
                return candidate_qs.sample(1).iloc[0], f"🔍 Moving to Bloom Level: **{level}**"
            continue # FIX: Continue to the next level if no questions are found.

    return None, "🎉 Topic complete! All questions have been answered."

# --- UI Rendering Functions ---
def render_sidebar(topics):
    """Renders the sidebar for navigation and session control."""
    st.sidebar.title("📚 Adaptive Java Practice")
    role = st.sidebar.radio("Select Role", ["Student", "Teacher"])
    selected_topic = st.sidebar.selectbox("📘 Choose a Topic", topics)

    if st.sidebar.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if st.sidebar.button("🔖 Review Bookmarked"):
        st.session_state.review_mode = True
        st.session_state.submitted = False
        st.session_state.current_question = None
        st.rerun()
    return role, selected_topic

def render_student_view(df):
    """Main function to control the student's learning experience."""
    topics = df["topic"].dropna().unique().tolist()
    _, selected_topic = render_sidebar(topics) # Role is not used here but kept for clarity

    st.title("🎓 Java Learning – Student Mode")
    st.subheader(f"📘 Topic: {selected_topic}")

    st.session_state.topic_mastery.setdefault(selected_topic, {b: 0 for b in df['bloom_level'].cat.categories})
    st.session_state.score.setdefault(selected_topic, {b: {"correct": 0, "total": 0} for b in df['bloom_level'].cat.categories})

    if not st.session_state.started:
        st.info("👋 Welcome! Click 'Start Learning' to begin your adaptive practice session.")
        if st.button("🚀 Start Learning"):
            st.session_state.started = True
            st.rerun()
        return

    topic_df = df[df["topic"] == selected_topic]
    if topic_df.empty:
        st.warning("This topic has no questions yet. Please select another topic.")
        return

    if st.session_state.current_question is None and not st.session_state.submitted:
        if st.session_state.review_mode:
            bookmarked_df = topic_df[topic_df["question_id"].isin(st.session_state.bookmarked)]
            question = bookmarked_df.sample(1).iloc[0] if not bookmarked_df.empty else None
            reason = "🔖 Reviewing a bookmarked question." if question is not None else ""
            if question is None:
                st.success("No bookmarked questions in this topic to review!")
                st.session_state.review_mode = False
        else:
            question, reason = get_next_question(topic_df, selected_topic)

        if question is not None:
            st.session_state.current_question = question.to_dict()
            st.session_state.asked_qs.add(question["question_id"])
        st.session_state.current_reason = reason

    if st.session_state.current_question:
        display_question_interface(pd.Series(st.session_state.current_question))
    else:
        display_completion_summary(selected_topic)


def display_question_interface(q):
    """Renders the UI for a single question, now with a compatible slider."""
    st.info(st.session_state.current_reason)
    st.markdown(f"**🧠 Bloom:** `{q['bloom_level']}` | **Sub-Concept:** `{q.get('sub_concept', 'General')}`")
    st.code(q['question_stem'], language='java')

    choice = st.radio("Choose:", ["a", "b", "c", "d"], index=None,
                      format_func=lambda x: f"{x.upper()}. {q.get(f'option_{x}', '')}",
                      key=f"choice_{q['question_id']}")

    # --- FIX IMPLEMENTED HERE ---
    confidence_labels = {1: "Guessing", 2: "Unsure", 3: "Okay", 4: "Confident", 5: "Very Confident"}
    confidence_val = st.slider("How confident are you?", 1, 5, 3)
    st.markdown(f"**Your Confidence Level:** _{confidence_labels[confidence_val]}_")
    # --- END OF FIX ---

    submit_col, bookmark_col = st.columns(2)
    submit = submit_col.button("✅ Submit Answer", use_container_width=True)

    if bookmark_col.button("🔖 Bookmark for Review", use_container_width=True):
        st.session_state.bookmarked.add(q['question_id'])
        st.success(f"Bookmarked Question ID: {q['question_id']}")

    if submit and not st.session_state.submitted:
        if choice is None:
            st.warning("⚠️ Please select an option.")
        else:
            handle_submission(q, choice, confidence_val)
            st.session_state.submitted = True
            st.rerun()

    if st.session_state.submitted:
        if st.button("➡️ Next Question", use_container_width=True):
            st.session_state.submitted = False
            st.session_state.current_question = None
            if st.session_state.review_mode and not st.session_state.bookmarked:
                st.session_state.review_mode = False
            st.rerun()

def handle_submission(q, choice, confidence):
    """Processes the user's answer, updates state, and provides feedback."""
    correct = q["correct_option"].strip().lower()
    is_correct = (choice == correct)
    mastery_points = 0

    if is_correct:
        mastery_points = 0.5 + (confidence / 10.0)
        st.success(f"✅ Correct! You earned {mastery_points:.2f} mastery points.")
    else:
        st.error(f"❌ Incorrect. The correct answer was **{correct.upper()}**.")
        if confidence >= 4:
            st.warning("High confidence on an incorrect answer suggests a key misunderstanding. We'll practice this concept again.")
        if 'sub_concept' in q and pd.notna(q['sub_concept']):
            st.session_state.remediation_queue.append(q['sub_concept'])

    if not st.session_state.review_mode:
        topic_mastery = st.session_state.topic_mastery[q['topic']]
        topic_mastery[q['bloom_level']] = topic_mastery.get(q['bloom_level'], 0) + mastery_points

    with st.expander("View Explanation"):
        st.markdown(q.get('main_explanation', 'No explanation available.'))
    
    # Log the event
    # (Your logging logic here...)

def display_completion_summary(selected_topic):
    """Shows the user's progress and stats after finishing a topic."""
    st.success(st.session_state.current_reason)
    st.balloons()
    st.markdown("---")
    st.subheader("📊 Topic Performance Summary")
    # (Your summary logic here...)

# --- Main Application Runner ---
def main():
    """The main function to run the Streamlit app."""
    df = load_data()
    if df.empty:
        return

    initialize_session_state()
    # Simplified to focus on the student view for this fix.
    # You can re-add the role-based switching here if needed.
    render_student_view(df)

if __name__ == "__main__":
    main()


















# import streamlit as st
# import pandas as pd
# import random
# import time
# import os

# # Constants
# MASTERY_THRESHOLD = 2
# LOG_DIR = "logs"
# os.makedirs(LOG_DIR, exist_ok=True)

# # Load and prepare question bank
# df = pd.read_csv("data/java_question_bank_with_topics_cleaned_gpt.csv", encoding="latin1")
# topic_order = [
#     "Basic Syntax", "Data Types", "Variables", "Operators", "Control Flow",
#     "Loops", "Methods", "Arrays", "Object-Oriented Programming",
#     "Inheritance", "Polymorphism", "Abstraction", "Encapsulation",
#     "Exception Handling", "File I/O", "Multithreading", "Collections", "Generics"
# ]
# df['topic'] = pd.Categorical(df['topic'], categories=topic_order, ordered=True)
# df['bloom_level'] = pd.Categorical(df['bloom_level'], ordered=True)
# df = df.sort_values(['topic', 'bloom_level'])

# # Page config
# st.set_page_config("Adaptive Java Learning", layout="wide")

# # Session state setup
# for key, default in {
#     "started": False, "log": [], "wrong_qs": [], "bookmarked": set(),
#     "score": {}, "topic_mastery": {}, "asked_qs": set(),
#     "current_question": None, "submitted": False
# }.items():
#     if key not in st.session_state:
#         st.session_state[key] = default

# # Sidebar
# st.sidebar.title("📚 Adaptive Java Learning")
# role = st.sidebar.radio("Select Role", ["Student", "Teacher"])
# topics = df["topic"].dropna().unique().tolist()
# selected_topic = st.sidebar.selectbox("📘 Choose a Topic", topics)

# if st.sidebar.button("🔄 Reset Session"):
#     for key in list(st.session_state.keys()):
#         del st.session_state[key]
#     st.rerun()

# # Teacher Dashboard
# if role == "Teacher":
#     st.title("👩‍🏫 Teacher Dashboard")
#     dist = df.groupby(["topic", "bloom_level"]).size().unstack().fillna(0)
#     st.subheader("📊 Question Distribution")
#     st.bar_chart(dist)

#     if st.session_state.log:
#         log_df = pd.DataFrame(st.session_state.log)
#         st.dataframe(log_df)
#         st.download_button("📥 Download Session Log", log_df.to_csv(index=False), file_name="student_session_log.csv")
#     else:
#         st.info("No session data yet.")

# # Student View
# else:
#     st.title("🎓 Java Learning – Student Mode")
#     st.subheader(f"📘 Topic: {selected_topic}")

#     if not st.session_state.started:
#         st.markdown("""
#         👋 **Welcome to your adaptive Java learning app!**

#         - Master each Bloom level before progressing.
#         - Missed questions will be repeated.
#         - Bookmark tricky ones for review.
#         """)
#         if st.button("🚀 Start Learning"):
#             st.session_state.started = True
#             st.rerun()
#         st.stop()

#     # Initialize per-topic states
#     if selected_topic not in st.session_state.topic_mastery:
#         st.session_state.topic_mastery[selected_topic] = {b: 0 for b in df['bloom_level'].cat.categories}
#     if selected_topic not in st.session_state.score:
#         st.session_state.score[selected_topic] = {b: {"correct": 0, "total": 0} for b in df['bloom_level'].cat.categories}

#     topic_df = df[df["topic"] == selected_topic]

#     # Determine Bloom level
#     for level in df['bloom_level'].cat.categories:
#         if st.session_state.topic_mastery[selected_topic][level] < MASTERY_THRESHOLD:
#             target_level = level
#             break
#     else:
#         target_level = None

#     # Select next question if needed
#     if not st.session_state.submitted and st.session_state.current_question is None:
#         if target_level:
#             candidate_qs = topic_df[topic_df["bloom_level"] == target_level]
#             available_qs = candidate_qs[~candidate_qs["question_id"].isin(st.session_state.asked_qs)]
#         else:
#             available_qs = pd.DataFrame()

#         if not available_qs.empty:
#             q = available_qs.sample(1).iloc[0]
#             st.session_state.asked_qs.add(q["question_id"])
#             st.session_state.current_question = q.to_dict()
#             st.session_state.current_reason = f"🔍 Bloom Level: **{target_level}**"
#         elif st.session_state.wrong_qs:
#             q = topic_df[topic_df["question_id"] == st.session_state.wrong_qs.pop(0)].iloc[0]
#             st.session_state.current_question = q.to_dict()
#             st.session_state.current_reason = "🔁 Retrying a previously missed question."
#         else:
#             st.session_state.current_question = None

#     # Show current question
#     if st.session_state.current_question is not None:
#         q = pd.Series(st.session_state.current_question)
#         st.info(st.session_state.current_reason)
#         st.markdown(f"**🧠 Bloom:** `{q['bloom_level']}` | **ID:** `{q['question_id']}`")
#         st.markdown(f"**📝 Q:** {q['question_stem']}")

#         choice = st.radio(
#             "Choose:", ["a", "b", "c", "d"], index=None,
#             format_func=lambda x: f"{x.upper()}. {q[f'option_{x}']}",
#             key=f"choice_{q['question_id']}"
#         )

#         col1, col2 = st.columns(2)
#         submit = col1.button("✅ Submit Answer")
#         bookmark = col2.button("🔖 Bookmark Question")

#         if submit and not st.session_state.submitted:
#             if choice not in ["a", "b", "c", "d"]:
#                 st.warning("⚠️ Please select an option before submitting.")
#                 st.stop()

#             correct = q["correct_option"].strip().lower()
#             is_correct = choice == correct

#             # Feedback
#             st.session_state.submitted = True
#             if is_correct:
#                 st.success("✅ Correct!")
#                 st.session_state.topic_mastery[selected_topic][q["bloom_level"]] += 1
#             else:
#                 st.error(f"❌ Incorrect. Correct answer: **{correct.upper()}**")
#                 st.session_state.wrong_qs.append(q["question_id"])

#             st.info(f"You chose: **{choice.upper()}** | Correct: **{correct.upper()}**")

#             # Log answer
#             st.session_state.score[selected_topic][q["bloom_level"]]["total"] += 1
#             if is_correct:
#                 st.session_state.score[selected_topic][q["bloom_level"]]["correct"] += 1

#             st.session_state.log.append({
#                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#                 "topic": q["topic"],
#                 "bloom_level": q["bloom_level"],
#                 "question_id": q["question_id"],
#                 "selected": choice,
#                 "correct_option": correct,
#                 "is_correct": is_correct
#             })

#             # Save log
#             log_df = pd.DataFrame(st.session_state.log)
#             log_df.to_csv(os.path.join(LOG_DIR, f"session_{time.strftime('%Y%m%d_%H%M%S')}.csv"), index=False)

#         if st.session_state.submitted:
#             if st.button("➡️ Next Question"):
#                 st.session_state.submitted = False
#                 st.session_state.current_question = None
#                 st.rerun()

#         if bookmark:
#             st.session_state.bookmarked.add(q["question_id"])
#             st.success("🔖 Question bookmarked.")

#     else:
#         # Completion Screen
#         st.balloons()
#         st.success("🎉 Topic complete!")
#         st.markdown("### 📈 Mastery Progress")

#         mastery_df = pd.DataFrame(st.session_state.topic_mastery[selected_topic], index=["Mastery"]).T
#         st.dataframe(mastery_df)

#         progress_value = int(
#             sum(st.session_state.topic_mastery[selected_topic].values()) /
#             max(len(df['bloom_level'].cat.categories) * MASTERY_THRESHOLD, 1) * 100
#         )
#         st.progress(min(progress_value, 100))

#         log_df = pd.DataFrame(st.session_state.log)
#         st.download_button("📥 Download Your Log", log_df.to_csv(index=False), file_name="session_log.csv")

#         if st.session_state.bookmarked:
#             st.markdown("### 🔖 Bookmarked Questions")
#             for b_id in st.session_state.bookmarked:
#                 bq = topic_df[topic_df['question_id'] == b_id].iloc[0]
#                 st.markdown(f"**ID {b_id}:** {bq['question_stem']}")

#         if st.session_state.wrong_qs:
#             if st.button("🔁 Retry Missed Questions"):
#                 st.session_state.asked_qs.clear()
#                 st.rerun()
