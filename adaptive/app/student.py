import streamlit as st
import pandas as pd
import os
import time
import matplotlib


# --- Session State ---
def initialize_session_state():
    defaults = {
        "started": False,
        "log": [],
        "bookmarked": set(),
        "score": {},
        "wrong_qs": [],
        "confidence_record": {},
        "topic_mastery": {},
        "asked_qs": set(),
        "submitted": False,
        "current_question": None,
        "current_reason": "",
        "review_mode": False,
        "remediation_queue": [],
        "session_id": str(int(time.time()))
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

initialize_session_state()



# --- DIFFICULTY MODELING ---
def calculate_difficulty(df_logs):
    """Assigns and refines difficulty weights for each question."""
    bloom_base = {
        "Remember": 0.2,
        "Understand": 0.4,
        "Apply": 0.6,
        "Analyze": 0.75,
        "Evaluate": 0.9,
        "Create": 1.0
    }

    # Bloom-weight baseline
    df_logs["bloom_weight"] = df_logs["bloom_level"].map(bloom_base).fillna(0.5)

    # Empirical difficulty (1 - average accuracy per question)
    empirical = df_logs.groupby("question_id")["correct"].mean().to_dict()
    df_logs["empirical_difficulty"] = df_logs["question_id"].map(lambda x: 1 - empirical.get(x, 0.5))

    # Combined difficulty model
    df_logs["difficulty_score"] = 0.6 * df_logs["bloom_weight"] + 0.4 * df_logs["empirical_difficulty"]

    return df_logs




def calculate_learning_gains(df_logs):
    """Compute learning gain per student per Bloom level."""
    # Sort logs by time for temporal grouping
    df_logs = df_logs.sort_values(["session_id", "timestamp"])

    # Rolling mean of accuracy for learning curve
    df_logs["rolling_accuracy"] = (
        df_logs.groupby(["session_id", "bloom_level"])["correct"]
        .apply(lambda x: x.rolling(5, min_periods=1).mean())
        .reset_index(level=[0,1], drop=True)
    )

    # Compute overall learning gain
    gain = (
        df_logs.groupby(["session_id", "bloom_level"])["rolling_accuracy"]
        .agg(["first", "last"])
        .reset_index()
    )
    gain["learning_gain"] = (gain["last"] - gain["first"]) * 100
    return df_logs, gain

  ##
# --- Configuration ---
st.set_page_config("Adaptive Java Learning", layout="wide")

# --- Data Loading ---
# @st.cache_data
# def load_data():
#     try:
#         df = pd.read_csv("data/java_question_bank_with_topics_cleaned_gpt.csv", encoding="latin1")
#     except FileNotFoundError:
#         st.error("Error: Data file not found.")
#         return pd.DataFrame()

#     topic_order = [
#         "Basic Syntax", "Data Types", "Variables", "Operators", "Control Flow",
#         "Loops", "Methods", "Arrays", "Object-Oriented Programming",
#         "Inheritance", "Polymorphism", "Abstraction", "Encapsulation",
#         "Exception Handling", "File I/O", "Multithreading", "Collections", "Generics"
#     ]
#     df['topic'] = pd.Categorical(df['topic'], categories=topic_order, ordered=True)
#     df['bloom_level'] = pd.Categorical(df['bloom_level'], categories=df['bloom_level'].unique(), ordered=True)
#     return df.sort_values(['topic', 'bloom_level'])
# --- Data Loading ---
@st.cache_data
def load_data():
    # Get path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "java_question_bank_with_topics_cleaned_gpt.csv")

    # if not os.path.exists(data_path):
    #     st.error(f"Error: Data file not found at {data_path}")
    #     return pd.DataFrame()


    # --- Log to Google Sheets Instead ---
    from gsheets_api import append_row_to_sheet   # your own helper module

    append_row_to_sheet(
        sheet_id="YOUR_GOOGLE_SHEET_ID",
        user_id=session_id,   # or student email, etc.
        row_data=log_entry
    )


    try:
        df = pd.read_csv(data_path, encoding="latin1")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

    # topic_order = [
    #     "Basic Syntax", "Data Types", "Variables", "Operators", "Control Flow",
    #     "Loops", "Methods", "Arrays", "Object-Oriented Programming",
    #     "Inheritance", "Polymorphism", "Abstraction", "Encapsulation",
    #     "Exception Handling", "File I/O", "Multithreading", "Collections", "Generics"
    # ]
    
    topic_order = [
    #level 1 beginner
    "Basic Syntax",
    "Control Structures",
    "Loops",
    "Arrays",
    "Strings",
    "Methods and Parameter Passing",
    "File I/O and Exception Handling",
    #level 2 intermediate
    "Classes and Objects",
    "Encapsulation and Access Modifiers",
    "Inheritance and Polymorphism",
    "Abstract Classes and Interfaces",
    #level 3 advanced
    "Collections",
    "Generics"
    ]

    df["topic"] = pd.Categorical(df["topic"], categories=topic_order, ordered=True)
    df["bloom_level"] = pd.Categorical(df["bloom_level"].astype(str),
                                       categories=df["bloom_level"].dropna().unique(),
                                       ordered=True)

    return df.sort_values(["topic", "bloom_level"])



# Inject CSS to remove syntax highlighting colors
st.markdown(
    """
    <style>
    /* Target the code blocks generated by st.code */
    pre code {
        color: unset !important; /* Remove specific text colors */
        background-color: unset !important; /* Remove specific background colors */
    }

    /* Target individual tokens within the code block if needed */
    pre code .token {
        color: unset !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

 

# --- Adaptive BBPS Logic ---
def get_next_question(topic_df, selected_topic):
    bloom_levels = topic_df['bloom_level'].cat.categories
    mastery = st.session_state.topic_mastery[selected_topic]
    confidence_log = st.session_state.confidence_record[selected_topic]

    # 1. Remediation
    if st.session_state.remediation_queue:
        target = st.session_state.remediation_queue.pop(0)
        qs = topic_df[(topic_df["sub_concept"] == target) &
                      (~topic_df["question_id"].isin(st.session_state.asked_qs))]
        if not qs.empty:
            return qs.sample(1).iloc[0], f"🎯 Targeting weak area: {target}"

    # 2. Bidirectional Bloom Logic
    for i, level in enumerate(bloom_levels):
        level_qs = topic_df[(topic_df["bloom_level"] == level) &
                            (~topic_df["question_id"].isin(st.session_state.asked_qs))]
        if level_qs.empty:
            continue

        records = confidence_log.get(level, [])
        high_conf_wrong = [r for r in records if not r['correct'] and r['confidence'] >= 4]
        low_conf_right = [r for r in records if r['correct'] and r['confidence'] <= 2]

        if high_conf_wrong and i > 0:
            lower = bloom_levels[i - 1]
            demotion_qs = topic_df[(topic_df["bloom_level"] == lower) &
                                   (~topic_df["question_id"].isin(st.session_state.asked_qs))]
            if not demotion_qs.empty:
                return demotion_qs.sample(1).iloc[0], f"🔻 Demoting to reinforce: {lower}"

        if low_conf_right:
            reinforce_qs = level_qs
            return reinforce_qs.sample(1).iloc[0], f"🔄 Reinforcing: {level} due to low confidence"

        if mastery.get(level, 0) < 2:
            return level_qs.sample(1).iloc[0], f"⬆️ Progressing to: {level}"

    return None, "🎉 You've completed this topic!"

# --- UI ---
def render_sidebar(topics):
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
        
    
    if role == "Teacher":
        st.session_state.started = False
        st.session_state.review_mode = False
        st.session_state.current_question = None
        render_teacher_view()
        st.stop()


    return role, selected_topic

##############################
# Student View
##############################

def render_student_view(df):
    topics = df["topic"].dropna().unique().tolist()
    role, selected_topic = render_sidebar(topics)

    st.title("🎓 Java Learning – Student Mode")
    st.subheader(f"📘 Topic: {selected_topic}")

    # Setup states
    st.session_state.topic_mastery.setdefault(selected_topic, {b: 0 for b in df['bloom_level'].cat.categories})
    st.session_state.score.setdefault(selected_topic, {b: {"correct": 0, "total": 0} for b in df['bloom_level'].cat.categories})
    st.session_state.confidence_record.setdefault(selected_topic, {})

    if not st.session_state.started:
        st.info("Click 'Start Learning' to begin.")
        if st.button("🚀 Start Learning"):
            st.session_state.started = True
            st.rerun()
        return

    topic_df = df[df["topic"] == selected_topic]
    if topic_df.empty:
        st.warning("No questions for this topic.")
        return

    # Question Flow
    if st.session_state.current_question is None and not st.session_state.submitted:
        if st.session_state.review_mode:
            bookmarked = topic_df[topic_df["question_id"].isin(st.session_state.bookmarked)]
            if not bookmarked.empty:
                q = bookmarked.sample(1).iloc[0]
                st.session_state.current_question = q.to_dict()
                st.session_state.current_reason = "🔖 Reviewing a bookmarked question."
            else:
                st.success("No bookmarked questions left!")
                st.session_state.review_mode = False
        else:
            q, reason = get_next_question(topic_df, selected_topic)
            if q is not None:
                st.session_state.current_question = q.to_dict()
                st.session_state.current_reason = reason
                st.session_state.asked_qs.add(q["question_id"])

    # Show UI
    if st.session_state.current_question:
        display_question(pd.Series(st.session_state.current_question), selected_topic)

def display_question(q, topic):
    st.info(st.session_state.current_reason)
    st.markdown(f"**Bloom:** '{q['bloom_level']}' | **Question Id:** '{q['question_id']}') | **Sub-Concept:** '{q.get('sub_concept', 'General')}' ")
    
    
    st.code(q['question_stem'], language='java')

    choice = st.radio("Choose:", ["a", "b", "c", "d"], index=None,
                      format_func=lambda x: f"{x.upper()}. {q[f'option_{x}']}",
                      key=f"choice_{q['question_id']}")
    # confidence = st.slider("Confidence?", 1, 5, 3, key=f"conf_{q['question_id']}",
    #                        format_func=lambda x: ["Guessing", "Unsure", "Okay", "Confident", "Very Confident"][x-1])

    confidence = st.slider("How confident are you?", 1, 5, 3, key=f"conf_{q['question_id']}")
    confidence_labels = {
        1: "😬 Guessing",
        2: "🤔 Unsure",
        3: "😐 Okay",
        4: "🙂 Confident",
        5: "😎 Very Confident"
    }
    st.markdown(f"**Selected Confidence:** {confidence_labels[confidence]}")

    submit_col, bookmark_col = st.columns(2)
    if bookmark_col.button("🔖 Bookmark", use_container_width=True):
        st.session_state.bookmarked.add(q['question_id'])
        st.success("Bookmarked.")

    if submit_col.button("✅ Submit", use_container_width=True) and not st.session_state.submitted:
        if choice is None:
            st.warning("Choose an option.")
        else:
            handle_submission(q, choice, confidence, topic)
            st.session_state.submitted = True
            # st.rerun()

    if st.session_state.submitted:
        if st.button("➡️ Next Question"):
            st.session_state.submitted = False
            st.session_state.current_question = None
            st.rerun()

# def handle_submission(q, choice, confidence, topic):
#     correct = q["correct_option"].strip().lower()
#     is_correct = (choice == correct)
#     bloom = q["bloom_level"]
#     points = 0.5 + (confidence / 10.0) if is_correct else 0

#     if is_correct:
#         st.success(f"Correct! +{points:.2f} points.")
#     else:
#         st.error(f"Wrong. Correct: **{correct.upper()}**")
#         if confidence >= 4 and pd.notna(q.get('sub_concept')):
#             st.session_state.remediation_queue.append(q['sub_concept'])

#     if not st.session_state.review_mode:
#         st.session_state.topic_mastery[topic][bloom] += points
#         st.session_state.score[topic][bloom]["total"] += 1
#         st.session_state.score[topic][bloom]["correct"] += int(is_correct)
#         st.session_state.confidence_record[topic].setdefault(bloom, []).append({
#             'question_id': q['question_id'],
#             'confidence': confidence,
#             'correct': is_correct
#         })

#     # Log result
#     st.session_state.log.append({
#         # "timestamp": time.time(),
#         "timestamp": pd.Timestamp.now().isoformat(),
#         "session_id": st.session_state.session_id,
#         "topic": topic,
#         "bloom_level": bloom,
#         "question_id": q['question_id'],
#         "confidence": confidence,
#         "correct": is_correct,
#         "reinforcement": st.session_state.current_reason
#     })

#     # Explanation
#     with st.expander("📘 Explanation"):
#         st.markdown(q['main_explanation'])

#     # Save logs to CSV
#     log_df = pd.DataFrame(st.session_state.log)
#     os.makedirs("logs", exist_ok=True)
#     log_df.to_csv(f"logs/session_{st.session_state.session_id}.csv", index=False)
#     check_for_bloom_badge(topic, bloom)


def handle_submission(q, choice, confidence, topic):
    """Handles the logic after a user submits an answer."""
    correct = q["correct_option"].strip().lower()
    is_correct = (choice == correct)
    bloom = q["bloom_level"]
    points = 0.5 + (confidence / 10.0) if is_correct else 0

    if is_correct:
        st.success("✅ Correct!")
        if not st.session_state.review_mode:
            st.session_state.topic_mastery[q['topic']][q['bloom_level']] += 1
    else:
        st.error(f"❌ Incorrect. The correct answer is **{correct.upper()}**.")
        if not st.session_state.review_mode:
            st.session_state.wrong_qs.append(q["question_id"])
            
    if not st.session_state.review_mode:
        st.session_state.topic_mastery[topic][bloom] += points
        st.session_state.score[topic][bloom]["total"] += 1
        st.session_state.score[topic][bloom]["correct"] += int(is_correct)
        st.session_state.confidence_record[topic].setdefault(bloom, []).append({
            'question_id': q['question_id'],
            'confidence': confidence,
            'correct': is_correct
        })

    with st.expander("View Explanation"):
        st.markdown(f"{q['main_explanation']}")

    # Update scores and log
    score = st.session_state.score[q['topic']][q['bloom_level']]
    score['total'] += 1
    if is_correct:
        score['correct'] += 1

    student_id = st.session_state.get("user_id", "unknown_student")
    session_id = st.session_state.get("session_id", "unknown_session")

    # Prepare log entry
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "student_id": student_id,
        "session_id": session_id,
        "topic": topic,
        "bloom_level": bloom,
        "question_id": q["question_id"],
        "selected": choice,
        "correct_option": correct,
        "is_correct": is_correct,
        "confidence": confidence,
        "reinforcement_reason": st.session_state.get("current_reason", "")
    }

    # Add this entry to in-memory log list
    st.session_state.log.append(log_entry)

    # === Absolute path: go back one directory from current file ===
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))  # one level up
    data_dir = os.path.join(project_root, "logs")
    os.makedirs(data_dir, exist_ok=True)

    # === Unified file per student ===
    student_csv = os.path.join(data_dir, f"student_{student_id}.csv")

    # Convert log to DataFrame and save (append mode)
    log_df = pd.DataFrame(st.session_state.log)

    if os.path.exists(student_csv):
        existing = pd.read_csv(student_csv)
        combined = pd.concat([existing, log_df], ignore_index=True)
        combined.to_csv(student_csv, index=False)
    else:
        log_df.to_csv(student_csv, index=False)

    # Save logs to CSV
    # log_df = pd.DataFrame(st.session_state.log)
    # os.makedirs("logs", exist_ok=True)
    # log_df.to_csv(f"logs/session_{st.session_state.session_id}.csv", index=False)
    # check_for_bloom_badge(topic, bloom)
    


def display_topic_completion_summary(selected_topic):
    """Displays a summary when a topic is completed."""
    st.success(f"🎉 You've completed the topic: {selected_topic}!")
    st.balloons()
    
    st.markdown("### 📈 Mastery Progress")
    mastery_df = pd.DataFrame(st.session_state.topic_mastery[selected_topic], index=["Mastery"]).T
    st.dataframe(mastery_df)
    
# --- MOTIVATION PANEL ---
def show_motivation_panel(topic):
    st.markdown("### 📈 Your Progress")
    stats = st.session_state.score[topic]
    total_attempted = sum([stats[b]['total'] for b in stats])
    total_correct = sum([stats[b]['correct'] for b in stats])
    percent = (total_correct / total_attempted * 100) if total_attempted else 0

    st.progress(int(percent), text=f"{int(percent)}% Correct")

    # Bloom-level breakdown
    with st.expander("🧠 Bloom Level Mastery"):
        for b in stats:
            correct = stats[b]["correct"]
            total = stats[b]["total"]
            pct = (correct / total * 100) if total else 0
            st.markdown(f"- **{b}**: {correct}/{total} ({int(pct)}%)")
            st.progress(int(pct))

# --- BLOOM BADGE CHECKER ---
def check_for_bloom_badge(topic, bloom_level):
    stats = st.session_state.score[topic][bloom_level]
    correct = stats["correct"]
    total = stats["total"]
    if total >= 3 and correct / total >= 0.8:
        if f"{topic}_{bloom_level}" not in st.session_state:
            st.success(f"🎉 Mastery Unlocked: `{bloom_level}` level in **{topic}**!")
            st.session_state[f"{topic}_{bloom_level}"] = True



# --- Main ---
def main():
    df = load_data()
    if df.empty:
        return
    initialize_session_state()
    render_student_view(df)


