
import streamlit as st
import pandas as pd
import os
import time
# import matplotlib


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
    data_path = os.path.join(script_dir, "..", "data", "comp_1050_fixed.csv")

    if not os.path.exists(data_path):
        st.error(f"Error: Data file not found at {data_path}")
        return pd.DataFrame()

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
    """Selects the next question using controlled Bloom progression/demotion."""
    bloom_levels = list(topic_df['bloom_level'].cat.categories)
    mastery = st.session_state.topic_mastery[selected_topic]
    confidence_log = st.session_state.confidence_record[selected_topic]
    current_bloom = st.session_state.current_bloom

    # --- 1. Remediation takes priority ---
    if st.session_state.remediation_queue:
        target = st.session_state.remediation_queue.pop(0)
        qs = topic_df[(topic_df["sub_concept"] == target) &
                      (~topic_df["question_id"].isin(st.session_state.asked_qs))]
        if not qs.empty:
            chosen = qs.sample(1).iloc[0]
            st.session_state.current_bloom = chosen["bloom_level"]
            return chosen, f"🎯 Targeting weak area: {target}"

    # --- 2. Freeze starting Bloom if any frozen questions remain ---
    if st.session_state.get("freeze_bloom_count", 0) > 0:
        freeze_qs = topic_df[(topic_df["bloom_level"] == current_bloom) &
                              (~topic_df["question_id"].isin(st.session_state.asked_qs))]
        if not freeze_qs.empty:
            q = freeze_qs.sample(1).iloc[0]
            st.session_state.freeze_bloom_count -= 1
            return q, f"🎯 Frozen question at {current_bloom}"
        # fallback if no frozen questions available
        st.session_state.freeze_bloom_count = 0

    # --- 3. Adaptive logic: one-level progression/demotion ---
    current_idx = bloom_levels.index(current_bloom)

    # Look at recent confidence log for current Bloom
    records = confidence_log.get(current_bloom, [])
    recent = records[-3:] if records else []

    # --- Check if demotion needed ---
    high_conf_wrong = [r for r in recent if not r['correct'] and r['confidence'] >= 4]
    if high_conf_wrong and current_idx > 0:
        lower_level = bloom_levels[current_idx - 1]
        demotion_qs = topic_df[(topic_df["bloom_level"] == lower_level) &
                               (~topic_df["question_id"].isin(st.session_state.asked_qs))]
        if not demotion_qs.empty:
            chosen = demotion_qs.sample(1).iloc[0]
            st.session_state.current_bloom = lower_level
            return chosen, f"🔻 Demoting to {lower_level} due to recent mistakes"

    # --- Check if promotion possible ---
    low_conf_right = [r for r in recent if r['correct'] and r['confidence'] <= 2]
    if not high_conf_wrong and len(recent) >= 2 and current_idx < len(bloom_levels)-1:
        # if mostly high confidence correct, move up one level
        if all(r['correct'] and r['confidence'] >= 3 for r in recent):
            higher_level = bloom_levels[current_idx + 1]
            level_qs = topic_df[(topic_df["bloom_level"] == higher_level) &
                                (~topic_df["question_id"].isin(st.session_state.asked_qs))]
            if not level_qs.empty:
                chosen = level_qs.sample(1).iloc[0]
                st.session_state.current_bloom = higher_level
                return chosen, f"⬆️ Progressing to {higher_level}"

    # --- 4. Stay at current Bloom if nothing else ---
    level_qs = topic_df[(topic_df["bloom_level"] == current_bloom) &
                        (~topic_df["question_id"].isin(st.session_state.asked_qs))]
    if not level_qs.empty:
        chosen = level_qs.sample(1).iloc[0]
        return chosen, f"➡️ Continuing at {current_bloom}"

    # --- 5. Fallback: pick any remaining question ---
    remaining_qs = topic_df[~topic_df["question_id"].isin(st.session_state.asked_qs)]
    if not remaining_qs.empty:
        chosen = remaining_qs.sample(1).iloc[0]
        st.session_state.current_bloom = chosen["bloom_level"]
        return chosen, f"🎯 Picking remaining question at {chosen['bloom_level']}"

    return None, "🎉 All questions completed!"



# --- UI ---
def render_sidebar(topics):
    st.sidebar.title("📚 Adaptive Java Practice")
    
    # Role selection
    role = st.sidebar.radio("Select Role", ["Student", "Teacher"])
    
    # Topic selection
    selected_topic = st.sidebar.selectbox("📘 Choose a Topic", topics)

    # Review bookmarked
    if st.sidebar.button("🔖 Review Bookmarked"):
        st.session_state.review_mode = True
        st.session_state.submitted = False
        st.session_state.current_question = None
        st.rerun()

    # Back to Main Menu
    if st.sidebar.button("⬅️ Back to Main Menu"):
        st.session_state.selected_mode = None
        st.session_state.started = False
        st.session_state.current_question = None
        st.session_state.submitted = False
        st.session_state.review_mode = False
        st.rerun()
    
    # Reset session
    if st.sidebar.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Teacher view
    if role == "Teacher":
        st.session_state.started = False
        st.session_state.review_mode = False
        st.session_state.current_question = None
        render_teacher_view()
        st.stop()

    return role,selected_topic


##############################
# Student View
##############################

def render_student_view(df):
    LEVEL_ORDER = ["Beginner", "Intermediate", "Advanced"]

    # --- Load student info ---
    if "student_level" not in st.session_state:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            student_csv_path = os.path.join(script_dir, "..", "data", "student_list.csv")

            # Load as strings ALWAYS
            students_df = pd.read_csv(student_csv_path, dtype=str)

            student_id = st.session_state.get("user_id", None)

            if student_id:
                # Compare as strings, never int()
                level_series = students_df.loc[
                    students_df["student_id"] == str(student_id),
                    "level"
                ]

                if not level_series.empty:
                    st.session_state.student_level = level_series.iloc[0]
                else:
                    st.session_state.student_level = "Beginner"
            else:
                st.session_state.student_level = "Beginner"

        except Exception as e:
            st.warning(f"Could not load student info CSV: {e}")
            st.session_state.student_level = "Beginner"


    student_level = st.session_state.student_level

    # --- Initialize current Bloom level in session state (starting point only) ---
    starting_bloom = {
        "Beginner": "Remember",
        "Intermediate": "Apply",
        "Advanced": "Evaluate"
    }
    if "current_bloom" not in st.session_state:
        st.session_state.current_bloom = starting_bloom.get(student_level, "Remember")
        # Freeze Bloom for first 2 questions (adjustable)
        st.session_state.freeze_bloom_count = 0

    # --- Map level to allowed topics ---
    level_mapping = {
        "Beginner": 7,
        "Intermediate": 11,
        "Advanced": len(df['topic'].cat.categories)
    }
    max_idx = level_mapping.get(student_level, 7)
    allowed_topics = df['topic'].cat.categories[:max_idx]

    # --- Sidebar & topic selection ---
    role, selected_topic = render_sidebar([t for t in df['topic'].dropna().unique() if t in allowed_topics])
    if selected_topic is None:
        return

    st.title(f"🎓 Java Learning – Student Mode")
    st.subheader(f"📘 Topic: {selected_topic} | Level: {student_level}")

    # --- Initialize persistent states ---
    st.session_state.topic_mastery.setdefault(selected_topic, {b: 0 for b in df['bloom_level'].cat.categories})
    st.session_state.score.setdefault(selected_topic, {b: {"correct": 0, "total": 0} for b in df['bloom_level'].cat.categories})
    st.session_state.confidence_record.setdefault(selected_topic, {})
    st.session_state.question_count = st.session_state.get("question_count", 0)
    st.session_state.session_done = st.session_state.get("session_done", False)
    st.session_state.current_question = st.session_state.get("current_question", None)
    st.session_state.submitted = st.session_state.get("submitted", False)
    st.session_state.review_mode = st.session_state.get("review_mode", False)
    st.session_state.incorrect_review_queue = st.session_state.get("incorrect_review_queue", [])
    st.session_state.asked_qs = st.session_state.get("asked_qs", set())

    # --- Start learning button ---
    if not st.session_state.get("started", False):
        st.info("Click 'Start Learning' to begin.")
        if st.button("🚀 Start Learning"):
            st.session_state.started = True
            st.rerun()
        return

    # --- Handle session completion ---
    if st.session_state.session_done:
        show_session_summary(selected_topic)
        return

    # --- Filter questions for current topic ---
    topic_df = df[df["topic"] == selected_topic]
    if topic_df.empty:
        st.warning("No questions available for this topic.")
        return

    # --- Question flow ---
    if st.session_state.current_question is None and not st.session_state.submitted:
        if st.session_state.review_mode and st.session_state.incorrect_review_queue:
            next_qid = st.session_state.incorrect_review_queue.pop(0)
            q = df[df["question_id"] == next_qid].iloc[0]
            st.session_state.current_question = q.to_dict()
            st.session_state.current_reason = "🔄 Reviewing an incorrect question."
            st.session_state.current_bloom = q["bloom_level"]
        else:
            if st.session_state.review_mode:
                st.success("✅ All incorrect questions reviewed!")
                st.session_state.review_mode = False
                st.session_state.session_done = True
                st.rerun()
            else:
                # --- Use freeze Bloom logic if any remaining ---
                if st.session_state.freeze_bloom_count > 0:
                    freeze_qs = topic_df[
                        (topic_df["bloom_level"] == st.session_state.current_bloom) &
                        (~topic_df["question_id"].isin(st.session_state.asked_qs))
                    ]
                    if not freeze_qs.empty:
                        q = freeze_qs.sample(1).iloc[0]
                        reason = f"🎯 Starting at {st.session_state.current_bloom} level for {student_level} learner."
                        st.session_state.freeze_bloom_count -= 1
                    else:
                        # fallback to adaptive logic if no questions available at starting Bloom
                        q, reason = get_next_question(topic_df, selected_topic)
                else:
                    q, reason = get_next_question(topic_df, selected_topic)

                if q is not None:
                    st.session_state.current_question = q.to_dict()
                    st.session_state.current_reason = reason
                    st.session_state.asked_qs.add(q["question_id"])
                    st.session_state.current_bloom = q["bloom_level"]
  # update Bloom for subsequent questions


    # --- Display current question ---
    if st.session_state.current_question:
        display_question(pd.Series(st.session_state.current_question), selected_topic)
        if st.session_state.submitted:
            st.session_state.question_count += 1
            # End session after 10 questions (normal mode)
            if st.session_state.question_count >= 10 and not st.session_state.review_mode:
                st.session_state.session_done = True
                st.rerun()



def show_session_summary(topic):
    """Displays session summary and handles promotion/demotion once per session."""

    # Filter session questions for current session
    session_log = [
        log for log in st.session_state.log
        if log['session_id'] == st.session_state.session_id
    ]

    if not session_log:
        st.info("No questions answered this session.")
        return

    total = len(session_log)
    correct = sum([l['is_correct'] for l in session_log])
    accuracy = correct / total * 100

    st.markdown(f"### 📊 Session Summary\n**Accuracy:** {accuracy:.1f}%\n")

    # Bloom-level breakdown
    bloom_counts = {}
    for log in session_log:
        b = log["bloom_level"]
        bloom_counts.setdefault(b, {"correct": 0, "total": 0})
        bloom_counts[b]["total"] += 1
        bloom_counts[b]["correct"] += int(log["is_correct"])

    st.markdown("**Bloom Level Breakdown:**\n")
    for b, stats in bloom_counts.items():
        pct = stats["correct"] / stats["total"] * 100
        st.markdown(f"- {b}: {stats['correct']}/{stats['total']} correct ({pct:.1f}%)")

    # Identify incorrect questions
    incorrect_questions = [log['question_id'] for log in session_log if not log['is_correct']]

    # Review incorrect questions button
    if incorrect_questions and not st.session_state.get("review_mode", False):
        if st.button("🔁 Review Incorrect Questions", key="review_incorrect"):
            st.session_state.review_mode = True
            st.session_state.incorrect_review_queue = incorrect_questions.copy()
            st.session_state.current_question = None
            st.session_state.submitted = False
            st.session_state.question_count = 0
            st.session_state.session_done = False
            st.rerun()

    # --- Student-level promotion/demotion (run once per session) ---
    if not st.session_state.get("session_summary_done", False):
        LEVEL_ORDER = ["Beginner", "Intermediate", "Advanced"]
        current_level = st.session_state.student_level
        idx = LEVEL_ORDER.index(current_level)

        if accuracy >= 70 and idx < len(LEVEL_ORDER) - 1:
            st.session_state.student_level = LEVEL_ORDER[idx + 1]
            st.info(f"🎉 Well done! You've been promoted to **{LEVEL_ORDER[idx + 1]}**.")
        elif accuracy <= 50 and idx > 0:
            st.session_state.student_level = LEVEL_ORDER[idx - 1]
            st.info(f"🙂 Let's reinforce your skills. You've been adjusted to **{LEVEL_ORDER[idx - 1]}**.")

        # --- Update current Bloom level in session and CSV ---
        # Use the last question's bloom level as the new current Bloom
        last_bloom = session_log[-1]["bloom_level"]
        st.session_state.current_bloom = last_bloom

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        student_csv_path = os.path.join(script_dir, "..", "data", "student_list.csv")

        # ALWAYS read IDs as strings
        students_df = pd.read_csv(student_csv_path, dtype=str)

        student_id = st.session_state.get("user_id")

        if student_id:
            # MATCH AS STRING ONLY — NO int()
            mask = students_df["student_id"] == str(student_id)

            students_df.loc[mask, "level"] = st.session_state.student_level
            students_df.loc[mask, "current_bloom"] = st.session_state.current_bloom

            students_df.to_csv(student_csv_path, index=False)

    except Exception as e:
        st.warning(f"Could not update student CSV: {e}")


        st.session_state.session_summary_done = True  # prevents double promotion/demotion

    # Start new session button
    if st.button("🔁 Start New Session", key="start_new_session"):
        reset_session_for_topic(topic)
        st.rerun()



def reset_session_for_topic(selected_topic):
    """Resets session states for a new session without affecting global login info."""
    st.session_state.started = False
    st.session_state.current_question = None
    st.session_state.submitted = False
    st.session_state.review_mode = False
    st.session_state.question_count = 0
    st.session_state.session_done = False
    st.session_state.asked_qs = set()
    
    # Mark that session summary is not yet applied
    st.session_state.session_summary_done = False
    
    # Remove old session log for this session to prevent double promotion/demotion
    current_session = st.session_state.session_id
    st.session_state.log = [log for log in st.session_state.log if log['session_id'] != current_session]
    
    # Increment session ID for a fresh session
    st.session_state.session_id = str(int(time.time()))



def display_question(q, topic):
    st.info(st.session_state.current_reason)
    st.markdown(f"**Bloom:** '{q['bloom_level']}' | **Question Id:** '{q['question_id']}') | **Sub-Concept:** '{q.get('sub_concept', 'General')}' ")
    
    
    st.code(q['question_stem'], language='java')

    col1, col2 = st.columns(2)

    with col1:
        choice = st.radio(
            "Choose:", 
            ["a", "b", "c", "d"], 
            index=None,
            format_func=lambda x: f"{x.upper()}. {q[f'option_{x}']}",
            key=f"choice_{q['question_id']}"
        )

    with col2:
        confidence = st.radio(
            "How confident are you?",
            ["😬 Guessing", "🤔 Unsure", "😐 Okay", "🙂 Confident", "😎 Very Confident"],
            index=None,
            key=f"conf_{q['question_id']}"
        )



    # CONFIDENCE_MAP = {
    #     "😬 Guessing": 1,
    #     "🤔 Unsure": 2,
    #     "😐 Okay": 3,
    #     "🙂 Confident": 4,
    #     "😎 Very Confident": 5
    # }



    submit_col, bookmark_col = st.columns(2)
    if bookmark_col.button("🔖 Bookmark", use_container_width=True):
        st.session_state.bookmarked.add(q['question_id'])
        st.success("Bookmarked.")

    if submit_col.button("✅ Submit", use_container_width=True) and not st.session_state.submitted:
        if choice is None:
            st.warning("Choose an option.")
        
        elif confidence is None:
            st.warning("Choose your confidence level.")

        else:
            # handle_submission(q, choice, confidence, topic)
            numeric_conf = CONFIDENCE_MAP.get(confidence, 3)  # default to neutral
            handle_submission(q, choice, numeric_conf, topic)

            st.session_state.submitted = True
            # st.rerun()

    if st.session_state.submitted:
        if st.button("➡️ Next Question"):
            st.session_state.submitted = False
            st.session_state.current_question = None
            st.rerun()



def handle_submission(q, choice, confidence, topic):
    """Handles logic after a user submits an answer, without tracking explanation clicks, and avoids duplicate CSV entries."""

    # --- Correctness and Scoring ---
    correct = q["correct_option"].strip().lower()
    is_correct = (choice == correct)
    bloom = q["bloom_level"]
    points = 0.5 + (confidence / 10.0) if is_correct else 0

    if is_correct:
        st.success("✅ Correct!")
    else:
        st.error(f"❌ Incorrect. The correct answer is **{correct.upper()}**.")
        st.session_state.wrong_qs.append(q["question_id"])

    # --- Update Scores ---
    st.session_state.topic_mastery[topic][bloom] += points
    st.session_state.score[topic][bloom]["total"] += 1
    st.session_state.score[topic][bloom]["correct"] += int(is_correct)

    # --- Confidence Logging ---
    st.session_state.confidence_record[topic].setdefault(bloom, []).append({
        'question_id': q['question_id'],
        'confidence': confidence,
        'correct': is_correct
    })

    # --- Explanation Section ---
    with st.expander("📘 View Explanation", expanded=False):
        st.markdown(q["main_explanation"])

    # --- More Info Section ---
    with st.expander("🔗 More Info"):
        st.markdown(
            "Learn more about Java concepts at 👉 [W3Schools Java Tutorial](https://www.w3schools.com/java/default.asp)",
            unsafe_allow_html=True
        )

    # --- Logging the Result ---
    student_id = st.session_state.get("user_id", "unknown_student")
    session_id = st.session_state.get("session_id", "unknown_session")

    # Initialize logged_questions to prevent duplicates
    if "logged_questions" not in st.session_state:
        st.session_state.logged_questions = set()

    qid = q["question_id"]
    if qid in st.session_state.logged_questions:
        return  # already logged this question, skip

    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "student_id": student_id,
        "session_id": session_id,
        "topic": topic,
        "bloom_level": bloom,
        "question_id": qid,
        "selected": choice,
        "correct_option": correct,
        "is_correct": is_correct,
        "confidence": confidence,
        "reinforcement_reason": st.session_state.get("current_reason", ""),
    }

    # Add to in-memory log and mark as logged
    st.session_state.log.append(log_entry)
    st.session_state.logged_questions.add(qid)

    # --- Save CSV (append only new row) ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    data_dir = os.path.join(project_root, "logs")
    os.makedirs(data_dir, exist_ok=True)

    student_csv = os.path.join(data_dir, f"student_{student_id}.csv")
    new_df = pd.DataFrame([log_entry])
    new_df.to_csv(student_csv, mode='a', header=not os.path.exists(student_csv), index=False)


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



import glob
# import matplotlib.pyplot as plt

def render_teacher_view():
    st.title("👩‍🏫 Teacher Dashboard")
    st.markdown("Explore learner performance across topics, Bloom levels, and time dimensions.")

    log_files = glob.glob("logs/session_*.csv")
    if not log_files:
        st.warning("No session logs found yet. Students must complete at least one session.")
        return

    df_logs = pd.concat([pd.read_csv(f) for f in log_files], ignore_index=True)
    # df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'], unit='s')
    # --- Timestamp normalization ---
    try:
        # Handle mixed formats: numeric epochs and human-readable strings
        df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'], errors='coerce', unit='s')
        if df_logs['timestamp'].isna().any():
            # Retry parsing those that failed (likely formatted datetimes)
            df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'], errors='coerce', infer_datetime_format=True)
    except Exception:
        df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'], errors='coerce', infer_datetime_format=True)

    # Drop any invalid timestamps
    df_logs = df_logs.dropna(subset=['timestamp'])

    
    df_logs = calculate_difficulty(df_logs)
    df_logs, gain = calculate_learning_gains(df_logs)


    # --- STUDENT FILTER ---
    st.sidebar.subheader("🧑‍🎓 Filter by Student ID")
    unique_students = df_logs['session_id'].unique().tolist()
    selected_student = st.sidebar.selectbox("Select a Student", ["All Students"] + unique_students)

    if selected_student != "All Students":
        df_logs = df_logs[df_logs['session_id'] == selected_student]
        st.info(f"Showing data for student session: `{selected_student}`")

    # --- BLOOM MASTER HEATMAP ---
    st.subheader("📘 Bloom Level Accuracy by Topic")
    if 'bloom_level' in df_logs.columns:
        pivot = df_logs.groupby(['topic', 'bloom_level'])['correct'].mean().unstack().fillna(0) * 100
        st.dataframe(pivot.style.background_gradient(cmap='YlGn').format("{:.0f}%"))

    # --- TOPIC COVERAGE ---
    st.subheader("📊 Topic Coverage (Questions Attempted)")
    topic_counts = df_logs.groupby('topic')['question_id'].nunique().sort_values(ascending=False)
    st.bar_chart(topic_counts)

    # --- CONFIDENCE VS ACCURACY ---
    st.subheader("🧠 Confidence vs Accuracy Trend")
    df_logs['conf_bin'] = pd.cut(df_logs['confidence'], bins=[0,2,3,4,5], labels=["Low", "Medium", "High", "Very High"])
    conf_acc = df_logs.groupby(['conf_bin'])['correct'].mean() * 100
    st.line_chart(conf_acc)

    # --- MISCONCEPTION HEATMAP ---
    st.subheader("🎯 Misconception Heatmap – Tricky Sub-Concepts")
    if 'sub_concept' in df_logs.columns:
        miscon_df = df_logs[df_logs['confidence'] >= 4]  # high confidence wrong answers
        miscon_df = miscon_df[miscon_df['correct'] == False]
        misconcept = miscon_df.groupby('sub_concept')['question_id'].count().sort_values(ascending=False).head(10)
        if not misconcept.empty:
            st.bar_chart(misconcept)
        else:
            st.info("No significant misconceptions detected yet.")

    # --- TIME ON TASK ---
    st.subheader("🕓 Time-on-Task Analysis")
    df_logs = df_logs.sort_values(['session_id', 'timestamp'])
    df_logs['time_diff'] = df_logs.groupby('session_id')['timestamp'].diff().dt.total_seconds()
    time_summary = df_logs.groupby('session_id')['time_diff'].sum().fillna(0)
    st.write("Average session duration (mins):", round(time_summary.mean() / 60, 2))
    st.bar_chart(time_summary / 60)

    # --- BLOOM PROGRESS OVER TIME ---
    st.subheader("📈 Per-Bloom Progress Timeline")
    df_time_bloom = (
        df_logs.groupby(['bloom_level', pd.Grouper(key='timestamp', freq='1min')])['correct']
        .mean().unstack(fill_value=None) * 100
    )
    if not df_time_bloom.empty:
        st.line_chart(df_time_bloom)
    else:
        st.info("Insufficient data to generate timeline trends.")

    st.subheader("⚖️ Item Difficulty Overview")
    difficulty_summary = df_logs.groupby("bloom_level")["difficulty_score"].mean().round(2)
    st.bar_chart(difficulty_summary)
    st.markdown("Higher bars = higher relative complexity (weighted by Bloom level + empirical error).")

    st.subheader("📈 Learning Gain per Bloom Level")
    st.markdown("Tracks student improvement across Bloom levels using rolling accuracy windows.")
    if not gain.empty:
        gain_summary = gain.groupby("bloom_level")["learning_gain"].mean().round(1)
        st.dataframe(gain_summary.rename("Avg Learning Gain (%)"))
        st.line_chart(gain_summary)
    else:
        st.info("Insufficient data for gain trend analysis.")


    st.subheader("🎯 Weighted Mastery Score (Cognitive Value Index)")
    df_logs["weighted_correct"] = df_logs["correct"] * df_logs["difficulty_score"]
    mastery = df_logs.groupby("session_id")["weighted_correct"].mean() * 100
    st.bar_chart(mastery)
    st.caption("This metric rewards success on harder questions more heavily.")


    # --- STUDENT SUMMARY TABLE ---
    st.subheader("🧾 Student Summary Overview")
    if "session_id" in df_logs.columns:
        student_summary = df_logs.groupby('session_id').agg({
            'question_id': 'count',
            'correct': 'mean',
            'confidence': 'mean',
            'time_diff': 'sum'
        }).rename(columns={
            'question_id': 'Questions',
            'correct': 'Accuracy',
            'confidence': 'Avg_Confidence',
            'time_diff': 'Time_Spent_Seconds'
        })
        student_summary['Time_Spent_Minutes'] = (student_summary['Time_Spent_Seconds'] / 60).round(2)
        st.dataframe(student_summary.style.format({
            "Accuracy": "{:.1%}",
            "Avg_Confidence": "{:.2f}",
            "Time_Spent_Minutes": "{:.2f}"
        }))



if __name__ == "__main__":
    main()
    
def run_student_mode():
    df = load_data()
    if df.empty:
        st.warning("No data available. Please check the CSV file.")
        return
    initialize_session_state()
    render_student_view(df)









