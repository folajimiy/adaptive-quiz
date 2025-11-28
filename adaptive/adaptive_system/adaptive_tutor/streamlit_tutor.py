import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime

# ==========================
# CONFIG
# ==========================

CSV_PATH = "data/java_questions_adaptive_clean.csv"

BLOOM_ORDER = ["Remember", "Understand", "Apply", "Analyze", "Evaluate"]

TOPIC_MAP = {
    "Java Fundamentals": [
        "Variables and data types",
        "Operators and expressions",
        "Basic syntax and structure",
        "Wrapper classes and autoboxing"
    ],
    "Control Flow": [
        "if/else branching",
        "switch statements",
        "boolean expressions"
    ],
    "Loops": [
        "for loop",
        "while loop",
        "do-while loop",
        "loop control (break/continue)"
    ],
    "Arrays": [
        "1D arrays",
        "Array iteration",
        "Common array errors"
    ],
    "Strings": [
        "String immutability",
        "Common String methods",
        "String comparison and interning"
    ],
    "Methods": [
        "Parameter passing",
        "Method signatures",
        "Return values"
    ],
    "Objects and Classes": [
        "Fields and methods",
        "Constructors",
        "Instance vs static members",
        "toString and equals overrides"
    ],
    "Encapsulation": [
        "Access modifiers",
        "Getters and setters",
        "Information hiding"
    ],
    "Inheritance": [
        "Superclass/subclass relationships",
        "Method overriding",
        "super keyword"
    ],
    "Polymorphism": [
        "Dynamic dispatch",
        "Upcasting and downcasting",
        "Method binding"
    ],
    "Abstract Classes": [
        "Abstract methods",
        "Partial implementation",
        "Concrete subclasses of abstract classes"
    ],
    "Interfaces": [
        "Interface contracts",
        "Implementing multiple interfaces",
        "Functional interfaces"
    ],
    "Generics": [
        "Generic classes",
        "Generic methods",
        "Type parameters and type safety"
    ],
    "Collections": [
        "Lists (e.g., ArrayList)",
        "Sets and uniqueness",
        "Maps (key-value pairs)",
        "Iteration over collections"
    ],
    "JavaFX": [
        "Scene graph",
        "UI controls and layout",
        "Stages and scenes"
    ],
    "Event-Driven Programming": [
        "Event handlers",
        "Listener patterns",
        "JavaFX event handling with lambdas"
    ]
}

MISCONCEPTIONS = {
    "string_mutable",
    "use_double_equals_for_string_content",
    "constructor_has_return_type",
    "interface_has_state",
    "override_vs_overload",
    "polymorphism_static_binding",
}

MISCONCEPTION_TO_CONCEPTS = {
    "string_mutable": ["Strings::String immutability"],
    "use_double_equals_for_string_content": ["Strings::String comparison and interning"],
    "constructor_has_return_type": ["Objects and Classes::Constructors"],
    "interface_has_state": ["Interfaces::Interface contracts"],
    "override_vs_overload": [
        "Methods::Method signatures",
        "Inheritance::Method overriding"
    ],
    "polymorphism_static_binding": [
        "Polymorphism::Dynamic dispatch"
    ]
}

# ==========================
# DATA LOADING
# ==========================

@st.cache_data
def load_item_bank(path: str):
    df = pd.read_csv(path)
    if "misconception_tags_per_option" not in df.columns:
        df["misconception_tags_per_option"] = "{}"
    return df

# ==========================
# STUDENT MODEL
# ==========================

def initialize_student_state():
    misconception_belief = {m: 0.1 for m in MISCONCEPTIONS}

    concept_mastery = {}
    bloom_profile = {}
    for topic, subs in TOPIC_MAP.items():
        for sub in subs:
            cid = f"{topic}::{sub}"
            concept_mastery[cid] = 0.4
            bloom_profile[cid] = "Remember"

    return {
        "misconception_belief": misconception_belief,
        "concept_mastery": concept_mastery,
        "bloom_profile": bloom_profile,
        "history": []
    }

# ==========================
# HELPER FUNCTIONS
# ==========================

def misconception_priority(student_state):
    scores = {}
    for m, p in student_state["misconception_belief"].items():
        scores[m] = p * (1 - p)
    return sorted(scores.keys(), key=lambda m: scores[m], reverse=True)

def concepts_for_misconceptions(target_miscs):
    concepts = set()
    for m in target_miscs:
        concepts.update(MISCONCEPTION_TO_CONCEPTS.get(m, []))
    return list(concepts)

def pick_weakest_concept(student_state, candidate_concepts):
    if not candidate_concepts:
        return None
    return min(
        candidate_concepts,
        key=lambda c: student_state["concept_mastery"].get(c, 0.4)
    )

def next_bloom_level(current_level, goal="Apply"):
    idx = BLOOM_ORDER.index(current_level)
    goal_idx = BLOOM_ORDER.index(goal)
    if idx >= goal_idx:
        return BLOOM_ORDER[min(idx + 1, len(BLOOM_ORDER) - 1)]
    else:
        return BLOOM_ORDER[min(idx + 1, len(BLOOM_ORDER) - 1)]

def allowed_bloom_levels(target_bloom, window=1):
    idx = BLOOM_ORDER.index(target_bloom)
    levels = {target_bloom}
    if idx - 1 >= 0:
        levels.add(BLOOM_ORDER[idx - 1])
    if idx + 1 < len(BLOOM_ORDER):
        levels.add(BLOOM_ORDER[idx + 1])
    return levels

def item_has_misconception(row, target_miscs):
    tags_json = row.get("misconception_tags_per_option", "{}")
    try:
        tags = json.loads(tags_json)
    except Exception:
        return False
    all_tags = set()
    for opt_tags in tags.values():
        all_tags.update(opt_tags)
    return len(all_tags.intersection(target_miscs)) > 0

def candidate_items_for(items_df, topic, subtopic, target_miscs, target_bloom):
    df = items_df[
        (items_df["topic"] == topic) &
        (items_df["subtopic"] == subtopic)
    ]
    df = df[df["bloom_level"].isin(allowed_bloom_levels(target_bloom))]
    if target_miscs:
        df = df[df.apply(lambda r: item_has_misconception(r, target_miscs), axis=1)]
    return df

def score_item(row, student_state, target_miscs):
    desired_diff = 3
    diff = row.get("predicted_difficulty_level", 3)
    if pd.isna(diff):
        diff = 3
    diff_score = 1.0 - abs(diff - desired_diff) / 4.0

    rel = row.get("eval_relevance", 3)
    acc = row.get("eval_accuracy", 3)
    expl = row.get("eval_explainability", 3)
    for v in (rel, acc, expl):
        if pd.isna(v):
            v = 3
    quality = (rel + acc + expl) / 15.0

    tags_json = row.get("misconception_tags_per_option", "{}")
    try:
        tags = json.loads(tags_json)
    except Exception:
        tags = {}
    all_tags = set()
    for opt_tags in tags.values():
        all_tags.update(opt_tags)
    coverage = len(all_tags.intersection(target_miscs))
    coverage_score = min(coverage, 3) / 3.0 if coverage > 0 else 0.0

    seen_ids = {h["item_id"] for h in student_state["history"]}
    novelty = 0.0 if row["id"] in seen_ids else 1.0

    return (
        0.4 * coverage_score +
        0.3 * quality +
        0.2 * diff_score +
        0.1 * novelty
    )

def select_next_item(student_state, items_df):
    miscs_by_priority = misconception_priority(student_state)
    target_miscs = miscs_by_priority[:3]

    candidate_concepts = concepts_for_misconceptions(target_miscs)
    if not candidate_concepts:
        candidate_concepts = list(student_state["concept_mastery"].keys())

    concept = pick_weakest_concept(student_state, candidate_concepts)
    if concept is None:
        return None, None, None

    topic, subtopic = concept.split("::")

    current_bloom = student_state["bloom_profile"].get(concept, "Remember")
    target_bloom = next_bloom_level(current_bloom, goal="Apply")

    df = candidate_items_for(items_df, topic, subtopic, set(target_miscs), target_bloom)
    if df.empty:
        df = items_df

    best_row = None
    best_idx = None
    best_score = -1
    for idx, row in df.iterrows():
        s = score_item(row, student_state, set(target_miscs))
        if s > best_score:
            best_score = s
            best_row = row
            best_idx = idx

    return best_row, best_idx, (topic, subtopic)

def infer_misconceptions_triggered(item_row, chosen_option):
    try:
        tags = json.loads(item_row.get("misconception_tags_per_option", "{}"))
    except Exception:
        return []
    return tags.get(chosen_option, [])

def update_misconceptions(student_state, item_row, chosen_option, is_correct):
    tags_json = item_row.get("misconception_tags_per_option", "{}")
    try:
        tags = json.loads(tags_json)
    except Exception:
        tags = {}

    all_miscs = set()
    for opt, miscs in tags.items():
        all_miscs.update(miscs)

    chosen_miscs = set(tags.get(chosen_option, []))

    for m in all_miscs:
        p = student_state["misconception_belief"].get(m, 0.1)
        if (m in chosen_miscs) and not is_correct:
            p_new = p + 0.3 * (1 - p)
        else:
            p_new = p - 0.1 * p
        student_state["misconception_belief"][m] = max(0.0, min(1.0, p_new))

def update_concept_mastery(student_state, topic, subtopic, is_correct):
    c = f"{topic}::{subtopic}"
    m = student_state["concept_mastery"].get(c, 0.4)
    if is_correct:
        m_new = m + 0.2 * (1 - m)
    else:
        m_new = m - 0.2 * m
    student_state["concept_mastery"][c] = max(0.0, min(1.0, m_new))

def update_bloom_profile(student_state, topic, subtopic, bloom_level, is_correct):
    if not is_correct:
        return
    c = f"{topic}::{subtopic}"
    current = student_state["bloom_profile"].get(c, "Remember")
    if BLOOM_ORDER.index(bloom_level) > BLOOM_ORDER.index(current):
        student_state["bloom_profile"][c] = bloom_level

def update_student_model(student_state, item_row, chosen_option, response_time):
    is_correct = (chosen_option == item_row["correct_answer"])
    topic = item_row["topic"]
    subtopic = item_row["subtopic"]
    bloom = item_row["bloom_level"]

    update_misconceptions(student_state, item_row, chosen_option, is_correct)
    update_concept_mastery(student_state, topic, subtopic, is_correct)
    update_bloom_profile(student_state, topic, subtopic, bloom, is_correct)

    student_state["history"].append({
        "item_id": item_row["id"],
        "chosen_option": chosen_option,
        "is_correct": is_correct,
        "response_time": response_time,
        "topic": topic,
        "subtopic": subtopic,
        "bloom": bloom,
        "timestamp": datetime.now().isoformat(),
        "misconceptions_triggered": infer_misconceptions_triggered(item_row, chosen_option),
    })

# ==========================
# MULTI-STUDENT SUPPORT
# ==========================

if "ALL_STUDENTS" not in st.session_state:
    st.session_state.ALL_STUDENTS = {}

def ensure_session_state(items_df):
    st.sidebar.subheader("Student Login")

    existing_ids = list(st.session_state.ALL_STUDENTS.keys())
    selected = st.sidebar.selectbox(
        "Choose existing student or <new>:",
        ["<new>"] + existing_ids
    )

    if selected == "<new>":
        new_id = st.sidebar.text_input("Enter new Student ID:", value="student_001")
        if new_id:
            st.session_state.student_id = new_id
    else:
        st.session_state.student_id = selected

    if st.session_state.student_id not in st.session_state.ALL_STUDENTS:
        st.session_state.ALL_STUDENTS[st.session_state.student_id] = initialize_student_state()

    st.session_state.student_state = st.session_state.ALL_STUDENTS[st.session_state.student_id]

    if "items_df" not in st.session_state:
        st.session_state.items_df = items_df

    if "current_item_idx" not in st.session_state:
        st.session_state.current_item_idx = None

    if "feedback" not in st.session_state:
        st.session_state.feedback = None

    if "chosen_option" not in st.session_state:
        st.session_state.chosen_option = None

    if "question_start_time" not in st.session_state:
        st.session_state.question_start_time = None

def restart_current_student():
    st.session_state.ALL_STUDENTS[st.session_state.student_id] = initialize_student_state()
    st.session_state.student_state = st.session_state.ALL_STUDENTS[st.session_state.student_id]
    st.session_state.current_item_idx = None
    st.session_state.feedback = None
    st.session_state.chosen_option = None
    st.session_state.question_start_time = None

# ==========================
# STREAMLIT APP
# ==========================

def main():
    st.set_page_config(page_title="Adaptive Java Tutor", layout="wide")
    st.title("🧠 Adaptive Java Tutor (Misconception-driven)")

    try:
        items_df = load_item_bank(CSV_PATH)
    except Exception as e:
        st.error(f"Could not load CSV at '{CSV_PATH}': {e}")
        st.stop()

    ensure_session_state(items_df)

    tabs = st.tabs(["Tutor", "Admin Dashboard"])

    # ============ TUTOR TAB ============
    with tabs[0]:
        student_state = st.session_state.student_state

        with st.sidebar:
            st.header("Session Controls")
            st.write(f"Active Student: **{st.session_state.student_id}**")
            if st.button("🔁 Restart This Student's Session"):
                restart_current_student()
                st.success("Session restarted for this student.")

            st.markdown("---")
            st.subheader("Debug / Info")
            st.write(f"Questions answered: {len(student_state['history'])}")

            # Export section
            st.markdown("### Export")

            import io
            if student_state["history"]:
                csv_buffer = io.StringIO()
                pd.DataFrame(student_state["history"]).to_csv(csv_buffer, index=False)
                st.download_button(
                    label="⬇️ Download This Student History",
                    data=csv_buffer.getvalue(),
                    file_name=f"{st.session_state.student_id}_history.csv",
                    mime="text/csv"
                )

            all_histories = []
            for sid, state in st.session_state.ALL_STUDENTS.items():
                for record in state["history"]:
                    r = record.copy()
                    r["student_id"] = sid
                    all_histories.append(r)

            if all_histories:
                csv_buffer_all = io.StringIO()
                pd.DataFrame(all_histories).to_csv(csv_buffer_all, index=False)
                st.download_button(
                    label="⬇️ Download ALL Students History",
                    data=csv_buffer_all.getvalue(),
                    file_name="all_students_history.csv",
                    mime="text/csv"
                )

        col_main, col_side = st.columns([2, 1])

        with col_main:
            st.subheader("Question")

            if st.session_state.current_item_idx is None:
                if st.button("➡️ Get Next Question"):
                    row, idx, (topic, subtopic) = select_next_item(student_state, st.session_state.items_df)
                    if row is None:
                        st.warning("No suitable items found. You might be done!")
                    else:
                        st.session_state.current_item_idx = idx
                        st.session_state.feedback = None
                        st.session_state.chosen_option = None
                        st.session_state.question_start_time = datetime.now()
                else:
                    st.info("Click 'Get Next Question' to begin.")
            else:
                row = st.session_state.items_df.loc[st.session_state.current_item_idx]
                st.markdown(f"**Topic:** {row['topic']}  \n**Subtopic:** {row['subtopic']}  \n**Bloom:** {row['bloom_level']}")
                st.markdown(f"### {row['question_stem']}")

                options = {
                    "A": row["option_a"],
                    "B": row["option_b"],
                    "C": row["option_c"],
                    "D": row["option_d"],
                }

                st.session_state.chosen_option = st.radio(
                    "Choose an option:",
                    options=list(options.keys()),
                    format_func=lambda k: f"{k}) {options[k]}",
                    index=["A", "B", "C", "D"].index(st.session_state.chosen_option)
                    if st.session_state.chosen_option in ["A", "B", "C", "D"] else 0
                )

                submit = st.button("✅ Submit Answer")

                if submit:
                    if st.session_state.chosen_option is None:
                        st.warning("Please select an option.")
                    else:
                        start_time = st.session_state.question_start_time or datetime.now()
                        response_time = (datetime.now() - start_time).total_seconds()

                        is_correct = (st.session_state.chosen_option == row["correct_answer"])
                        update_student_model(
                            student_state,
                            row,
                            st.session_state.chosen_option,
                            response_time
                        )

                        st.session_state.feedback = {
                            "is_correct": is_correct,
                            "correct_answer": row["correct_answer"],
                            "explanation": row["main_explanation"],
                            "response_time": response_time
                        }

                if st.session_state.feedback:
                    fb = st.session_state.feedback
                    st.markdown("---")
                    if fb["is_correct"]:
                        st.success(f"✅ Correct! (in {fb['response_time']:.1f}s)")
                    else:
                        st.error(f"❌ Incorrect. Correct answer: {fb['correct_answer']}  \nAnswered in {fb['response_time']:.1f}s")
                    st.markdown(f"**Explanation:** {fb['explanation']}")

                    if st.button("➡️ Next Question"):
                        st.session_state.current_item_idx = None
                        st.session_state.feedback = None
                        st.session_state.chosen_option = None
                        st.session_state.question_start_time = None

        with col_side:
            st.subheader("Learning Analytics")

            mis_beliefs = student_state["misconception_belief"]
            st.markdown("**Top Misconceptions (belief)**")
            if mis_beliefs:
                mis_df = pd.DataFrame(
                    sorted(mis_beliefs.items(), key=lambda kv: kv[1], reverse=True)[:5],
                    columns=["misconception", "belief"]
                )
                st.dataframe(mis_df, use_container_width=True)
            else:
                st.write("No misconceptions tracked yet.")

            st.markdown("**Weakest Concepts**")
            cm = student_state["concept_mastery"]
            if cm:
                cm_sorted = sorted(cm.items(), key=lambda kv: kv[1])[:5]
                cm_df = pd.DataFrame(cm_sorted, columns=["concept", "mastery"])
                st.dataframe(cm_df, use_container_width=True)
            else:
                st.write("No concept mastery data yet.")

            st.markdown("**Sample Bloom Profile**")
            bp = student_state["bloom_profile"]
            if bp:
                sample_bp = list(bp.items())[:5]
                for c, level in sample_bp:
                    st.write(f"- {c}: {level}")
            else:
                st.write("No Bloom data yet.")

    # ============ ADMIN TAB ============
    with tabs[1]:
        st.title("📊 Admin Analytics Dashboard")

        all_students = st.session_state.ALL_STUDENTS

        if not all_students:
            st.info("No student data yet.")
        else:
            rows = []
            for sid, state in all_students.items():
                for r in state["history"]:
                    rr = r.copy()
                    rr["student_id"] = sid
                    rows.append(rr)

            if not rows:
                st.info("No interaction history yet.")
            else:
                df_hist = pd.DataFrame(rows)
                st.subheader("Most Commonly Triggered Misconceptions")
                mis_counts = {}

                for state in all_students.values():
                    for h in state["history"]:
                        for m in h.get("misconceptions_triggered", []):
                            mis_counts[m] = mis_counts.get(m, 0) + 1

                if mis_counts:
                    mis_df = pd.DataFrame(
                        sorted(mis_counts.items(), key=lambda kv: kv[1], reverse=True),
                        columns=["misconception", "count"]
                    )
                    st.bar_chart(mis_df.set_index("misconception"))
                else:
                    st.write("No misconception data yet.")

                st.subheader("Concept Mastery Distribution (Across Students)")
                mastery_rows = []
                for sid, state in all_students.items():
                    for concept, val in state["concept_mastery"].items():
                        mastery_rows.append({"student_id": sid, "concept": concept, "mastery": val})
                mastery_df = pd.DataFrame(mastery_rows)
                if not mastery_df.empty:
                    pivot_mastery = mastery_df.pivot_table(
                        index="concept", columns="student_id", values="mastery"
                    )
                    st.line_chart(pivot_mastery)
                else:
                    st.write("No mastery data available.")

                st.subheader("Bloom Readiness per Concept")
                bloom_rows = []
                for sid, state in all_students.items():
                    for concept, b in state["bloom_profile"].items():
                        bloom_rows.append({"student_id": sid, "concept": concept, "bloom": b})
                bloom_df = pd.DataFrame(bloom_rows)
                if not bloom_df.empty:
                    bloom_df["bloom_score"] = bloom_df["bloom"].apply(lambda x: BLOOM_ORDER.index(x))
                    pivot_bloom = bloom_df.pivot_table(
                        index="concept", columns="student_id", values="bloom_score"
                    )
                    st.area_chart(pivot_bloom)
                else:
                    st.write("No Bloom data available.")

if __name__ == "__main__":
    main()
