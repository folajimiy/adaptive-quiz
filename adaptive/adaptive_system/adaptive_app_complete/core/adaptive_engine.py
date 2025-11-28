# core/adaptive_engine.py

import pandas as pd
import streamlit as st
from core.mastery_engine import detect_weak_subconcepts
from core.skill_engine import get_skill_graph 
from core.data_access import load_student_logs




def _filter_unseen(df):
    """Filter only unseen questions."""
    asked = st.session_state.asked_qs
    return df[~df["question_id"].isin(asked)]
    


def _choose(df):
    """Return a random question or None."""
    if df.empty:
        return None
    q = df.sample(1).iloc[0]
    return q


import random  # put this at top of file, not inside the function

def get_next_question(topic_df: pd.DataFrame, topic: str):
    """
    Fully adaptive engine:
    1. Subconcept remediation
    2. Bloom progression (promotion/demotion)
    3. Skill-Graph readiness layer
    4. Ability-based difficulty selection (ELO-like)
    5. Bloom-tier fallback
    6. Global fallback
    """

    # ------------------------------------------------
    # 0. BOOTSTRAP
    # ------------------------------------------------
    if topic_df.empty:
        return None, "⚠️ No questions available for this topic."

    current_bloom = st.session_state.get("current_bloom", None)
    if current_bloom is None:
        # pick the lowest bloom in this topic
        try:
            current_bloom = topic_df["bloom_level"].cat.categories[0]
        except:
            current_bloom = topic_df["bloom_level"].unique()[0]
        st.session_state.current_bloom = current_bloom

    # Utility to filter unseen items
    def _filter_unseen(df):
        asked = st.session_state.asked_qs
        return df[~df["question_id"].isin(asked)] if not df.empty else df

    # Random selector
    def _choose(df):
        return df.sample(1).iloc[0]

    # ------------------------------------------------
    # 1. SUBCONCEPT REMEDIATION
    # ------------------------------------------------
    session_log = st.session_state.log
    weak_subconcepts = detect_weak_subconcepts(session_log, min_hits=2)

    if weak_subconcepts:
        target = weak_subconcepts[0]
        rem_df = _filter_unseen(topic_df[topic_df["subtopic"] == target])
        if not rem_df.empty:
            q = _choose(rem_df)
            return q, f"🎯 Targeting weak subconcept: {target}"

    # ------------------------------------------------
    # 2. BLOOM PROGRESSION (promotion/demotion)
    # ------------------------------------------------
    bloom_levels = (
        list(topic_df['bloom_level'].cat.categories)
        if hasattr(topic_df['bloom_level'], "cat")
        else sorted(topic_df['bloom_level'].unique())
    )

    # safety
    if current_bloom not in bloom_levels:
        current_bloom = bloom_levels[0]
        st.session_state.current_bloom = current_bloom

    idx = bloom_levels.index(current_bloom)

    # get recent confidence log
    conf_log = st.session_state.confidence_record.get(topic, {}).get(current_bloom, [])
    recent = conf_log[-3:] if conf_log else []

    # High-confidence errors → demotion
    if recent:
        high_conf_wrong = [r for r in recent if not r["correct"] and r["confidence"] >= 4]
        if high_conf_wrong and idx > 0:
            lower = bloom_levels[idx - 1]
            df_lower = _filter_unseen(topic_df[topic_df["bloom_level"] == lower])
            if not df_lower.empty:
                st.session_state.current_bloom = lower
                return _choose(df_lower), f"🔻 Demoting to easier Bloom: {lower}"

    # High-confidence success → promotion
    if len(recent) >= 2 and all(r["correct"] and r["confidence"] >= 3 for r in recent):
        if idx < len(bloom_levels) - 1:
            higher = bloom_levels[idx + 1]
            df_higher = _filter_unseen(topic_df[topic_df["bloom_level"] == higher])
            if not df_higher.empty:
                st.session_state.current_bloom = higher
                return _choose(df_higher), f"⬆️ Progressing to Bloom: {higher}"

    # ------------------------------------------------
    # 3. SKILL-GRAPH READINESS FILTERING (NEW)
    # ------------------------------------------------
    sg = get_skill_graph()


    readiness = sg.readiness(load_student_logs(st.session_state.user_id), topic)

    # 🔧 Low readiness → remediation & easy items
    if readiness < 0.3:
        easy_pool = _filter_unseen(topic_df[topic_df["predicted_difficulty_level"] <= 2])
        if not easy_pool.empty:
            return _choose(easy_pool), f"🔧 Readiness low ({readiness:.2f}) → Easy remediation"

    # 📘 Moderate readiness → medium difficulty
    if readiness < 0.6:
        medium_pool = _filter_unseen(topic_df[topic_df["predicted_difficulty_level"] == 3])
        if not medium_pool.empty:
            return _choose(medium_pool), f"📘 Readiness moderate ({readiness:.2f}) → Medium items"

    # 🔥 High readiness → challenge
    hard_pool = _filter_unseen(topic_df[topic_df["predicted_difficulty_level"] >= 4])
    if readiness >= 0.6 and not hard_pool.empty:
        return _choose(hard_pool), f"🔥 High readiness ({readiness:.2f}) → Challenging concepts"

    # ------------------------------------------------
    # 4. ABILITY-BASED DIFFICULTY (ELO-like θ)
    # ------------------------------------------------
    ability = st.session_state.ability_score.get(topic, 0.0)

    # Map ability (−2 to +2) → difficulty bucket (1..5)
    target_base = int(round((ability + 2) * (5 / 4)))
    target_base = max(1, min(5, target_base))

    r = random.random()
    if r < 0.30:
        target_diffs = [max(1, target_base - 1)]
        diff_label = f"easy near θ={ability:.2f}"
    elif r < 0.70:
        target_diffs = [target_base]
        diff_label = f"medium θ={ability:.2f}"
    else:
        target_diffs = [min(5, target_base + 1)]
        diff_label = f"hard near θ={ability:.2f}"

    diff_pool = _filter_unseen(topic_df[topic_df["predicted_difficulty_level"].isin(target_diffs)])

    # relax difficulty if empty
    if diff_pool.empty:
        neighborhood = [
            max(1, target_base - 1),
            target_base,
            min(5, target_base + 1),
        ]
        diff_pool = _filter_unseen(
            topic_df[topic_df["predicted_difficulty_level"].isin(neighborhood)]
        )
        diff_label += " (neighborhood)"

    if diff_pool.empty:
        diff_pool = _filter_unseen(topic_df)
        diff_label += " (fallback)"

    if not diff_pool.empty:
        q = _choose(diff_pool)
        return q, f"🎯 Ability-based difficulty: {diff_label}"

    # ------------------------------------------------
    # 5. BLOOM-TIER FALLBACK
    # ------------------------------------------------
    same_bloom_df = _filter_unseen(topic_df[topic_df["bloom_level"] == current_bloom])
    if not same_bloom_df.empty:
        return _choose(same_bloom_df), f"➡️ Staying at Bloom {current_bloom}"

    # ------------------------------------------------
    # 6. GLOBAL FALLBACK (any unseen)
    # ------------------------------------------------
    remaining = _filter_unseen(topic_df)
    if not remaining.empty:
        q = _choose(remaining)
        st.session_state.current_bloom = q["bloom_level"]
        return q, f"🎯 Fallback to {q['bloom_level']}"

    # ------------------------------------------------
    # 7. NOTHING LEFT
    # ------------------------------------------------
    return None, "🎉 No more questions available!"



