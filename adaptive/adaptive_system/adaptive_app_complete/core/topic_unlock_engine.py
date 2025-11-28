# core/topic_unlock_engine.py

from typing import Dict, List, Any
import pandas as pd

from core.skill_graph import ALL_TOPICS, get_prereq_topics
from core.readiness_engine import compute_topic_readiness


# thresholds (you can tune these empirically)
READINESS_READY = 75.0
READINESS_ALMOST = 55.0


def compute_all_topic_readiness(student_logs: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute readiness profile for every topic for a student.
    Returns dict: {topic: readiness_profile}
    """
    profiles = {}
    if student_logs is None or student_logs.empty:
        for t in ALL_TOPICS:
            profiles[t] = {
                "accuracy": 0.0,
                "confidence_calibration": 50.0,
                "bloom_coverage": 0.0,
                "misconception_penalty": 0.0,
                "readiness_score": 0.0,
            }
        return profiles

    for topic in ALL_TOPICS:
        tdf = student_logs[student_logs["topic"] == topic]
        profiles[topic] = compute_topic_readiness(tdf)

    return profiles


def classify_topic_readiness(topic: str, profile: Dict[str, Any], prereq_ok: bool) -> str:
    """
    Return 'ready', 'almost', or 'not_ready' label.
    """
    score = profile["readiness_score"]

    if not prereq_ok:
        return "not_ready"

    if score >= READINESS_READY:
        return "ready"
    elif score >= READINESS_ALMOST:
        return "almost"
    else:
        return "not_ready"


def compute_prereq_satisfaction(topic: str, profiles: Dict[str, Dict[str, Any]]) -> bool:
    """A topic's prereqs are 'satisfied' if all prereq topics have readiness above READINESS_ALMOST."""
    prereqs = get_prereq_topics(topic)
    if not prereqs:
        return True

    for p in prereqs:
        prof = profiles.get(p)
        if not prof:
            return False
        if prof["readiness_score"] < READINESS_ALMOST:
            return False
    return True


def generate_topic_path(student_logs: pd.DataFrame) -> Dict[str, Any]:
    """
    Core engine:
      - classify topics into ready / almost / not_ready
      - compute recommended next topics
      - allow multifurcated paths (several 'ready' topics)
    """
    profiles = compute_all_topic_readiness(student_logs)

    topic_states: Dict[str, str] = {}
    for t in ALL_TOPICS:
        prereq_ok = compute_prereq_satisfaction(t, profiles)
        topic_states[t] = classify_topic_readiness(t, profiles[t], prereq_ok)

    # Recommendations:
    #  - primary: ready topics whose prereqs are satisfied
    #  - secondary: almost topics that are 'on deck'
    ready_topics = [t for t, s in topic_states.items() if s == "ready"]
    almost_topics = [t for t, s in topic_states.items() if s == "almost"]
    not_ready_topics = [t for t, s in topic_states.items() if s == "not_ready"]

    # simple heuristic: recommend up to 3 best ready topics
    ready_sorted = sorted(ready_topics, key=lambda t: profiles[t]["readiness_score"], reverse=True)
    recommended = ready_sorted[:3]

    return {
        "profiles": profiles,
        "states": topic_states,
        "recommended_topics": recommended,
        "ready_topics": ready_topics,
        "almost_topics": almost_topics,
        "not_ready_topics": not_ready_topics,
    }
