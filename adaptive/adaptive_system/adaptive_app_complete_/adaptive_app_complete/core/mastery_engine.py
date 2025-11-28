# core/mastery_engine.py

from typing import List, Dict, Any
import numpy as np
from config import MIN_ITEMS_FOR_MASTERY, MASTERY_THRESHOLD


# ------------------------------
# BASIC METRICS
# ------------------------------

def compute_session_accuracy(session_log: List[Dict[str, Any]]) -> float:
    """Return accuracy percentage across a session."""
    total = len(session_log)
    if total == 0:
        return 0.0
    correct = sum(int(l["is_correct"]) for l in session_log)
    return (correct / total) * 100.0


def bloom_breakdown(session_log: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Return correct/total stats per Bloom level."""
    stats = {}
    for entry in session_log:
        b = entry["bloom_level"]
        stats.setdefault(b, {"correct": 0, "total": 0})
        stats[b]["total"] += 1
        stats[b]["correct"] += int(entry["is_correct"])
    return stats


# ------------------------------
# MASTERY DECISIONING
# ------------------------------

def is_topic_mastered(session_log: List[Dict[str, Any]]) -> bool:
    """Determine if topic is mastered based on thresholds."""
    acc = compute_session_accuracy(session_log)
    total = len(session_log)
    return total >= MIN_ITEMS_FOR_MASTERY and acc >= MASTERY_THRESHOLD


# ------------------------------
# LEVEL PROGRESSION
# ------------------------------

def suggest_level_change(current_level: str, accuracy: float) -> str:
    """Suggest movement between Beginner → Intermediate → Advanced."""
    order = ["Beginner", "Intermediate", "Advanced"]
    i = order.index(current_level)

    if accuracy >= 75 and i < len(order) - 1:
        return order[i + 1]  # promote
    if accuracy <= 50 and i > 0:
        return order[i - 1]  # demote
    return current_level


# ------------------------------
# SUB-CONCEPT WEAKNESS DETECTOR
# ------------------------------

def detect_weak_subconcepts(session_log: List[Dict[str, Any]], min_hits=3) -> List[str]:
    """
    Identify subconcepts with repeated errors.
    Returns a sorted list of weak areas.
    """
    errors = {}
    for entry in session_log:
        if not entry["is_correct"]:
            sc = entry.get("sub_concept") or "General"
            errors.setdefault(sc, 0)
            errors[sc] += 1

    # Apply threshold
    weak = [sc for sc, count in errors.items() if count >= min_hits]

    # Sort by severity
    weak = sorted(weak, key=lambda sc: errors[sc], reverse=True)

    return weak


# ------------------------------
# CONFIDENCE ANALYTICS
# ------------------------------

def confidence_quality(session_log: List[Dict[str, Any]]) -> float:
    """
    Measure calibration: high-confidence correct answers = good.
    """
    if not session_log:
        return 0.0
    score = 0
    total = 0
    for entry in session_log:
        conf = entry["confidence"]
        corr = entry["is_correct"]
        if corr:
            score += conf
        else:
            score -= conf
        total += 1
    return score / total
