# core/readiness_engine.py

from typing import Dict, Any
import pandas as pd
import numpy as np

from config import MIN_ITEMS_FOR_MASTERY, MASTERY_THRESHOLD
from core.mastery_engine import bloom_breakdown


def _accuracy_component(topic_df: pd.DataFrame) -> float:
    if topic_df.empty:
        return 0.0
    return float(topic_df["is_correct"].mean()) * 100.0


def _confidence_calibration_component(topic_df: pd.DataFrame) -> float:
    """
    High confidence + correctness is good.
    High confidence + wrong is bad.
    Returns score centered ~50, scaled 0–100.
    """
    if topic_df.empty or "confidence" not in topic_df.columns:
        return 50.0

    score = 0
    n = 0
    for _, row in topic_df.iterrows():
        conf = row.get("confidence", 3)
        correct = row.get("is_correct", False)
        score += conf if correct else (6 - conf) * -1
        n += 1

    if n == 0:
        return 50.0

    raw = score / n  # can be negative
    # normalize to 0-100 (rough heuristic)
    return float(max(0, min(100, 50 + raw * 10)))


def _bloom_coverage_component(topic_df: pd.DataFrame) -> float:
    """
    Reward having correct answers across multiple Bloom levels.
    """
    if topic_df.empty:
        return 0.0

    stats = bloom_breakdown(topic_df.to_dict(orient="records"))
    levels = len(stats)
    if levels == 0:
        return 0.0

    # fraction of Bloom levels with >= 60% accuracy
    good_levels = 0
    for b, val in stats.items():
        total = val["total"]
        correct = val["correct"]
        if total >= 3 and correct / total >= 0.6:
            good_levels += 1

    return float(good_levels / levels) * 100.0


def _misconception_penalty_component(topic_df: pd.DataFrame) -> float:
    """
    Penalty based on misconception tags present in WRONG answers.
    Expect a column 'misconception_tags' or 'misconception_tags_per_option' in logs.
    """
    if topic_df.empty:
        return 0.0

    wrong = topic_df[topic_df["is_correct"] == 0].copy()
    if wrong.empty:
        return 0.0  # no penalty

    # simple heuristic: more misconception-tagged wrong answers => bigger penalty
    penalties = 0
    n = 0
    for _, row in wrong.iterrows():
        tags = row.get("misconception_tags", "") or row.get("misconception_tags_per_option", "")
        if isinstance(tags, str) and tags.strip():
            count = len([t for t in tags.split(",") if t.strip()])
            penalties += count
            n += 1

    if n == 0:
        return 0.0

    avg_penalty = penalties / n
    # scale penalty 0–40
    return float(min(40.0, avg_penalty * 10))


def compute_topic_readiness(topic_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a readiness profile for a single topic for a single student.
    Returns components + combined readiness_score 0–100.
    """
    acc = _accuracy_component(topic_df)
    conf = _confidence_calibration_component(topic_df)
    bloom = _bloom_coverage_component(topic_df)
    penalty = _misconception_penalty_component(topic_df)

    # weighted combination
    # You can tweak these weights as part of your study
    readiness_raw = (
        0.4 * acc +
        0.2 * conf +
        0.3 * bloom -
        0.3 * penalty
    )

    readiness_score = float(max(0.0, min(100.0, readiness_raw)))

    return {
        "accuracy": acc,
        "confidence_calibration": conf,
        "bloom_coverage": bloom,
        "misconception_penalty": penalty,
        "readiness_score": readiness_score,
    }
