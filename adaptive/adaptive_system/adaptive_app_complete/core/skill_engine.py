import json
import pandas as pd
import numpy as np

from config import (
    SKILL_GRAPH_JSON,
    SKILL_MIN_ATTEMPTS,
    SKILL_MASTERY_THRESHOLD,
)


# =======================================================
#  Skill Graph Structure
# =======================================================

class SkillGraph:
    """
    Represents the full skill dependency graph.
    Provides:
      - prerequisite lookup
      - readiness calculation
      - skill-level mastery estimates
      - class readiness distributions
    """

    def __init__(self, json_path: str):
        self.graph = self._load_graph(json_path)
        self.skills = list(self.graph.keys())

    # ---------------------------------------------------
    # Load JSON skill graph
    # ---------------------------------------------------
    def _load_graph(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph = {}
        for skill in data.get("skills", []):
            graph[skill["id"]] = {
                "name": skill["name"],
                "prereqs": skill.get("prerequisites", []),
                "topic": skill.get("topic", None)
            }
        return graph

    # ---------------------------------------------------
    # Get prerequisites for a skill
    # ---------------------------------------------------
    def prerequisites(self, skill_id: str):
        return self.graph.get(skill_id, {}).get("prereqs", [])

    # ---------------------------------------------------
    # Get all ancestors recursively (for full dependency)
    # ---------------------------------------------------
    def all_prereqs(self, skill_id: str):
        visited = set()
        stack = list(self.prerequisites(skill_id))

        while stack:
            s = stack.pop()
            if s not in visited:
                visited.add(s)
                stack.extend(self.prerequisites(s))

        return list(visited)

    # ---------------------------------------------------
    # Compute skill readiness from logs
    # ---------------------------------------------------
    def readiness(self, df_logs: pd.DataFrame, skill_id: str) -> float:
        """
        Readiness is based on:
            recency-weighted accuracy
            response-time normalization
            confidence bonus
        Returns a value in [0,1].
        """
        if df_logs is None or df_logs.empty:
            return 0.0

        df = df_logs[df_logs["topic"] == skill_id]

        if df.empty or len(df) < SKILL_MIN_ATTEMPTS:
            return 0.0

        # 1. Accuracy
        acc = df["is_correct"].mean()

        # 2. Confidence (normalize to [0,1])
        conf = df["confidence"].mean() / 5

        # 3. Response time quality (lower is better)
        rt = df["response_time_sec"].dropna()
        if len(rt) > 0:
            rt_score = 1 / (1 + np.log1p(rt.mean()))
        else:
            rt_score = 0.5

        # Combine with weights
        readiness = (
            0.50 * acc +
            0.30 * conf +
            0.20 * rt_score
        )

        return float(max(0, min(1, readiness)))

    # ---------------------------------------------------
    # Skill mastery (boolean)
    # ---------------------------------------------------
    def mastered(self, df_logs: pd.DataFrame, skill_id: str) -> bool:
        r = self.readiness(df_logs, skill_id)
        return r >= SKILL_MASTERY_THRESHOLD

    # ---------------------------------------------------
    # Full skill readiness table for a student
    # ---------------------------------------------------
    # def stats(self, df_logs: pd.DataFrame) -> pd.DataFrame:
    #     rows = []
    #     for skill in self.skills:
    #         r = self.readiness(df_logs, skill)
    #         rows.append({
    #             "skill_id": skill,
    #             "skill_name": self.graph[skill]["name"],
    #             "readiness": r,
    #             "mastered": r >= SKILL_MASTERY_THRESHOLD,
    #             "prereqs": self.graph[skill]["prereqs"],
    #         })

    #     return pd.DataFrame(rows)
    def stats(self, df_logs: pd.DataFrame):
        """
        Returns:
        skill_id → {
            attempts: int,
            accuracy: float,
            mastered: bool
        }
        """
        # Initialize output for all skills
        out = {
            sid: {"attempts": 0, "accuracy": 0.0, "mastered": False}
            for sid in self.skills
        }

        if df_logs.empty:
            return out

        df = df_logs.copy()
        df["question_id"] = df["question_id"].astype(str)

        # Parse KC tags (skills per question)
        import ast
        def parse_tags(raw):
            if pd.isna(raw) or raw == "":
                return []
            try:
                if isinstance(raw, str):
                    if "[" in raw or "{" in raw:
                        return list(ast.literal_eval(raw))
                    return [t.strip() for t in raw.split(",")]
            except:
                return []
            return []

        df["kc_list"] = df["misconception_tags"].apply(parse_tags)

        # Aggregate attempts + correct counts per KC
        for _, row in df.iterrows():
            for kc in row["kc_list"]:
                if kc not in out:
                    continue
                out[kc]["attempts"] += 1
                out[kc]["accuracy"] += (1 if row["is_correct"] else 0)

        # Normalize accuracy and determine mastery
        for sid, d in out.items():
            if d["attempts"] > 0:
                d["accuracy"] = d["accuracy"] / d["attempts"]
                d["mastered"] = d["accuracy"] >= SKILL_MASTERY_THRESHOLD
            else:
                d["accuracy"] = 0.0
                d["mastered"] = False

        return out



    # ---------------------------------------------------
    # Class readiness distribution (teacher analytics)
    # ---------------------------------------------------
    def class_readiness(self, df_all_logs: pd.DataFrame) -> pd.DataFrame:
        if df_all_logs is None or df_all_logs.empty:
            return pd.DataFrame()

        rows = []
        for sid, df_stu in df_all_logs.groupby("student_id"):
            for skill in self.skills:
                r = self.readiness(df_stu, skill)
                rows.append({
                    "student_id": sid,
                    "skill_id": skill,
                    "readiness": r,
                })

        return pd.DataFrame(rows)

    # ---------------------------------------------------
    # Skills blocked by missing prerequisites
    # ---------------------------------------------------
    def blocked_skills(self, df_logs: pd.DataFrame) -> list:
        blocked = []
        for skill in self.skills:
            prereqs = self.all_prereqs(skill)
            if not prereqs:
                continue

            # if ANY prereq is < 30% readiness → locked
            for p in prereqs:
                if self.readiness(df_logs, p) < 0.3:
                    blocked.append(skill)
                    break
        return blocked

    # ---------------------------------------------------
    # Recommend next skills to unlock
    # ---------------------------------------------------
    def next_unlocks(self, df_logs: pd.DataFrame) -> list:
        next_skills = []
        for skill in self.skills:
            if self.mastered(df_logs, skill):
                continue

            prereqs = self.all_prereqs(skill)

            if all(self.readiness(df_logs, p) >= 0.4 for p in prereqs):
                next_skills.append(skill)

        return next_skills



    def unlocks(self, df_logs: pd.DataFrame):
        """
        A skill is unlocked if all of its prerequisites are mastered.
        """
        stats = self.stats(df_logs)
        unlock_map = {}

        for sid in self.skills:
            prereqs = self.graph[sid]["prereqs"]

            # No prerequisites → always unlocked
            if not prereqs:
                unlock_map[sid] = True
                continue

            # Mastery check: all prereqs must be mastered
            prereq_mastery = [
                stats[p]["mastered"] if p in stats else False
                for p in prereqs
            ]

            unlock_map[sid] = all(prereq_mastery)

        return unlock_map
    

    def misconceptions(self, df_logs: pd.DataFrame):
        """
        Aggregate misconception tags from logs.
        Returns dict: {misconception_tag: count}
        """
        if df_logs.empty or "misconception_tags" not in df_logs.columns:
            return {}

        tags = (
            df_logs["misconception_tags"]
            .dropna()
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
        )

        tags = tags[tags != ""]

        if tags.empty:
            return {}

        return tags.value_counts().to_dict()



# =======================================================
#  Singleton Loader
# =======================================================

_skill_graph_cache = None

def get_skill_graph() -> SkillGraph:
    """
    Returns a singleton SkillGraph instance.
    Ensures SKILL_GRAPH_JSON is loaded only once.
    """
    global _skill_graph_cache
    if _skill_graph_cache is None:
        _skill_graph_cache = SkillGraph(SKILL_GRAPH_JSON)
    return _skill_graph_cache
