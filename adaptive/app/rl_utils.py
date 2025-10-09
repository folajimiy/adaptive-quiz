import pandas as pd
import random

bloom_levels = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
bloom_to_num = {b: i+1 for i, b in enumerate(bloom_levels)}
num_to_bloom = {v: k for k, v in bloom_to_num.items()}

def run_adaptive_rl_simulation(df, start_level="Understand", total_rounds=30):
    df["bloom_numeric"] = df["bloom_level"].map(bloom_to_num)
    history = []
    proficiency = bloom_to_num[start_level]
    correct_streak = 0
    fail_streak = 0

    for round_num in range(1, total_rounds + 1):
        pool = df[df["bloom_numeric"].between(proficiency - 1, proficiency + 1)]
        if pool.empty:
            pool = df

        q = pool.sample(1).iloc[0]
        chosen_level = q["bloom_numeric"]
        chosen_bloom = q["bloom_level"]
        qid = q["question_id"]
        qstem = q.get("question_stem", "")[:120]

        success_prob = 0.9 if chosen_level < proficiency else 0.7 if chosen_level == proficiency else 0.3
        correct = random.random() < success_prob

        # Reward and adaptive logic
        reward = 10 if correct else -5
        delta = 0

        if correct:
            correct_streak += 1
            fail_streak = 0
            if chosen_level > proficiency:
                reward += 5
            if correct_streak >= 3:
                delta = 2  # Boost level if on a roll
        else:
            correct_streak = 0
            fail_streak += 1
            if chosen_level < proficiency:
                reward -= 5
            if fail_streak >= 2:
                delta = -2

        proficiency = max(1, min(6, proficiency + (1 if delta > 0 else -1 if delta < 0 else 0)))

        history.append({
            "Round": round_num,
            "Question_ID": qid,
            "Bloom_Level": chosen_bloom,
            "Correct": int(correct),
            "Reward": reward,
            "Updated_Proficiency": proficiency,
            "Updated_Bloom": num_to_bloom[proficiency],
            "Question_Snippet": qstem,
            "Confusion": chosen_level - proficiency
        })

    return pd.DataFrame(history)
