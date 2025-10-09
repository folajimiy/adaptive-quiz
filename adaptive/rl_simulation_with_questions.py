import random
import pandas as pd
import os

# 1. Load your question bank
# Update this path to your actual question bank file
questions_df = pd.read_csv("LLM-Generated MCQ Dataset with Bloom_Level Annotations_Clean.csv")

# Ensure you have these columns in your file:
# - 'question_id' or unique ID column
# - 'bloom_level'

# If you don't have a question_id, create one
if "question_id" not in questions_df.columns:
    questions_df["question_id"] = range(1, len(questions_df) + 1)

# 2. Bloom taxonomy mapping
bloom_levels = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
bloom_to_num = {b: i+1 for i, b in enumerate(bloom_levels)}
num_to_bloom = {v: k for k, v in bloom_to_num.items()}

# Map Bloom to numeric difficulty
questions_df["bloom_numeric"] = questions_df["bloom_level"].map(bloom_to_num)

# 3. RL Simulation
total_rounds = 30
proficiency = 2  # Start at 'Understand'
history = []

for round_num in range(1, total_rounds + 1):
    # Choose a question near current proficiency
    question_pool = questions_df[questions_df["bloom_numeric"].between(proficiency - 1, proficiency + 1)]
    if question_pool.empty:
        question_pool = questions_df  # fallback

    question = question_pool.sample(1).iloc[0]

    chosen_bloom = question["bloom_level"]
    chosen_level = question["bloom_numeric"]
    question_id = question["question_id"]

    # Simulate success probability
    if chosen_level < proficiency:
        success_prob = 0.9
    elif chosen_level == proficiency:
        success_prob = 0.7
    else:
        success_prob = 0.3

    correct = random.random() < success_prob

    # Reward logic
    if correct:
        reward = 10
        if chosen_level > proficiency:
            reward += 5
        proficiency = min(6, proficiency + 1)
    else:
        reward = -5
        if chosen_level < proficiency:
            reward -= 5
        proficiency = max(1, proficiency - 1)

    history.append({
        "Round": round_num,
        "Question_ID": question_id,
        "Bloom_Level": chosen_bloom,
        "Correct": int(correct),
        "Reward": reward,
        "Updated_Proficiency": proficiency,
        "Updated_Bloom": num_to_bloom[proficiency],
        "Question_Snippet": question.get("question_stem", "")[:120]  # optional snippet
    })

# 4. Save to CSV
os.makedirs("data", exist_ok=True)
output_path = "data/RL_simulation_output.csv"
pd.DataFrame(history).to_csv(output_path, index=False)

print(f"✅ Simulation complete. Output saved to {output_path}")
