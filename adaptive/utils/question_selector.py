import pandas as pd
import random

def get_next_question(df: pd.DataFrame, proficiency: dict):
    min_level = min(proficiency.items(), key=lambda x: x[1])[0]
    subset = df[df["bloom_level"] == min_level]
    return subset.sample(1).iloc[0]
