from core.skill_engine import SkillGraph
import pandas as pd

g = SkillGraph()

# test no logs
df_empty = pd.DataFrame()

print("All skills:", g.subskills)
# print("Unlocks:", g.unlocks(df_empty))
# print("Stats:", g.stats(df_empty))