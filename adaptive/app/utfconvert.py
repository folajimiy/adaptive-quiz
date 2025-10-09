import pandas as pd
df = pd.read_csv("java_question_bank_with_topics.csv", encoding_errors="ignore")
df.to_csv("java_question_bank_with_topics_utf8.csv", index=False)