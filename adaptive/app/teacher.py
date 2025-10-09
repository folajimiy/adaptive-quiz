# app/teacher.py
import streamlit as st
import pandas as pd
from utils import load_data, load_logs

def run_teacher_mode():
    st.subheader("📊 Teacher Dashboard")
    df = load_data()

    # Topic & Bloom-level distribution
    topic_bloom = df.groupby(["topic", "bloom_level"]).size().unstack().fillna(0)
    st.markdown("### 🧠 Topic-wise Bloom Distribution")
    st.bar_chart(topic_bloom)

    # Display logs if available
    log_df = load_logs()
    if log_df is not None:
        st.markdown("### 📋 Recent Student Attempts")
        st.dataframe(log_df)

        if st.download_button("📥 Download Logs", log_df.to_csv(index=False), "performance_log.csv"):
            st.success("Log exported!")
    else:
        st.info("No logs yet.")
