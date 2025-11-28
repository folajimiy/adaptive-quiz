import streamlit as st
import sys
import os

# Ensure your package root is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from adaptive_app_complete.main import main  # ← uses your existing main() entry

if __name__ == "__main__":
    st.set_page_config(
        page_title="Adaptive Java Learning System",
        layout="wide"
    )
    main()
