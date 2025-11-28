# Adaptive Java Tutor (Streamlit)

This is a prototype adaptive tutor for Java based on misconception-driven adaptation
and a simple concept topology.

## Structure

- `streamlit_tutor.py` — main Streamlit app (includes adaptive engine + UI)
- `data/java_questions_adaptive.csv` — question bank with metadata (placeholder sample row included)
- `requirements.txt` — Python dependencies

## Running

```bash
pip install -r requirements.txt
streamlit run streamlit_tutor.py
```

Place your own `java_questions_adaptive.csv` in the `data/` folder with the same columns
to use your full question bank.
