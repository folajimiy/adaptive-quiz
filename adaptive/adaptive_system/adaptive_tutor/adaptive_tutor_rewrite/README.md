# Adaptive Java Tutor (Rewritten Streamlit App)

This is a cleaned, role-separated adaptive tutor built with Streamlit.

## Features

- Landing page with role selection:
  - **Student**: clean question-and-feedback interface
  - **Instructor**: password-protected analytics dashboard
- Student access controlled by `student_roster.csv` (whitelist).
- SQLite-backed logging of all responses in `adaptive_tutor.db`.
- Simple adaptive selection:
  - Finds weakest topic/subtopic for each student based on accuracy
  - Picks an unseen question from that area, favoring medium difficulty
- Instructor dashboard:
  - Class overview with accuracy by topic
  - Per-student history and accuracy
  - CSV exports of all responses and per-student responses

## Directory Structure

- `app.py` — main Streamlit app
- `student_roster.csv` — whitelist of allowed students (edit this)
- `adaptive_tutor.db` — SQLite database created at runtime
- `data/java_questions_adaptive.csv` — your item bank (copy yours here)
- `.env.example` — sample env file (rename to `.env`)
- `requirements.txt` — dependencies

## Setup

1. Create a virtual environment (optional) and install requirements:

    ```bash
    pip install -r requirements.txt
    ```

2. Copy your generated `java_questions_adaptive.csv` into the `data/` folder.

3. Set the admin password:

    - Copy `.env.example` to `.env`
    - Edit `.env` and set `ADMIN_PASSWORD` to a secret value
    - Ensure your environment loads it, e.g. exporting it manually:

      ```bash
      export ADMIN_PASSWORD="yourpassword"
      ```

      On Windows PowerShell:

      ```powershell
      setx ADMIN_PASSWORD "yourpassword"
      ```

      Then open a new terminal.

4. Edit `student_roster.csv` to contain the Student IDs you want to allow.

5. Run the app:

    ```bash
    streamlit run app.py
    ```

6. Open the URL shown in the terminal (e.g. http://localhost:8501).

## Notes

- The student interface only shows question, options, and feedback — no analytics.
- The instructor dashboard is only available after entering the correct admin password.
- All response data is logged in `adaptive_tutor.db`, which you can analyze separately if desired.
