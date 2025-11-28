# Question Generation Pipeline (Hybrid LLM + Slides-Restricted)

This folder contains `question_generation.py`, a standalone script that:

- Loads all `.pptx` files from the `slides/` directory.
- Uses simple retrieval to build a context window from your lecture slides.
- Calls **OpenAI GPT-4.1** to generate Java MCQs + adaptive metadata.
- Calls **Claude Sonnet** to refine, validate, and score each item.
- Writes output to `data/java_questions_adaptive.csv` with a schema
  compatible with the Streamlit adaptive tutor.

## Directory structure

- `question_generation.py`  — main script to run
- `slides/`                 — put all your `.pptx` slide decks here
- `data/`                   — output folder for the CSV
- `requirements.txt`        — Python dependencies

## Setup

1. Create and activate a virtual environment (optional but recommended).

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set your API keys as environment variables:

    **Linux / macOS (bash/zsh):**
    ```bash
    export OPENAI_API_KEY="your-openai-key"
    export ANTHROPIC_API_KEY="your-claude-key"
    ```

    **Windows PowerShell:**
    ```powershell
    setx OPENAI_API_KEY "your-openai-key"
    setx ANTHROPIC_API_KEY "your-claude-key"
    # then open a NEW PowerShell window so the vars are loaded
    ```

4. Copy all your lecture slides (`.pptx`) into the `slides/` folder.

5. Run the generator:

    ```bash
    python question_generation.py
    ```

6. The script will append questions to:

    `data/java_questions_adaptive.csv`

You can then move / copy that CSV into the `adaptive_tutor/data/` folder
for use with the Streamlit adaptive tutor.




Start a tmux session:
tmux new -s quizgen


You’ll see a new terminal.

Run your script inside tmux:
python slide_question_gen.py


Now you can detach safely by pressing:

Ctrl + B, then D


Your program continues running even if:

✔ You close PuTTY
✔ Power off your laptop
✔ Lose internet

To reconnect later:
tmux attach -t quizgen