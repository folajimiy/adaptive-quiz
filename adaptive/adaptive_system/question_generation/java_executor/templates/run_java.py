import subprocess
import tempfile
import os
from pathlib import Path

WRAPPER_TEMPLATE = Path(__file__).parent / "templates" / "Wrapper.java"

def execute_java_snippet(snippet: str, timeout_sec=2) -> Dict[str, str]:
    """
    Compiles and executes the Java snippet inside Wrapper.java.
    Returns stdout, stderr, success flag.
    """

    with open(WRAPPER_TEMPLATE, "r") as f:
        template = f.read()

    java_code = template.replace("{{CODE_SNIPPET}}", snippet)

    with tempfile.TemporaryDirectory() as temp_dir:
        wrapper_path = Path(temp_dir) / "Wrapper.java"
        with open(wrapper_path, "w") as f:
            f.write(java_code)

        # Compile
        compile_proc = subprocess.run(
            ["javac", "Wrapper.java"],
            cwd=temp_dir,
            capture_output=True,
            text=True
        )

        if compile_proc.returncode != 0:
            return {
                "success": False,
                "stdout": "",
                "stderr": compile_proc.stderr
            }

        # Run
        try:
            run_proc = subprocess.run(
                ["java", "Wrapper"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )
            return {
                "success": run_proc.returncode == 0,
                "stdout": run_proc.stdout,
                "stderr": run_proc.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Timeout"
            }
