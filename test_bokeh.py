import subprocess
import os
import pytest

INPUT_IMAGE = "IMG_2037.jpeg"
OUTPUT_DIR = "output"
EXPECTED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "IMG_2037_bokeh.jpg")

def test_run_bokeh_script():
    """
    Tests the basic execution of the dof_bokeh1.py script.
    """
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clean up any previous output file
    if os.path.exists(EXPECTED_OUTPUT_FILE):
        os.remove(EXPECTED_OUTPUT_FILE)

    # Construct the command to run the script
    command = [
        "python",
        "dof_bokeh1.py",
        "--input",
        INPUT_IMAGE,
        "--outdir",
        OUTPUT_DIR
    ]

    # Run the script
    result = subprocess.run(command, capture_output=True, text=True)

    # 1. Check if the script ran successfully
    assert result.returncode == 0, f"Script failed to run with exit code {result.returncode}. Stderr: {result.stderr}"

    # 2. Check if the output file was created
    assert os.path.exists(EXPECTED_OUTPUT_FILE), f"Output file was not created at {EXPECTED_OUTPUT_FILE}"

    # 3. Optional: Check if the file is a valid image (simple check for size)
    assert os.path.getsize(EXPECTED_OUTPUT_FILE) > 0, "Output file is empty."

    # Clean up the created file
    os.remove(EXPECTED_OUTPUT_FILE)
    print(f"\nTest passed. Cleaned up {EXPECTED_OUTPUT_FILE}.")

def test_run_with_different_model():
    """
    Tests running the script with a different model type.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(EXPECTED_OUTPUT_FILE):
        os.remove(EXPECTED_OUTPUT_FILE)

    command = [
        "python",
        "dof_bokeh1.py",
        "--input",
        INPUT_IMAGE,
        "--outdir",
        OUTPUT_DIR,
        "--model_type",
        "MiDaS_small"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with --model_type MiDaS_small. Stderr: {result.stderr}"
    assert os.path.exists(EXPECTED_OUTPUT_FILE)
    assert os.path.getsize(EXPECTED_OUTPUT_FILE) > 0
    os.remove(EXPECTED_OUTPUT_FILE)
    print(f"\nTest with MiDaS_small passed. Cleaned up {EXPECTED_OUTPUT_FILE}.")
