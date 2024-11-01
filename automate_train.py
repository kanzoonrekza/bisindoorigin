import subprocess
import time
import datetime
import os
from pathlib import Path


def run_notebook(notebook_path, output_dir):
    """
    Run a Jupyter notebook and save its output to a specified directory

    Args:
        notebook_path (str): Path to the input notebook
        output_dir (str): Directory to save the executed notebook
    """
    # Create timestamp for unique output name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    notebook_name = Path(notebook_path).stem
    output_path = os.path.join(
        output_dir, f"{notebook_name}_{timestamp}.ipynb")

    print(f"Starting execution at {datetime.datetime.now()}")

    try:
        # Execute the notebook
        subprocess.run([
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--output", output_path,
            notebook_path
        ], check=True)
        print(
            f"Execution completed successfully. Output saved to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing notebook: {e}")
        return False


def automate_notebook_runs(notebook_path, num_runs=5, wait_time=0):
    """
    Automate multiple runs of a Jupyter notebook

    Args:
        notebook_path (str): Path to the notebook
        num_runs (int): Number of times to run the notebook
        wait_time (int): Time to wait between runs in seconds
    """
    # Create output directory
    output_dir = "notebook_runs"
    os.makedirs(output_dir, exist_ok=True)

    successful_runs = 0
    failed_runs = 0

    for i in range(num_runs):
        print(f"\nStarting run {i+1}/{num_runs}")

        if run_notebook(notebook_path, output_dir):
            successful_runs += 1
        else:
            failed_runs += 1

        if i < num_runs - 1:  # Don't wait after the last run
            if wait_time > 0:
                print(f"Waiting {wait_time} seconds before next run...")
                time.sleep(wait_time)

    print(f"\nAutomation completed:")
    print(f"Total runs attempted: {num_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")


if __name__ == "__main__":
    # These files that trianed below just a copy of legacy-training-holistic.ipynb with different parameters
    automate_notebook_runs(
        "train-126-1X.ipynb",
        num_runs=10,
        wait_time=5
    )
    automate_notebook_runs(
        "train-126-2X.ipynb",
        num_runs=10,
        wait_time=5
    )
    automate_notebook_runs(
        "train-1662-1X.ipynb",
        num_runs=10,
        wait_time=5
    )
    automate_notebook_runs(
        "train-1662-2X.ipynb",
        num_runs=10,
        wait_time=5
    )
