"""
Tests for the standalone plotter tool (`tools/plotter.py`).

This module contains integration tests that execute the plotter tool as a
subprocess, simulating real-world command-line usage. It verifies that the
tool can correctly parse arguments, process data, and generate plot files.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
import pytest

# --- Test Fixtures ---


@pytest.fixture
def setup_plotter_test_env(tmp_path: Path) -> Path:
    """
    Sets up a temporary environment with fake monitoring data for plotter tests.
    """
    log_dir = tmp_path / "plotter_run_logs"
    log_dir.mkdir()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = f"test_project_j4_mem_pss_psutil_{timestamp}"

    # Create a fake Parquet data file that mimics the real application's output.
    fake_data = {
        "Timestamp_epoch": [1672531200, 1672531200, 1672531201, 1672531201, 1672531202, 1672531202],
        "Record_Type": ["PROCESS", "ALL_SUM", "PROCESS", "PROCESS", "ALL_SUM", "PROCESS"],
        "PID": ["101", None, "102", "103", None, "104"],
        "Major_Category": ["Compiler", "All", "Linker", "Compiler", "All", "BuildSystem"],
        "Minor_Category": ["gcc", "All", "ld", "gcc", "All", "make"],
        "Command_Name": ["gcc", None, "ld", "gcc", None, "make"],
        "Full_Command": ["gcc -c file1.c", None, "ld -o app", "gcc -c file2.c", None, "make"],
        "PSS_KB": [10000.0, None, 5000.0, 15000.0, None, 3000.0],
        "RSS_KB": [12000.0, None, 6000.0, 18000.0, None, 3500.0],
        "Sum_Value": [None, 10000.0, None, None, 20000.0, None],
    }
    df = pl.DataFrame(fake_data)
    data_filepath = log_dir / f"{base_filename}.parquet"
    df.write_parquet(data_filepath)

    # Create a fake summary log file to provide context.
    summary_log_filepath = log_dir / f"{base_filename}_summary.log"
    summary_log_filepath.write_text(
        "Run Summary\n"
        "Project: test_project\n"
        "Peak Overall Memory (PSS_KB): 20000.0 KB\n"
    )

    return log_dir


# --- Test Cases ---


def run_plotter_tool(log_dir: Path, extra_args: list[str] = None) -> subprocess.CompletedProcess:
    """Helper function to execute the plotter.py script as a subprocess."""
    if extra_args is None:
        extra_args = []

    project_root = Path(__file__).parent.parent
    package_root = project_root / "src"
    plotter_script_path = project_root / "tools" / "plotter.py"
    command = [
        sys.executable,
        str(plotter_script_path),
        "--log-dir",
        str(log_dir),
    ] + extra_args

    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{package_root}{os.pathsep}{python_path}"

    result = subprocess.run(command, capture_output=True, text=True, env=env)

    # --- DEBUGGING STEP ---
    # Always print the output from the subprocess to see what it's doing.
    # This will help diagnose silent failures where the script exits with code 0
    # but doesn't produce the expected files.
    if result.stdout or result.stderr:
        print(f"\n--- Output from plotter.py for test in {log_dir.name} ---")
        print(f"Return Code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        print("--- End of plotter.py output ---")
    # --- END DEBUGGING STEP ---

    return result


def find_file_by_suffix(directory: Path, suffix: str) -> Path:
    """Helper to find the first file in a directory with a given suffix."""
    try:
        return next(directory.glob(f"*{suffix}"))
    except StopIteration:
        raise FileNotFoundError(f"No file with suffix '{suffix}' found in {directory}")


def test_plotter_basic_run(setup_plotter_test_env: Path):
    """
    Tests the default behavior of the plotter tool.
    """
    log_dir = setup_plotter_test_env
    result = run_plotter_tool(log_dir)

    assert result.returncode == 0, f"Plotter tool failed!\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert find_file_by_suffix(log_dir, "_PSS_KB_lines_plot.html").exists()
    assert find_file_by_suffix(log_dir, "_PSS_KB_stacked_plot.html").exists()


def test_plotter_filter_chart_type(setup_plotter_test_env: Path):
    """
    Tests the `--chart-type` argument.
    """
    log_dir = setup_plotter_test_env
    result = run_plotter_tool(log_dir, extra_args=["--chart-type", "line"])

    assert result.returncode == 0
    assert find_file_by_suffix(log_dir, "_PSS_KB_lines_plot.html").exists()
    with pytest.raises(FileNotFoundError):
        find_file_by_suffix(log_dir, "_PSS_KB_stacked_plot.html")


def test_plotter_filter_by_category(setup_plotter_test_env: Path):
    """
    Tests the `--category` filter.
    """
    log_dir = setup_plotter_test_env
    result = run_plotter_tool(log_dir, extra_args=["--category", "Linker_ld"])

    assert result.returncode == 0
    assert find_file_by_suffix(log_dir, "_PSS_KB_lines_plot.html").exists()
    assert "Filtering for user-specified categories: ['Linker_ld']" in result.stdout
    assert "Plotting for categories: ['Linker_ld']" in result.stdout


def test_plotter_filter_top_n(setup_plotter_test_env: Path):
    """
    Tests the `--top-n` filter.
    """
    log_dir = setup_plotter_test_env
    result = run_plotter_tool(log_dir, extra_args=["--top-n", "1"])

    assert result.returncode == 0
    assert find_file_by_suffix(log_dir, "_PSS_KB_lines_plot.html").exists()
    assert "Filtering for top 1 categories by peak memory" in result.stdout
    assert "Plotting for categories: ['Compiler_gcc']" in result.stdout


def test_plotter_no_data_graceful_exit(tmp_path: Path):
    """
    Tests that the plotter exits gracefully when the log directory is empty.
    """
    log_dir = tmp_path / "empty_log_dir"
    log_dir.mkdir()
    result = run_plotter_tool(log_dir)

    assert result.returncode == 0
    # Make the assertion less strict to accommodate log formatting.
    assert "No Parquet data files found" in result.stdout
