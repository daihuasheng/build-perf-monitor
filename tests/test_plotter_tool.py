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
from typing import Callable, List

import polars as pl
import pytest

# --- Test Fixtures ---


@pytest.fixture
def plotter_test_env_factory(tmp_path: Path) -> Callable[[List[int]], Path]:
    """
    A factory fixture to create a test environment for the plotter tool.

    This fixture returns a function that can be called with a list of
    parallelism levels (job_levels). For each level, it creates a corresponding
    .parquet data file and a _summary.log file, simulating a multi-job run.

    Args:
        tmp_path: The temporary directory path provided by pytest.

    Returns:
        A function that takes a list of integers (job levels) and returns the
        path to the created temporary log directory.
    """

    def _create_env(job_levels: List[int]) -> Path:
        log_dir = tmp_path / "plotter_run_logs"
        log_dir.mkdir(exist_ok=True)

        for jobs in job_levels:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_filename = f"test_project_j{jobs}_mem_pss_psutil_{timestamp}"

            # --- Create a fake Parquet data file ---
            fake_data = {
                "Timestamp_epoch": [1672531200, 1672531201],
                "Record_Type": ["PROCESS", "ALL_SUM"],
                "Major_Category": ["Compiler", "All"],
                "Minor_Category": ["gcc", "All"],
                "PSS_KB": [10000.0 * jobs, None],
                "Sum_Value": [None, 10000.0 * jobs],
            }
            df = pl.DataFrame(fake_data)
            data_filepath = log_dir / f"{base_filename}.parquet"
            df.write_parquet(data_filepath)

            # --- Create a fake summary log file ---
            summary_log_filepath = log_dir / f"{base_filename}_summary.log"
            # Simulate that higher parallelism reduces build time but increases peak memory.
            duration = 100 / jobs
            peak_mem = 15000 * jobs
            summary_log_filepath.write_text(
                f"Run Summary\n"
                f"Project: test_project\n"
                f"Parallelism: -j{jobs}\n"
                f"Total Build & Monitoring Duration: {duration:.2f}s (some text) ({duration:.2f} seconds)\n"
                f"Peak Overall Memory (PSS_KB): {peak_mem} KB\n"
            )
        return log_dir

    return _create_env


# --- Test Cases ---


def run_plotter_tool(
    log_dir: Path, extra_args: list[str] = None
) -> subprocess.CompletedProcess:
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
    return result


def find_file_by_suffix(directory: Path, suffix: str) -> Path:
    """Helper to find the first file in a directory with a given suffix."""
    try:
        return next(directory.glob(f"*{suffix}"))
    except StopIteration:
        raise FileNotFoundError(f"No file with suffix '{suffix}' found in {directory}")


def test_plotter_basic_run(plotter_test_env_factory: Callable[[List[int]], Path]):
    """
    Tests the default behavior of the plotter tool for a single run.
    """
    log_dir = plotter_test_env_factory(job_levels=[4])
    result = run_plotter_tool(log_dir)

    assert result.returncode == 0, (
        f"Plotter tool failed!\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert find_file_by_suffix(log_dir, "_PSS_KB_lines_plot.html").exists()
    assert find_file_by_suffix(log_dir, "_PSS_KB_stacked_plot.html").exists()


def test_plotter_filter_chart_type(
    plotter_test_env_factory: Callable[[List[int]], Path],
):
    """
    Tests the `--chart-type` argument.
    """
    log_dir = plotter_test_env_factory(job_levels=[4])
    result = run_plotter_tool(log_dir, extra_args=["--chart-type", "line"])

    assert result.returncode == 0
    assert find_file_by_suffix(log_dir, "_PSS_KB_lines_plot.html").exists()
    with pytest.raises(FileNotFoundError):
        find_file_by_suffix(log_dir, "_PSS_KB_stacked_plot.html")


def test_plotter_filter_by_category(
    plotter_test_env_factory: Callable[[List[int]], Path],
):
    """
    Tests the `--category` filter.
    """
    log_dir = plotter_test_env_factory(job_levels=[4])
    result = run_plotter_tool(log_dir, extra_args=["--category", "Compiler_gcc"])

    assert result.returncode == 0
    assert find_file_by_suffix(log_dir, "_PSS_KB_lines_plot.html").exists()
    assert "Filtering for user-specified categories: ['Compiler_gcc']" in result.stdout


def test_plotter_filter_top_n(plotter_test_env_factory: Callable[[List[int]], Path]):
    """
    Tests the `--top-n` filter.
    """
    log_dir = plotter_test_env_factory(job_levels=[4])
    result = run_plotter_tool(log_dir, extra_args=["--top-n", "1"])

    assert result.returncode == 0
    assert find_file_by_suffix(log_dir, "_PSS_KB_lines_plot.html").exists()
    assert "Filtering for top 1 categories by peak memory" in result.stdout


def test_plotter_no_data_graceful_exit(tmp_path: Path):
    """
    Tests that the plotter exits gracefully when the log directory is empty.
    """
    log_dir = tmp_path / "empty_log_dir"
    log_dir.mkdir()
    result = run_plotter_tool(log_dir)

    assert result.returncode == 0
    assert "No Parquet data files found" in result.stdout


# --- NEW TEST CASE for the summary plot feature ---
def test_plotter_summary_plot_generation(
    plotter_test_env_factory: Callable[[List[int]], Path],
):
    """
    Tests the `--summary-plot` feature.

    It verifies that the plotter can correctly parse multiple summary logs
    from a single run and generate a single summary comparison plot.
    """
    # 1. Setup: Create an environment with data for multiple job levels
    log_dir = plotter_test_env_factory(job_levels=[4, 8, 16])

    # 2. Execution: Run the plotter with the --summary-plot flag
    result = run_plotter_tool(log_dir, extra_args=["--summary-plot"])

    # 3. Verification
    assert result.returncode == 0, (
        f"Plotter tool failed for summary plot!\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # Check that the summary plot file was created
    summary_plot_file = find_file_by_suffix(log_dir, "_build_summary_plot.html")
    assert summary_plot_file.exists()

    # Check that detailed plots were NOT created in this mode
    with pytest.raises(FileNotFoundError):
        find_file_by_suffix(log_dir, "_lines_plot.html")
    with pytest.raises(FileNotFoundError):
        find_file_by_suffix(log_dir, "_stacked_plot.html")
