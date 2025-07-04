"""
Tests for the basic mymonitor functionality end-to-end.

This module tests the integration of the MyMonitor application from
command-line invocation through configuration loading, monitoring execution,
and output file generation.
"""

import os
import sys
import toml
import pandas as pd
import pytest
from pathlib import Path

# Import the main entry point of the application to be tested.
from mymonitor.cli import main_cli

# --- Constants for Test Setup ---

# Path to the fake build script used in tests
FAKE_BUILD_SCRIPT_PATH = Path(__file__).parent / "fake_build_script.sh"


@pytest.fixture
def setup_test_config_files(tmp_path: Path):
    """
    Fixture that sets up temporary config files for testing.

    This fixture creates a complete set of temporary TOML configuration files
    required by the mymonitor application, including:
    - config.toml (main configuration)
    - projects.toml (project definitions)
    - rules.toml (categorization rules)

    The configuration is designed to work with the fake_build_script.sh test script
    and includes appropriate settings for testing (short durations, simple output paths).

    Args:
        tmp_path: pytest's temporary directory fixture

    Yields:
        Path: Path to the main config.toml file that can be used to configure
              the application for testing

    Note:
        This fixture also sets the MYMONITOR_SKIP_PATH_VALIDATION environment
        variable to bypass path validation during testing.
    """
    # Set environment variable to skip path validation for testing
    old_env_value = os.environ.get("MYMONITOR_SKIP_PATH_VALIDATION")
    os.environ["MYMONITOR_SKIP_PATH_VALIDATION"] = "1"
    
    try:
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()

        # Define paths for the config files.
        config_file = conf_dir / "config.toml"
        projects_file = conf_dir / "projects.toml"
        rules_file = conf_dir / "rules.toml"
        log_output_dir = tmp_path / "logs_output"

        # --- Configuration Content ---
        # Main config.toml content
        config_content = {
            "paths": {
                "projects_config": str(projects_file),
                "rules_config": str(rules_file),
            },
            "monitor": {
                "general": {
                    "log_root_dir": str(log_output_dir),
                    "default_jobs": [1],
                    "skip_plots": True,
                    "categorization_cache_size": 128,
                },
                "collection": {
                    "metric_type": "pss_psutil",
                    "interval_seconds": 0.1,
                    "pss_collector_mode": "full_scan",
                },
                "scheduling": {
                    "scheduling_policy": "adaptive",
                    "monitor_core": 0,
                    "manual_build_cores": "",
                    "manual_monitoring_cores": "",
                },
            },
        }

        # Projects projects.toml content
        projects_content = {
            "projects": [
                {
                    "name": "FakeProject",
                    "dir": str(tmp_path),
                    "process_pattern": "fake_build_script.sh|sleep",
                    "setup_command_template": "echo 'Fake setup complete'",
                    "clean_command_template": "echo 'Fake clean complete'",
                    "build_command_template": f"{str(FAKE_BUILD_SCRIPT_PATH)} <N>",
                }
            ]
        }

        # Rules rules.toml content
        # Note: Case matters in the minor category.
        rules_content = """
[[rules]]
priority = 100
major_category = "Scripting"
category = "ShellScriptFile"
match_field = "current_cmd_name"
match_type = "in_list"
patterns = ["fake_build_script.sh"]
comment = "Rule for the main test script."

[[rules]]
priority = 90
major_category = "build"
category = "Sleep"
match_field = "current_cmd_name"
match_type = "in_list"
patterns = ["sleep"]
comment = "Rule for the sleep command used in the test script."
"""

        # --- Write Content to Files ---
        with open(config_file, "w") as f:
            toml.dump(config_content, f)

        with open(projects_file, "w") as f:
            toml.dump(projects_content, f)

        with open(rules_file, "w") as f:
            f.write(rules_content)

        yield config_file
    
    finally:
        # Cleanup environment variable
        if old_env_value is None:
            os.environ.pop("MYMONITOR_SKIP_PATH_VALIDATION", None)
        else:
            os.environ["MYMONITOR_SKIP_PATH_VALIDATION"] = old_env_value


# --- Test Cases ---


def test_basic_monitoring_run(setup_test_config_files: Path, monkeypatch, capsys):
    """
    Tests a basic, end-to-end run of the mymonitor CLI.

    This test simulates running the tool from the command line without any
    arguments, relying on the temporary configuration files created by the
    `setup_test_config_files` fixture. It verifies that the application
    runs to completion and that the primary output artifacts (Parquet data file
    and summary log) are created and contain the expected content.

    Args:
        setup_test_config_files: The fixture that provides the path to the temp config.
        monkeypatch: The pytest fixture for modifying modules, classes, or functions.
        capsys: The pytest fixture for capturing stdout/stderr.
    """
    # Ensure the fake build script is executable before running the test.
    if not os.access(FAKE_BUILD_SCRIPT_PATH, os.X_OK):
        os.chmod(FAKE_BUILD_SCRIPT_PATH, 0o755)

    # Set environment variable to skip path validation (in case fixture didn't)
    monkeypatch.setenv("MYMONITOR_SKIP_PATH_VALIDATION", "1")

    # --- Test Setup: Monkeypatching ---
    # Force the application's config loader to use our temporary config file.
    monkeypatch.setattr("mymonitor.config.manager._CONFIG_FILE_PATH", setup_test_config_files)
    # Force a reload of the config singleton to pick up the new path.
    monkeypatch.setattr("mymonitor.config.manager._CONFIG", None)

    # Simulate the command-line arguments. Passing just the script name is
    # equivalent to running with no extra arguments.
    test_args = ["mymonitor"]
    monkeypatch.setattr(sys, "argv", test_args)

    # --- Test Execution ---
    # Run the main CLI function and assert that it exits cleanly (code 0).
    try:
        main_cli()
    except SystemExit as e:
        assert e.code == 0, f"main_cli exited with non-zero code: {e.code}"

    # --- Verification ---
    # Define the expected root directory for log outputs.
    log_root_dir = setup_test_config_files.parent.parent / "logs_output"

    # Find the unique, timestamped run directory created by the application.
    run_dirs = list(log_root_dir.glob("run_*"))
    assert len(run_dirs) == 1, (
        f"Expected 1 run directory in {log_root_dir}, found {len(run_dirs)}"
    )
    run_timestamp_dir = run_dirs[0]
    assert run_timestamp_dir.is_dir()

    # The actual run data is now inside another subdirectory
    project_run_dirs = list(run_timestamp_dir.glob("FakeProject_j1_pss_psutil_*"))
    assert len(project_run_dirs) == 1, (
         f"Expected 1 project run directory in {run_timestamp_dir}, found {len(project_run_dirs)}"
    )
    run_specific_output_dir = project_run_dirs[0]
    assert run_specific_output_dir.is_dir()

    # 1. Verify the Parquet data file.
    parquet_file = run_specific_output_dir / "memory_samples.parquet"
    assert parquet_file.exists(), f"Expected Parquet file not found at {parquet_file}"

    # Check the content of the Parquet file.
    df = pd.read_parquet(parquet_file)
    assert not df.empty, "Parquet file should not be empty."
    expected_cols = {
        "epoch",
        "major_category",
        "minor_category",
        "pid",
        "command_name",
        "PSS_KB",
    }
    assert expected_cols.issubset(df.columns), f"Expected columns are not a subset of Parquet columns: {df.columns}"
    assert "Generic_Path" in df["minor_category"].unique()

    # 2. Verify the summary log file.
    summary_log_file = run_specific_output_dir / "summary.log"
    assert summary_log_file.exists(), f"Expected summary log not found at {summary_log_file}"

    summary_content = summary_log_file.read_text()
    assert "Build Exit Code: 0" in summary_content
    assert "Peak Overall Memory (PSS_KB):" in summary_content
    assert " GB" in summary_content  # Verify memory is now in GB
    assert "Total Build & Monitoring Duration:" in summary_content
    assert "Project: FakeProject" in summary_content
    assert "Parallelism: -j1" in summary_content
    assert "OSUtilities:" in summary_content, (
        "Expected 'OSUtilities' major category in summary log."
    )
    assert "Generic_Path:" in summary_content, (
        "Expected 'Generic_Path' minor category in summary log."
    )
    assert "Total Peak Memory:" in summary_content, (
        "Expected 'Total Peak Memory' for major categories in summary log."
    )

    # 3. Verify auxiliary logs
    assert (run_specific_output_dir / "build_stdout.log").exists()
    assert (run_specific_output_dir / "build_stderr.log").exists()
    assert (run_specific_output_dir / "metadata.log").exists()
    assert (run_specific_output_dir / "clean.log").exists()
    # This log is only created by the 'rss_pidstat' collector, not 'pss_psutil' used in this test.
    # assert (run_specific_output_dir / "collector_aux.log").exists()


def test_run_with_no_project_found(monkeypatch, caplog):
    """
    Tests the behavior when a specified project is not in the config.

    This test simulates running the tool from the command line with a project
    name that does not exist in the configuration files. It verifies that the
    application exits with a non-zero code and provides an appropriate error message.

    Args:
        monkeypatch: The pytest fixture for modifying modules, classes, or functions.
        caplog: The pytest fixture for capturing log output.
    """
    # Set environment variable to skip path validation for testing
    monkeypatch.setenv("MYMONITOR_SKIP_PATH_VALIDATION", "1")
    
    # Simulate the command-line arguments. Passing a non-existent project name.
    project_name = "NonExistentProject"
    test_args = ["mymonitor", "--project", project_name]
    monkeypatch.setattr(sys, "argv", test_args)

    # --- Test Execution ---
    # Run the main CLI function and assert that it exits with a non-zero code.
    with pytest.raises(SystemExit) as e:
        main_cli()

    # Verify that the exit code is non-zero and that an appropriate error
    # message was logged.
    assert e.value.code != 0
    assert "not found in configuration" in caplog.text


def test_unrecognized_argument(monkeypatch):
    """
    Tests the behavior when an unrecognized argument is provided.
    """
    # Set environment variable to skip path validation for testing
    monkeypatch.setenv("MYMONITOR_SKIP_PATH_VALIDATION", "1")
    
    # Simulate the command-line arguments with an unrecognized argument.
    test_args = ["mymonitor", "--unrecognized-argument"]
    monkeypatch.setattr(sys, "argv", test_args)

    # --- Test Execution ---
    with pytest.raises(SystemExit) as e:
        main_cli()

    # Verify that the exit code is non-zero (typically 2 for argument errors).
    assert e.value.code != 0


def test_invalid_jobs_argument(monkeypatch, caplog):
    """
    Tests the behavior when an invalid --jobs argument is provided.

    This test simulates running the tool from the command line with an invalid
    jobs argument. It verifies that the application exits with a non-zero code
    and provides an appropriate error message.

    Args:
        monkeypatch: The pytest fixture for modifying modules, classes, or functions.
        caplog: The pytest fixture for capturing log output.
    """
    # Set environment variable to skip path validation for testing
    monkeypatch.setenv("MYMONITOR_SKIP_PATH_VALIDATION", "1")
    
    # Simulate the command-line arguments. Passing an invalid jobs argument.
    invalid_jobs_str = "invalid_jobs_format"
    test_args = ["mymonitor", "--jobs", invalid_jobs_str]
    monkeypatch.setattr(sys, "argv", test_args)

    # --- Test Execution ---
    with pytest.raises(SystemExit) as e:
        main_cli()

    # Verify that the exit code is non-zero and an appropriate error
    # message was logged.
    assert e.value.code != 0
    assert "validation" in caplog.text
