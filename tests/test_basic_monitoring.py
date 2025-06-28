"""
End-to-end integration tests for the MyMonitor application.

This test module focuses on verifying the main command-line entry point (`main_cli`)
and the entire monitoring workflow. It simulates a user running the application
with a controlled, temporary configuration and inspects the output artifacts
to ensure they are created correctly and contain the expected data.
"""

import os
import sys
from pathlib import Path

import polars as pl
import pytest
import toml

# Import the main entry point of the application to be tested.
from mymonitor.main import main_cli

# --- Module Constants ---

# Resolve the absolute path to the fake build script used in tests.
# This ensures the test can locate the script regardless of where pytest is run from.
FAKE_BUILD_SCRIPT_PATH = (Path(__file__).parent / "fake_build_script.sh").resolve()


# --- Fixtures ---


@pytest.fixture
def setup_test_config_files(tmp_path: Path) -> Path:
    """
    Creates a temporary, valid TOML configuration structure for testing.

    This fixture sets up an isolated environment within a temporary directory
    provided by pytest. It creates `conf/config.toml`, `conf/projects.toml`,
    and `conf/rules.toml`, mimicking the real application structure. This ensures
    that tests are self-contained and do not depend on or alter the actual
    configuration files in the project.

    Args:
        tmp_path: The temporary directory path provided by the pytest fixture.

    Returns:
        The path to the main `config.toml` file, which will be used to
        drive the application during the test.
    """
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()

    # 1. Create projects.toml with a single fake project.
    # This project uses the fake build script and includes setup/clean commands
    # to test the full execution flow.
    projects_data = {
        "projects": [
            {
                "name": "FakeProject",
                "dir": str(tmp_path),
                "build_command_template": f"{str(FAKE_BUILD_SCRIPT_PATH)}",
                "process_pattern": "fake_build_script.sh|sleep",
                "clean_command_template": "echo 'Fake clean complete'",
                "setup_command_template": "echo 'Fake setup complete'",
            }
        ]
    }
    projects_toml_path = conf_dir / "projects.toml"
    with open(projects_toml_path, "w") as f:
        toml.dump(projects_data, f)

    # 2. Create rules.toml with a minimal set of rules needed for this test.
    # These rules ensure that the processes spawned by the fake build script
    # are categorized correctly.
    rules_data = {
        "rules": [
            {
                "major_category": "Scripting",
                "category": "ShellScriptFile",
                "priority": 62,
                "match_field": "current_cmd_full",
                "match_type": "regex",
                "pattern": r"fake_build_script\.sh",
            },
            {
                "major_category": "OSUtilities",
                "category": "Sleep",
                "priority": 5,
                "match_field": "current_cmd_name",
                "match_type": "exact",
                "pattern": "sleep",
            },
        ]
    }
    rules_toml_path = conf_dir / "rules.toml"
    with open(rules_toml_path, "w") as f:
        toml.dump(rules_data, f)

    # 3. Create the main config.toml that points to the other config files.
    # It also defines test-specific settings like a short monitoring interval
    # and the output log directory, following the new nested structure.
    main_config_data = {
        "monitor": {
            "general": {
                "default_jobs": [1],
                "skip_plots": True,
                "log_root_dir": str(tmp_path / "logs_output"),
            },
            "collection": {
                "interval_seconds": 0.1,
                "metric_type": "pss_psutil",
            },
            "scheduling": {
                "monitor_core": -1,
                "build_cores_policy": "none",
            },
        },
        "paths": {
            "projects_config": "projects.toml",
            "rules_config": "rules.toml",
        },
    }
    main_config_toml_path = conf_dir / "config.toml"
    with open(main_config_toml_path, "w") as f:
        toml.dump(main_config_data, f)

    return main_config_toml_path


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

    # --- Test Setup: Monkeypatching ---
    # Force the application's config loader to use our temporary config file.
    monkeypatch.setattr("mymonitor.config._CONFIG_FILE_PATH", setup_test_config_files)
    # Force a reload of the config singleton to pick up the new path.
    monkeypatch.setattr("mymonitor.config._CONFIG", None)

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
    assert (
        len(run_dirs) == 1
    ), f"Expected 1 run directory in {log_root_dir}, found {len(run_dirs)}"
    run_specific_output_dir = run_dirs[0]
    assert run_specific_output_dir.is_dir()

    # 1. Verify the Parquet data file.
    parquet_files = list(
        run_specific_output_dir.glob("FakeProject_j1_mem_pss_psutil_*.parquet")
    )
    assert (
        len(parquet_files) == 1
    ), f"Expected 1 Parquet file, found {len(parquet_files)} in {run_specific_output_dir}"
    output_parquet_file = parquet_files[0]
    assert output_parquet_file.exists() and output_parquet_file.stat().st_size > 0

    # Validate the content of the Parquet file to ensure data was collected.
    try:
        df = pl.read_parquet(output_parquet_file)
        assert df.height > 0, "Parquet file loaded but contains no data rows"
        # Check for both individual process samples and interval summary rows.
        assert "PROCESS" in df["Record_Type"].unique().to_list()
        assert "ALL_SUM" in df["Record_Type"].unique().to_list()
        # Check that our fake script was categorized correctly based on the test rules.
        scripting_rows = df.filter(pl.col("Major_Category") == "Scripting")
        assert scripting_rows.height > 0, "No 'Scripting' category rows found."
    except Exception as e:
        pytest.fail(
            f"Failed to read or validate Parquet file {output_parquet_file}: {e}"
        )

    # 2. Verify the summary log file.
    summary_files = list(
        run_specific_output_dir.glob("FakeProject_j1_mem_pss_psutil_*_summary.log")
    )
    assert (
        len(summary_files) == 1
    ), f"Expected 1 summary log file, found {len(summary_files)} in {run_specific_output_dir}"
    summary_log_file = summary_files[0]
    assert summary_log_file.exists() and summary_log_file.stat().st_size > 0

    # Validate the content of the summary log.
    with open(summary_log_file, "r") as f:
        summary_content = f.read()
        # Check that the build was reported as successful.
        assert (
            "Final Build Exit Code: 0" in summary_content
        ), "Build did not exit successfully according to summary log"
        # Check for the prefixed output from the main build script.
        assert (
            "[STDOUT] Fake build process finished." in summary_content
        ), "Prefixed fake build output not found in summary log"
        # Check for the prefixed output from the setup command, confirming that
        # the logging format is consistent across all executed commands.
        assert (
            "[STDOUT] Fake setup complete" in summary_content
        ), "Prefixed setup command output not found in summary log"
