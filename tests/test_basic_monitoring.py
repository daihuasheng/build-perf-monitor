import pytest
import os
import polars as pl
import sys
import toml
from pathlib import Path

from mymonitor.main import main_cli

# Path to the fake build script
FAKE_BUILD_SCRIPT_PATH = (Path(__file__).parent / "fake_build_script.sh").resolve()

@pytest.fixture
def setup_test_config_files(tmp_path: Path) -> Path:
    """
    Creates a temporary, valid TOML configuration structure for testing.
    This includes config.toml, projects.toml, and rules.toml.
    """
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()

    # 1. Create projects.toml
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

    # 2. Create rules.toml (minimal set for the test)
    rules_data = {
        "rules": [
            {
                "major_category": "Scripting",
                "category": "ShellScriptFile",
                "priority": 45,
                "match_field": "current_cmd_name",
                "match_type": "endswith",
                "pattern": ".sh",
            }
        ]
    }
    rules_toml_path = conf_dir / "rules.toml"
    with open(rules_toml_path, "w") as f:
        toml.dump(rules_data, f)

    # 3. Create the main config.toml that points to the others
    main_config_data = {
        "monitor": {
            "interval_seconds": 1,
            "default_jobs": [1], # Keep it simple for tests
            "metric_type": "pss_psutil",
            "monitor_core": -1, # Disable pinning for simplicity in test
            "build_cores_policy": "none",
            "skip_plots": True,
            "log_root_dir": str(tmp_path / "logs_output")
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

@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory for logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir

def test_basic_monitoring_run(
    setup_test_config_files: Path, # Use the new fixture
    monkeypatch,
    capsys
):
    """
    Tests a basic run of the mymonitor CLI with a fake build using the new TOML config system.
    Checks if output Parquet and summary log files are created.
    """
    # Ensure the fake build script is executable
    if not os.access(FAKE_BUILD_SCRIPT_PATH, os.X_OK):
        pytest.skip(f"Fake build script {FAKE_BUILD_SCRIPT_PATH} is not executable. Run chmod +x on it.")

    # --- Monkeypatching the new config system ---
    # 1. Force the config module to use our temporary config.toml file.
    monkeypatch.setattr("mymonitor.config._CONFIG_FILE_PATH", setup_test_config_files)
    # 2. IMPORTANT: Reset the global _CONFIG singleton to None so that get_config() is forced to reload
    #    from the new path we just set.
    monkeypatch.setattr("mymonitor.config._CONFIG", None)

    # Simulate command-line arguments for main_cli.
    # Note: Most config is now in the TOML files, so we don't need many CLI args.
    test_args = [
        "mymonitor", # Script name, usually sys.argv[0]
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    try:
        main_cli()
    except SystemExit as e:
        assert e.code == 0, f"main_cli exited with non-zero code: {e.code}"

    # --- Verification ---
    # The log root dir is now defined inside our temporary config.toml
    log_root_dir = setup_test_config_files.parent.parent / "logs_output"
    
    # Check that a run-specific directory was created inside the log root
    run_dirs = list(log_root_dir.glob("run_*"))
    assert len(run_dirs) == 1, f"Expected 1 run directory in {log_root_dir}, found {len(run_dirs)}"
    run_specific_output_dir = run_dirs[0]
    assert run_specific_output_dir.is_dir()

    # Check for the Parquet file
    parquet_files = list(run_specific_output_dir.glob("FakeProject_j1_mem_pss_psutil_*.parquet"))
    assert len(parquet_files) == 1, f"Expected 1 Parquet file, found {len(parquet_files)} in {run_specific_output_dir}"
    output_parquet_file = parquet_files[0]
    assert output_parquet_file.exists()
    assert output_parquet_file.stat().st_size > 0, "Parquet file is empty"

    # Optionally, try to read the Parquet file to ensure it's valid
    try:
        df = pl.read_parquet(output_parquet_file)
        assert df.height > 0, "Parquet file loaded but contains no data rows"
        assert "PROCESS" in df["Record_Type"].unique().to_list(), "PROCESS records missing"
        assert "ALL_SUM" in df["Record_Type"].unique().to_list(), "ALL_SUM records missing"
    except Exception as e:
        pytest.fail(f"Failed to read or validate Parquet file {output_parquet_file}: {e}")

    # Check for the summary log file
    summary_files = list(run_specific_output_dir.glob("FakeProject_j1_mem_pss_psutil_*_summary.log"))
    assert len(summary_files) == 1, f"Expected 1 summary log file, found {len(summary_files)} in {run_specific_output_dir}"
    summary_log_file = summary_files[0]
    assert summary_log_file.exists()
    assert summary_log_file.stat().st_size > 0, "Summary log file is empty"

    # Check content of summary log for build success
    with open(summary_log_file, "r") as f:
        summary_content = f.read()
        assert "Final Build Exit Code: 0" in summary_content, "Build did not exit successfully according to summary log"
        assert "Fake build process finished." in summary_content, "Fake build output not found in summary log"