import pytest
import json
import sys
from pathlib import Path
import os
import polars as pl

from mymonitor.main import main_cli

# Path to the fake build script
FAKE_BUILD_SCRIPT_PATH = (Path(__file__).parent / "fake_build_script.sh").resolve()

@pytest.fixture
def temp_project_config(tmp_path: Path) -> Path:
    """
    Creates a temporary projects_config.json file for testing.
    The build command uses the fake_build_script.sh.
    """
    config_data = [
        {
            "NAME": "FakeProject",
            "DIR": str(tmp_path), # Build will run in the temp directory itself
            "BUILD_COMMAND_TEMPLATE": f"{str(FAKE_BUILD_SCRIPT_PATH)}", # Use absolute path
            "PROCESS_PATTERN": "fake_build_script.sh|sleep", # Pattern to match our script and its children
            "CLEAN_COMMAND_TEMPLATE": "echo 'Fake clean complete'",
            "SETUP_COMMAND_TEMPLATE": "echo 'Fake setup complete'"
        }
    ]
    config_file = tmp_path / "temp_projects_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f)
    return config_file

@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """
    Creates a temporary directory for logs.
    This is separate from pytest's tmp_path for the config
    to simulate a more realistic log output structure.
    """
    log_dir = tmp_path / "test_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def test_basic_monitoring_run(
    temp_project_config: Path,
    temp_log_dir: Path,
    monkeypatch,
    capsys
):
    """
    Tests a basic run of the mymonitor CLI with a fake build.
    Checks if output Parquet and summary log files are created.
    """
    # Ensure the fake build script is executable
    if not os.access(FAKE_BUILD_SCRIPT_PATH, os.X_OK):
        pytest.skip(f"Fake build script {FAKE_BUILD_SCRIPT_PATH} is not executable. Run chmod +x on it.")

    # Mock sys.argv to simulate command line arguments
    # We need to tell main_cli where to find the config and where to put logs.
    # For simplicity, we'll make main.py use these paths directly by patching.
    # A more advanced way would be to pass them as CLI args if main.py supported it for these paths.

    # Patch global config paths in main.py if they are used directly for loading/output
    # For this test, we will assume main.py can be influenced by environment or args
    # or we can directly call a function that takes these paths.
    # Here, we'll rely on the fact that main_cli creates a run_specific_log_dir_name
    # inside LOG_ROOT_DIR. We'll patch LOG_ROOT_DIR.

    monkeypatch.setattr("mymonitor.main.CONFIG_FILE_PATH", temp_project_config)
    monkeypatch.setattr("mymonitor.main.LOG_ROOT_DIR", temp_log_dir)

    # Simulate command-line arguments for main_cli
    # Running for project "FakeProject", with 1 job, PSS_Psutil collector, and skipping plots.
    test_args = [
        "mymonitor", # Script name, usually sys.argv[0]
        "--projects", "FakeProject",
        "--jobs", "1", # A single, small parallelism level for quick test
        "--metric-type", "pss_psutil", # Use psutil collector as it has fewer external deps than pidstat
        "--skip-plots" # Skip plotting for this basic test
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    try:
        main_cli()
    except SystemExit as e:
        # main_cli calls sys.exit(0) on successful completion.
        # We expect this, so we catch it. Any other SystemExit might be an error.
        assert e.code == 0, f"main_cli exited with non-zero code: {e.code}"

    # Check that a run-specific directory was created inside temp_log_dir
    run_dirs = list(temp_log_dir.glob("run_*"))
    assert len(run_dirs) == 1, f"Expected 1 run directory in {temp_log_dir}, found {len(run_dirs)}"
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
        # Check for expected record types
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

    # Clean up the specific run directory after test to keep temp_log_dir clean for other potential tests
    # shutil.rmtree(run_specific_output_dir) # Or let pytest's tmp_path handling do its work