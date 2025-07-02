"""
Tests for the process tree verification tool (`tools/verify_process_tree.py`).

This module contains integration tests that simulate running the verification
tool against different types of build scripts to ensure it correctly identifies
whether the 'descendants_only' optimization is safe.
"""

import os
import sys
from pathlib import Path

import pytest
import toml

# Add the project root to the path to allow importing the tool's modules.
# This makes the test environment behave like a real execution.
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.verify_process_tree import verify_project


# --- Fixtures ---


@pytest.fixture
def setup_verifier_test_env(tmp_path: Path) -> Path:
    """
    Sets up a temporary environment for the verifier tool tests.

    This fixture creates:
    1. Two fake build scripts: one for a success case (all children are descendants)
       and one for a failure case (creates a detached, "orphan" process).
    2. A temporary configuration structure (`config.toml`, `projects.toml`) that
       defines projects using these fake scripts.

    Args:
        tmp_path: The temporary directory path provided by pytest.

    Returns:
        The path to the temporary main `config.toml` file.
    """
    # --- Create Fake Build Scripts ---
    # Success case: A simple script that spawns a child process and waits for it.
    success_script_path = tmp_path / "fake_build_success.sh"
    success_script_path.write_text(
        "#!/bin/bash\necho 'Success case: Starting sleep...'\nsleep 0.2\necho 'Success case: Finished.'"
    )

    # Failure case: This script starts a process in the background and detaches it.
    # The orphan process now sleeps for 0.5s, which is longer than the
    # verifier's grace period (0.2s), eliminating the race condition.
    failure_script_path = tmp_path / "fake_build_failure.sh"
    failure_script_path.write_text(
        "#!/bin/bash\necho 'Failure case: Starting detached sleep...'\n(sleep 0.5 &)\nsleep 0.1\necho 'Failure case: Main script finished.'"
    )

    # Make the scripts executable.
    os.chmod(success_script_path, 0o755)
    os.chmod(failure_script_path, 0o755)

    # --- Create Configuration Files ---
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()

    # Create projects.toml defining two projects for our test cases.
    projects_data = {
        "projects": [
            {
                "name": "SuccessProject",
                "dir": str(tmp_path),
                "build_command_template": f"{str(success_script_path)} <N>",
                "process_pattern": r"fake_build_success\.sh|sleep 0\.2",
            },
            {
                "name": "FailureProject",
                "dir": str(tmp_path),
                "build_command_template": f"{str(failure_script_path)} <N>",
                "process_pattern": r"fake_build_failure\.sh|sleep 0\.5",
            },
        ]
    }
    projects_toml_path = conf_dir / "projects.toml"
    with open(projects_toml_path, "w") as f:
        toml.dump(projects_data, f)

    # Create the main config.toml pointing to the projects file.
    main_config_data = {
        "paths": {"projects_config": "projects.toml", "rules_config": "rules.toml"}
    }
    main_config_toml_path = conf_dir / "config.toml"
    with open(main_config_toml_path, "w") as f:
        toml.dump(main_config_data, f)

    # Create an empty rules.toml, as the config loader expects it.
    (conf_dir / "rules.toml").touch()

    return main_config_toml_path


# --- Test Cases ---


def test_verifier_success_case(setup_verifier_test_env: Path, monkeypatch, capsys):
    """
    Tests the success scenario where all build processes are descendants.
    The verifier tool should report that the 'descendants_only' mode is safe.
    """
    # Force the config loader to use our temporary config file.
    monkeypatch.setattr("mymonitor.config._CONFIG_FILE_PATH", setup_verifier_test_env)
    monkeypatch.setattr("mymonitor.config._CONFIG", None)

    # Call the function directly and check its boolean return value instead
    # of expecting a SystemExit. The function is a library function, not a CLI entrypoint.
    is_safe = verify_project(project_name="SuccessProject", parallelism=1)

    # A successful verification should return True.
    assert is_safe is True

    # Capture the printed output and verify the success message.
    captured = capsys.readouterr()
    assert "[SUCCESS] Verification passed!" in captured.out


def test_verifier_failure_case(setup_verifier_test_env: Path, monkeypatch, capsys):
    """
    Tests the failure scenario where a detached process is created.
    The verifier tool should detect this "orphan" process and report failure.
    """
    # Force the config loader to use our temporary config file.
    monkeypatch.setattr("mymonitor.config._CONFIG_FILE_PATH", setup_verifier_test_env)
    monkeypatch.setattr("mymonitor.config._CONFIG", None)

    is_safe = verify_project(project_name="FailureProject", parallelism=1)

    # A failed verification should return False.
    assert is_safe is False

    # Capture the printed output and verify the failure message and details.
    captured = capsys.readouterr()
    assert "[FAILURE] Verification failed." in captured.out
    # The failure script creates one detached 'sleep' process.
    assert "Found 1 orphan process(es)" in captured.out
