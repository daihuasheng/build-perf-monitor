"""
Pytest configuration and shared fixtures for MyMonitor test suite.

This module provides common fixtures, test utilities, and configuration
for all test modules in the MyMonitor project.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

import pytest

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Test Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# ============================================================================
# Core Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_cpu_cores():
    """Mock available CPU cores for testing."""
    return list(range(8))  # 8-core system for testing


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
        "general": {
            "default_jobs": [4, 8, 16],
            "log_level": "INFO",
            "enable_detailed_logging": False,
        },
        "collection": {
            "interval_seconds": 0.05,
            "collector_type": "pss_psutil",
            "pss_collector_mode": "full_scan",
        },
        "scheduling": {
            "scheduling_policy": "adaptive",
            "enable_cpu_affinity": True,
            "max_concurrent_monitors": 4,
            "thread_name_prefix": "MonitorWorker",
            "manual_build_cores": "",
            "manual_monitoring_cores": "",
        },
        "hybrid": {
            "hybrid_discovery_interval": 0.01,
            "hybrid_sampling_workers": 4,
            "hybrid_task_queue_size": 1000,
            "hybrid_result_queue_size": 2000,
            "hybrid_enable_prioritization": True,
            "hybrid_max_retry_count": 3,
            "hybrid_queue_timeout": 0.1,
            "hybrid_worker_timeout": 5.0,
            "hybrid_enable_queue_monitoring": True,
            "hybrid_batch_result_size": 50,
        },
    }


@pytest.fixture
def sample_project_config():
    """Sample project configuration for testing."""
    return {
        "name": "test_project",
        "dir": "/tmp/test_project",
        "build_command_template": "make -j<N>",
        "clean_command_template": "make clean",
        "process_pattern": ".*",
        "description": "Test project for unit testing",
    }


@pytest.fixture
def sample_rules_config():
    """Sample classification rules for testing."""
    return [
        {
            "priority": 100,
            "major_category": "Compiler",
            "category": "gcc",
            "match_type": "regex",
            "match_field": "current_cmd_name",
            "patterns": "^gcc.*|^g\\+\\+.*",
        },
        {
            "priority": 90,
            "major_category": "Linker",
            "category": "ld",
            "match_type": "regex",
            "match_field": "current_cmd_name",
            "patterns": "^ld$|^ld\\..*",
        },
        {
            "priority": 80,
            "major_category": "Build_Tool",
            "category": "make",
            "match_type": "in_list",
            "match_field": "current_cmd_name",
            "patterns": ["make", "gmake", "ninja"],
        },
    ]


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_psutil():
    """Mock psutil for testing without system dependencies."""
    with (
        patch("psutil.cpu_count") as mock_cpu_count,
        patch("psutil.Process") as mock_process_class,
    ):

        mock_cpu_count.return_value = 8

        # Mock process instance
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.name.return_value = "test_process"
        mock_process.cmdline.return_value = ["test_process", "--arg1", "--arg2"]
        mock_process.memory_info.return_value = Mock(rss=1024 * 1024, vms=2048 * 1024)
        mock_process.memory_full_info.return_value = Mock(
            rss=1024 * 1024, vms=2048 * 1024, pss=1536 * 1024
        )
        mock_process.children.return_value = []
        mock_process.is_running.return_value = True

        mock_process_class.return_value = mock_process

        yield {
            "cpu_count": mock_cpu_count,
            "Process": mock_process_class,
            "process_instance": mock_process,
        }


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for testing command execution."""
    with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:

        # Mock successful command execution
        mock_run.return_value = Mock(
            returncode=0, stdout="Mock command output", stderr=""
        )

        # Mock process for Popen
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("Mock output", "")
        mock_process.poll.return_value = 0

        mock_popen.return_value = mock_process

        yield {"run": mock_run, "Popen": mock_popen, "process": mock_process}


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def config_files(
    temp_dir, sample_config_data, sample_project_config, sample_rules_config
):
    """Create temporary configuration files for testing."""
    import toml

    # Create config.toml
    config_file = temp_dir / "config.toml"
    config_data = {
        "monitor": sample_config_data,
        "paths": {
            "projects_config": str(temp_dir / "projects.toml"),
            "rules_config": str(temp_dir / "rules.toml"),
        },
    }
    with open(config_file, "w") as f:
        toml.dump(config_data, f)

    # Create projects.toml
    projects_file = temp_dir / "projects.toml"
    with open(projects_file, "w") as f:
        toml.dump({"projects": [sample_project_config]}, f)

    # Create rules.toml
    rules_file = temp_dir / "rules.toml"
    with open(rules_file, "w") as f:
        toml.dump({"rules": sample_rules_config}, f)

    return {
        "config": config_file,
        "projects": projects_file,
        "rules": rules_file,
        "dir": temp_dir,
    }


# ============================================================================
# Test Utilities
# ============================================================================


class TestUtils:
    """Utility functions for testing."""

    @staticmethod
    def create_mock_process_sample(
        pid: int, cmd_name: str, full_cmd: str, memory_mb: float = 10.0
    ):
        """Create a mock process memory sample."""
        from mymonitor.collectors.base import ProcessMemorySample

        return ProcessMemorySample(
            timestamp=1234567890.0,
            pid=pid,
            command_name=cmd_name,
            full_command=full_cmd,
            memory_mb=memory_mb,
        )

    @staticmethod
    def create_mock_cpu_allocation_plan(
        build_cores: List[int], monitoring_cores: List[int], **kwargs
    ):
        """Create a mock CPU allocation plan."""
        from mymonitor.models.runtime import CpuAllocationPlan

        return CpuAllocationPlan(
            build_cores=build_cores,
            monitoring_cores=monitoring_cores,
            build_cores_desc=(
                f"{build_cores[0]}-{build_cores[-1]}" if build_cores else ""
            ),
            monitoring_cores_desc=(
                f"{monitoring_cores[0]}-{monitoring_cores[-1]}"
                if monitoring_cores
                else ""
            ),
            taskset_prefix=(
                f"taskset -c {build_cores[0]}-{build_cores[-1]}" if build_cores else ""
            ),
            taskset_available=True,
            **kwargs,
        )


@pytest.fixture
def test_utils():
    """Provide test utility functions."""
    return TestUtils


@pytest.fixture(autouse=True)
def clear_config_after_test():
    """Automatically clear configuration cache after each test."""
    from pathlib import Path

    # Store original config path (default path)
    original_config_path = Path(__file__).parent.parent / "conf" / "config.toml"

    yield  # Run the test

    # Clean up after test
    from mymonitor.config import clear_config_cache, set_config_path
    from mymonitor.classification.classifier import clear_categorization_cache

    clear_config_cache()
    clear_categorization_cache()

    # Always reset to original config path
    set_config_path(original_config_path)


# ============================================================================
# Performance Test Fixtures
# ============================================================================


@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        "max_memory_mb": 100,  # Maximum memory usage in MB
        "max_cpu_percent": 50,  # Maximum CPU usage percentage
        "max_execution_time": 5.0,  # Maximum execution time in seconds
        "sample_size": 1000,  # Number of samples for performance tests
    }
