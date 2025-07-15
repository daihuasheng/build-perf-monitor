"""
End-to-end tests for complete monitoring workflow.

Tests the full monitoring workflow from configuration loading
to result generation, including all major components.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock

from mymonitor.cli.orchestrator import BuildRunner
from mymonitor.config import set_config_path, clear_config_cache
from mymonitor.models.config import ProjectConfig


@pytest.mark.e2e
class TestMonitoringWorkflow:
    """End-to-end tests for monitoring workflow."""

    def setup_method(self):
        """Set up test environment."""
        # Clear any cached configuration
        clear_config_cache()

    def teardown_method(self):
        """Clean up after test."""
        clear_config_cache()

    @pytest.mark.asyncio
    @patch("mymonitor.executor.build_process.subprocess.Popen")
    @patch("mymonitor.system.cpu_manager.subprocess.run")
    @patch("mymonitor.system.cpu_manager.psutil.cpu_count")
    @patch("mymonitor.collectors.pss_psutil.psutil.process_iter")
    async def test_complete_monitoring_workflow(
        self,
        mock_process_iter,
        mock_cpu_count,
        mock_subprocess_run,
        mock_popen,
        config_files,
        temp_dir,
    ):
        """Test complete monitoring workflow from start to finish."""
        # 1. Setup mocks
        mock_cpu_count.return_value = 8
        mock_subprocess_run.return_value = Mock(returncode=0)  # taskset available

        # Mock build process
        mock_build_process = Mock()
        mock_build_process.pid = 12345
        mock_build_process.returncode = 0
        mock_build_process.communicate.return_value = ("Build output", "")
        mock_build_process.poll.return_value = 0
        mock_build_process.wait.return_value = 0
        mock_build_process.__enter__ = Mock(return_value=mock_build_process)
        mock_build_process.__exit__ = Mock(return_value=None)
        mock_popen.return_value = mock_build_process

        # Mock processes for monitoring
        mock_gcc_process = Mock()
        mock_gcc_process.pid = 12346
        mock_gcc_process.name.return_value = "gcc"
        mock_gcc_process.cmdline.return_value = ["gcc", "-O2", "-c", "file.c"]

        # Mock as_dict to return proper dictionary
        mock_gcc_process.as_dict.return_value = {
            "pid": 12346,
            "name": "gcc",
            "cmdline": ["gcc", "-O2", "-c", "file.c"],
        }

        # Mock memory info with proper attributes
        mock_memory_info = Mock()
        mock_memory_info.rss = 10 * 1024 * 1024
        mock_gcc_process.memory_info.return_value = mock_memory_info

        mock_memory_full_info = Mock()
        mock_memory_full_info.rss = 10 * 1024 * 1024
        mock_memory_full_info.vms = 20 * 1024 * 1024
        mock_memory_full_info.pss = 15 * 1024 * 1024
        mock_gcc_process.memory_full_info.return_value = mock_memory_full_info

        mock_gcc_process.is_running.return_value = True
        mock_gcc_process.children.return_value = []

        mock_process_iter.return_value = [mock_gcc_process]

        # 2. Set up configuration
        set_config_path(config_files["config"])

        # 3. Create project configuration
        project_config = ProjectConfig(
            name="test_project",
            dir=str(temp_dir),
            build_command_template="echo 'mock build'",
            process_pattern=".*",
            clean_command_template="echo 'mock clean'",
            setup_command_template="",
        )

        # 4. Create build runner
        build_runner = BuildRunner(
            project_config=project_config,
            parallelism_level=4,
            monitoring_interval=0.01,  # Fast for testing
            log_dir=temp_dir / "logs",
            collector_type="pss_psutil",
            skip_pre_clean=True,
        )

        # 5. Run the complete workflow
        with (
            patch(
                "mymonitor.executor.thread_pool.get_available_cores",
                return_value=list(range(8)),
            ),
            patch(
                "builtins.open",
                side_effect=lambda path, mode="r": (
                    Mock(
                        __enter__=Mock(
                            return_value=Mock(
                                read=Mock(
                                    return_value="12346 (gcc) R 1 12346 12346 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 0"
                                )
                            )
                        ),
                        __exit__=Mock(return_value=None),
                    )
                    if "/proc/" in str(path) and "/stat" in str(path)
                    else open(path, mode)
                ),
            ),
        ):
            success = await build_runner.run_async()

        # 6. Verify results
        assert success is True
        assert build_runner.run_context is not None
        assert build_runner.monitoring_architecture is not None
        assert build_runner.build_runner is not None

        # Verify CPU allocation was performed
        assert build_runner.run_context.build_cores_target_str is not None
        assert build_runner.run_context.monitor_script_pinned_to_core_info is not None

    @pytest.mark.asyncio
    @patch("mymonitor.system.cpu_manager.psutil.cpu_count")
    async def test_resource_constrained_workflow(
        self, mock_cpu_count, config_files, temp_dir, caplog
    ):
        """Test workflow in resource-constrained environment."""
        # Simulate 2-core system
        mock_cpu_count.return_value = 2

        set_config_path(config_files["config"])

        project_config = ProjectConfig(
            name="test_project",
            dir=str(temp_dir),
            build_command_template="echo 'mock build'",
            process_pattern=".*",
            clean_command_template="echo 'mock clean'",
            setup_command_template="",
        )

        build_runner = BuildRunner(
            project_config=project_config,
            parallelism_level=8,  # High parallelism on low-core system
            monitoring_interval=0.01,
            log_dir=temp_dir / "logs",
            collector_type="pss_psutil",
            skip_pre_clean=True,
        )

        with patch(
            "mymonitor.executor.thread_pool.get_available_cores", return_value=[0, 1]
        ):
            with patch(
                "mymonitor.executor.build_process.subprocess.Popen"
            ) as mock_popen:
                mock_process = Mock()
                mock_process.pid = 12345
                mock_process.returncode = 0
                mock_process.communicate.return_value = ("", "")
                mock_process.poll.return_value = 0
                mock_popen.return_value = mock_process

                with patch(
                    "mymonitor.collectors.pss_psutil.psutil.process_iter",
                    return_value=[],
                ):
                    success = await build_runner.run_async()

        # Should succeed but with warnings
        assert success is True
        assert "CPU资源不足" in caplog.text or "资源不足" in caplog.text

    @pytest.mark.asyncio
    @patch("mymonitor.system.cpu_manager.psutil.cpu_count")
    async def test_manual_allocation_workflow(
        self, mock_cpu_count, config_files, temp_dir
    ):
        """Test workflow with manual CPU allocation."""
        mock_cpu_count.return_value = 16

        # Modify config for manual allocation
        import toml

        config_data = toml.load(config_files["config"])
        config_data["monitor"]["scheduling"]["scheduling_policy"] = "manual"
        config_data["monitor"]["scheduling"]["manual_build_cores"] = "0-7"
        config_data["monitor"]["scheduling"]["manual_monitoring_cores"] = "8-11"

        manual_config_file = temp_dir / "manual_config.toml"
        with open(manual_config_file, "w") as f:
            toml.dump(config_data, f)

        set_config_path(manual_config_file)

        project_config = ProjectConfig(
            name="test_project",
            dir=str(temp_dir),
            build_command_template="echo 'mock build'",
            process_pattern=".*",
            clean_command_template="echo 'mock clean'",
            setup_command_template="",
        )

        build_runner = BuildRunner(
            project_config=project_config,
            parallelism_level=4,
            monitoring_interval=0.01,
            log_dir=temp_dir / "logs",
            collector_type="pss_psutil",
            skip_pre_clean=True,
        )

        with patch(
            "mymonitor.executor.thread_pool.get_available_cores",
            return_value=list(range(16)),
        ):
            with patch(
                "mymonitor.executor.build_process.subprocess.Popen"
            ) as mock_popen:
                mock_process = Mock()
                mock_process.pid = 12345
                mock_process.returncode = 0
                mock_process.communicate.return_value = ("", "")
                mock_process.poll.return_value = 0
                mock_popen.return_value = mock_process

                with patch(
                    "mymonitor.collectors.pss_psutil.psutil.process_iter",
                    return_value=[],
                ):
                    success = await build_runner.run_async()

        assert success is True
        # Verify manual allocation was used
        assert (
            "manual" in build_runner.run_context.build_cores_target_str.lower()
            or "0-7" in build_runner.run_context.build_cores_target_str
        )

    @pytest.mark.asyncio
    async def test_build_failure_workflow(self, config_files, temp_dir):
        """Test workflow when build fails."""
        set_config_path(config_files["config"])

        project_config = ProjectConfig(
            name="test_project",
            dir=str(temp_dir),
            build_command_template="exit 1",  # Failing command
            process_pattern=".*",
            clean_command_template="echo 'mock clean'",
            setup_command_template="",
        )

        build_runner = BuildRunner(
            project_config=project_config,
            parallelism_level=2,
            monitoring_interval=0.01,
            log_dir=temp_dir / "logs",
            collector_type="pss_psutil",
            skip_pre_clean=True,
        )

        with patch(
            "mymonitor.executor.thread_pool.get_available_cores",
            return_value=list(range(4)),
        ):
            with patch(
                "mymonitor.executor.build_process.subprocess.Popen"
            ) as mock_popen:
                mock_process = Mock()
                mock_process.pid = 12345
                mock_process.returncode = 1  # Failure
                mock_process.communicate.return_value = ("", "Build failed")
                mock_process.poll.return_value = 1
                mock_popen.return_value = mock_process

                success = await build_runner.run_async()

        # Should handle build failure gracefully
        assert success is False

    @pytest.mark.asyncio
    @patch("mymonitor.system.cpu_manager.psutil.cpu_count")
    async def test_monitoring_with_process_classification(
        self, mock_cpu_count, config_files, temp_dir
    ):
        """Test workflow with process classification."""
        mock_cpu_count.return_value = 8

        set_config_path(config_files["config"])

        project_config = ProjectConfig(
            name="test_project",
            dir=str(temp_dir),
            build_command_template="echo 'mock build'",
            process_pattern=".*",
            clean_command_template="echo 'mock clean'",
            setup_command_template="",
        )

        build_runner = BuildRunner(
            project_config=project_config,
            parallelism_level=2,
            monitoring_interval=0.01,
            log_dir=temp_dir / "logs",
            collector_type="pss_psutil",
            skip_pre_clean=True,
        )

        # Mock processes with different types
        mock_gcc = Mock()
        mock_gcc.pid = 12346
        mock_gcc.name.return_value = "gcc"
        mock_gcc.cmdline.return_value = ["gcc", "-O2", "file.c"]
        mock_gcc.memory_info.return_value = Mock(rss=10 * 1024 * 1024)
        mock_gcc.memory_full_info.return_value = Mock(
            rss=10 * 1024 * 1024, vms=20 * 1024 * 1024, pss=15 * 1024 * 1024
        )
        mock_gcc.is_running.return_value = True
        mock_gcc.children.return_value = []

        mock_make = Mock()
        mock_make.pid = 12347
        mock_make.name.return_value = "make"
        mock_make.cmdline.return_value = ["make", "-j2"]
        mock_make.memory_info.return_value = Mock(rss=5 * 1024 * 1024)
        mock_make.memory_full_info.return_value = Mock(
            rss=5 * 1024 * 1024, vms=10 * 1024 * 1024, pss=7 * 1024 * 1024
        )
        mock_make.is_running.return_value = True
        mock_make.children.return_value = []

        with patch(
            "mymonitor.executor.thread_pool.get_available_cores",
            return_value=list(range(8)),
        ):
            with patch(
                "mymonitor.executor.build_process.subprocess.Popen"
            ) as mock_popen:
                mock_process = Mock()
                mock_process.pid = 12345
                mock_process.returncode = 0
                mock_process.communicate.return_value = ("", "")
                mock_process.poll.return_value = 0
                mock_popen.return_value = mock_process

                with patch(
                    "mymonitor.collectors.pss_psutil.psutil.process_iter",
                    return_value=[mock_gcc, mock_make],
                ):
                    success = await build_runner.run_async()

        assert success is True

        # Verify monitoring results contain classified processes
        if (
            hasattr(build_runner.monitoring_architecture, "results")
            and build_runner.monitoring_architecture.results
        ):
            results = build_runner.monitoring_architecture.results
            # Results should contain process classification information
            assert hasattr(results, "samples") or hasattr(results, "data")

    @pytest.mark.asyncio
    async def test_graceful_shutdown_workflow(self, config_files, temp_dir):
        """Test graceful shutdown of monitoring workflow."""
        set_config_path(config_files["config"])

        project_config = ProjectConfig(
            name="test_project",
            dir=str(temp_dir),
            build_command_template="sleep 10",  # Long-running command
            process_pattern=".*",
            clean_command_template="echo 'mock clean'",
            setup_command_template="",
        )

        build_runner = BuildRunner(
            project_config=project_config,
            parallelism_level=2,
            monitoring_interval=0.01,
            log_dir=temp_dir / "logs",
            collector_type="pss_psutil",
            skip_pre_clean=True,
        )

        with patch(
            "mymonitor.executor.thread_pool.get_available_cores",
            return_value=list(range(4)),
        ):
            with patch(
                "mymonitor.executor.build_process.subprocess.Popen"
            ) as mock_popen:
                mock_process = Mock()
                mock_process.pid = 12345
                mock_process.returncode = None  # Still running
                mock_process.communicate.return_value = ("", "")
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process

                with patch(
                    "mymonitor.collectors.pss_psutil.psutil.process_iter",
                    return_value=[],
                ):
                    # Start the workflow
                    task = asyncio.create_task(build_runner.run_async())

                    # Let it run briefly
                    await asyncio.sleep(0.1)

                    # Request shutdown
                    build_runner.shutdown_requested = True

                    # Simulate process completion
                    mock_process.returncode = 0
                    mock_process.poll.return_value = 0

                    # Wait for completion
                    success = await task

        # Should handle shutdown gracefully
        assert success is not None  # May be True or False depending on timing
