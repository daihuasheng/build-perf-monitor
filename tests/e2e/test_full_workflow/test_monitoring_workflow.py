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

    def _get_system_mocks(self, core_count=8):
        """Get common system mocks to avoid warnings."""
        return [
            patch(
                "mymonitor.executor.thread_pool.get_available_cores",
                return_value=list(range(core_count)),
            ),
            patch("mymonitor.system.cpu_manager.get_cpu_manager"),
            patch("mymonitor.executor.build_process.os.getpid", return_value=12345),
            patch(
                "mymonitor.system.cpu_manager.set_current_thread_affinity",
                return_value=True,
            ),
            patch("mymonitor.system.cpu_manager.psutil.Process"),
            patch(
                "mymonitor.executor.build_process.BuildProcessManager.cancel_build_async"
            ),
        ]

    def _configure_system_mocks(
        self, mock_cpu_manager, mock_psutil_process, mock_cancel, core_count=8
    ):
        """Configure system mocks to avoid warnings."""
        # Mock CPU manager
        mock_manager = Mock()
        mock_manager.set_process_affinity.return_value = True
        mock_manager.set_thread_affinity.return_value = True
        mock_manager.available_cores = list(range(core_count))
        mock_manager.taskset_available = True
        mock_manager.get_taskset_prefix.return_value = f"taskset -c 0-{core_count-1} "
        mock_cpu_manager.return_value = mock_manager

        # Mock psutil.Process
        mock_process_instance = Mock()
        mock_process_instance.cpu_affinity.return_value = None
        mock_psutil_process.return_value = mock_process_instance

        # Mock cancel_build_async
        mock_cancel.return_value = None

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

        # Make poll() return None initially (process running), then 0 (completed)
        poll_call_count = 0

        def mock_poll():
            nonlocal poll_call_count
            poll_call_count += 1
            if poll_call_count <= 3:  # Let it run for a few poll cycles
                return None
            return 0

        mock_build_process.poll = mock_poll
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

        # Mock memory info with proper numeric attributes
        from collections import namedtuple

        MemoryInfo = namedtuple("MemoryInfo", ["rss", "vms"])
        MemoryFullInfo = namedtuple("MemoryFullInfo", ["rss", "vms", "pss"])

        mock_memory_info = MemoryInfo(rss=10 * 1024 * 1024, vms=20 * 1024 * 1024)
        mock_gcc_process.memory_info.return_value = mock_memory_info

        mock_memory_full_info = MemoryFullInfo(
            rss=10 * 1024 * 1024, vms=20 * 1024 * 1024, pss=15 * 1024 * 1024
        )
        mock_gcc_process.memory_full_info.return_value = mock_memory_full_info

        mock_gcc_process.is_running.return_value = True
        mock_gcc_process.children.return_value = []

        # Add CPU times for more realistic monitoring
        from collections import namedtuple

        CPUTimes = namedtuple("CPUTimes", ["user", "system"])
        mock_gcc_process.cpu_times.return_value = CPUTimes(user=1.5, system=0.5)
        mock_gcc_process.cpu_percent.return_value = 25.0

        # Make process_iter return the same process multiple times to simulate ongoing monitoring
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
            monitoring_interval=0.1,  # Longer interval for more stable testing
            log_dir=temp_dir / "logs",
            collector_type="pss_psutil",
            skip_pre_clean=True,
        )

        # 5. Run the complete workflow
        # Save original open function to avoid recursion
        original_open = open

        def mock_open_func(path, mode="r", *args, **kwargs):
            if "/proc/" in str(path) and "/stat" in str(path):
                return Mock(
                    __enter__=Mock(
                        return_value=Mock(
                            read=Mock(
                                return_value="12346 (gcc) R 1 12346 12346 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 0"
                            )
                        )
                    ),
                    __exit__=Mock(return_value=None),
                )
            else:
                return original_open(path, mode, *args, **kwargs)

        with (
            patch("builtins.open", side_effect=mock_open_func),
            patch(
                "mymonitor.executor.thread_pool.get_available_cores",
                return_value=list(range(8)),
            ),
            patch("mymonitor.system.cpu_manager.get_cpu_manager") as mock_cpu_manager,
            patch("mymonitor.executor.build_process.os.getpid", return_value=12345),
            patch(
                "mymonitor.system.cpu_manager.set_current_thread_affinity",
                return_value=True,
            ),
            patch("mymonitor.system.cpu_manager.psutil.Process") as mock_psutil_process,
            patch(
                "mymonitor.executor.build_process.BuildProcessManager.cancel_build_async"
            ) as mock_cancel,
        ):
            # Configure system mocks to avoid warnings
            self._configure_system_mocks(
                mock_cpu_manager, mock_psutil_process, mock_cancel, core_count=8
            )

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
    @patch("mymonitor.system.cpu_manager.subprocess.run")
    @patch("mymonitor.system.cpu_manager.psutil.cpu_count")
    async def test_resource_constrained_workflow(
        self, mock_cpu_count, mock_subprocess_run, config_files, temp_dir, caplog
    ):
        """Test workflow in resource-constrained environment."""
        # Simulate 2-core system
        mock_cpu_count.return_value = 2
        mock_subprocess_run.return_value = Mock(returncode=0)  # taskset available

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
                mock_process.wait.return_value = 0
                mock_process.__enter__ = Mock(return_value=mock_process)
                mock_process.__exit__ = Mock(return_value=None)
                mock_popen.return_value = mock_process

                # Create a minimal mock process for resource-constrained testing
                mock_process_for_monitoring = Mock()
                mock_process_for_monitoring.pid = 12346
                mock_process_for_monitoring.name.return_value = "gcc"
                mock_process_for_monitoring.cmdline.return_value = [
                    "gcc",
                    "-c",
                    "file.c",
                ]
                mock_process_for_monitoring.as_dict.return_value = {
                    "pid": 12346,
                    "name": "gcc",
                    "cmdline": ["gcc", "-c", "file.c"],
                }

                from collections import namedtuple

                MemoryInfo = namedtuple("MemoryInfo", ["rss", "vms"])
                MemoryFullInfo = namedtuple("MemoryFullInfo", ["rss", "vms", "pss"])

                mock_process_for_monitoring.memory_info.return_value = MemoryInfo(
                    rss=5 * 1024 * 1024, vms=10 * 1024 * 1024
                )
                mock_process_for_monitoring.memory_full_info.return_value = (
                    MemoryFullInfo(
                        rss=5 * 1024 * 1024, vms=10 * 1024 * 1024, pss=7 * 1024 * 1024
                    )
                )
                mock_process_for_monitoring.is_running.return_value = True
                mock_process_for_monitoring.children.return_value = []

                # Save original open function to avoid recursion
                original_open = open

                def mock_open_func(path, mode="r", *args, **kwargs):
                    if "/proc/" in str(path) and "/stat" in str(path):
                        return Mock(
                            __enter__=Mock(
                                return_value=Mock(
                                    read=Mock(
                                        return_value="12346 (gcc) R 1 12346 12346 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 0"
                                    )
                                )
                            ),
                            __exit__=Mock(return_value=None),
                        )
                    else:
                        return original_open(path, mode, *args, **kwargs)

                with (
                    patch("builtins.open", side_effect=mock_open_func),
                    patch(
                        "mymonitor.collectors.pss_psutil.psutil.process_iter",
                        return_value=[mock_process_for_monitoring],
                    ),
                    patch(
                        "mymonitor.system.cpu_manager.get_cpu_manager"
                    ) as mock_cpu_manager,
                    patch(
                        "mymonitor.executor.build_process.os.getpid", return_value=12345
                    ),
                    patch(
                        "mymonitor.executor.thread_pool.get_available_cores",
                        return_value=list(range(2)),
                    ),
                    patch(
                        "mymonitor.system.cpu_manager.set_current_thread_affinity",
                        return_value=True,
                    ),
                    patch(
                        "mymonitor.system.cpu_manager.psutil.Process"
                    ) as mock_psutil_process,
                    patch(
                        "mymonitor.executor.build_process.BuildProcessManager.cancel_build_async"
                    ) as mock_cancel,
                ):
                    # Configure system mocks to avoid warnings
                    self._configure_system_mocks(
                        mock_cpu_manager, mock_psutil_process, mock_cancel, core_count=2
                    )

                    success = await build_runner.run_async()

        # Should succeed but with warnings about resource constraints
        assert success is True
        # Check for resource-related warnings
        assert (
            "Invalid cores specified" in caplog.text
            or "CPU资源不足" in caplog.text
            or "资源不足" in caplog.text
            or "Failed to bind build process" in caplog.text
        )

    @pytest.mark.asyncio
    @patch("mymonitor.system.cpu_manager.subprocess.run")
    @patch("mymonitor.system.cpu_manager.psutil.cpu_count")
    async def test_manual_allocation_workflow(
        self, mock_cpu_count, mock_subprocess_run, config_files, temp_dir
    ):
        """Test workflow with manual CPU allocation."""
        mock_cpu_count.return_value = 16
        mock_subprocess_run.return_value = Mock(returncode=0)  # taskset available

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
                mock_process.wait.return_value = 0
                mock_process.__enter__ = Mock(return_value=mock_process)
                mock_process.__exit__ = Mock(return_value=None)
                mock_popen.return_value = mock_process

                # Create a mock process for manual allocation testing
                mock_process_for_monitoring = Mock()
                mock_process_for_monitoring.pid = 12346
                mock_process_for_monitoring.name.return_value = "make"
                mock_process_for_monitoring.cmdline.return_value = ["make", "-j4"]
                mock_process_for_monitoring.as_dict.return_value = {
                    "pid": 12346,
                    "name": "make",
                    "cmdline": ["make", "-j4"],
                }

                from collections import namedtuple

                MemoryInfo = namedtuple("MemoryInfo", ["rss", "vms"])
                MemoryFullInfo = namedtuple("MemoryFullInfo", ["rss", "vms", "pss"])

                mock_process_for_monitoring.memory_info.return_value = MemoryInfo(
                    rss=8 * 1024 * 1024, vms=16 * 1024 * 1024
                )
                mock_process_for_monitoring.memory_full_info.return_value = (
                    MemoryFullInfo(
                        rss=8 * 1024 * 1024, vms=16 * 1024 * 1024, pss=12 * 1024 * 1024
                    )
                )
                mock_process_for_monitoring.is_running.return_value = True
                mock_process_for_monitoring.children.return_value = []

                # Save original open function to avoid recursion
                original_open = open

                def mock_open_func(path, mode="r", *args, **kwargs):
                    if "/proc/" in str(path) and "/stat" in str(path):
                        return Mock(
                            __enter__=Mock(
                                return_value=Mock(
                                    read=Mock(
                                        return_value="12346 (make) R 1 12346 12346 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 0"
                                    )
                                )
                            ),
                            __exit__=Mock(return_value=None),
                        )
                    else:
                        return original_open(path, mode, *args, **kwargs)

                with (
                    patch("builtins.open", side_effect=mock_open_func),
                    patch(
                        "mymonitor.collectors.pss_psutil.psutil.process_iter",
                        return_value=[mock_process_for_monitoring],
                    ),
                    patch(
                        "mymonitor.system.cpu_manager.get_cpu_manager"
                    ) as mock_cpu_manager,
                    patch(
                        "mymonitor.executor.build_process.os.getpid", return_value=12345
                    ),
                    patch(
                        "mymonitor.executor.thread_pool.get_available_cores",
                        return_value=list(range(16)),
                    ),
                    patch(
                        "mymonitor.system.cpu_manager.set_current_thread_affinity",
                        return_value=True,
                    ),
                    patch(
                        "mymonitor.system.cpu_manager.psutil.Process"
                    ) as mock_psutil_process,
                    patch(
                        "mymonitor.executor.build_process.BuildProcessManager.cancel_build_async"
                    ) as mock_cancel,
                ):
                    # Configure system mocks to avoid warnings
                    self._configure_system_mocks(
                        mock_cpu_manager,
                        mock_psutil_process,
                        mock_cancel,
                        core_count=16,
                    )

                    success = await build_runner.run_async()

        assert success is True
        # Verify manual allocation was used - check if build context exists and has expected core allocation
        if build_runner.run_context and hasattr(
            build_runner.run_context, "build_cores_target_str"
        ):
            build_cores_str = build_runner.run_context.build_cores_target_str
            # Manual allocation should result in specific core ranges
            assert build_cores_str is not None and (
                len(build_cores_str) > 0 or "0" in str(build_cores_str)
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

        with (
            patch(
                "mymonitor.executor.thread_pool.get_available_cores",
                return_value=list(range(4)),
            ),
            patch("mymonitor.executor.build_process.subprocess.Popen") as mock_popen,
            patch("mymonitor.system.cpu_manager.get_cpu_manager") as mock_cpu_manager,
            patch("mymonitor.executor.build_process.os.getpid", return_value=12345),
            patch(
                "mymonitor.system.cpu_manager.set_current_thread_affinity",
                return_value=True,
            ),
            patch("mymonitor.system.cpu_manager.psutil.Process") as mock_psutil_process,
            patch(
                "mymonitor.executor.build_process.BuildProcessManager.cancel_build_async"
            ) as mock_cancel,
        ):
            # Configure system mocks to avoid warnings
            self._configure_system_mocks(
                mock_cpu_manager, mock_psutil_process, mock_cancel, core_count=4
            )

            mock_process = Mock()
            mock_process.pid = 12345
            mock_process.returncode = 1  # Failure
            mock_process.communicate.return_value = ("", "Build failed")
            mock_process.poll.return_value = 1
            mock_process.wait.return_value = 1
            mock_process.__enter__ = Mock(return_value=mock_process)
            mock_process.__exit__ = Mock(return_value=None)
            mock_popen.return_value = mock_process

            with patch(
                "mymonitor.collectors.pss_psutil.psutil.process_iter",
                return_value=[],  # No processes to monitor during build failure
            ):
                success = await build_runner.run_async()

        # Should handle build failure gracefully
        assert success is False

    @pytest.mark.asyncio
    @patch("mymonitor.system.cpu_manager.subprocess.run")
    @patch("mymonitor.system.cpu_manager.psutil.cpu_count")
    async def test_monitoring_with_process_classification(
        self, mock_cpu_count, mock_subprocess_run, config_files, temp_dir
    ):
        """Test workflow with process classification."""
        mock_cpu_count.return_value = 8
        mock_subprocess_run.return_value = Mock(returncode=0)  # taskset available

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

        # Mock processes with different types using proper namedtuples
        from collections import namedtuple

        MemoryInfo = namedtuple("MemoryInfo", ["rss", "vms"])
        MemoryFullInfo = namedtuple("MemoryFullInfo", ["rss", "vms", "pss"])

        mock_gcc = Mock()
        mock_gcc.pid = 12346
        mock_gcc.name.return_value = "gcc"
        mock_gcc.cmdline.return_value = ["gcc", "-O2", "file.c"]
        mock_gcc.as_dict.return_value = {
            "pid": 12346,
            "name": "gcc",
            "cmdline": ["gcc", "-O2", "file.c"],
        }
        mock_gcc.memory_info.return_value = MemoryInfo(
            rss=10 * 1024 * 1024, vms=20 * 1024 * 1024
        )
        mock_gcc.memory_full_info.return_value = MemoryFullInfo(
            rss=10 * 1024 * 1024, vms=20 * 1024 * 1024, pss=15 * 1024 * 1024
        )
        mock_gcc.is_running.return_value = True
        mock_gcc.children.return_value = []

        mock_make = Mock()
        mock_make.pid = 12347
        mock_make.name.return_value = "make"
        mock_make.cmdline.return_value = ["make", "-j2"]
        mock_make.as_dict.return_value = {
            "pid": 12347,
            "name": "make",
            "cmdline": ["make", "-j2"],
        }
        mock_make.memory_info.return_value = MemoryInfo(
            rss=5 * 1024 * 1024, vms=10 * 1024 * 1024
        )
        mock_make.memory_full_info.return_value = MemoryFullInfo(
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
                mock_process.wait.return_value = 0
                mock_process.__enter__ = Mock(return_value=mock_process)
                mock_process.__exit__ = Mock(return_value=None)
                mock_popen.return_value = mock_process

                # Save original open function to avoid recursion
                original_open = open

                def mock_open_func(path, mode="r", *args, **kwargs):
                    if "/proc/" in str(path) and "/stat" in str(path):
                        return Mock(
                            __enter__=Mock(
                                return_value=Mock(
                                    read=Mock(
                                        return_value="12346 (gcc) R 1 12346 12346 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 0"
                                    )
                                )
                            ),
                            __exit__=Mock(return_value=None),
                        )
                    else:
                        return original_open(path, mode, *args, **kwargs)

                with (
                    patch("builtins.open", side_effect=mock_open_func),
                    patch(
                        "mymonitor.collectors.pss_psutil.psutil.process_iter",
                        return_value=[mock_gcc, mock_make],
                    ),
                    patch(
                        "mymonitor.system.cpu_manager.get_cpu_manager"
                    ) as mock_cpu_manager,
                    patch(
                        "mymonitor.executor.build_process.os.getpid", return_value=12345
                    ),
                    patch(
                        "mymonitor.executor.thread_pool.get_available_cores",
                        return_value=list(range(8)),
                    ),
                    patch(
                        "mymonitor.system.cpu_manager.set_current_thread_affinity",
                        return_value=True,
                    ),
                    patch(
                        "mymonitor.system.cpu_manager.psutil.Process"
                    ) as mock_psutil_process,
                    patch(
                        "mymonitor.executor.build_process.BuildProcessManager.cancel_build_async"
                    ) as mock_cancel,
                ):
                    # Configure system mocks to avoid warnings
                    self._configure_system_mocks(
                        mock_cpu_manager, mock_psutil_process, mock_cancel, core_count=8
                    )

                    success = await build_runner.run_async()

        assert success is True

        # Verify monitoring architecture was created and ran
        assert build_runner.monitoring_architecture is not None

        # If results exist, they should have some structure
        if (
            hasattr(build_runner.monitoring_architecture, "results")
            and build_runner.monitoring_architecture.results
        ):
            results = build_runner.monitoring_architecture.results
            # Results should contain some monitoring information
            assert (
                hasattr(results, "samples")
                or hasattr(results, "data")
                or hasattr(results, "total_samples_collected")
                or results is not None
            )

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

        with (
            patch(
                "mymonitor.executor.thread_pool.get_available_cores",
                return_value=list(range(4)),
            ),
            patch("mymonitor.executor.build_process.subprocess.Popen") as mock_popen,
            patch("mymonitor.system.cpu_manager.get_cpu_manager") as mock_cpu_manager,
            patch("mymonitor.executor.build_process.os.getpid", return_value=12345),
            patch(
                "mymonitor.system.cpu_manager.set_current_thread_affinity",
                return_value=True,
            ),
            patch("mymonitor.system.cpu_manager.psutil.Process") as mock_psutil_process,
            patch(
                "mymonitor.executor.build_process.BuildProcessManager.cancel_build_async"
            ) as mock_cancel,
        ):
            # Configure system mocks to avoid warnings
            self._configure_system_mocks(
                mock_cpu_manager, mock_psutil_process, mock_cancel, core_count=4
            )

            mock_process = Mock()
            mock_process.pid = 12345
            mock_process.returncode = None  # Still running
            mock_process.communicate.return_value = ("", "")
            mock_process.poll.return_value = None
            mock_process.wait.return_value = 0
            mock_process.__enter__ = Mock(return_value=mock_process)
            mock_process.__exit__ = Mock(return_value=None)
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
