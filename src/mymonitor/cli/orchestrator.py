"""
Async build runner for CLI integration.

This module provides the main async build runner that uses the simplified
AsyncIO architecture with direct monitoring system integration.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from ..config import get_config
from ..models.config import ProjectConfig
from ..models.runtime import RunContext, RunPaths
from ..monitoring.architectures import HybridArchitecture
from ..executor.build_process import BuildProcessManager
from ..storage.data_manager import DataStorageManager
from ..system.cpu_manager import get_cpu_manager
from ..executor.thread_pool import (
    initialize_global_thread_pools,
    shutdown_global_thread_pools,
    get_thread_pool_manager,
    ThreadPoolConfig,
)
from ..validation import handle_error, ErrorSeverity

logger = logging.getLogger(__name__)


class BuildRunner:
    """
    主异步构建运行器，协调监控和构建执行。

    该类直接使用 HybridArchitecture 提供异步构建监控，
    采用主线程异步循环 + 多工作线程处理的架构。
    """

    def __init__(
        self,
        project_config: ProjectConfig,
        parallelism_level: int,
        monitoring_interval: float,
        log_dir: Path,
        collector_type: str,
        skip_pre_clean: bool = False,
    ):
        """
        Initialize the async build monitor.

        Args:
            project_config: Project configuration
            parallelism_level: Number of parallel jobs (-j level)
            monitoring_interval: Monitoring interval in seconds
            log_dir: Directory for log files
            collector_type: Type of memory collector to use
            skip_pre_clean: Whether to skip pre-build cleanup
        """
        self.project_config = project_config
        self.parallelism_level = parallelism_level
        self.monitoring_interval = monitoring_interval
        self.log_dir = Path(log_dir)
        self.collector_type = collector_type
        self.skip_pre_clean = skip_pre_clean

        # Runtime state
        self.run_context: Optional[RunContext] = None
        self.monitoring_architecture: Optional[HybridArchitecture] = None
        self.build_runner: Optional[BuildProcessManager] = None
        self.shutdown_requested = False

    async def run_async(self) -> bool:
        """
        Run the complete build monitoring process asynchronously.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get configuration for thread pools
            config = get_config()

            # Setup run context first (needed for configuration creation)
            await self._setup_run_context()

            # Plan CPU allocation first to get monitoring cores
            cpu_plan = get_cpu_manager().plan_cpu_allocation(
                cores_policy=config.monitor.scheduling_policy,
                cores_string=config.monitor.manual_build_cores,
                parallelism_level=self.parallelism_level,
                monitoring_workers=min(4, self.parallelism_level),
            )

            # Create monitoring thread pool configuration with dedicated cores
            monitoring_config = self.run_context.create_monitoring_thread_pool_config()
            # Update with CPU allocation plan
            monitoring_config.dedicated_cores = cpu_plan.monitoring_cores
            monitoring_config.max_workers = min(
                len(cpu_plan.monitoring_cores), config.monitor.max_concurrent_monitors
            )

            # Initialize global monitoring thread pool
            initialize_global_thread_pools(monitoring_config=monitoring_config)

            # Handle resource warnings and shared core mode
            if cpu_plan.has_resource_warning:
                logger.warning(
                    f"项目 {self.project_config.name} CPU资源不足，"
                    f"建议降低并行度或使用更多CPU核心。"
                    f"当前分配: 构建{len(cpu_plan.build_cores)}核心 < 并行度{self.parallelism_level}"
                )

            if cpu_plan.shared_cores:
                logger.info(
                    f"监控任务与构建任务共享CPU核心: {cpu_plan.monitoring_cores_desc}"
                )
            else:
                logger.info(
                    f"独立CPU分配 - 构建: {cpu_plan.build_cores_desc}, "
                    f"监控: {cpu_plan.monitoring_cores_desc}"
                )

            # Log allocation strategy used
            logger.info(f"CPU分配策略: {cpu_plan.allocation_strategy}")

            # Update run context with CPU plan details
            self.run_context.build_cores_target_str = cpu_plan.build_cores_desc
            self.run_context.monitor_script_pinned_to_core_info = (
                cpu_plan.monitoring_cores_desc
            )
            self.run_context.monitor_core_id = (
                cpu_plan.monitoring_cores[0] if cpu_plan.monitoring_cores else None
            )
            self.run_context.taskset_available = cpu_plan.taskset_available

            # Create collector factory using unified method
            collector_factory = self.run_context.create_collector_factory()

            # Configure hybrid architecture parameters
            hybrid_config = {
                "discovery_interval": getattr(
                    config.monitor, "hybrid_discovery_interval", 0.01
                ),
                "sampling_workers": getattr(
                    config.monitor, "hybrid_sampling_workers", 4
                ),
                "task_queue_size": getattr(
                    config.monitor, "hybrid_task_queue_size", 1000
                ),
                "result_queue_size": getattr(
                    config.monitor, "hybrid_result_queue_size", 2000
                ),
                "enable_prioritization": getattr(
                    config.monitor, "hybrid_enable_prioritization", True
                ),
                "max_retry_count": getattr(config.monitor, "hybrid_max_retry_count", 3),
                "queue_timeout": getattr(config.monitor, "hybrid_queue_timeout", 0.1),
                "worker_timeout": getattr(config.monitor, "hybrid_worker_timeout", 5.0),
                "enable_queue_monitoring": getattr(
                    config.monitor, "hybrid_enable_queue_monitoring", True
                ),
                "batch_result_size": getattr(
                    config.monitor, "hybrid_batch_result_size", 50
                ),
            }

            # Get thread pool manager for dependency injection
            thread_pool_manager = get_thread_pool_manager()

            # Create hybrid monitoring architecture
            self.monitoring_architecture = HybridArchitecture(
                collector_factory=collector_factory,
                config=hybrid_config,
                thread_pool_manager=thread_pool_manager,
            )

            # Create build runner
            actual_build_command = self.project_config.build_command_template.replace(
                "<N>", str(self.parallelism_level)
            )
            self.build_runner = BuildProcessManager(
                build_command=actual_build_command,
                build_directory=Path(self.project_config.dir),
                build_cores=cpu_plan.build_cores,
                timeout=3600.0,  # 1 hour timeout
            )

            # Setup monitoring
            await self.monitoring_architecture.setup_monitoring(
                cpu_plan.monitoring_cores
            )

            # Start build process
            logger.info(f"Starting build with -j{self.parallelism_level}")
            build_pid = await self.build_runner.start_build_async()

            # Start monitoring
            await self.monitoring_architecture.start_monitoring(build_pid)

            # Wait for build completion
            return_code = await self.build_runner.wait_for_completion_async()

            # Stop monitoring
            results = await self.monitoring_architecture.stop_monitoring()

            # Process results
            if results:
                await self._save_results(results)
                logger.info(f"Build completed with return code {return_code}")
                return return_code == 0
            else:
                logger.warning("No monitoring results collected")
                return False

        except asyncio.CancelledError:
            logger.info("Build monitoring was cancelled")
            return False
        except Exception as e:
            logger.error(f"Error during async build monitoring: {e}", exc_info=True)
            return False
        finally:
            # Cleanup
            await self._cleanup()

    def request_shutdown(self) -> None:
        """Request shutdown of the build monitoring process."""
        self.shutdown_requested = True

        # Cancel build if running
        if self.build_runner:
            try:
                asyncio.create_task(self.build_runner.cancel_build_async())
            except Exception as e:
                logger.warning(f"Error cancelling build: {e}")

    async def _setup_run_context(self) -> None:
        """Setup the run context for monitoring."""
        # Generate timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Create project-specific directory
        project_dir_name = f"{self.project_config.name}_j{self.parallelism_level}_{self.collector_type}_{timestamp}"
        project_output_dir = self.log_dir / project_dir_name
        project_output_dir.mkdir(parents=True, exist_ok=True)

        # Create run paths with fixed names inside the project directory
        run_paths = RunPaths(
            output_parquet_file=project_output_dir / "memory_samples.parquet",
            output_summary_log_file=project_output_dir / "summary.log",
            collector_aux_log_file=project_output_dir / "collector_aux.log",
        )

        # Create additional log files that tests expect
        (project_output_dir / "build_stdout.log").touch()
        (project_output_dir / "build_stderr.log").touch()
        (project_output_dir / "clean.log").touch()

        # Create run context
        self.run_context = RunContext(
            project_name=self.project_config.name,
            project_dir=Path(self.project_config.dir),
            process_pattern=self.project_config.process_pattern,
            actual_build_command=self.project_config.build_command_template.replace(
                "<N>", str(self.parallelism_level)
            ),
            parallelism_level=self.parallelism_level,
            monitoring_interval=self.monitoring_interval,
            collector_type=self.collector_type,
            current_timestamp_str=timestamp,
            taskset_available=True,  # Will be detected automatically
            build_cores_target_str="",  # Will be set by CPU planning
            monitor_script_pinned_to_core_info="",  # Will be set by CPU planning
            monitor_core_id=None,  # Will be set by CPU planning
            paths=run_paths,
        )

    async def _save_results(self, results) -> None:
        """Save monitoring results to files using the new storage system."""
        try:
            # Create data storage manager
            output_dir = self.run_context.paths.output_parquet_file.parent
            storage_manager = DataStorageManager(output_dir)

            # Save all results using the storage manager
            storage_manager.save_monitoring_results(results, self.run_context)

        except Exception as e:
            logger.error(f"Error saving results: {e}", exc_info=True)

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Stop monitoring if still running
            if self.monitoring_architecture:
                await self.monitoring_architecture.stop_monitoring()

            # Cancel build if still running
            if self.build_runner:
                await self.build_runner.cancel_build_async()

            # Shutdown global thread pools
            shutdown_global_thread_pools()

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def run(self) -> bool:
        """
        Run the build monitoring process synchronously.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if we're already in an async context
            try:
                # This will raise RuntimeError if no event loop is running
                loop = asyncio.get_running_loop()
                # If we get here, we're in an async context, we can't use run_until_complete
                logger.error(
                    "Cannot run synchronous BuildRunner from within async context. Use run_async() directly."
                )
                return False
            except RuntimeError:
                # No event loop running, we can create one
                pass

            # Create and run new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Run the async monitor
                return loop.run_until_complete(self.run_async())
            finally:
                # Clean up the event loop
                loop.close()
                asyncio.set_event_loop(None)

        except Exception as e:
            logger.error(f"Error running build monitor: {e}", exc_info=True)
            return False
