"""
Runtime data models.

This module contains data structures used during the execution of a monitoring run,
including paths, context, and CPU allocation plans.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..collectors.factory import CollectorFactory
    from ..executor.thread_pool import ThreadPoolConfig


@dataclass
class RunPaths:
    """
    A container for all generated file paths for a single monitoring run.
    """

    # Path to the main data output file in Parquet format.
    output_parquet_file: Path
    # Path to the human-readable summary log file.
    output_summary_log_file: Path
    # Path to a temporary file for auxiliary collector output (e.g., pidstat stderr).
    collector_aux_log_file: Path


@dataclass
class RunContext:
    """
    Encapsulates all configuration and context for a single, specific monitoring run
    (i.e., one project at one parallelism level).
    """

    # --- Project-specific details ---
    project_name: str
    project_dir: Path
    process_pattern: str
    actual_build_command: str

    # --- Run-specific parameters ---
    parallelism_level: int
    monitoring_interval: float
    collector_type: str
    current_timestamp_str: str

    # --- CPU affinity details ---
    taskset_available: bool
    build_cores_target_str: str
    monitor_script_pinned_to_core_info: str
    monitor_core_id: Optional[int]

    # --- Generated paths for the run ---
    paths: RunPaths

    build_process_pid: Optional[int] = None

    def create_collector_factory(self) -> "CollectorFactory":
        """
        创建配置好的 CollectorFactory 实例。

        Returns:
            配置好的 CollectorFactory 实例
        """
        from ..collectors.factory import CollectorFactory
        from ..config import get_config

        config = get_config()

        return CollectorFactory(
            metric_type=self.collector_type,
            process_pattern=self.process_pattern,
            monitoring_interval=self.monitoring_interval,
            pss_collector_mode=getattr(
                config.monitor, "pss_collector_mode", "full_scan"
            ),
            taskset_available=self.taskset_available,
            pidstat_stderr_file=self.paths.collector_aux_log_file,
        )

    def create_monitoring_thread_pool_config(self) -> "ThreadPoolConfig":
        """
        创建监控线程池配置。

        Returns:
            配置好的 ThreadPoolConfig 实例
        """
        from ..executor.thread_pool import ThreadPoolConfig
        from ..config import get_config

        config = get_config()

        return ThreadPoolConfig(
            max_workers=config.monitor.max_concurrent_monitors,
            thread_name_prefix="MonitorWorker",
            enable_cpu_affinity=config.monitor.enable_cpu_affinity,
            shutdown_timeout=config.monitor.graceful_shutdown_timeout,
        )


@dataclasses.dataclass
class CpuAllocationPlan:
    """Dataclass to hold the results of a CPU allocation planning."""

    build_cores: List[int]
    monitoring_cores: List[int]
    build_cores_desc: str
    monitoring_cores_desc: str
    taskset_prefix: str
    taskset_available: bool

    # New fields for enhanced adaptive allocation
    has_resource_warning: bool = False  # True if build cores < parallelism level
    shared_cores: bool = False  # True if monitoring shares cores with build
    allocation_strategy: str = "adaptive"  # Strategy used for allocation
    monitoring_taskset_prefix: str = ""  # Separate taskset prefix for monitoring

    # Backward compatibility properties
    @property
    def build_command_prefix(self) -> str:
        """Backward compatibility for old API."""
        return self.taskset_prefix

    @property
    def build_taskset_prefix(self) -> str:
        """Get taskset prefix for build processes."""
        return self.taskset_prefix
