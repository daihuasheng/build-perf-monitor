"""
Data models and structures for the monitoring application.

This module defines the dataclasses used to structure data throughout the
monitoring and analysis process, ensuring type safety and clear data contracts
between different parts of the application.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class RunPaths:
    """Holds paths for a single monitoring run."""

    output_parquet_file: Path
    output_summary_log_file: Path
    collector_aux_log_file: Path


@dataclass
class RunContext:
    """Encapsulates all configuration and context for a single monitoring run."""

    # Project-specific details
    project_name: str
    project_dir: Path
    process_pattern: str
    actual_build_command: str

    # Run-specific parameters
    parallelism_level: int
    monitoring_interval: int
    collector_type: str
    current_timestamp_str: str

    # CPU affinity details
    taskset_available: bool
    build_cores_target_str: str
    monitor_script_pinned_to_core_info: str
    monitor_core_id: Optional[int]

    # Generated paths for the run
    paths: RunPaths


@dataclass
class MonitoringResults:
    """Holds the results from a monitoring loop."""

    all_samples_data: List[Dict[str, Any]]
    category_stats: Dict[str, Dict[str, Any]]
    peak_overall_memory_kb: int
    peak_overall_memory_epoch: int
    category_peak_sum: Dict[str, int]
    category_pid_set: Dict[str, Set[str]]