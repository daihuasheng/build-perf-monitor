"""
Runtime data models.

This module contains data structures used during the execution of a monitoring run,
including paths, context, and CPU allocation plans.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


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


@dataclasses.dataclass
class CpuAllocationPlan:
    """Dataclass to hold the results of a CPU allocation planning."""
    build_cores: List[int]
    monitoring_cores: List[int]
    build_cores_desc: str
    monitoring_cores_desc: str
    taskset_prefix: str
    taskset_available: bool
    
    # Backward compatibility properties
    @property
    def build_command_prefix(self) -> str:
        """Backward compatibility for old API."""
        return self.taskset_prefix
