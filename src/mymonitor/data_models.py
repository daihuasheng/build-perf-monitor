"""
Data models and structures for the monitoring application.

This module defines the dataclasses used to structure data throughout the
monitoring and analysis process. Using dataclasses ensures type safety and
provides clear, self-documenting data contracts between different parts of
the application, such as configuration loading, runtime monitoring, and
data reporting.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# --- Configuration Models ---
# These models represent the static configuration loaded from TOML files.


@dataclass
class MonitorConfig:
    """
    Configuration for the monitor's global behavior, loaded from `config.toml`.
    """

    # The time in seconds between each memory sample. Can be a float.
    interval_seconds: int
    # A list of parallelism levels (-j values) to use when no specific level is provided via CLI.
    default_jobs: List[int]
    # The type of memory collector to use (e.g., 'pss_psutil', 'rss_pidstat').
    metric_type: str
    # The CPU core to pin the main monitor script to (-1 to disable).
    monitor_core: int
    # The policy for assigning CPU cores to the build process (e.g., 'all_others', 'specific', 'none').
    build_cores_policy: str
    # A string defining specific cores for the build process (e.g., '1,2,4-7') if policy is 'specific'.
    specific_build_cores: str
    # If True, plot generation will be skipped after the monitoring run.
    skip_plots: bool
    # The root directory where all run logs and plots will be saved.
    log_root_dir: Path
    # Categorization cache size
    categorization_cache_size: int = 4096
    # Collector optimization mode, could be 'full_scan' or 'descendants_only'
    pss_collector_mode: str = "full_scan"


@dataclass
class ProjectConfig:
    """
    Configuration for a single project to be monitored, loaded from `projects.toml`.
    """

    # A unique, descriptive name for the project (e.g., "qemu", "chromium").
    name: str
    # The root directory of the project where commands will be executed.
    dir: Path
    # The command used to build the project. '<N>' is a placeholder for the parallelism level.
    build_command_template: str
    # A regex pattern used by the memory collector to identify relevant processes for this project.
    process_pattern: str
    # The command used to clean build artifacts. Can be an empty string.
    clean_command_template: str
    # An optional command to run before the build (e.g., 'source env.sh'). Can be an empty string.
    setup_command_template: str


@dataclass
class RuleConfig:
    """
    Configuration for a single process categorization rule, loaded from `rules.toml`.
    """

    # The broad category for the process (e.g., 'CPP_Compile', 'BuildSystem').
    major_category: str
    # The specific sub-category (e.g., 'GCCInternalCompiler', 'Ninja').
    category: str
    # The rule's priority (higher numbers are checked first).
    priority: int
    # The process attribute to match against (e.g., 'current_cmd_name', 'current_cmd_full').
    match_field: str
    # The type of match to perform (e.g., 'exact', 'regex', 'contains').
    match_type: str
    # The string pattern to use for matching (for most match types).
    pattern: Optional[str] = None
    # A list of strings for 'in_list' matching.
    patterns: Optional[List[str]] = None
    # An optional comment describing the rule's purpose.
    comment: Optional[str] = ""


@dataclass
class AppConfig:
    """
    The root configuration object that aggregates all loaded settings.
    """

    # The global monitor configuration.
    monitor: MonitorConfig
    # A list of all configured projects.
    projects: List[ProjectConfig]
    # A list of all categorization rules, sorted by priority.
    rules: List[RuleConfig]


# --- Runtime Data Models ---
# These models represent data created and used during a single application run.


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
    monitoring_interval: int
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


@dataclass
class MonitoringResults:
    """
    Holds all the aggregated results from a completed monitoring loop.
    """

    # A list of dictionaries, where each dict is a raw data row for the Parquet file.
    all_samples_data: List[Dict[str, Any]]
    # A dictionary holding peak memory stats for individual processes, keyed by category.
    category_stats: Dict[str, Dict[str, Any]]
    # The peak sum of memory across all monitored processes at any single interval.
    peak_overall_memory_kb: int
    # The timestamp (epoch seconds) when the overall peak memory was observed.
    peak_overall_memory_epoch: int
    # A dictionary holding the peak summed memory for each category, keyed by category.
    category_peak_sum: Dict[str, int]
    # A dictionary holding the set of unique PIDs observed for each category.
    category_pid_set: Dict[str, Set[str]]
