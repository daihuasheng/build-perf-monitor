"""
Data models and structures for the monitoring application.

This module defines the dataclasses used to structure data throughout the
monitoring and analysis process. Using dataclasses ensures type safety and
provides clear, self-documenting data contracts between different parts of
the application, such as configuration loading, runtime monitoring, and
data reporting.
"""

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

# --- Configuration Models ---
# These models represent the static configuration loaded from TOML files.


@dataclass
class MonitorConfig:
    """
    Configuration for the monitor's global behavior, loaded from `config.toml`.
    """

    # [monitor.general]
    default_jobs: List[int]
    skip_plots: bool
    log_root_dir: Path
    categorization_cache_size: int

    # [monitor.collection]
    interval_seconds: float
    metric_type: str
    pss_collector_mode: str

    # [monitor.scheduling] - Unified Policy
    scheduling_policy: str
    monitor_core: int
    manual_build_cores: str
    manual_monitoring_cores: str


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


@dataclasses.dataclass
class CpuAllocationPlan:
    """Dataclass to hold the results of a CPU allocation planning."""
    build_command_prefix: str
    build_cores_desc: str
    monitoring_cores: List[int]
    monitoring_cores_desc: str


# --- Error Handling Models and Utilities ---
# Centralized error handling infrastructure for consistent error management.


class ErrorSeverity:
    """Define error severity levels for consistent error handling across all modules."""
    CRITICAL = "critical"    # System failure, should exit
    ERROR = "error"         # Operation failure, should log and potentially retry
    WARNING = "warning"     # Unexpected but recoverable condition
    DEBUG = "debug"         # Minor issue, for debugging purposes


def handle_error(
    error: Exception,
    context: str,
    severity: str = ErrorSeverity.ERROR,
    include_traceback: bool = True,
    reraise: bool = False,
    fallback_value: Any = None,
    logger: Optional[logging.Logger] = None
) -> Any:
    """
    Standardized error handling function for consistent error processing.
    
    Args:
        error: The caught exception
        context: Description of what operation was being performed
        severity: Error severity level (use ErrorSeverity constants)
        include_traceback: Whether to include full traceback in logs
        reraise: Whether to re-raise the exception after logging
        fallback_value: Value to return if not re-raising
        logger: Optional logger instance; if None, uses root logger
        
    Returns:
        fallback_value if reraise=False, otherwise raises the exception
        
    Raises:
        The original exception if reraise=True
    """
    # Use provided logger or fall back to root logger
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Map severity to log levels
    severity_to_level = {
        ErrorSeverity.CRITICAL: logging.CRITICAL,
        ErrorSeverity.ERROR: logging.ERROR,
        ErrorSeverity.WARNING: logging.WARNING,
        ErrorSeverity.DEBUG: logging.DEBUG,
    }
    
    log_level = severity_to_level.get(severity, logging.ERROR)
    error_type = type(error).__name__
    
    # Create comprehensive error message
    msg = f"{context}: {error_type}: {error}"
    
    # Log with appropriate level and traceback
    logger.log(log_level, msg, exc_info=include_traceback)
    
    if reraise:
        raise error
    
    return fallback_value


def handle_file_error(
    error: Exception,
    file_path: Union[str, Path],
    operation: str,
    reraise: bool = True,
    logger: Optional[logging.Logger] = None
) -> Optional[bool]:
    """
    Specialized error handler for file operations.
    
    Args:
        error: The file-related exception
        file_path: Path to the file being operated on
        operation: Description of the file operation (e.g., "reading", "writing")
        reraise: Whether to re-raise the exception
        logger: Optional logger instance
        
    Returns:
        None if reraise=True, False if reraise=False
    """
    context = f"File {operation} operation on '{file_path}'"
    
    # Classify common file errors
    if isinstance(error, FileNotFoundError):
        severity = ErrorSeverity.ERROR
    elif isinstance(error, PermissionError):
        severity = ErrorSeverity.ERROR
    elif isinstance(error, OSError):
        severity = ErrorSeverity.ERROR
    else:
        severity = ErrorSeverity.WARNING
    
    return handle_error(
        error=error,
        context=context,
        severity=severity,
        include_traceback=True,
        reraise=reraise,
        fallback_value=False,
        logger=logger
    )


def handle_subprocess_error(
    error: Exception,
    command: str,
    reraise: bool = False,
    logger: Optional[logging.Logger] = None
) -> Optional[bool]:
    """
    Specialized error handler for subprocess operations.
    
    Args:
        error: The subprocess-related exception
        command: The command that was being executed
        reraise: Whether to re-raise the exception
        logger: Optional logger instance
        
    Returns:
        None if reraise=True, False if reraise=False
    """
    import subprocess
    
    context = f"Subprocess execution of command '{command[:100]}...'"
    
    # Classify subprocess errors
    if isinstance(error, subprocess.TimeoutExpired):
        severity = ErrorSeverity.WARNING
    elif isinstance(error, subprocess.CalledProcessError):
        severity = ErrorSeverity.ERROR
    elif isinstance(error, FileNotFoundError):
        severity = ErrorSeverity.ERROR
    else:
        severity = ErrorSeverity.WARNING
    
    return handle_error(
        error=error,
        context=context,
        severity=severity,
        include_traceback=True,
        reraise=reraise,
        fallback_value=False,
        logger=logger
    )


def handle_config_error(
    error: Exception,
    context: str,
    severity: str = ErrorSeverity.ERROR,
    reraise: bool = True,
    logger: Optional[logging.Logger] = None
) -> Union[None, Any]:
    """
    Specialized error handler for configuration operations.
    
    Args:
        error: The caught exception
        context: Description of what configuration operation was being performed
        severity: Error severity level
        reraise: Whether to re-raise the exception after logging
        logger: Optional logger instance
        
    Returns:
        None if reraise=False, otherwise raises the exception
    """
    config_context = f"Configuration {context}"
    
    return handle_error(
        error=error,
        context=config_context,
        severity=severity,
        include_traceback=True,
        reraise=reraise,
        fallback_value=None,
        logger=logger
    )


def handle_cli_error(
    error: Exception,
    context: str,
    exit_code: int = 1,
    include_traceback: bool = True,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Specialized error handler for CLI operations that exits the program.
    
    Args:
        error: The caught exception
        context: Description of what operation was being performed
        exit_code: Exit code for sys.exit()
        include_traceback: Whether to include full traceback in logs
        logger: Optional logger instance
    """
    import sys
    
    cli_context = f"CLI {context}"
    
    handle_error(
        error=error,
        context=cli_context,
        severity=ErrorSeverity.CRITICAL,
        include_traceback=include_traceback,
        reraise=False,
        logger=logger
    )
    
    sys.exit(exit_code)
