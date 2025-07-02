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
import os
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
    Configuration for a categorization rule, loaded from `rules.toml`.
    """

    # Priority level for the rule (higher numbers processed first).
    priority: int
    # The major category this rule assigns.
    major_category: str
    # The minor category this rule assigns.
    category: str
    # The field to match against ('current_cmd_name' or 'current_full_cmd').
    match_field: str
    # The type of match to perform ('exact', 'in_list', 'regex', 'contains').
    match_type: str
    # For backward compatibility, support both pattern (string) and patterns (Union)
    # patterns can be either a string (for exact/regex/contains) or list (for in_list).
    patterns: Union[str, List[str]] = ""
    # Legacy field for single pattern (maintained for compatibility)
    pattern: Optional[str] = None
    # Optional comment describing the rule.
    comment: str = ""
    
    def __post_init__(self):
        """Post-initialization processing to handle pattern/patterns compatibility."""
        # If both pattern and patterns are provided, prefer patterns
        if self.pattern is not None and self.patterns == "":
            self.patterns = self.pattern
        # If patterns is empty and pattern is None, this will be caught by validation
        # Ensure pattern field stays in sync for backward compatibility
        if isinstance(self.patterns, str) and self.pattern is None:
            self.pattern = self.patterns


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


# --- Input Validation Infrastructure ---
# Centralized validation utilities for consistent input validation across all modules.


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class ValidationSeverity:
    """Define validation severity levels for different types of validation failures."""
    ERROR = "error"        # Invalid input that should cause failure
    WARNING = "warning"    # Questionable input that should log a warning
    INFO = "info"         # Informative validation message


def validate_positive_integer(
    value: Union[int, str],
    min_value: int = 1,
    max_value: Optional[int] = None,
    field_name: str = "value"
) -> int:
    """
    Validate that a value is a positive integer within specified bounds.
    
    Args:
        value: The value to validate (int or string representation)
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive), None for no limit
        field_name: Name of the field being validated (for error messages)
        
    Returns:
        The validated integer value
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        if isinstance(value, str):
            int_value = int(value.strip())
        else:
            int_value = int(value)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"{field_name} must be a valid integer, got '{value}': {e}")
    
    if int_value < min_value:
        raise ValidationError(f"{field_name} must be >= {min_value}, got {int_value}")
    
    if max_value is not None and int_value > max_value:
        raise ValidationError(f"{field_name} must be <= {max_value}, got {int_value}")
    
    return int_value


def validate_positive_float(
    value: Union[float, str],
    min_value: float = 0.0,
    max_value: Optional[float] = None,
    field_name: str = "value"
) -> float:
    """
    Validate that a value is a positive float within specified bounds.
    
    Args:
        value: The value to validate (float or string representation)
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive), None for no limit
        field_name: Name of the field being validated (for error messages)
        
    Returns:
        The validated float value
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        if isinstance(value, str):
            float_value = float(value.strip())
        else:
            float_value = float(value)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"{field_name} must be a valid number, got '{value}': {e}")
    
    if float_value < min_value:
        raise ValidationError(f"{field_name} must be >= {min_value}, got {float_value}")
    
    if max_value is not None and float_value > max_value:
        raise ValidationError(f"{field_name} must be <= {max_value}, got {float_value}")
    
    return float_value


def validate_path_exists(
    path: Union[str, Path],
    must_be_dir: bool = False,
    must_be_file: bool = False,
    check_readable: bool = True,
    check_writable: bool = False,
    field_name: str = "path"
) -> Path:
    """
    Validate that a path exists and has required properties.
    
    Args:
        path: The path to validate
        must_be_dir: If True, path must be a directory
        must_be_file: If True, path must be a file
        check_readable: If True, check if path is readable
        check_writable: If True, check if path is writable
        field_name: Name of the field being validated (for error messages)
        
    Returns:
        The validated Path object
        
    Raises:
        ValidationError: If validation fails
    """
    if isinstance(path, str):
        path_obj = Path(path)
    else:
        path_obj = Path(path)
    
    # Allow skipping path existence checks for testing
    skip_path_validation = os.environ.get("MYMONITOR_SKIP_PATH_VALIDATION", "").lower() in ("1", "true", "yes")
    
    if not skip_path_validation and not path_obj.exists():
        raise ValidationError(f"{field_name} does not exist: {path_obj}")
    
    if not skip_path_validation and must_be_dir and not path_obj.is_dir():
        raise ValidationError(f"{field_name} must be a directory: {path_obj}")
    
    if not skip_path_validation and must_be_file and not path_obj.is_file():
        raise ValidationError(f"{field_name} must be a file: {path_obj}")
    
    if not skip_path_validation and check_readable and not os.access(path_obj, os.R_OK):
        raise ValidationError(f"{field_name} is not readable: {path_obj}")
    
    if not skip_path_validation and check_writable and not os.access(path_obj, os.W_OK):
        raise ValidationError(f"{field_name} is not writable: {path_obj}")
    
    return path_obj


def validate_enum_choice(
    value: str,
    valid_choices: List[str],
    field_name: str = "value",
    case_sensitive: bool = True
) -> str:
    """
    Validate that a value is one of the allowed choices.
    
    Args:
        value: The value to validate
        valid_choices: List of valid choices
        field_name: Name of the field being validated (for error messages)
        case_sensitive: Whether the comparison should be case sensitive
        
    Returns:
        The validated value (possibly normalized for case)
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string, got {type(value).__name__}")
    
    if case_sensitive:
        if value not in valid_choices:
            raise ValidationError(f"{field_name} must be one of {valid_choices}, got '{value}'")
        return value
    else:
        value_lower = value.lower()
        choices_lower = [choice.lower() for choice in valid_choices]
        if value_lower not in choices_lower:
            raise ValidationError(f"{field_name} must be one of {valid_choices}, got '{value}'")
        # Return the original case from valid_choices
        return valid_choices[choices_lower.index(value_lower)]


def validate_regex_pattern(
    pattern: str,
    field_name: str = "pattern"
) -> str:
    """
    Validate that a string is a valid regular expression pattern.
    
    Args:
        pattern: The regex pattern to validate
        field_name: Name of the field being validated (for error messages)
        
    Returns:
        The validated pattern
        
    Raises:
        ValidationError: If the pattern is invalid
    """
    import re
    
    if not isinstance(pattern, str):
        raise ValidationError(f"{field_name} must be a string, got {type(pattern).__name__}")
    
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValidationError(f"{field_name} is not a valid regex pattern: {e}")
    
    return pattern


def validate_cpu_core_range(
    core_str: str,
    max_cores: Optional[int] = None,
    field_name: str = "CPU core range"
) -> str:
    """
    Validate CPU core range string format (e.g., "1,2,4-7").
    
    Args:
        core_str: The core range string to validate
        max_cores: Maximum number of cores available, None to skip check
        field_name: Name of the field being validated (for error messages)
        
    Returns:
        The validated core range string
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(core_str, str):
        raise ValidationError(f"{field_name} must be a string, got {type(core_str).__name__}")
    
    if not core_str.strip():
        return core_str  # Empty string is valid (means no specific cores)
    
    cores = set()
    try:
        for part in core_str.split(","):
            part = part.strip()
            if "-" in part:
                start_str, end_str = part.split("-")
                start = int(start_str.strip())
                end = int(end_str.strip())
                if start < 0 or end < 0:
                    raise ValueError("Core numbers must be non-negative")
                if start > end:
                    raise ValueError(f"Invalid range: {start}-{end}")
                cores.update(range(start, end + 1))
            else:
                core = int(part)
                if core < 0:
                    raise ValueError("Core numbers must be non-negative")
                cores.add(core)
    except ValueError as e:
        raise ValidationError(f"Invalid {field_name} format '{core_str}': {e}")
    
    if max_cores is not None:
        invalid_cores = [c for c in cores if c >= max_cores]
        if invalid_cores:
            raise ValidationError(
                f"{field_name} contains invalid core numbers {invalid_cores}, "
                f"system has {max_cores} cores (0-{max_cores-1})"
            )
    
    return core_str


def validate_jobs_list(
    jobs_str: str,
    min_jobs: int = 1,
    max_jobs: int = 1024,
    field_name: str = "--jobs argument"
) -> List[int]:
    """
    Validate and parse a comma-separated list of job numbers.
    
    Args:
        jobs_str: Comma-separated job numbers (e.g., "1,4,8,16")
        min_jobs: Minimum allowed job count
        max_jobs: Maximum allowed job count
        field_name: Name of the field being validated (for error messages)
        
    Returns:
        List of validated job numbers
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(jobs_str, str):
        raise ValidationError(f"{field_name} must be a string, got {type(jobs_str).__name__}")
    
    jobs = []
    try:
        for job_str in jobs_str.split(","):
            job_str = job_str.strip()
            if not job_str:
                continue
            job = validate_positive_integer(
                job_str, 
                min_value=min_jobs, 
                max_value=max_jobs, 
                field_name=f"{field_name} item"
            )
            jobs.append(job)
    except ValidationError:
        raise  # Re-raise validation errors as-is
    except Exception as e:
        raise ValidationError(f"Failed to parse {field_name} '{jobs_str}': {e}")
    
    if not jobs:
        raise ValidationError(f"{field_name} cannot be empty")
    
    # Remove duplicates while preserving order
    unique_jobs = []
    seen = set()
    for job in jobs:
        if job not in seen:
            unique_jobs.append(job)
            seen.add(job)
    
    return unique_jobs


def validate_command_template(
    template: str,
    required_placeholders: Optional[List[str]] = None,
    field_name: str = "command template"
) -> str:
    """
    Validate a command template string.
    
    Args:
        template: The command template to validate
        required_placeholders: List of required placeholder names (supports both <name> and {name} formats)
        field_name: Name of the field being validated (for error messages)
        
    Returns:
        The validated template
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(template, str):
        raise ValidationError(f"{field_name} must be a string, got {type(template).__name__}")
    
    if not template.strip():
        raise ValidationError(f"{field_name} cannot be empty")
    
    if required_placeholders:
        for placeholder in required_placeholders:
            # Support both <N> and {j_level} formats for backward compatibility
            legacy_pattern = f"<{placeholder}>"
            new_pattern = f"{{{placeholder}}}"
            
            # Check if either format is present
            if legacy_pattern not in template and new_pattern not in template:
                raise ValidationError(
                    f"{field_name} must contain placeholder '{legacy_pattern}' or '{new_pattern}'"
                )
    
    # Basic shell command validation - check for dangerous patterns
    dangerous_patterns = [
        r';\s*rm\s+-rf\s+/',  # Dangerous rm commands
        r':\(\)\{\s*:\|\:&\s*\}',  # Fork bombs
        r'>\s*/dev/sd[a-z]',  # Writing to raw devices
    ]
    
    import re
    for pattern in dangerous_patterns:
        if re.search(pattern, template, re.IGNORECASE):
            raise ValidationError(f"{field_name} contains potentially dangerous pattern")
    
    return template.strip()


def validate_project_name(
    name: str,
    existing_names: Optional[List[str]] = None,
    field_name: str = "project name"
) -> str:
    """
    Validate a project name.
    
    Args:
        name: The project name to validate
        existing_names: List of existing project names to check uniqueness
        field_name: Name of the field being validated (for error messages)
        
    Returns:
        The validated project name
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(name, str):
        raise ValidationError(f"{field_name} must be a string, got {type(name).__name__}")
    
    name = name.strip()
    if not name:
        raise ValidationError(f"{field_name} cannot be empty")
    
    # Check for valid characters (alphanumeric, hyphen, underscore)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValidationError(
            f"{field_name} can only contain letters, numbers, hyphens, and underscores, got '{name}'"
        )
    
    # Check uniqueness if existing names provided
    if existing_names and name in existing_names:
        raise ValidationError(f"{field_name} '{name}' already exists")
    
    return name


def validate_and_handle_error(
    validation_func,
    *args,
    severity: str = ValidationSeverity.ERROR,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> Any:
    """
    Execute a validation function and handle errors according to severity.
    
    Args:
        validation_func: The validation function to call
        *args: Arguments to pass to the validation function
        severity: How to handle validation failures
        logger: Optional logger for warning/info messages
        **kwargs: Keyword arguments to pass to the validation function
        
    Returns:
        The result of the validation function
        
    Raises:
        ValidationError: If severity is ERROR and validation fails
    """
    try:
        return validation_func(*args, **kwargs)
    except ValidationError as e:
        if severity == ValidationSeverity.ERROR:
            raise e
        elif severity == ValidationSeverity.WARNING:
            if logger:
                logger.warning(f"Validation warning: {e}")
            return None
        elif severity == ValidationSeverity.INFO:
            if logger:
                logger.info(f"Validation info: {e}")
            return None
        else:
            raise e  # Unknown severity, treat as error
