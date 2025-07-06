"""
Validation and error handling for the mymonitor package.

This module provides simplified input validation and error handling
with consistent error reporting across the application.
"""

# Core exception classes and error handling
from .exceptions import (
    ErrorSeverity,
    ValidationError,
    ValidationSeverity,  # Legacy compatibility
    handle_error,
    handle_config_error,
    handle_file_error,
    handle_subprocess_error,
    handle_cli_error,
    validate_and_handle_error,
    validate_with_handler,
)

# Simplified error handling strategies
from .strategies import (
    simple_retry,
    with_simple_retry,
    SimpleRetryStrategy,
    get_error_recovery_strategy,
    with_error_recovery,
    create_file_operation_strategy,
    create_monitoring_operation_strategy,
    create_network_operation_strategy,
    create_process_operation_strategy,
    register_error_recovery_strategy,
)

# Validation functions
from .validators import (
    validate_command_template,
    validate_simple_command,
    validate_cpu_core_range,
    validate_enum_choice,
    validate_jobs_list,
    validate_path_exists,
    validate_positive_float,
    validate_positive_integer,
    validate_project_name,
    validate_regex_pattern,
)

__all__ = [
    # Core functionality
    "ErrorSeverity",
    "ValidationError", 
    "ValidationSeverity",
    "handle_error",
    "handle_config_error",
    "handle_file_error", 
    "handle_subprocess_error",
    "handle_cli_error",
    "validate_and_handle_error",
    "validate_with_handler",
    # Simplified strategies
    "simple_retry",
    "with_simple_retry",
    "SimpleRetryStrategy",
    "get_error_recovery_strategy",
    "with_error_recovery",
    "create_file_operation_strategy",
    "create_monitoring_operation_strategy",
    "create_network_operation_strategy",
    "create_process_operation_strategy",
    "register_error_recovery_strategy",
    # Validators
    "validate_command_template",
    "validate_simple_command",
    "validate_cpu_core_range",
    "validate_enum_choice", 
    "validate_jobs_list",
    "validate_path_exists",
    "validate_positive_float",
    "validate_positive_integer",
    "validate_project_name",
    "validate_regex_pattern",
]
