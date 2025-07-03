"""
Validation and error handling for the mymonitor package.

This module provides input validation, error handling, and exception management
with consistent error reporting across the application.
"""

# Exception classes
from .exceptions import (
    ErrorSeverity,
    ValidationError,
    ValidationSeverity,
    handle_cli_error,
    handle_config_error,
    handle_error,
    handle_file_error,
    handle_subprocess_error,
    validate_and_handle_error,
)

# Validation functions
from .validators import (
    validate_command_template,
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
    # Exception classes
    "ErrorSeverity",
    "ValidationError", 
    "ValidationSeverity",
    # Error handlers
    "handle_cli_error",
    "handle_config_error",
    "handle_error",
    "handle_file_error", 
    "handle_subprocess_error",
    "validate_and_handle_error",
    # Validators
    "validate_command_template",
    "validate_cpu_core_range",
    "validate_enum_choice", 
    "validate_jobs_list",
    "validate_path_exists",
    "validate_positive_float",
    "validate_positive_integer",
    "validate_project_name",
    "validate_regex_pattern",
]
