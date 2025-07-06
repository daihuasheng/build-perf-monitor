"""
Simplified exception handling and error management.

This module provides essential error handling functionality without over-engineering,
focusing on the patterns that are actually used throughout the application.
"""

import logging
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Severity levels for error handling."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationError(Exception):
    """
    Exception raised when validation fails.
    
    This is the main exception type used throughout the validation system.
    """
    
    def __init__(self, message: str, field_name: Optional[str] = None, 
                 value: Any = None, severity: ErrorSeverity = ErrorSeverity.ERROR):
        super().__init__(message)
        self.field_name = field_name
        self.value = value
        self.severity = severity


def handle_error(
    error: Exception,
    context: str,
    severity: Union[ErrorSeverity, str] = ErrorSeverity.ERROR,
    reraise: bool = True,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Handle errors with consistent logging and optional re-raising.
    
    Args:
        error: The exception that occurred
        context: Context description of where the error occurred
        severity: Severity level for logging
        reraise: Whether to re-raise the exception after logging
        logger: Logger instance to use (defaults to module logger)
    """
    effective_logger = logger or globals()['logger']
    
    error_msg = f"Error in {context}: {error}"
    
    # Handle both enum and string severity values
    if isinstance(severity, str):
        severity_str = severity.lower()
    else:
        severity_str = severity.value
    
    # Log based on severity
    if severity_str == "debug":
        effective_logger.debug(error_msg, exc_info=True)
    elif severity_str == "info":
        effective_logger.info(error_msg)
    elif severity_str == "warning":
        effective_logger.warning(error_msg)
    elif severity_str == "error":
        effective_logger.error(error_msg)
    elif severity_str == "critical":
        effective_logger.critical(error_msg, exc_info=True)
    
    if reraise:
        raise error


def validate_with_handler(
    validation_func: Callable[[Any], T],
    value: Any,
    field_name: str,
    context: str,
    logger: Optional[logging.Logger] = None
) -> T:
    """
    Validate a value with automatic error handling.
    
    Args:
        validation_func: Function to call for validation
        value: Value to validate
        field_name: Name of the field being validated
        context: Context description for error messages
        logger: Logger instance to use
        
    Returns:
        Result from validation_func if successful
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        return validation_func(value)
    except ValidationError:
        # Re-raise ValidationError as-is
        raise
    except Exception as e:
        # Convert other exceptions to ValidationError
        error_msg = f"Validation failed for {field_name} in {context}: {e}"
        if logger:
            logger.error(error_msg)
        raise ValidationError(error_msg, field_name=field_name, value=value)


# Convenience aliases for specific error types
def handle_config_error(error: Exception, context: str, **kwargs) -> None:
    """Handle configuration-related errors."""
    handle_error(error, f"config {context}", **kwargs)


def handle_file_error(error: Exception, context: str, **kwargs) -> None:
    """Handle file-related errors."""
    handle_error(error, f"file {context}", **kwargs)


def handle_subprocess_error(error: Exception, command: str, **kwargs) -> None:
    """Handle subprocess-related errors."""
    handle_error(error, f"subprocess command '{command}'", **kwargs)


def handle_cli_error(error: Exception, context: str, **kwargs) -> None:
    """Handle CLI-related errors."""
    # Extract exit_code and include_traceback from kwargs
    exit_code = kwargs.pop('exit_code', 1)
    include_traceback = kwargs.pop('include_traceback', False)
    
    # Handle the error with logging
    severity = kwargs.get('severity', ErrorSeverity.ERROR)
    handle_error(error, f"CLI {context}", severity=severity, reraise=False, **kwargs)
    
    # Exit with the specified code
    import sys
    sys.exit(exit_code)


# Aliases for compatibility
validate_and_handle_error = validate_with_handler

# Legacy compatibility class
class ValidationSeverity:
    """Legacy compatibility class."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
