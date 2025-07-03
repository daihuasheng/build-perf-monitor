"""
Exception handling and error management.

This module provides centralized error handling infrastructure with
consistent error management across all modules.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any, Optional, Union


class ErrorSeverity:
    """Define error severity levels for consistent error handling across all modules."""
    CRITICAL = "critical"    # System failure, should exit
    ERROR = "error"         # Operation failure, should log and potentially retry
    WARNING = "warning"     # Unexpected but recoverable condition
    DEBUG = "debug"         # Minor issue, for debugging purposes


class ValidationSeverity:
    """Define validation severity levels for different types of validation failures."""
    ERROR = "error"        # Invalid input that should cause failure
    WARNING = "warning"    # Questionable input that should log a warning
    INFO = "info"         # Informative validation message


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


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
    
    # Build error message
    error_msg = f"{context}: {error_type}: {error}"
    
    # Log with appropriate level
    if include_traceback and log_level >= logging.ERROR:
        logger.log(log_level, error_msg, exc_info=True)
    else:
        logger.log(log_level, error_msg)
    
    # Re-raise or return fallback
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
    Specialized error handler for configuration-related operations.
    
    Args:
        error: The configuration-related exception
        context: Description of the configuration operation
        severity: Error severity level
        reraise: Whether to re-raise the exception
        logger: Optional logger instance
        
    Returns:
        None if reraise=True, appropriate fallback if reraise=False
    """
    return handle_error(
        error=error,
        context=f"Configuration {context}",
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
    Specialized error handler for CLI operations that should exit the program.
    
    Args:
        error: The CLI-related exception
        context: Description of the CLI operation
        exit_code: Exit code for the program
        include_traceback: Whether to include traceback in error output
        logger: Optional logger instance
    """
    import sys
    
    handle_error(
        error=error,
        context=f"CLI {context}",
        severity=ErrorSeverity.CRITICAL,
        include_traceback=include_traceback,
        reraise=False,
        logger=logger
    )
    
    sys.exit(exit_code)


def validate_and_handle_error(
    validation_func,
    *args,
    severity: str = ValidationSeverity.ERROR,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> Any:
    """
    Wrapper to handle validation errors with consistent error processing.
    
    Args:
        validation_func: The validation function to call
        *args: Positional arguments for the validation function
        severity: Validation severity level
        logger: Optional logger instance
        **kwargs: Keyword arguments for the validation function
        
    Returns:
        Result from validation_func if successful
        
    Raises:
        ValidationError: If validation fails and severity is ERROR
    """
    try:
        return validation_func(*args, **kwargs)
    except ValidationError as e:
        if severity == ValidationSeverity.ERROR:
            handle_error(
                error=e,
                context="Validation",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger
            )
        elif severity == ValidationSeverity.WARNING:
            handle_error(
                error=e,
                context="Validation warning",
                severity=ErrorSeverity.WARNING,
                reraise=False,
                logger=logger
            )
        else:  # INFO
            if logger:
                logger.info(f"Validation info: {e}")
        
        # Return None for non-ERROR severities
        return None
