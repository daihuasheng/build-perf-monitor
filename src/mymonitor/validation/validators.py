"""
Validation functions for input validation and data integrity checks.

This module contains all validation logic for user inputs, configuration values,
and system constraints.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Union

from .exceptions import ValidationError


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
    Validate that a path exists and meets specified criteria.
    
    Args:
        path: The path to validate
        must_be_dir: If True, path must be a directory
        must_be_file: If True, path must be a file
        check_readable: If True, check that path is readable
        check_writable: If True, check that path is writable
        field_name: Name of the field being validated (for error messages)
        
    Returns:
        The validated Path object
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        path_obj = Path(path).resolve()
    except Exception as e:
        raise ValidationError(f"{field_name} is not a valid path '{path}': {e}")
    
    if not path_obj.exists():
        raise ValidationError(f"{field_name} does not exist: {path_obj}")
    
    if must_be_dir and not path_obj.is_dir():
        raise ValidationError(f"{field_name} must be a directory: {path_obj}")
    
    if must_be_file and not path_obj.is_file():
        raise ValidationError(f"{field_name} must be a file: {path_obj}")
    
    if check_readable and not os.access(path_obj, os.R_OK):
        raise ValidationError(f"{field_name} is not readable: {path_obj}")
    
    if check_writable and not os.access(path_obj, os.W_OK):
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
        case_sensitive: Whether the comparison should be case-sensitive
        
    Returns:
        The validated value (potentially normalized to match case)
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string, got {type(value).__name__}")
    
    if case_sensitive:
        if value in valid_choices:
            return value
    else:
        value_lower = value.lower()
        for choice in valid_choices:
            if choice.lower() == value_lower:
                return choice  # Return the canonical case
    
    raise ValidationError(
        f"{field_name} must be one of {valid_choices}, got '{value}'"
    )


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
        The validated pattern string
        
    Raises:
        ValidationError: If the pattern is invalid
    """
    if not isinstance(pattern, str):
        raise ValidationError(f"{field_name} must be a string, got {type(pattern).__name__}")
    
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValidationError(f"{field_name} is not a valid regex pattern '{pattern}': {e}")
    
    return pattern


def validate_cpu_core_range(
    core_str: str,
    max_cores: Optional[int] = None,
    field_name: str = "CPU core range"
) -> str:
    """
    Validate a CPU core range string (e.g., "0-3", "1,3,5", "").
    
    Args:
        core_str: The core range string to validate
        max_cores: Maximum number of available cores (for range validation)
        field_name: Name of the field being validated (for error messages)
        
    Returns:
        The validated core range string
        
    Raises:
        ValidationError: If the core range is invalid
    """
    if not isinstance(core_str, str):
        raise ValidationError(f"{field_name} must be a string, got {type(core_str).__name__}")
    
    # Empty string is valid (means no specific assignment)
    if not core_str.strip():
        return core_str.strip()
    
    if max_cores is None:
        max_cores = os.cpu_count() or 1
    
    try:
        # Parse the core specification to validate it
        cores = set()
        for part in core_str.split(','):
            part = part.strip()
            if '-' in part:
                # Range specification (e.g., "0-3")
                start_str, end_str = part.split('-', 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                if start > end:
                    raise ValidationError(f"Invalid range in {field_name}: {part} (start > end)")
                cores.update(range(start, end + 1))
            else:
                # Single core specification
                cores.add(int(part))
        
        # Validate core numbers are within system limits
        invalid_cores = [c for c in cores if c < 0 or c >= max_cores]
        if invalid_cores:
            raise ValidationError(
                f"Invalid CPU cores in {field_name}: {invalid_cores}. "
                f"Valid range is 0-{max_cores-1}"
            )
            
    except ValueError as e:
        raise ValidationError(f"Invalid {field_name} format '{core_str}': {e}")
    
    return core_str.strip()


def validate_jobs_list(
    jobs_str: str,
    min_jobs: int = 1,
    max_jobs: int = 1024,
    field_name: str = "--jobs argument"
) -> List[int]:
    """
    Validate and parse a jobs list string (e.g., "1,2,4,8").
    
    Args:
        jobs_str: The jobs string to validate and parse
        min_jobs: Minimum allowed job count
        max_jobs: Maximum allowed job count
        field_name: Name of the field being validated (for error messages)
        
    Returns:
        List of validated job counts
        
    Raises:
        ValidationError: If the jobs string is invalid
    """
    if not isinstance(jobs_str, str):
        raise ValidationError(f"{field_name} must be a string, got {type(jobs_str).__name__}")
    
    if not jobs_str.strip():
        raise ValidationError(f"{field_name} cannot be empty")
    
    try:
        jobs = []
        for part in jobs_str.split(','):
            job_count = int(part.strip())
            if job_count < min_jobs:
                raise ValidationError(f"Job count {job_count} is below minimum {min_jobs}")
            if job_count > max_jobs:
                raise ValidationError(f"Job count {job_count} exceeds maximum {max_jobs}")
            jobs.append(job_count)
        
        if not jobs:
            raise ValidationError(f"{field_name} resulted in empty job list")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_jobs = []
        for job in jobs:
            if job not in seen:
                seen.add(job)
                unique_jobs.append(job)
        
        return unique_jobs
        
    except ValueError as e:
        raise ValidationError(f"Invalid {field_name} format '{jobs_str}': {e}")


def validate_command_template(
    template: str,
    required_placeholders: Optional[List[str]] = None,
    field_name: str = "command template"
) -> str:
    """
    Validate a command template string for required placeholders.
    
    Args:
        template: The command template to validate
        required_placeholders: List of required placeholder names (e.g., ['<N>'])
        field_name: Name of the field being validated (for error messages)
        
    Returns:
        The validated template string
        
    Raises:
        ValidationError: If the template is invalid
    """
    if not isinstance(template, str):
        raise ValidationError(f"{field_name} must be a string, got {type(template).__name__}")
    
    if not template.strip():
        raise ValidationError(f"{field_name} cannot be empty")
    
    if required_placeholders:
        for placeholder in required_placeholders:
            if placeholder not in template:
                raise ValidationError(
                    f"{field_name} must contain placeholder '{placeholder}'. "
                    f"Template: '{template}'"
                )
    
    # Basic shell injection detection (very simple)
    dangerous_patterns = [';', '&&', '||', '`', '$(']
    for pattern in dangerous_patterns:
        if pattern in template:
            # This is just a warning - we allow these but note the risk
            pass
    
    return template.strip()


def validate_project_name(
    name: str,
    existing_names: Optional[List[str]] = None,
    field_name: str = "project name"
) -> str:
    """
    Validate a project name for safety and uniqueness.
    
    Args:
        name: The project name to validate
        existing_names: List of existing project names (for uniqueness check)
        field_name: Name of the field being validated (for error messages)
        
    Returns:
        The validated project name
        
    Raises:
        ValidationError: If the name is invalid
    """
    if not isinstance(name, str):
        raise ValidationError(f"{field_name} must be a string, got {type(name).__name__}")
    
    if not name.strip():
        raise ValidationError(f"{field_name} cannot be empty")
    
    name = name.strip()
    
    # Check for valid characters (alphanumeric, hyphens, underscores)
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValidationError(
            f"{field_name} can only contain letters, numbers, hyphens, and underscores. "
            f"Got: '{name}'"
        )
    
    # Check length
    if len(name) > 50:
        raise ValidationError(f"{field_name} cannot be longer than 50 characters. Got: '{name}'")
    
    # Check uniqueness
    if existing_names and name in existing_names:
        raise ValidationError(f"{field_name} '{name}' already exists")
    
    return name
