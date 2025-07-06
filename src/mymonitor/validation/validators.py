"""
Simplified validation functions.

This module provides essential validation functions without over-engineering,
focusing on the validations that are actually used in the application.
"""

import os
import re
from pathlib import Path
from typing import Any, List, Union, Optional

from .exceptions import ValidationError


def validate_positive_integer(
    value: Any, 
    min_value: int = 1,
    max_value: Optional[int] = None,
    field_name: str = "value"
) -> int:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive), None for no limit
        field_name: Name of the field being validated
        
    Returns:
        Validated integer value
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        int_value = int(value)
        if int_value < min_value:
            raise ValidationError(
                f"{field_name} must be >= {min_value}, got {int_value}",
                field_name=field_name,
                value=value
            )
        if max_value is not None and int_value > max_value:
            raise ValidationError(
                f"{field_name} must be <= {max_value}, got {int_value}",
                field_name=field_name,
                value=value
            )
        return int_value
    except (ValueError, TypeError):
        raise ValidationError(
            f"{field_name} must be a valid integer, got {value}",
            field_name=field_name,
            value=value
        )


def validate_positive_float(
    value: Any, 
    min_value: float = 0.0,
    max_value: Optional[float] = None,
    field_name: str = "value"
) -> float:
    """
    Validate that a value is a positive float.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive), None for no limit
        field_name: Name of the field being validated
        
    Returns:
        Validated float value
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        float_value = float(value)
        if float_value < min_value:
            raise ValidationError(
                f"{field_name} must be >= {min_value}, got {float_value}",
                field_name=field_name,
                value=value
            )
        if max_value is not None and float_value > max_value:
            raise ValidationError(
                f"{field_name} must be <= {max_value}, got {float_value}",
                field_name=field_name,
                value=value
            )
        return float_value
    except (ValueError, TypeError):
        raise ValidationError(
            f"{field_name} must be a valid number, got {value}",
            field_name=field_name,
            value=value
        )


def validate_path_exists(path: Union[str, Path], field_name: str = "path") -> str:
    """
    Validate that a path exists.
    
    Args:
        path: Path to validate
        field_name: Name of the field being validated
        
    Returns:
        Validated path string
        
    Raises:
        ValidationError: If path doesn't exist
    """
    path_str = str(path)
    if not os.path.exists(path_str):
        raise ValidationError(
            f"{field_name} does not exist: {path_str}",
            field_name=field_name,
            value=path_str
        )
    return path_str


def validate_project_name(
    name: str, 
    existing_names: Optional[List[str]] = None,
    field_name: str = "project_name"
) -> str:
    """
    Validate project name format.
    
    Args:
        name: Project name to validate
        existing_names: List of existing project names (for uniqueness check)
        field_name: Name of the field being validated
        
    Returns:
        Validated project name
        
    Raises:
        ValidationError: If name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValidationError(
            f"{field_name} must be a non-empty string",
            field_name=field_name,
            value=name
        )
    
    # Allow alphanumeric, underscore, hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValidationError(
            f"{field_name} must contain only alphanumeric characters, underscores, and hyphens: {name}",
            field_name=field_name,
            value=name
        )
    
    # Check uniqueness if existing names provided
    if existing_names and name in existing_names:
        raise ValidationError(
            f"{field_name} must be unique, '{name}' already exists",
            field_name=field_name,
            value=name
        )
    
    return name


def validate_command_template(template: str, field_name: str = "command_template", require_placeholder: bool = True) -> str:
    """
    Validate build command template format.
    
    Args:
        template: Command template to validate
        field_name: Name of the field being validated
        require_placeholder: Whether to require a placeholder (for build commands)
        
    Returns:
        Validated command template
        
    Raises:
        ValidationError: If template is invalid
    """
    if not template or not isinstance(template, str):
        raise ValidationError(
            f"{field_name} must be a non-empty string",
            field_name=field_name,
            value=template
        )
    
    # Check for required placeholder only if required (for build commands)
    if require_placeholder and '{j_level}' not in template and '<N>' not in template:
        raise ValidationError(
            f"{field_name} must contain either '{{j_level}}' or '<N>' placeholder",
            field_name=field_name,
            value=template
        )
    
    return template


def validate_simple_command(command: str, field_name: str = "command") -> str:
    """
    Validate simple command format (no placeholders required).
    
    Args:
        command: Command to validate
        field_name: Name of the field being validated
        
    Returns:
        Validated command
        
    Raises:
        ValidationError: If command is invalid
    """
    return validate_command_template(command, field_name, require_placeholder=False)


def validate_regex_pattern(pattern: str, field_name: str = "regex_pattern") -> str:
    """
    Validate regex pattern format.
    
    Args:
        pattern: Regex pattern to validate
        field_name: Name of the field being validated
        
    Returns:
        Validated pattern
        
    Raises:
        ValidationError: If pattern is invalid
    """
    if not pattern or not isinstance(pattern, str):
        raise ValidationError(
            f"{field_name} must be a non-empty string",
            field_name=field_name,
            value=pattern
        )
    
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValidationError(
            f"{field_name} is not a valid regex pattern: {e}",
            field_name=field_name,
            value=pattern
        )
    
    return pattern


def validate_cpu_core_range(cores: str, field_name: str = "cpu_cores") -> str:
    """
    Validate CPU core range format.
    
    Args:
        cores: CPU core range string (e.g., "0-3", "1,3,5-7")
        field_name: Name of the field being validated
        
    Returns:
        Validated core range string
        
    Raises:
        ValidationError: If format is invalid
    """
    if not cores or not isinstance(cores, str):
        raise ValidationError(
            f"{field_name} must be a non-empty string",
            field_name=field_name,
            value=cores
        )
    
    # Basic format validation - allow numbers, commas, and hyphens
    if not re.match(r'^[0-9,-]+$', cores):
        raise ValidationError(
            f"{field_name} must contain only numbers, commas, and hyphens: {cores}",
            field_name=field_name,
            value=cores
        )
    
    return cores


def validate_jobs_list(
    jobs: Union[int, List[int], str], 
    field_name: str = "jobs",
    min_jobs: int = 1,
    max_jobs: int = 1024
) -> List[int]:
    """
    Validate jobs list format.
    
    Args:
        jobs: Single job level, list of job levels, or comma-separated string
        field_name: Name of the field being validated
        min_jobs: Minimum allowed job level
        max_jobs: Maximum allowed job level
        
    Returns:
        Validated list of job levels
        
    Raises:
        ValidationError: If format is invalid
    """
    # Handle string input (e.g., "8,16")
    if isinstance(jobs, str):
        jobs = jobs.strip()
        if not jobs:
            raise ValidationError(
                f"{field_name} cannot be empty",
                field_name=field_name,
                value=jobs
            )
        
        # Split by comma and convert to integers
        try:
            jobs_list = [int(j.strip()) for j in jobs.split(",")]
        except ValueError:
            raise ValidationError(
                f"{field_name} must be comma-separated integers (e.g., '8,16')",
                field_name=field_name,
                value=jobs
            )
        jobs = jobs_list
    
    if isinstance(jobs, int):
        jobs = [jobs]
    
    if not isinstance(jobs, list) or not jobs:
        raise ValidationError(
            f"{field_name} must be a positive integer or non-empty list of positive integers",
            field_name=field_name,
            value=jobs
        )
    
    validated_jobs = []
    for i, job in enumerate(jobs):
        validated_job = validate_positive_integer(
            job, 
            min_value=min_jobs,
            max_value=max_jobs,
            field_name=f"{field_name} item {i}"
        )
        validated_jobs.append(validated_job)
    
    return validated_jobs


def validate_enum_choice(
    value: Any, 
    valid_choices: List[str] = None,
    choices: List[str] = None,
    field_name: str = "value",
    case_sensitive: bool = True
) -> str:
    """
    Validate that a value is one of the allowed choices.
    
    Args:
        value: Value to validate
        valid_choices: List of allowed choices (legacy parameter name)
        choices: List of allowed choices (new parameter name)
        field_name: Name of the field being validated
        case_sensitive: Whether the comparison should be case-sensitive
        
    Returns:
        Validated choice
        
    Raises:
        ValidationError: If value is not in choices
    """
    # Handle both parameter names for compatibility
    choice_list = valid_choices or choices
    if choice_list is None:
        raise ValidationError(
            f"No valid choices provided for {field_name}",
            field_name=field_name,
            value=value
        )
    
    # Convert value to string for comparison
    str_value = str(value)
    
    # Handle case sensitivity
    if case_sensitive:
        if str_value not in choice_list:
            raise ValidationError(
                f"{field_name} must be one of {choice_list}, got {value}",
                field_name=field_name,
                value=value
            )
        return str_value
    else:
        # Case insensitive comparison
        lower_choices = [choice.lower() for choice in choice_list]
        lower_value = str_value.lower()
        if lower_value not in lower_choices:
            raise ValidationError(
                f"{field_name} must be one of {choice_list}, got {value}",
                field_name=field_name,
                value=value
            )
        # Return the original case from valid choices
        index = lower_choices.index(lower_value)
        return choice_list[index]
