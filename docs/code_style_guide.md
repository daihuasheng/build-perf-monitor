# Code Style Guide

> **Languages**: [English](code_style_guide.md) | [中文](code_style_guide.zh-CN.md)

This document outlines the coding standards and documentation practices for the MyMonitor project.

## Python Code Style

### General Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters (Black default)
- Use UTF-8 encoding for all Python files
- End files with a single newline
- Use consistent naming conventions

### Imports

- Group imports in the following order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library specific imports
- Use absolute imports for clarity
- Use `import` statements for packages and modules
- Use `from ... import` for classes and functions
- Avoid wildcard imports (`from module import *`)

Example:
```python
# Standard library
import os
import sys
from typing import Dict, List, Optional

# Third-party libraries
import polars as pl
import psutil

# Local modules
from mymonitor.storage.base import DataStorage
from mymonitor.config import get_config
```

### Naming Conventions

- `snake_case` for functions, methods, variables, and modules
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Prefix private attributes with underscore: `_private_var`
- Use descriptive names that reflect purpose

### Type Hints

- Use type hints for all function parameters and return values
- Use `Optional[Type]` for parameters that can be None
- Use `Union[Type1, Type2]` for parameters that can be multiple types
- Use `Any` sparingly, only when truly necessary
- Use `List`, `Dict`, `Set`, etc. from `typing` module

Example:
```python
def process_data(
    input_data: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    max_rows: int = 1000
) -> pl.DataFrame:
    """Process input data into a DataFrame."""
    # Implementation
```

## Documentation Standards

### Module Docstrings

Every module should have a docstring at the top that explains its purpose and contents:

```python
"""
Module name and primary function.

Detailed description of what the module does, its key components,
and how it fits into the larger system. May include examples of usage
if appropriate.

Key features:
- Feature 1: Brief description
- Feature 2: Brief description
"""
```

### Class Docstrings

Classes should have docstrings explaining their purpose, behavior, and usage:

```python
class ExampleClass:
    """
    Short description of the class purpose.
    
    Detailed explanation of what the class does, its key features,
    and how it should be used. May include design patterns or
    architectural considerations.
    
    Attributes:
        attr1: Description of first attribute
        attr2: Description of second attribute
    
    Note:
        Any special considerations or limitations
    """
```

### Function and Method Docstrings

Functions and methods should have docstrings following Google style:

```python
def example_function(param1: str, param2: Optional[int] = None) -> bool:
    """
    Short description of function purpose.
    
    Detailed explanation of what the function does, its algorithm,
    and any side effects.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter, mention default value
            if parameter is complex, can use indented continuation
    
    Returns:
        Description of return value
    
    Raises:
        ExceptionType: When and why this exception is raised
    
    Examples:
        >>> example_function("test", 42)
        True
    """
```

### Comments

- Use comments sparingly to explain "why" not "what"
- Keep comments up-to-date when code changes
- Use complete sentences with proper capitalization and punctuation
- Avoid obvious comments that don't add value

Good comment:
```python
# Skip validation for empty inputs to improve performance
if not input_data:
    return default_value
```

Poor comment:
```python
# Increment counter
counter += 1
```

## Code Organization

### File Structure

- One class per file when the class is complex
- Group related functions in modules
- Keep files focused on a single responsibility
- Limit file size (aim for under 500 lines)

### Function and Method Organization

- Keep functions focused on a single task
- Limit function length (aim for under 50 lines)
- Use helper functions to break down complex logic
- Order methods logically:
  1. Special methods (`__init__`, `__str__`, etc.)
  2. Public methods
  3. Protected/private methods

## Error Handling

- Use specific exception types
- Handle exceptions at the appropriate level
- Log exceptions with context information
- Provide helpful error messages
- Use context managers for resource cleanup

Example:
```python
try:
    result = process_data(input_data)
except ValueError as e:
    logger.error(f"Invalid data format: {e}")
    raise
except IOError as e:
    logger.error(f"Failed to read input file: {e}")
    return default_result
finally:
    cleanup_resources()
```

## Logging

- Use the standard `logging` module
- Create a logger per module
- Use appropriate log levels:
  - `DEBUG`: Detailed information for debugging
  - `INFO`: Confirmation that things are working
  - `WARNING`: Something unexpected but not an error
  - `ERROR`: An error that prevents normal operation
  - `CRITICAL`: A serious error that may prevent the program from continuing
- Include context in log messages

Example:
```python
import logging

logger = logging.getLogger(__name__)

def process_file(file_path: str) -> None:
    """Process a file."""
    logger.info(f"Processing file: {file_path}")
    try:
        # Process file
        logger.debug(f"File {file_path} processed successfully")
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
```

## Testing

- Write unit tests for all functionality
- Use pytest for testing
- Aim for high test coverage
- Use descriptive test names
- Structure tests using the Arrange-Act-Assert pattern
- Mock external dependencies

Example:
```python
def test_data_storage_manager_saves_results():
    """Test that DataStorageManager correctly saves monitoring results."""
    # Arrange
    manager = DataStorageManager(tmp_path)
    results = create_test_results()
    
    # Act
    manager.save_monitoring_results(results, mock_context)
    
    # Assert
    assert (tmp_path / "memory_samples.parquet").exists()
    loaded_df = manager.load_memory_samples()
    assert len(loaded_df) == len(results.all_samples_data)
```

## Version Control

- Write clear, concise commit messages
- Use conventional commit format:
  - `feat(component): add new feature`
  - `fix(component): fix bug in feature`
  - `docs(component): update documentation`
  - `refactor(component): refactor code without changing behavior`
- Keep commits focused on a single change
- Reference issue numbers in commit messages

## Tools

- Use Black for code formatting
- Use isort for import sorting
- Use flake8 for linting
- Use mypy for type checking
- Use pytest for testing

## Conclusion

Following these guidelines ensures code consistency, readability, and maintainability across the project. When in doubt, prioritize clarity and readability over cleverness or optimization.
