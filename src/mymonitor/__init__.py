"""
MyMonitor: Build performance monitoring and analysis tool.

This package provides comprehensive build performance monitoring capabilities,
memory usage tracking, and process categorization for software build systems.

The package is organized into specialized modules:
- config: Configuration management and validation
- models: Data structures and type definitions  
- validation: Input validation and error handling
- system: System interaction and process management
- classification: Process categorization and rules
- collectors: Memory data collection implementations
- monitoring: Memory monitoring coordination
- executor: Build process execution
- cli: Command-line interface and orchestration

Usage:
    From command line:
        python -m mymonitor.cli.main [options]
    
    Programmatically:
        from mymonitor import BuildRunner, get_config
        config = get_config()
        runner = BuildRunner(...)
        runner.run()
"""

# Main interfaces
from .config import get_config, clear_config_cache, set_config_path
from .cli.orchestrator import BuildRunner
from .cli import main_cli

# Model classes for external use
from .models import (
    AppConfig,
    MonitorConfig, 
    ProjectConfig,
    RuleConfig,
    RunContext,
    RunPaths,
    MonitoringResults,
    CpuAllocationPlan,
)

# Validation utilities
from .validation import (
    ValidationError,
    validate_positive_integer,
    validate_positive_float,
    validate_path_exists,
    validate_enum_choice,
    validate_regex_pattern,
)

# System utilities
from .system import (
    check_pidstat_installed,
    plan_cpu_allocation,
    run_command,
)

# Classification utilities
from .classification import get_process_category

__version__ = "2.0.0"

__all__ = [
    # Main interfaces
    "get_config",
    "clear_config_cache", 
    "set_config_path",
    "BuildRunner",
    "main_cli",
    # Models
    "AppConfig",
    "MonitorConfig",
    "ProjectConfig", 
    "RuleConfig",
    "RunContext",
    "RunPaths",
    "MonitoringResults",
    "CpuAllocationPlan",
    # Validation
    "ValidationError",
    "validate_positive_integer",
    "validate_positive_float",
    "validate_path_exists",
    "validate_enum_choice",
    "validate_regex_pattern", 
    # System utilities
    "check_pidstat_installed",
    "plan_cpu_allocation",
    "run_command",
    # Classification
    "get_process_category",
]
