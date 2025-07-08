"""
System interaction utilities for the mymonitor package.

This module provides functionality for command execution, process management,
and CPU allocation planning.
"""

# Command execution
from .commands import (
    check_pidstat_installed,
    prepare_command_with_setup,
    prepare_full_build_command,
    run_command,
)

# CPU allocation and scheduling
from .cpu_manager import plan_cpu_allocation, get_cpu_manager, parse_cpu_cores

# Process parsing and analysis
from .processes import parse_shell_wrapper_command

__all__ = [
    # Commands
    "check_pidstat_installed",
    "prepare_command_with_setup", 
    "prepare_full_build_command",
    "run_command",
    # CPU allocation
    "plan_cpu_allocation",
    "get_cpu_manager",
    "parse_cpu_cores",
    # Process parsing
    "parse_shell_wrapper_command",
]
