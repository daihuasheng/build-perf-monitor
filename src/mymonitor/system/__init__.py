"""
System interaction utilities for process monitoring and resource management.

This module provides comprehensive system-level functionality including:

- Command execution with proper error handling and logging
- CPU core allocation and affinity management for optimal performance
- Process parsing and shell wrapper command analysis
- System resource detection and validation
- Build command preparation with CPU pinning support

The utilities are designed to work across different operating systems and
handle various system configurations, providing consistent interfaces for
the monitoring system to interact with the underlying OS resources.

Key features:
- Intelligent CPU allocation strategies (adaptive, manual)
- Process affinity management for consistent performance measurements
- Shell command parsing for accurate process classification
- Resource availability checking and validation
"""

# Command execution
from .commands import (
    check_pidstat_installed,
    prepare_command_with_setup,
    prepare_full_build_command,
    run_command,
)

# CPU allocation and scheduling
from .cpu_manager import get_cpu_manager

# Process parsing and analysis
from .processes import parse_shell_wrapper_command

__all__ = [
    # Commands
    "check_pidstat_installed",
    "prepare_command_with_setup",
    "prepare_full_build_command",
    "run_command",
    # CPU allocation
    "get_cpu_manager",
    # Process parsing
    "parse_shell_wrapper_command",
]
