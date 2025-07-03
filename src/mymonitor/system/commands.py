"""
Command execution and process management utilities.

This module provides functions for executing system commands, preparing command
lines with setup scripts, and checking for system dependencies.
"""

import logging
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def run_command(
    command: str, cwd: Path, shell: bool = False, executable_shell: Optional[str] = None
) -> Tuple[int, str, str]:
    """Execute a command and capture its output with robust error handling.

    Runs a subprocess command in the specified directory, capturing both stdout
    and stderr while handling various error conditions gracefully.

    Args:
        command: The command string to execute.
        cwd: Working directory path for command execution.
        shell: Whether to use shell for execution (default: False).
        executable_shell: Specific shell executable path (e.g., '/bin/bash').

    Returns:
        Tuple of (return_code, stdout_string, stderr_string).
        return_code is -1 for execution errors.

    Note:
        Uses UTF-8 encoding with error replacement for robust text handling.
        Logs command execution details and any errors encountered.
    """
    logger.debug(f"Executing command: '{command}' in '{cwd}'")
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            shell=shell,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            executable=executable_shell,
            check=False,
        )
        return process.returncode, process.stdout, process.stderr
    except FileNotFoundError as e:
        error_msg = f"Command not found: {shlex.split(command)[0]}"
        logger.error(f"{error_msg}: {type(e).__name__}: {e}")
        return -1, "", f"Error: Command not found '{shlex.split(command)[0]}'"
    except Exception as e:
        error_msg = f"Unexpected error while running command '{command[:50]}...'"
        logger.error(f"{error_msg}: {type(e).__name__}: {e}", exc_info=True)
        return -1, "", f"An unexpected error occurred: {e}"


def prepare_command_with_setup(
    main_command: str, setup_command: Optional[str]
) -> Tuple[str, Optional[str]]:
    """Combine a main command with an optional setup command.

    Handles the combination of setup scripts (like environment sourcing) with
    the main command, determining the appropriate shell executable if needed.

    Args:
        main_command: The primary command to execute.
        setup_command: Optional setup command to source before main command
            (e.g., "source env.sh").

    Returns:
        Tuple of (final_command_string, shell_executable).
        shell_executable is the required shell path or None for default shell.

    Examples:
        >>> prepare_command_with_setup("make", "source env.sh")
        ("source env.sh && make", None)
        >>> prepare_command_with_setup("make", None)
        ("make", None)
    """
    if setup_command:
        # If a setup command is provided, we need to use a shell to source it.
        final_command = f"{setup_command} && {main_command}"
        # Check if the setup command explicitly requires /bin/bash
        if " /bin/bash" in setup_command:
            executable = "/bin/bash"
        else:
            executable = None  # Use the system's default shell
        return final_command, executable
    else:
        # No setup command, so no special shell or executable is needed.
        # The command will be executed directly.
        return main_command, None


def prepare_full_build_command(
    main_command_template: str,
    j_level: int,
    taskset_prefix: str,
    setup_command: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Prepare the complete build command with all components.

    Combines the template command with parallelism level, CPU affinity (taskset),
    and setup scripts to create the final command ready for execution.

    Args:
        main_command_template: Build command template with {j_level} placeholder.
        j_level: Parallelism level to substitute in the template.
        taskset_prefix: CPU affinity prefix (e.g., "taskset -c 0-3").
        setup_command: Optional setup command to source before building.

    Returns:
        Tuple of (final_command_string, shell_executable).
        shell_executable is None unless a specific shell is required.

    Note:
        Supports both modern {j_level} and legacy <N> placeholder formats
        for backward compatibility.
    """
    # Convert legacy <N> placeholder to modern {j_level} format for backward compatibility
    normalized_template = main_command_template.replace("<N>", "{j_level}")
    
    # Format the core build command with the parallelism level.
    # The template is expected to contain the argument name, e.g., "make -j{j_level}".
    build_command = normalized_template.format(j_level=j_level)

    # Prepend the taskset prefix if it exists to the core build command.
    if taskset_prefix:
        command_with_affinity = f"{taskset_prefix} {build_command}"
    else:
        command_with_affinity = build_command

    # Combine with setup script if necessary.
    return prepare_command_with_setup(command_with_affinity, setup_command)


def check_pidstat_installed() -> bool:
    """Check if the 'pidstat' command is available on the system.
    
    Returns:
        True if pidstat is found in the system PATH, False otherwise.
        
    Note:
        The pidstat tool is part of the sysstat package and is required
        for RSS memory collection using the rss_pidstat collector.
    """
    return shutil.which("pidstat") is not None
