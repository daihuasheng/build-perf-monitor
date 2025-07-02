"""
General utility functions for process and system interaction.

This module provides helpers for:
- Determining CPU affinity for processes based on configured policies.
- Executing external commands robustly and capturing their output.
- Categorizing processes into meaningful groups (e.g., 'Compiler', 'Linker')
  by applying a set of configurable rules.
- Checking for the presence of system dependencies like 'pidstat'.
"""

import logging
import math
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Set, Tuple, Dict

import psutil

# Local application imports
from . import config
from .data_models import CpuAllocationPlan

logger = logging.getLogger(__name__)

# Cache for process categorization to avoid repeated rule evaluation
_categorization_cache: Dict[Tuple[str, str], Tuple[str, str]] = {}


def parse_shell_wrapper_command(cmd_name: str, cmd_full: str) -> Tuple[str, str]:
    """Parse shell wrapper commands to extract the actual executed command.
    
    This function identifies common shell wrapper patterns (like sh -c, bash -c, etc.)
    and extracts the actual command being executed, while excluding cases where
    the shell is used to launch scripts.
    
    Args:
        cmd_name: The process command name (e.g., 'sh', 'bash').
        cmd_full: The complete command line string.
        
    Returns:
        A tuple of (parsed_command_name, parsed_full_command).
        If not a shell wrapper or parsing fails, returns the original (cmd_name, cmd_full).
        
    Examples:
        >>> parse_shell_wrapper_command('sh', 'sh -c "gcc -o test test.c"')
        ('gcc', 'gcc -o test test.c')
        >>> parse_shell_wrapper_command('bash', 'bash script.sh')
        ('bash', 'bash script.sh')
    """
    
    # Only handle common shell commands
    if cmd_name not in ['sh', 'bash', 'zsh', 'dash']:
        return cmd_name, cmd_full
    
    # Check if this is a shell -c pattern
    if ' -c ' not in cmd_full:
        return cmd_name, cmd_full
    
    try:
        # Use shlex to safely parse the command line
        parts = shlex.split(cmd_full)
        
        # Find the position of the -c argument
        c_index = -1
        for i, part in enumerate(parts):
            if part == '-c' and i + 1 < len(parts):
                c_index = i
                break
        
        if c_index == -1:
            return cmd_name, cmd_full
            
        # Get the command string after -c
        wrapped_command = parts[c_index + 1]
        
        # Parse the wrapped command again
        try:
            wrapped_parts = shlex.split(wrapped_command)
        except ValueError as e:
            # If parsing fails, it might be complex shell syntax, keep original
            logger.debug(f"Failed to parse shell command '{wrapped_command}': {e}")
            return cmd_name, cmd_full
            
        if not wrapped_parts:
            return cmd_name, cmd_full
            
        # Get the base name of the wrapped command
        wrapped_cmd_name = Path(wrapped_parts[0]).name
        
        # Check if this is a script file - if so, keep the original shell classification
        if (wrapped_cmd_name.endswith('.sh') or 
            wrapped_cmd_name.endswith('.py') or 
            wrapped_cmd_name.endswith('.pl') or
            wrapped_cmd_name.endswith('.rb') or
            wrapped_cmd_name.endswith('.js') or
            wrapped_parts[0].endswith('.sh') or
            wrapped_parts[0].endswith('.py') or
            wrapped_parts[0].endswith('.pl') or
            wrapped_parts[0].endswith('.rb') or
            wrapped_parts[0].endswith('.js')):
            return cmd_name, cmd_full
            
        # Check if this contains obvious script content (shell syntax)
        if any(syntax in wrapped_command for syntax in ['&&', '||', '|', ';', '$(', '`']):
            # If it contains shell syntax, keep original shell classification as this is script logic
            return cmd_name, cmd_full
                
        # If the wrapped command looks like a simple program call, return the parsed result
        logger.debug(f"Shell wrapper detected: '{cmd_full}' -> unwrapped: '{wrapped_cmd_name}', '{wrapped_command}'")
        return wrapped_cmd_name, wrapped_command
        
    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to parse shell wrapper command '{cmd_full}': {e}")
        return cmd_name, cmd_full


def _parse_core_range_str(core_str: str) -> Set[int]:
    """Parse a core range string into a set of integer core IDs.
    
    Parses strings like "1,2,4-7" into sets of core IDs. Supports individual
    cores and ranges separated by commas and hyphens.
    
    Args:
        core_str: String representation of core ranges (e.g., "1,2,4-7").
        
    Returns:
        Set of integer core IDs. Empty set if parsing fails.
        
    Examples:
        >>> _parse_core_range_str("1,2,4-7")
        {1, 2, 4, 5, 6, 7}
        >>> _parse_core_range_str("")
        set()
    """
    cores = set()
    if not core_str:
        return cores
    try:
        for part in core_str.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                cores.update(range(start, end + 1))
            else:
                cores.add(int(part))
    except ValueError as e:
        logger.warning(
            f"Failed to parse core range string '{core_str}': {type(e).__name__}: {e}. "
            f"Returning empty set."
        )
        return set()
    return cores


def plan_cpu_allocation(
    policy: str,
    j_level: int,
    manual_build_cores_str: str,
    manual_monitor_cores_str: str,
    main_monitor_core: int,
) -> CpuAllocationPlan:
    """Create a comprehensive CPU allocation plan based on the selected policy.

    This is the central function for intelligent, adaptive CPU scheduling. It determines
    which cores to assign to build processes and monitoring workers based on the
    configured scheduling policy, available system cores, and build parallelism level.

    Args:
        policy: Scheduling policy ("adaptive" or "manual").
        j_level: Build parallelism level (number of parallel jobs).
        manual_build_cores_str: Core range string for manual build allocation.
        manual_monitor_cores_str: Core range string for manual monitor allocation.
        main_monitor_core: Core ID for the main monitor process.

    Returns:
        CpuAllocationPlan containing taskset prefix, core descriptions, and
        monitoring worker core assignments.

    Note:
        Requires 'taskset' command to be available for CPU pinning. Falls back
        to shared core allocation if taskset is not found.
    """
    total_cores = psutil.cpu_count() or 1
    taskset_available = bool(shutil.which("taskset"))

    if not taskset_available:
        logger.warning("`taskset` command not found. CPU pinning is disabled.")
        return CpuAllocationPlan(
            build_command_prefix="",
            build_cores_desc="All (taskset not available)",
            monitoring_cores=[],
            monitoring_cores_desc="Shared (taskset not available)",
        )

    if policy == "manual":
        return _plan_manual_allocation(manual_build_cores_str, manual_monitor_cores_str)

    if policy == "adaptive":
        return _plan_adaptive_allocation(total_cores, j_level, main_monitor_core)

    logger.warning(f"Unknown scheduling policy '{policy}'. Disabling CPU pinning.")
    return CpuAllocationPlan(
        build_command_prefix="",
        build_cores_desc=f"All (unknown policy: {policy})",
        monitoring_cores=[],
        monitoring_cores_desc=f"Shared (unknown policy: {policy})",
    )


def _plan_manual_allocation(
    build_cores_str: str, monitor_cores_str: str
) -> CpuAllocationPlan:
    """Handle the 'manual' CPU allocation strategy.
    
    Creates CPU allocation plan based on user-specified core ranges for
    both build and monitoring processes.
    
    Args:
        build_cores_str: Core range string for build processes.
        monitor_cores_str: Core range string for monitoring workers.
        
    Returns:
        CpuAllocationPlan with manual core assignments.
    """
    build_prefix = ""
    build_desc = "All (manual with no cores specified)"
    if build_cores_str:
        build_prefix = f"taskset -c {build_cores_str} "
        build_desc = f"Manual: cores {build_cores_str}"

    monitor_cores = sorted(list(_parse_core_range_str(monitor_cores_str)))
    monitor_desc = (
        f"Manual: cores {monitor_cores_str}" if monitor_cores_str else "Shared"
    )

    return CpuAllocationPlan(
        build_command_prefix=build_prefix,
        build_cores_desc=build_desc,
        monitoring_cores=monitor_cores,
        monitoring_cores_desc=monitor_desc,
    )


def _plan_adaptive_allocation(
    total_cores: int, j_level: int, main_monitor_core: int
) -> CpuAllocationPlan:
    """Implement the 'adaptive' CPU allocation strategy (V2.1).
    
    This version prioritizes giving surplus cores to the build process
    and uses a max(static, proportional) buffer for build cores. It automatically
    determines optimal core allocation based on system resources and build requirements.
    
    The algorithm considers:
    - Build parallelism level and required buffer cores
    - Monitoring worker requirements
    - System core availability
    - Core isolation vs. shared allocation strategies
    
    Args:
        total_cores: Total number of CPU cores available on the system.
        j_level: Build parallelism level (number of parallel jobs).
        main_monitor_core: Core ID reserved for the main monitor process.
        
    Returns:
        CpuAllocationPlan with optimized core assignments for build-priority isolation
        or mixed-shared mode depending on core availability.
    """
    # --- V2.1 Heuristics ---
    BUILD_CORE_FACTOR = 1.25
    BUILD_CORE_STATIC_BUFFER = 4
    MONITOR_WORKER_CAP = 16

    # --- V2.1 Core Requirement Calculation ---
    num_build_cores_needed = max(
        j_level + BUILD_CORE_STATIC_BUFFER, math.ceil(j_level * BUILD_CORE_FACTOR)
    )
    num_monitor_workers_ideal = min(MONITOR_WORKER_CAP, max(1, math.floor(j_level / 2)))
    min_cores_for_isolation = num_build_cores_needed + num_monitor_workers_ideal + 1

    all_cores = set(range(total_cores))
    build_cores: Set[int] = set()
    monitor_worker_cores: Set[int] = set()

    # --- Decision Logic ---
    if total_cores < min_cores_for_isolation:
        # SCENARIO A: Mixed-Shared Mode (Not enough cores for isolation)
        logger.info(
            f"Core scarcity detected (total: {total_cores}, required for isolation: {min_cores_for_isolation})."
            " Entering mixed-shared mode."
        )
        build_cores = all_cores - {main_monitor_core}
        monitor_worker_cores = {main_monitor_core}
    else:
        # SCENARIO B: Build-Priority Isolated Mode
        logger.info(
            f"Sufficient cores for isolation (total: {total_cores}, required: {min_cores_for_isolation})."
            " Entering build-priority isolated mode."
        )
        # 1. Reserve for Monitor Workers from the top
        monitor_worker_cores = set(
            range(total_cores - num_monitor_workers_ideal, total_cores)
        )
        # 2. Reserve for Main Monitor & get remaining for build
        # (handle edge case where monitor core is also a worker core)
        cores_for_build = all_cores - monitor_worker_cores - {main_monitor_core}
        build_cores = cores_for_build

    # --- Boundary check: ensure build process has at least one core ---
    if not build_cores:
        logger.warning(
            f"No cores available for build process after allocation. "
            f"Falling back to shared mode with all cores except monitor core {main_monitor_core}."
        )
        build_cores = all_cores - {main_monitor_core}
        # If even this doesn't work, give the build process at least one core
        if not build_cores:
            logger.warning("Extreme core scarcity: assigning all cores to build.")
            build_cores = all_cores

    # --- Generate final plan object ---
    build_core_str = _format_core_set_to_str(build_cores)
    build_prefix = f"taskset -c {build_core_str} " if build_core_str else ""
    build_desc = f"Adaptive: cores {build_core_str}"

    final_monitor_list = sorted(list(monitor_worker_cores))
    monitor_core_str = _format_core_set_to_str(monitor_worker_cores)
    monitor_desc = f"Adaptive: cores {monitor_core_str}"

    # Add a log message for the shared core case, but do not modify the list.
    if len(final_monitor_list) == 1 and final_monitor_list[0] == main_monitor_core:
        logger.info(
            "Monitoring workers will run shared with the main monitor process on core %d.",
            main_monitor_core,
        )

    return CpuAllocationPlan(
        build_command_prefix=build_prefix,
        build_cores_desc=build_desc,
        monitoring_cores=final_monitor_list,
        monitoring_cores_desc=monitor_desc,
    )


def _format_core_set_to_str(cores: Set[int]) -> str:
    """Format a set of core integers into a compact string representation.
    
    Converts a set of core IDs into a string format suitable for taskset,
    using ranges where possible for compactness.
    
    Args:
        cores: Set of integer core IDs.
        
    Returns:
        Formatted string like "0,2-4" or empty string if no cores.
        
    Examples:
        >>> _format_core_set_to_str({0, 2, 3, 4})
        "0,2-4"
        >>> _format_core_set_to_str({1, 5})
        "1,5"
        >>> _format_core_set_to_str(set())
        ""
    """
    if not cores:
        return ""

    sorted_cores = sorted(list(cores))
    ranges = []
    start = end = sorted_cores[0]

    for i in range(1, len(sorted_cores)):
        if sorted_cores[i] == end + 1:
            end = sorted_cores[i]
        else:
            ranges.append(str(start) if start == end else f"{start}-{end}")
            start = end = sorted_cores[i]
    ranges.append(str(start) if start == end else f"{start}-{end}")

    return ",".join(ranges)


def prepare_full_build_command(
    main_command_template: str,
    j_level: int,
    taskset_prefix: str,
    setup_command: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Construct the full build command including taskset prefix and parallelism.

    Combines the build command template with parallelism level, CPU affinity prefix,
    and optional setup commands to create the final executable command string.

    Args:
        main_command_template: Build command template with placeholders
            (e.g., "make -j{j_level}" or legacy "make -j<N>").
        j_level: The parallelism level to substitute in the template.
        taskset_prefix: CPU affinity prefix command (e.g., "taskset -c 0-7").
        setup_command: Optional setup command to source before build.

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


def get_process_category(cmd_name: str, cmd_full: str) -> Tuple[str, str]:
    """Categorize a process based on configured classification rules.

    This function applies a set of prioritized rules to classify processes into
    major and minor categories (e.g., 'Compiler', 'gcc'). It uses caching to
    avoid repeated rule evaluation for identical process command lines, which
    significantly improves performance on builds with many similar processes.

    The function also handles shell wrapper commands by attempting to extract
    the actual executed command from shell wrappers like 'sh -c "gcc ..."'.

    Args:
        cmd_name: Base name of the command executable (e.g., 'gcc').
        cmd_full: Complete command line with all arguments.

    Returns:
        Tuple of (major_category, minor_category) strings.
        Returns ('Other', 'Other_<cmd_name>') if no rules match.

    Note:
        Rules are loaded from the application configuration and applied in
        priority order. Results are cached up to the configured cache size limit.

    Examples:
        >>> get_process_category('gcc', 'gcc -O2 -c file.c')
        ('Compiler', 'gcc')
        >>> get_process_category('unknown_tool', 'unknown_tool --help')
        ('Other', 'Other_unknown_tool')
    """
    # Check cache first
    cache_key = (cmd_name, cmd_full)
    if cache_key in _categorization_cache:
        return _categorization_cache[cache_key]
    
    app_config = config.get_config()
    
    # Try to parse shell wrapper commands
    current_cmd_name, current_cmd_full = parse_shell_wrapper_command(cmd_name, cmd_full)

    for rule in app_config.rules:
        target_field_value = (
            current_cmd_name
            if rule.match_field == "current_cmd_name"
            else current_cmd_full
        )

        match = False
        if rule.match_type == "exact":
            # For exact match, patterns should be a string
            pattern_to_match = rule.patterns if isinstance(rule.patterns, str) else rule.patterns[0] if rule.patterns else ""
            match = target_field_value == pattern_to_match
        elif rule.match_type == "contains":
            # For contains match, patterns should be a string
            pattern_to_match = rule.patterns if isinstance(rule.patterns, str) else rule.patterns[0] if rule.patterns else ""
            match = pattern_to_match and pattern_to_match in target_field_value
        elif rule.match_type == "regex":
            # For regex match, patterns should be a string
            pattern_to_match = rule.patterns if isinstance(rule.patterns, str) else rule.patterns[0] if rule.patterns else ""
            if pattern_to_match:
                import re
                match = bool(re.search(pattern_to_match, target_field_value))
        elif rule.match_type == "in_list":
            # For in_list match, patterns should be a list
            patterns_to_check = rule.patterns if isinstance(rule.patterns, list) else [rule.patterns] if rule.patterns else []
            match = target_field_value in patterns_to_check

        if match:
            result = (rule.major_category, rule.category)
            # Cache the result, but respect the cache size limit
            if len(_categorization_cache) < app_config.monitor.categorization_cache_size:
                _categorization_cache[cache_key] = result
            return result

    # If no rules match, generate a classification based on the original command name
    result = ("Other", f"Other_{current_cmd_name}")
    # Cache the result, but respect the cache size limit
    if len(_categorization_cache) < app_config.monitor.categorization_cache_size:
        _categorization_cache[cache_key] = result
    return result


def check_pidstat_installed() -> bool:
    """Check if the 'pidstat' command is available on the system.
    
    Returns:
        True if pidstat is found in the system PATH, False otherwise.
        
    Note:
        The pidstat tool is part of the sysstat package and is required
        for RSS memory collection using the rss_pidstat collector.
    """
    return shutil.which("pidstat") is not None
