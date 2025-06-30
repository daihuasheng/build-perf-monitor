"""
General utility functions for process and system interaction.

This module provides helpers for:
- Determining CPU affinity for processes based on configured policies.
- Executing external commands robustly and capturing their output.
- Categorizing processes into meaningful groups (e.g., 'Compiler', 'Linker')
  by applying a set of configurable rules.
- Checking for the presence of system dependencies like 'pidstat'.
"""

import functools
import logging
import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

# Local application imports
from . import config

logger = logging.getLogger(__name__)


def determine_build_cpu_affinity(
    build_cpu_cores_policy: str,
    specific_build_cores_str: Optional[str],
    monitor_core_id: Optional[int],
    taskset_available: bool,
    total_cores_available: Optional[int],
) -> Tuple[str, str]:
    """
    Determines the build command prefix for CPU affinity and a descriptive string.

    Based on the configured policy, this function generates a `taskset` command
    prefix to pin the build process to specific CPU cores. It also returns a
    human-readable string describing which cores are being used.

    Args:
        build_cpu_cores_policy: The policy name (e.g., 'all_others', 'specific', 'none').
        specific_build_cores_str: A string defining cores if policy is 'specific' (e.g., "1,2,4-7").
        monitor_core_id: The ID of the core the main monitor script is pinned to, to be excluded.
        taskset_available: A boolean indicating if the `taskset` command is available.
        total_cores_available: The total number of CPU cores on the system.

    Returns:
        A tuple containing:
        - The command prefix string (e.g., "taskset -c 1-7 ") or an empty string.
        - A human-readable description of the core allocation strategy.
    """
    build_command_prefix = ""
    build_cores_target_str = "All Available (taskset not used or policy is 'none')"

    if not taskset_available or build_cpu_cores_policy == "none":
        build_cores_target_str = (
            "All Available (taskset not available)"
            if build_cpu_cores_policy != "none"
            else build_cores_target_str
        )
        return build_command_prefix, build_cores_target_str

    cores_for_build_taskset_str: Optional[str] = None

    if build_cpu_cores_policy == "all_others":
        if total_cores_available and total_cores_available > 1:
            all_cores = set(range(total_cores_available))
            if monitor_core_id is not None and monitor_core_id in all_cores:
                all_cores.remove(monitor_core_id)

            # Generate a comma-separated list of core ranges (e.g., "0,2-7")
            if all_cores:
                sorted_cores = sorted(list(all_cores))
                ranges = []
                start = end = sorted_cores[0]
                for i in range(1, len(sorted_cores)):
                    if sorted_cores[i] == end + 1:
                        end = sorted_cores[i]
                    else:
                        ranges.append(f"{start}-{end}" if start != end else str(start))
                        start = end = sorted_cores[i]
                ranges.append(f"{start}-{end}" if start != end else str(start))
                cores_for_build_taskset_str = ",".join(ranges)
                build_cores_target_str = (
                    f"All Other Cores (cores: {cores_for_build_taskset_str})"
                )
            else:
                build_cores_target_str = "All Available (no other cores left)"
        else:
            build_cores_target_str = "All Available (not enough cores to isolate)"

    elif build_cpu_cores_policy == "specific":
        if specific_build_cores_str:
            cores_for_build_taskset_str = specific_build_cores_str
            build_cores_target_str = f"Specific (cores: {cores_for_build_taskset_str})"
        else:
            logger.warning(
                "Policy is 'specific' but 'specific_build_cores' is not defined in config. Build will not be pinned."
            )
            build_cores_target_str = "All Available (specific_build_cores not set)"

    if cores_for_build_taskset_str:
        build_command_prefix = f"taskset -c {cores_for_build_taskset_str} "

    return build_command_prefix, build_cores_target_str


def determine_monitoring_cpu_affinity(
    monitoring_cores_policy: str,
    num_monitoring_cores: int,
    specific_monitoring_cores: str,
    build_cpu_cores_str: str,
    total_cores_available: int,
) -> Tuple[List[int], str]:
    """
    Determines the list of CPU cores to be used by monitoring worker processes.

    Args:
        monitoring_cores_policy: The policy ('auto', 'specific', 'shared').
        num_monitoring_cores: The desired number of cores if policy is 'auto'.
        specific_monitoring_cores: A string defining cores if policy is 'specific'.
        build_cpu_cores_str: The string of cores used by the build process.
        total_cores_available: The total number of CPU cores on the system.

    Returns:
        A tuple containing:
        - A list of integer core IDs for the monitoring workers.
        - A human-readable description of the allocation.
    """
    if monitoring_cores_policy == "shared" or total_cores_available <= 1:
        return (
            [],
            "Shared with build process (no specific pinning for monitoring workers)",
        )

    all_cores = set(range(total_cores_available))
    build_cores = set()
    if build_cpu_cores_str:
        try:
            # A simple parser for "1,2,4-7" style strings
            for part in build_cpu_cores_str.split(","):
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    build_cores.update(range(start, end + 1))
                else:
                    build_cores.add(int(part))
        except ValueError:
            logger.warning(
                f"Could not parse build_cpu_cores_str '{build_cpu_cores_str}'. Monitoring may overlap."
            )
            build_cores = set()

    available_for_monitoring = sorted(list(all_cores - build_cores))
    final_monitor_cores = []
    desc = ""

    if monitoring_cores_policy == "specific":
        # Logic to parse specific_monitoring_cores string
        if specific_monitoring_cores:
            try:
                for part in specific_monitoring_cores.split(","):
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        final_monitor_cores.extend(range(start, end + 1))
                    else:
                        final_monitor_cores.append(int(part))
                desc = f"Specific cores: {specific_monitoring_cores}"
            except ValueError:
                logger.warning(
                    f"Invalid 'specific_monitoring_cores' format: '{specific_monitoring_cores}'. Falling back to sharing cores."
                )
                return [], "Shared (invalid specific_monitoring_cores format)"
        else:
            logger.warning(
                "Policy is 'specific' but 'specific_monitoring_cores' is empty. Sharing cores."
            )
            return [], "Shared (specific_monitoring_cores is empty)"

    elif monitoring_cores_policy == "auto":
        if num_monitoring_cores == 0:
            # Default to a quarter of available cores, with a minimum of 1.
            num_to_take = max(1, len(available_for_monitoring) // 4)
        else:
            num_to_take = num_monitoring_cores

        final_monitor_cores = available_for_monitoring[:num_to_take]
        core_str = ",".join(map(str, final_monitor_cores))
        desc = f"Auto-selected {len(final_monitor_cores)} cores: [{core_str}]"

    if not final_monitor_cores:
        return [], "Shared (no available cores found for monitoring)"

    return final_monitor_cores, desc


def prepare_command_with_setup(
    main_command: str, setup_command: Optional[str]
) -> Tuple[str, Optional[str]]:
    """
    Prepares a final command string and determines if a specific shell is needed.

    If a setup command is provided, it's prepended to the main command
    using '&&', and '/bin/bash' is designated as the required shell to
    correctly handle the command chaining.

    Args:
        main_command: The primary command to execute.
        setup_command: An optional command to run before the main command.

    Returns:
        A tuple containing:
        - The final command string to be executed.
        - The path to the shell executable ('/bin/bash') if needed, otherwise None.
    """
    if setup_command:
        final_command = f"{setup_command} && {main_command}"
        executable_shell = "/bin/bash"
    else:
        final_command = main_command
        executable_shell = None
    return final_command, executable_shell


def prepare_full_build_command(
    main_command_template: str,
    j_level: int,
    taskset_prefix: str,
    setup_command: Optional[str],
) -> Tuple[str, Optional[str]]:
    """
    Assembles the final, fully-formed build command string from all its parts.

    This function handles formatting the command template with the parallelism
    level, prepending the CPU affinity prefix (`taskset`), and chaining an
    optional setup script.

    Args:
        main_command_template: The command string with a {j_level} placeholder.
        j_level: The parallelism level to inject into the template.
        taskset_prefix: The `taskset -c ...` command prefix, if any.
        setup_command: An optional setup command to run before the main command.

    Returns:
        A tuple containing:
        - The final, executable command string.
        - The path to the shell executable ('/bin/bash') if needed, otherwise None.
    """
    # 1. Format the main command with the parallelism level.
    formatted_main_cmd = main_command_template.format(j_level=j_level)

    # 2. Prepend the taskset prefix for CPU pinning.
    command_with_affinity = taskset_prefix + formatted_main_cmd

    # 3. Use the existing helper to prepend the setup script.
    final_cmd, executable = prepare_command_with_setup(
        command_with_affinity, setup_command
    )

    return final_cmd, executable


def run_command(
    command: str, cwd: Path, shell: bool = False, executable_shell: Optional[str] = None
) -> Tuple[int, str, str]:
    """
    Executes a given command and returns its exit code, stdout, and stderr.

    This is a wrapper around `subprocess.run` for robust command execution.

    Args:
        command: The command string to execute.
        cwd: The working directory for the command.
        shell: If True, execute the command through the shell.
        executable_shell: The path to the shell to use if `shell` is True.

    Returns:
        A tuple containing:
        - The integer exit code of the command.
        - The captured standard output as a string.
        - The captured standard error as a string.
    """
    logger.info(f"Executing command in {cwd}: {command}")
    if shell and executable_shell:
        logger.info(f"Using shell: {executable_shell}")
    try:
        # Use shlex.split for safety when not using a shell.
        cmd_to_run = command if shell else shlex.split(command)
        process = subprocess.run(
            cmd_to_run,
            cwd=cwd,
            capture_output=True,
            text=True,
            shell=shell,
            executable=(executable_shell if shell else None),
            check=False,  # Do not raise exception on non-zero exit codes.
        )
        if process.returncode != 0:
            logger.warning(
                f"Command '{command}' exited with code {process.returncode}."
            )
        return process.returncode, process.stdout, process.stderr
    except Exception as e:
        logger.error(f"Failed to execute command '{command}': {e}", exc_info=True)
        return -1, "", str(e)


# PERFORMANCE FIX: Use an LRU cache to memoize the results of categorization.
# Process command lines are highly repetitive during a build. Caching the
# result of the expensive rule-matching logic for a given command name and
# full command string significantly reduces CPU usage in this hot path.
# A maxsize of 4096 should be more than sufficient to store all unique
# command lines encountered during a large build.
@functools.lru_cache(maxsize=config.get_config().monitor.categorization_cache_size)
def get_process_category(cmd_name: str, cmd_full: str) -> Tuple[str, str]:
    """
    Categorizes a process based on its command name and full command line.

    This function is the core of the process classification engine. It iterates
    through a prioritized list of rules loaded from `rules.toml`. For each
    process, it attempts to match the rules against different attributes of the
    command (e.g., name, full command line). It also handles unwrapping commands
    executed via `sh -c "..."` to classify the actual underlying command.

    Args:
        cmd_name: The base name of the process command (e.g., 'gcc').
        cmd_full: The full command line with all arguments.

    Returns:
        A tuple containing the determined (major_category, minor_category).
        If no rule matches, it returns ('Other', 'Other_<sanitized_cmd_name>').
    """
    # config.get_config().rules returns a List[RuleConfig], pre-sorted by priority.
    rules = config.get_config().rules
    orig_cmd_name = cmd_name
    orig_cmd_full = cmd_full

    # --- Command Unwrapping Logic ---
    unwrapped_cmd_name = orig_cmd_name
    unwrapped_cmd_full = orig_cmd_full
    was_sh_bash_c_call = False

    # This regex is intentionally broad to catch `sh -c`, `bash -c`, `/bin/sh -c`, etc.
    sh_bash_c_pattern = r"^(?:.*/)?(sh|bash)\s+-c\s+"
    is_sh_bash_c_match = re.match(sh_bash_c_pattern, orig_cmd_full)

    if is_sh_bash_c_match:
        was_sh_bash_c_call = True
        # Extract the real command that comes after 'sh -c '
        command_part = orig_cmd_full[is_sh_bash_c_match.end() :]

        # Remove surrounding quotes if they exist (e.g., sh -c "command...")
        if (command_part.startswith('"') and command_part.endswith('"')) or (
            command_part.startswith("'") and command_part.endswith("'")
        ):
            command_part = command_part[1:-1]

        # Heuristically find the actual command by splitting
        try:
            # Use shlex for robustness against nested quotes
            inner_parts = shlex.split(command_part)
            if inner_parts:
                unwrapped_cmd_name = inner_parts[0].split("/")[-1]  # Get base name
                unwrapped_cmd_full = (
                    command_part  # The full command is the unwrapped part
                )
        except ValueError:
            # Fallback for shlex errors
            parts = command_part.split()
            if parts:
                unwrapped_cmd_name = parts[0].split("/")[-1]
                unwrapped_cmd_full = command_part

    # --- Rule Matching Logic ---
    # The `rules` list is already sorted by priority (descending).
    for rule in rules:
        target_value = ""
        if rule.match_field == "current_cmd_name":
            target_value = unwrapped_cmd_name
        elif rule.match_field == "current_cmd_full":
            target_value = unwrapped_cmd_full
        elif rule.match_field == "orig_cmd_name":
            target_value = orig_cmd_name
        elif rule.match_field == "orig_cmd_full":
            target_value = orig_cmd_full
        else:
            continue

        match_found = False
        if rule.match_type == "exact" and rule.pattern is not None:
            match_found = target_value == rule.pattern
        elif rule.match_type == "contains" and rule.pattern is not None:
            match_found = rule.pattern in target_value
        elif rule.match_type == "startswith" and rule.pattern is not None:
            match_found = target_value.startswith(rule.pattern)
        elif rule.match_type == "endswith" and rule.pattern is not None:
            match_found = target_value.endswith(rule.pattern)
        elif rule.match_type == "regex" and rule.pattern is not None:
            match_found = bool(re.search(rule.pattern, target_value))
        elif rule.match_type == "in_list" and rule.patterns is not None:
            match_found = target_value in rule.patterns

        # Special case: Don't match the 'sh'/'bash' process itself if it was just a wrapper.
        if (
            was_sh_bash_c_call
            and rule.major_category == "Scripting"
            and rule.category == "Shell"
        ):
            match_found = False

        if match_found:
            return rule.major_category, rule.category

    # --- Fallback Logic ---
    sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", unwrapped_cmd_name)
    return "Other", f"Other_{sanitized_name}"


def check_pidstat_installed() -> bool:
    """
    Checks if the 'pidstat' command is installed and available in the system PATH.

    Returns:
        True if 'pidstat' is found, False otherwise.
    """
    try:
        # Use 'which' command to check for the existence of 'pidstat' in PATH.
        subprocess.run(
            ["which", "pidstat"],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # This will fail if 'which' returns a non-zero exit code or is not found.
        return False
