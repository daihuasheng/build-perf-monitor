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
from typing import Optional, Tuple

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
    rules = config.get_config().rules
    orig_cmd_name = cmd_name
    orig_cmd_full = cmd_full

    # --- Command Unwrapping Logic ---
    # This section handles commands wrapped in `sh -c "..."` or `bash -c "..."`.
    # It extracts the inner command to allow for more accurate classification.
    unwrapped_cmd_name = orig_cmd_name
    unwrapped_cmd_full = orig_cmd_full
    was_sh_bash_c_call = False

    sh_bash_c_pattern = r"^(?:.*/)?(sh|bash)\s+-c\s+"
    is_sh_bash_c_match = re.match(sh_bash_c_pattern, orig_cmd_full)

    if is_sh_bash_c_match:
        was_sh_bash_c_call = True
        # Extract the command part after 'sh -c '.
        command_part = orig_cmd_full[is_sh_bash_c_match.end() :]
        # Remove surrounding quotes if they exist.
        if (command_part.startswith('"') and command_part.endswith('"')) or (
            command_part.startswith("'") and command_part.endswith("'")
        ):
            command_part = command_part[1:-1]

        # Strip leading environment variable assignments (e.g., "VAR=val command...").
        processed_command_part = re.sub(
            r"^((?:export\s+)?[A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|\"[^\"]*\"|[^\"'\s]+)\s+)*",
            "",
            command_part.strip(),
        )

        if processed_command_part:
            unwrapped_cmd_full = processed_command_part
            # Use shlex.split for robust parsing of wrapped commands.
            # This correctly handles commands with quoted paths that contain spaces.
            try:
                unwrapped_cmd_parts = shlex.split(unwrapped_cmd_full)
                unwrapped_cmd_name = (
                    unwrapped_cmd_parts[0] if unwrapped_cmd_parts else ""
                )
            except ValueError:
                # Fallback for shlex errors (e.g., unmatched quotes)
                unwrapped_cmd_name = unwrapped_cmd_full.split()[0]

    # Use the potentially unwrapped command for classification.
    current_cmd_name = unwrapped_cmd_name
    # Handle cases where tools like `ccache` create symlinks ending in `.real`.
    if current_cmd_name.endswith(".real"):
        current_cmd_name = current_cmd_name[:-5]
    current_cmd_full = unwrapped_cmd_full

    logger.debug(
        f"--- Starting categorization for: name='{cmd_name}', full='{cmd_full}' ---"
    )
    logger.debug(
        f"    Using effective command: name='{current_cmd_name}', full='{current_cmd_full}'"
    )

    # --- Rule Matching Loop ---
    # Iterate through rules, which are pre-sorted by priority (descending).
    for rule in rules:
        major_category_from_rule = rule.major_category
        minor_category_from_rule = rule.category
        match_field_key = rule.match_field
        match_type = rule.match_type
        pattern_str = rule.pattern
        patterns_list = rule.patterns

        # Determine which command attribute to use for matching based on the rule.
        target_value = ""
        if match_field_key == "current_cmd_name":
            target_value = current_cmd_name
        elif match_field_key == "current_cmd_full":
            target_value = current_cmd_full
        elif match_field_key == "orig_cmd_name":
            target_value = orig_cmd_name
        elif match_field_key == "orig_cmd_full":
            target_value = orig_cmd_full
        else:
            continue  # Skip rule if match_field is invalid.

        # Perform the match based on the rule's match_type.
        match_found = False
        if match_type == "exact" and pattern_str is not None:
            match_found = target_value == pattern_str
        elif match_type == "contains" and pattern_str is not None:
            match_found = pattern_str in target_value
        elif match_type == "startswith" and pattern_str is not None:
            match_found = target_value.startswith(pattern_str)
        elif match_type == "endswith" and pattern_str is not None:
            match_found = target_value.endswith(pattern_str)
        elif match_type == "regex" and pattern_str is not None:
            match_found = bool(re.search(pattern_str, target_value))
        elif match_type == "in_list" and patterns_list is not None:
            match_found = target_value in patterns_list

        # Special case: If a command was `sh -c "..."`, we don't want to categorize
        # the `sh` process itself as "ShellInteractiveOrDirect". We only care about
        # the inner command, which has already been processed.
        if (
            major_category_from_rule == "Scripting"
            and minor_category_from_rule == "ShellInteractiveOrDirect"
            and was_sh_bash_c_call
        ):
            match_found = False

        if match_found:
            logger.debug(
                f"    SUCCESS: Matched rule '{rule.category}' -> ({major_category_from_rule}, {minor_category_from_rule})"
            )
            return major_category_from_rule, minor_category_from_rule

    # --- Fallback Categorization ---
    # If no rule matched, create a generic 'Other' category.
    safe_cmd_name_part = re.sub(r"[^a-zA-Z0-9_.-]", "_", current_cmd_name[:30])
    fallback_category = f"Other_{safe_cmd_name_part}"
    logger.debug(f"--- No rule matched. Categorizing as Other: {fallback_category} ---")
    return "Other", fallback_category


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
