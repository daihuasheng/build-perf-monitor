"""
General utility functions for process and system interaction.

This module provides helpers for:
- Determining CPU affinity for processes.
- Executing external commands.
- Categorizing processes based on rules.
- Checking for system dependencies like 'pidstat'.
"""

import logging
import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Tuple
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
    (Content of the original _determine_build_cpu_affinity function)
    """
    build_command_prefix = ""
    build_cores_target_str = "All Available (taskset not used or policy is 'none')"

    if taskset_available and build_cpu_cores_policy != "none":
        cores_for_build_taskset_str: Optional[str] = None
        if build_cpu_cores_policy == "all_others":
            if total_cores_available and total_cores_available > 1:
                all_cores = set(range(total_cores_available))
                if monitor_core_id is not None and monitor_core_id in all_cores:
                    all_cores.remove(monitor_core_id)

                # Create a comma-separated list of core ranges
                if all_cores:
                    sorted_cores = sorted(list(all_cores))
                    ranges = []
                    start = sorted_cores[0]
                    end = start
                    for i in range(1, len(sorted_cores)):
                        if sorted_cores[i] == end + 1:
                            end = sorted_cores[i]
                        else:
                            ranges.append(
                                f"{start}-{end}" if start != end else str(start)
                            )
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
                build_cores_target_str = (
                    f"Specific (cores: {cores_for_build_taskset_str})"
                )
            else:
                logger.warning(
                    f"Invalid or missing specific_build_cores ('{specific_build_cores_str}') for 'specific' policy. Build will use all available cores via taskset if possible, or no affinity."
                )
                if total_cores_available and total_cores_available > 0:
                    cores_for_build_taskset_str = f"0-{total_cores_available - 1}"
                    build_cores_target_str = (
                        "All Available (invalid specific_build_cores)"
                    )
                else:
                    build_cores_target_str = "All Available (core count unknown, invalid specific_build_cores)"

        if cores_for_build_taskset_str:
            build_command_prefix = f"taskset -c {cores_for_build_taskset_str} "
    elif (
        build_cpu_cores_policy != "none"
    ):  # taskset not available but policy is not 'none'
        build_cores_target_str = "All Available (taskset not available)"

    return build_command_prefix, build_cores_target_str


def run_command(
    command: str, cwd: Path, shell: bool = False, executable_shell: Optional[str] = None
) -> Tuple[int, str, str]:
    """
    Executes a given shell command and returns its exit code, stdout, and stderr.
    (Content of the original run_command function)
    """
    logger.info(f"Executing command in {cwd}: {command}")
    if shell and executable_shell:
        logger.info(f"Using shell: {executable_shell}")
    try:
        cmd_to_run = command if shell else shlex.split(command)
        process = subprocess.run(
            cmd_to_run,
            cwd=cwd,
            capture_output=True,
            text=True,
            shell=shell,
            executable=(executable_shell if shell else None),
            check=False,
        )
        if process.returncode != 0:
            logger.warning(
                f"Command '{command}' exited with code {process.returncode}."
            )
        else:
            logger.debug(f"Command '{command}' executed successfully.")
        return process.returncode, process.stdout, process.stderr
    except Exception as e:
        logger.error(f"Failed to execute command '{command}': {e}", exc_info=True)
        return -1, "", str(e)


def get_process_category(cmd_name: str, cmd_full: str) -> Tuple[str, str]:
    """
    Categorizes a process based on its command name and full command line.
    (Content of the original get_process_category function)
    """
    # Get rules from the new config system
    rules = config.get_config().rules
    orig_cmd_name = cmd_name
    orig_cmd_full = cmd_full

    unwrapped_cmd_name = orig_cmd_name
    unwrapped_cmd_full = orig_cmd_full
    was_sh_bash_c_call = False

    sh_bash_c_pattern = r"^(?:.*/)?(sh|bash)\s+-c\s+"
    is_sh_bash_c_match = re.match(sh_bash_c_pattern, orig_cmd_full)

    if is_sh_bash_c_match:
        was_sh_bash_c_call = True
        command_part = orig_cmd_full[is_sh_bash_c_match.end() :]
        if (command_part.startswith('"') and command_part.endswith('"')) or (
            command_part.startswith("'") and command_part.endswith("'")
        ):
            command_part = command_part[1:-1]

        processed_command_part = re.sub(
            r"^((?:export\s+)?[A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|\"[^\"]*\"|[^\"'\s]+)\s+)*",
            "",
            command_part.strip(),
        )

        if processed_command_part:
            unwrapped_cmd_full = processed_command_part
            unwrapped_cmd_name = unwrapped_cmd_full.split()[0]

    current_cmd_name = unwrapped_cmd_name
    if current_cmd_name.endswith(".real"):
        current_cmd_name = current_cmd_name[:-5]
    current_cmd_full = unwrapped_cmd_full

    # DEBUG BEGIN: Debug log to trace matching process
    logger.debug(f"--- Starting categorization for: name='{cmd_name}', full='{cmd_full}' ---")
    logger.debug(f"    Using effective command: name='{current_cmd_name}', full='{current_cmd_full}'")
    # DEBUG END

    for rule in rules:
        minor_category_from_rule = rule.category
        major_category_from_rule = rule.major_category
        match_field_key = rule.match_field
        match_type = rule.match_type
        pattern_str = rule.pattern
        patterns_list = rule.patterns

        # The dataclass ensures these fields exist, so extensive checking is not needed
        # but we can keep a basic sanity check.
        if not all(
            [
                minor_category_from_rule,
                major_category_from_rule,
                match_field_key,
                match_type,
            ]
        ):
            logger.warning(f"Skipping invalid rule object: {rule}")
            continue

        # DEBUG BEGIN: Detailed debug log inside the loop
        if match_type == "regex" and pattern_str is not None:
            is_match = bool(re.search(pattern_str, target_value))
            logger.debug(
                f"  - Rule '{rule.category}' (regex): "
                f"Target='{target_value}', Pattern='{pattern_str}', Match={is_match}"
            )
            match_found = is_match
        # DEBUG END

        target_value = None
        if match_field_key == "current_cmd_name":
            target_value = current_cmd_name
        elif match_field_key == "current_cmd_full":
            target_value = current_cmd_full
        elif match_field_key == "orig_cmd_name":
            target_value = orig_cmd_name
        elif match_field_key == "orig_cmd_full":
            target_value = orig_cmd_full
        else:
            continue

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

        if (
            major_category_from_rule == "Scripting"
            and minor_category_from_rule == "ShellInteractiveOrDirect"
            and was_sh_bash_c_call
        ):
            match_found = False

        if match_found:
            return major_category_from_rule, minor_category_from_rule

    safe_cmd_name_part = re.sub(r"[^a-zA-Z0-9_.-]", "_", current_cmd_name[:30])
    
    # DEBUG BEGIN: Debug log for when no rule is matched
    logger.debug(f"--- No rule matched. Categorizing as Other: Other_{safe_cmd_name_part} ---")
    # DEBUG END
    
    return "Other", f"Other_{safe_cmd_name_part}"


def check_pidstat_installed() -> bool:
    """
    Checks if the 'pidstat' command is installed and available in the system PATH.
    """
    try:
        subprocess.run(
            ["which", "pidstat"],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
