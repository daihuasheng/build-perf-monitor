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
    """
    解析shell包装命令，提取实际执行的命令。
    
    这个函数识别常见的shell包装模式（如 sh -c, bash -c等），
    并提取其中实际执行的命令，但排除用shell启动脚本的情况。
    
    Args:
        cmd_name: 进程的命令名称 (如 'sh', 'bash')
        cmd_full: 完整的命令行字符串
        
    Returns:
        Tuple[str, str]: (解析后的命令名称, 解析后的完整命令)
        如果不是shell包装或解析失败，返回原始的(cmd_name, cmd_full)
    """
    
    # 只处理常见的shell命令
    if cmd_name not in ['sh', 'bash', 'zsh', 'dash']:
        return cmd_name, cmd_full
    
    # 检查是否是 shell -c 模式
    if ' -c ' not in cmd_full:
        return cmd_name, cmd_full
    
    try:
        # 使用shlex安全解析命令行
        parts = shlex.split(cmd_full)
        
        # 寻找 -c 参数的位置
        c_index = -1
        for i, part in enumerate(parts):
            if part == '-c' and i + 1 < len(parts):
                c_index = i
                break
        
        if c_index == -1:
            return cmd_name, cmd_full
            
        # 获取 -c 后面的命令字符串
        wrapped_command = parts[c_index + 1]
        
        # 再次解析被包装的命令
        try:
            wrapped_parts = shlex.split(wrapped_command)
        except ValueError:
            # 如果解析失败，可能是复杂的shell语法，保持原样
            return cmd_name, cmd_full
            
        if not wrapped_parts:
            return cmd_name, cmd_full
            
        # 获取被包装命令的基本名称
        wrapped_cmd_name = Path(wrapped_parts[0]).name
        
        # 检查是否是脚本文件 - 如果是，保持原始的shell分类
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
            
        # 检查是否是明显的脚本内容（包含shell语法）
        if any(syntax in wrapped_command for syntax in ['&&', '||', '|', ';', '$(', '`']):
            # 如果包含shell语法，保持原始shell分类，因为这是脚本逻辑
            return cmd_name, cmd_full
                
        # 如果被包装的命令看起来是一个简单的程序调用，返回解析后的结果
        logger.debug(f"Shell wrapper detected: '{cmd_full}' -> unwrapped: '{wrapped_cmd_name}', '{wrapped_command}'")
        return wrapped_cmd_name, wrapped_command
        
    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to parse shell wrapper command '{cmd_full}': {e}")
        return cmd_name, cmd_full


def _parse_core_range_str(core_str: str) -> Set[int]:
    """Parses a core range string (e.g., "1,2,4-7") into a set of integers."""
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
    except ValueError:
        logger.warning(
            f"Could not parse core range string '{core_str}'. Returning empty set."
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
    """
    Creates a comprehensive CPU allocation plan based on the selected policy.

    This is the central function for the intelligent, adaptive CPU scheduling.
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
    """Handles the 'manual' CPU allocation strategy."""
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
    """
    Implements the 'adaptive' CPU allocation strategy (V2.1).
    This version prioritizes giving surplus cores to the build process
    and uses a max(static, proportional) buffer for build cores.
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
    """Formats a set of core integers into a compact string like "0,2-4"."""
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
    """
    Constructs the full build command, including taskset prefix and parallelism.

    Args:
        main_command_template: The build command template (e.g., "make -j{j_level}").
        j_level: The parallelism level.
        taskset_prefix: The prefix command for CPU affinity (e.g., "taskset -c 0-7").
        setup_command: An optional setup command to be sourced.

    Returns:
        A tuple containing the final command string and the executable for shell.
    """
    # Format the core build command with the parallelism level.
    # The template is expected to contain the argument name, e.g., "make -j{j_level}".
    build_command = main_command_template.format(j_level=j_level)

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
    """
    Combines a main command with an optional setup command (e.g., 'source').

    It determines if a specific shell executable is needed.

    Args:
        main_command: The primary command to execute.
        setup_command: An optional setup command to be sourced before the main command.

    Returns:
        A tuple containing the final command string and the executable for shell.
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
    """
    Executes a command and captures its output, handling potential exceptions.

    Args:
        command: The command string to execute.
        cwd: The working directory to run the command in.
        shell: Whether to use a shell for execution.
        executable_shell: The specific shell to use (e.g., '/bin/bash').

    Returns:
        A tuple containing the return code, stdout, and stderr.
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
    except FileNotFoundError:
        logger.error(f"Command not found: {shlex.split(command)[0]}")
        return -1, "", f"Error: Command not found '{shlex.split(command)[0]}'"
    except Exception as e:
        logger.error(f"An unexpected error occurred while running command: {e}")
        return -1, "", f"An unexpected error occurred: {e}"


def get_process_category(cmd_name: str, cmd_full: str) -> Tuple[str, str]:
    """
    Categorizes a process based on a set of rules.

    This function uses an internal cache to avoid re-evaluating the same process
    command line repeatedly, which can significantly speed up monitoring on
    builds that spawn many identical processes.

    The rules are loaded from `rules.toml` and applied in order of priority.

    Args:
        cmd_name: The base name of the command (e.g., 'gcc').
        cmd_full: The full command line with all arguments.

    Returns:
        A tuple of (major_category, minor_category).
        Defaults to ('Unknown', 'Unknown') if no rules match.
    """
    # Check cache first
    cache_key = (cmd_name, cmd_full)
    if cache_key in _categorization_cache:
        return _categorization_cache[cache_key]
    
    app_config = config.get_config()
    
    # 尝试解析shell包装命令
    current_cmd_name, current_cmd_full = parse_shell_wrapper_command(cmd_name, cmd_full)

    for rule in app_config.rules:
        target_field_value = (
            current_cmd_name
            if rule.match_field == "current_cmd_name"
            else current_cmd_full
        )

        match = False
        if rule.match_type == "exact":
            match = target_field_value == rule.pattern
        elif rule.match_type == "contains":
            match = rule.pattern and rule.pattern in target_field_value
        elif rule.match_type == "regex":
            if rule.pattern and re.search(rule.pattern, target_field_value):
                match = True
        elif rule.match_type == "in_list":
            if rule.patterns and target_field_value in rule.patterns:
                match = True
        elif rule.match_type == "starts_with":
            if rule.pattern and target_field_value.startswith(rule.pattern):
                match = True
        elif rule.match_type == "endswith":
            if rule.pattern and target_field_value.endswith(rule.pattern):
                match = True

        if match:
            result = (rule.major_category, rule.category)
            # Cache the result, but respect the cache size limit
            if len(_categorization_cache) < app_config.monitor.categorization_cache_size:
                _categorization_cache[cache_key] = result
            return result

    # 如果没有规则匹配，生成基于原始命令名称的分类
    result = ("Other", f"Other_{current_cmd_name}")
    # Cache the result, but respect the cache size limit
    if len(_categorization_cache) < app_config.monitor.categorization_cache_size:
        _categorization_cache[cache_key] = result
    return result


def check_pidstat_installed() -> bool:
    """Checks if the 'pidstat' command is available on the system."""
    return shutil.which("pidstat") is not None
