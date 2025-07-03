"""
CPU allocation and scheduling utilities.

This module provides intelligent CPU core allocation for build processes and
monitoring workers, supporting both adaptive and manual scheduling policies.
"""

import logging
import math
import shutil
from typing import Optional, Set, Tuple

import psutil

from ..models.runtime import CpuAllocationPlan

logger = logging.getLogger(__name__)


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
