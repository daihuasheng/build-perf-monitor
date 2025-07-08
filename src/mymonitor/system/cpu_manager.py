"""
Consolidated CPU management utilities.

This module combines CPU affinity setting and core allocation planning
to provide a unified interface for CPU management in the monitoring system.
"""

import logging
import math
import platform
import shutil
import subprocess
from typing import Optional, Set, Tuple, Union, List

import psutil

from ..models.runtime import CpuAllocationPlan
from ..validation import handle_error, ErrorSeverity

logger = logging.getLogger(__name__)


class CPUManager:
    """
    Unified CPU management for monitoring system.
    
    This class provides both CPU affinity setting and core allocation planning,
    supporting both psutil-based affinity and taskset-based process binding.
    """
    
    def __init__(self):
        """Initialize CPU manager."""
        self.platform = platform.system().lower()
        self.available_cores = self._get_available_cores()
        self.taskset_available = self._check_taskset_available()
        
    def _get_available_cores(self) -> List[int]:
        """Get list of available CPU cores."""
        try:
            return list(range(psutil.cpu_count(logical=True)))
        except Exception as e:
            logger.warning(f"Failed to get CPU count: {e}")
            return list(range(4))  # Fallback to 4 cores
    
    def _check_taskset_available(self) -> bool:
        """Check if taskset command is available."""
        if self.platform != 'linux':
            return False
            
        try:
            result = subprocess.run(
                ['taskset', '--help'],
                capture_output=True,
                check=False,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            return False
    
    def set_process_affinity(self, process_id: int, core_ids: Union[int, List[int]]) -> bool:
        """
        Set CPU affinity for a process using psutil.
        
        Args:
            process_id: Process ID
            core_ids: CPU core ID(s) to bind to
            
        Returns:
            True if successful, False otherwise
        """
        if isinstance(core_ids, int):
            core_ids = [core_ids]
            
        if not self._validate_core_ids(core_ids):
            return False
            
        try:
            process = psutil.Process(process_id)
            process.cpu_affinity(core_ids)
            return True
        except Exception as e:
            logger.warning(f"Failed to set process affinity for PID {process_id}: {e}")
            return False
    
    def set_thread_affinity(self, core_ids: Union[int, List[int]], thread_id: Optional[int] = None) -> bool:
        """
        Set CPU affinity for a thread.
        
        Args:
            core_ids: CPU core ID(s) to bind to
            thread_id: Thread ID (None for current thread)
            
        Returns:
            True if successful, False otherwise
        """
        if isinstance(core_ids, int):
            core_ids = [core_ids]
            
        if not self._validate_core_ids(core_ids):
            return False
            
        try:
            if self.platform == 'linux':
                return self._set_linux_thread_affinity(core_ids, thread_id)
            elif self.platform == 'windows':
                return self._set_windows_thread_affinity(core_ids, thread_id)
            else:
                logger.warning(f"Thread affinity not supported on platform: {self.platform}")
                return False
        except Exception as e:
            logger.warning(f"Failed to set thread affinity: {e}")
            return False
    
    def plan_cpu_allocation(
        self,
        cores_policy: str,
        cores_string: Optional[str],
        parallelism_level: int,
        monitoring_workers: int = 4
    ) -> CpuAllocationPlan:
        """
        Plan CPU core allocation for build and monitoring processes.
        
        Args:
            cores_policy: CPU allocation policy ('adaptive' or 'manual')
            cores_string: Manual core specification (for manual policy)
            parallelism_level: Number of parallel build jobs
            monitoring_workers: Number of monitoring workers
            
        Returns:
            CpuAllocationPlan with core assignments
        """
        if cores_policy == "manual" and cores_string:
            return self._plan_manual_allocation(cores_string, parallelism_level, monitoring_workers)
        else:
            return self._plan_adaptive_allocation(parallelism_level, monitoring_workers)
    
    def get_taskset_prefix(self, core_ids: List[int]) -> str:
        """
        Get taskset command prefix for process binding.
        
        Args:
            core_ids: List of CPU core IDs
            
        Returns:
            Taskset command prefix or empty string if not available
        """
        if not self.taskset_available or not core_ids:
            return ""
            
        core_str = self._format_core_list(core_ids)
        return f"taskset -c {core_str} " if core_str else ""
    
    def _validate_core_ids(self, core_ids: List[int]) -> bool:
        """Validate core IDs against available cores."""
        if not core_ids:
            return False
            
        for core_id in core_ids:
            if core_id not in self.available_cores:
                logger.warning(f"Invalid core ID: {core_id}, available: {self.available_cores}")
                return False
        return True
    
    def _plan_adaptive_allocation(self, parallelism_level: int, monitoring_workers: int) -> CpuAllocationPlan:
        """Plan adaptive CPU allocation based on available cores."""
        total_cores = len(self.available_cores)
        
        if total_cores <= 2:
            # Very few cores, share everything
            build_cores = self.available_cores
            monitoring_cores = self.available_cores[:1]
        elif total_cores <= 4:
            # Few cores, give most to build
            build_cores = self.available_cores
            monitoring_cores = self.available_cores[-1:]
        else:
            # More cores available, can dedicate some to monitoring
            monitoring_core_count = min(monitoring_workers, max(1, total_cores // 4))
            build_core_count = total_cores - monitoring_core_count
            
            build_cores = self.available_cores[:build_core_count]
            monitoring_cores = self.available_cores[-monitoring_core_count:]
        
        return CpuAllocationPlan(
            build_cores=build_cores,
            monitoring_cores=monitoring_cores,
            build_cores_desc=self._format_core_list(build_cores),
            monitoring_cores_desc=self._format_core_list(monitoring_cores),
            taskset_prefix=self.get_taskset_prefix(build_cores),
            taskset_available=self.taskset_available
        )
    
    def _plan_manual_allocation(self, cores_string: str, parallelism_level: int, monitoring_workers: int) -> CpuAllocationPlan:
        """Plan manual CPU allocation based on user specification."""
        specified_cores = self._parse_core_range_str(cores_string)
        
        if not specified_cores:
            logger.warning(f"Invalid core specification: {cores_string}, falling back to adaptive")
            return self._plan_adaptive_allocation(parallelism_level, monitoring_workers)
        
        # Validate specified cores
        valid_cores = [c for c in specified_cores if c in self.available_cores]
        if not valid_cores:
            logger.warning(f"No valid cores in specification: {cores_string}")
            return self._plan_adaptive_allocation(parallelism_level, monitoring_workers)
        
        # Split cores between build and monitoring
        if len(valid_cores) == 1:
            build_cores = valid_cores
            monitoring_cores = valid_cores
        else:
            # Reserve one core for monitoring if possible
            monitoring_core_count = min(monitoring_workers, max(1, len(valid_cores) // 4))
            build_cores = valid_cores[:-monitoring_core_count] if monitoring_core_count < len(valid_cores) else valid_cores
            monitoring_cores = valid_cores[-monitoring_core_count:]
        
        return CpuAllocationPlan(
            build_cores=build_cores,
            monitoring_cores=monitoring_cores,
            build_cores_desc=self._format_core_list(build_cores),
            monitoring_cores_desc=self._format_core_list(monitoring_cores),
            taskset_prefix=self.get_taskset_prefix(build_cores),
            taskset_available=self.taskset_available
        )
    
    def _parse_core_range_str(self, core_str: str) -> List[int]:
        """Parse core range string like '1,2,4-7' into list of integers."""
        cores = []
        if not core_str or not core_str.strip():
            return cores
            
        try:
            for part in core_str.split(","):
                part = part.strip()
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    cores.extend(range(start, end + 1))
                else:
                    cores.append(int(part))
        except ValueError as e:
            logger.warning(f"Failed to parse core range string '{core_str}': {e}")
            return []
        
        return cores
    
    def _format_core_list(self, cores: List[int]) -> str:
        """Format core list into compact string representation."""
        if not cores:
            return ""
        
        # Sort cores for consistent output
        sorted_cores = sorted(cores)
        
        # Group consecutive cores into ranges
        ranges = []
        start = sorted_cores[0]
        end = start
        
        for i in range(1, len(sorted_cores)):
            if sorted_cores[i] == end + 1:
                end = sorted_cores[i]
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = sorted_cores[i]
                end = start
        
        # Add the last range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        return ",".join(ranges)
    
    def _set_linux_thread_affinity(self, core_ids: List[int], thread_id: Optional[int] = None) -> bool:
        """Set thread affinity on Linux using sched_setaffinity."""
        try:
            import os
            import ctypes
            import ctypes.util
            
            # Get libc
            libc = ctypes.CDLL(ctypes.util.find_library('c'))
            
            # Define cpu_set_t structure
            CPU_SETSIZE = 1024
            cpu_set_t = ctypes.c_ulong * (CPU_SETSIZE // 64)
            
            # Create CPU set
            cpu_set = cpu_set_t()
            
            # Set bits for specified cores
            for core_id in core_ids:
                cpu_set[core_id // 64] |= (1 << (core_id % 64))
            
            # Set affinity
            tid = thread_id if thread_id is not None else 0
            result = libc.sched_setaffinity(tid, ctypes.sizeof(cpu_set), ctypes.byref(cpu_set))
            
            return result == 0
            
        except Exception as e:
            logger.warning(f"Failed to set Linux thread affinity: {e}")
            return False
    
    def _set_windows_thread_affinity(self, core_ids: List[int], thread_id: Optional[int] = None) -> bool:
        """Set thread affinity on Windows using SetThreadAffinityMask."""
        try:
            import ctypes
            from ctypes import wintypes
            
            # Get Windows API
            kernel32 = ctypes.windll.kernel32
            
            # Calculate affinity mask
            affinity_mask = 0
            for core_id in core_ids:
                affinity_mask |= (1 << core_id)
            
            # Get thread handle
            if thread_id is not None:
                thread_handle = kernel32.OpenThread(0x0040, False, thread_id)
            else:
                thread_handle = kernel32.GetCurrentThread()
            
            if thread_handle:
                # Set affinity
                result = kernel32.SetThreadAffinityMask(thread_handle, affinity_mask)
                
                if thread_id is not None:
                    kernel32.CloseHandle(thread_handle)
                
                return result != 0
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to set Windows thread affinity: {e}")
            return False
    
    def parse_core_range_string(self, core_str: str) -> List[int]:
        """
        Public interface for parsing CPU core range strings.
        
        Args:
            core_str: String like "1,2,4-7" or "0-3"
            
        Returns:
            List of CPU core IDs
            
        Example:
            >>> manager = get_cpu_manager()
            >>> manager.parse_core_range_string("1,2,4-7")
            [1, 2, 4, 5, 6, 7]
        """
        return self._parse_core_range_str(core_str)


# Global CPU manager instance
_cpu_manager: Optional[CPUManager] = None


def get_cpu_manager() -> CPUManager:
    """Get the global CPU manager instance."""
    global _cpu_manager
    if _cpu_manager is None:
        _cpu_manager = CPUManager()
    return _cpu_manager


# Convenience functions for backward compatibility
def set_current_thread_affinity(core_ids: Union[int, List[int]]) -> bool:
    """Set CPU affinity for the current thread."""
    return get_cpu_manager().set_thread_affinity(core_ids)


def get_available_cores() -> List[int]:
    """Get list of available CPU cores."""
    return get_cpu_manager().available_cores
