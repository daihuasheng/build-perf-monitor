"""
Asynchronous build runner for executing builds with monitoring support.

This module provides an AsyncBuildRunner that can execute build processes
while coordinating with async monitoring, supporting CPU affinity and
proper resource management.
"""

import asyncio
import logging
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from ..system.cpu_manager import get_cpu_manager
from ..validation import handle_error, ErrorSeverity

logger = logging.getLogger(__name__)


class AsyncBuildRunner:
    """
    Asynchronous build runner that coordinates build execution with monitoring.
    
    This class manages the execution of build processes with proper CPU affinity,
    timeout handling, and integration with async monitoring systems.
    """
    
    def __init__(
        self,
        build_command: str,
        build_directory: Path,
        build_cores: Optional[List[int]] = None,
        timeout: float = 3600.0,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        """
        Initialize the async build runner.
        
        Args:
            build_command: The build command to execute
            build_directory: Directory where to run the build
            build_cores: CPU cores to bind the build process to
            timeout: Maximum time to wait for build completion (seconds)
            executor: ThreadPoolExecutor to use for async operations
        """
        self.build_command = build_command
        self.build_directory = Path(build_directory)
        self.build_cores = build_cores or []
        self.timeout = timeout
        self.executor = executor
        
        # Process management
        self.process: Optional[subprocess.Popen] = None
        self.build_pid: Optional[int] = None
        self.is_running = False
        self.return_code: Optional[int] = None
        self.stdout_data: Optional[str] = None
        self.stderr_data: Optional[str] = None
        
        # Timing tracking
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Cancellation support
        self._cancelled = False
        self._cancel_event = asyncio.Event()
        
    async def start_build_async(self) -> int:
        """
        Start the build process asynchronously.
        
        Returns:
            Process ID of the started build process
            
        Raises:
            RuntimeError: If build is already running or fails to start
        """
        if self.is_running:
            raise RuntimeError("Build is already running")
            
        try:
            self.is_running = True
            self._cancelled = False
            self._cancel_event.clear()
            
            # Record start time
            self.start_time = time.time()
            
            # Start the build process in a thread pool
            loop = asyncio.get_event_loop()
            
            if self.executor:
                self.process, self.build_pid = await loop.run_in_executor(
                    self.executor, self._start_build_process
                )
            else:
                self.process, self.build_pid = await loop.run_in_executor(
                    None, self._start_build_process
                )
                
            logger.info(f"Build process started with PID {self.build_pid}")
            return self.build_pid
            
        except Exception as e:
            self.is_running = False
            handle_error(
                error=e,
                context="starting async build process",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger
            )
    
    async def wait_for_completion_async(self) -> int:
        """
        Wait for the build process to complete asynchronously.
        
        Returns:
            Return code of the build process
            
        Raises:
            asyncio.TimeoutError: If build times out
            RuntimeError: If build is not running
        """
        if not self.is_running or not self.process:
            raise RuntimeError("Build is not running")
            
        try:
            # Wait for process completion with timeout and cancellation support
            loop = asyncio.get_event_loop()
            
            # Create a task to wait for the process
            wait_task = asyncio.create_task(
                self._wait_for_process_async()
            )
            
            # Create a task to wait for cancellation
            cancel_task = asyncio.create_task(
                self._cancel_event.wait()
            )
            
            # Wait for either completion or cancellation
            done, pending = await asyncio.wait(
                [wait_task, cancel_task],
                timeout=self.timeout,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Check what completed
            if cancel_task in done:
                # Build was cancelled
                await self._terminate_build_process()
                raise asyncio.CancelledError("Build was cancelled")
            elif wait_task in done:
                # Build completed normally
                self.return_code = await wait_task
            else:
                # Timeout occurred
                await self._terminate_build_process()
                raise asyncio.TimeoutError(f"Build timed out after {self.timeout} seconds")
                
            # Record end time
            self.end_time = time.time()
            logger.info(f"Build completed with return code {self.return_code}")
            return self.return_code
            
        except Exception as e:
            if not isinstance(e, (asyncio.TimeoutError, asyncio.CancelledError)):
                handle_error(
                    error=e,
                    context="waiting for async build completion",
                    severity=ErrorSeverity.ERROR,
                    reraise=True,
                    logger=logger
                )
            raise
        finally:
            self.is_running = False
    
    async def cancel_build_async(self) -> None:
        """
        Cancel the running build process asynchronously.
        """
        if not self.is_running:
            logger.warning("No build process to cancel")
            return
            
        try:
            self._cancelled = True
            self._cancel_event.set()
            
            logger.info("Build cancellation requested")
            
        except Exception as e:
            handle_error(
                error=e,
                context="cancelling async build",
                severity=ErrorSeverity.WARNING,
                reraise=False,
                logger=logger
            )
    
    async def get_build_output_async(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the build output asynchronously.
        
        Returns:
            Tuple of (stdout, stderr) or (None, None) if not available
        """
        if not self.process:
            return None, None
            
        try:
            loop = asyncio.get_event_loop()
            
            if self.executor:
                stdout, stderr = await loop.run_in_executor(
                    self.executor, self._get_process_output
                )
            else:
                stdout, stderr = await loop.run_in_executor(
                    None, self._get_process_output
                )
                
            self.stdout_data = stdout
            self.stderr_data = stderr
            
            return stdout, stderr
            
        except Exception as e:
            handle_error(
                error=e,
                context="getting async build output",
                severity=ErrorSeverity.WARNING,
                reraise=False,
                logger=logger
            )
            return None, None
    
    def _start_build_process(self) -> Tuple[subprocess.Popen, int]:
        """
        Start the build process (synchronous version for thread pool).
        
        Returns:
            Tuple of (process, pid)
        """
        # Set CPU affinity for the current process if specified
        if self.build_cores:
            cpu_manager = get_cpu_manager()
            success = cpu_manager.set_process_affinity(os.getpid(), self.build_cores)
            if success:
                logger.debug(f"Build process bound to cores {self.build_cores}")
            else:
                logger.warning(f"Failed to bind build process to cores {self.build_cores}")
        
        # Prepare environment
        env = os.environ.copy()
        
        # Start the build process
        process = subprocess.Popen(
            self.build_command,
            shell=True,
            cwd=self.build_directory,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,  # Line buffered
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Create new process group
        )
        
        return process, process.pid
    
    async def _wait_for_process_async(self) -> int:
        """
        Wait for the process to complete asynchronously.
        
        Returns:
            Process return code
        """
        if not self.process:
            raise RuntimeError("No process to wait for")
            
        loop = asyncio.get_event_loop()
        
        if self.executor:
            return_code = await loop.run_in_executor(
                self.executor, self.process.wait
            )
        else:
            return_code = await loop.run_in_executor(
                None, self.process.wait
            )
            
        return return_code
    
    async def _terminate_build_process(self) -> None:
        """
        Terminate the build process forcefully.
        """
        if not self.process:
            return
            
        try:
            # Try graceful termination first
            if hasattr(os, 'killpg') and self.process.pid:
                # Kill the entire process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
                
            # Wait a bit for graceful termination
            try:
                await asyncio.wait_for(
                    self._wait_for_process_async(),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                # Force kill if graceful termination failed
                if hasattr(os, 'killpg') and self.process.pid:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
                    
                # Wait for final termination
                await self._wait_for_process_async()
                
            logger.info("Build process terminated")
            
        except Exception as e:
            logger.warning(f"Error terminating build process: {e}")
    
    def _get_process_output(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get process output (synchronous version for thread pool).
        
        Returns:
            Tuple of (stdout, stderr)
        """
        if not self.process:
            return None, None
            
        try:
            stdout, stderr = self.process.communicate()
            return stdout, stderr
        except Exception as e:
            logger.warning(f"Error getting process output: {e}")
            return None, None
    
    def get_build_pid(self) -> Optional[int]:
        """
        Get the build process PID.
        
        Returns:
            Process PID or None if not running
        """
        return self.build_pid
    
    def is_build_running(self) -> bool:
        """
        Check if the build is currently running.
        
        Returns:
            True if build is running, False otherwise
        """
        return self.is_running
    
    def get_return_code(self) -> Optional[int]:
        """
        Get the build return code.
        
        Returns:
            Return code or None if build hasn't completed
        """
        return self.return_code
    
    @property
    def _process_pid(self) -> Optional[int]:
        """Get the process PID for compatibility."""
        return self.build_pid
    
    @property
    def _return_code(self) -> Optional[int]:
        """Get the return code for compatibility."""
        return self.return_code
    
    @property
    def _duration_seconds(self) -> float:
        """Get the build duration in seconds."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.is_running:
            await self.cancel_build_async()


class AsyncBuildRunnerFactory:
    """
    Factory for creating async build runners with various configurations.
    """
    
    @staticmethod
    def create_runner(
        build_command: str,
        build_directory: Path,
        parallelism_level: int = 1,
        timeout: float = 3600.0,
        cpu_scheduling_policy: str = "adaptive",
        executor: Optional[ThreadPoolExecutor] = None,
        **kwargs
    ) -> AsyncBuildRunner:
        """
        Create an async build runner with appropriate CPU allocation.
        
        Args:
            build_command: The build command to execute
            build_directory: Directory where to run the build
            parallelism_level: Build parallelism level (-j value)
            timeout: Build timeout in seconds
            cpu_scheduling_policy: CPU scheduling policy ("adaptive" or "manual")
            executor: ThreadPoolExecutor to use
            **kwargs: Additional arguments for build runner
            
        Returns:
            Configured AsyncBuildRunner instance
        """
        # Determine CPU cores for build process
        build_cores = None
        
        if cpu_scheduling_policy == "adaptive":
            build_cores = AsyncBuildRunnerFactory._get_adaptive_build_cores(parallelism_level)
        elif cpu_scheduling_policy == "manual":
            manual_cores = kwargs.get('manual_build_cores', '')
            if manual_cores:
                build_cores = AsyncBuildRunnerFactory._parse_core_list(manual_cores)
        
        return AsyncBuildRunner(
            build_command=build_command,
            build_directory=build_directory,
            build_cores=build_cores,
            timeout=timeout,
            executor=executor
        )
    
    @staticmethod
    def _get_adaptive_build_cores(parallelism_level: int) -> List[int]:
        """
        Get adaptive CPU core allocation for build process.
        
        Args:
            parallelism_level: Build parallelism level
            
        Returns:
            List of CPU core IDs for build process
        """
        available_cores = get_cpu_manager().available_cores
        total_cores = len(available_cores)
        
        if total_cores <= 2:
            # On systems with few cores, use all available
            return available_cores
        
        # Reserve core 0 for monitoring, allocate others for build
        monitor_cores = 1
        build_cores_count = min(parallelism_level, total_cores - monitor_cores)
        
        # Allocate cores starting from core 1
        build_cores = available_cores[monitor_cores:monitor_cores + build_cores_count]
        
        return build_cores
    
    @staticmethod
    def _parse_core_list(core_string: str) -> List[int]:
        """
        Parse a core list string into a list of integers.
        
        Args:
            core_string: String like "1,2,4-7"
            
        Returns:
            List of CPU core IDs
        """
        cores = []
        
        for part in core_string.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                cores.extend(range(start, end + 1))
            else:
                cores.append(int(part))
        
        return cores


# Convenience function for simple usage
async def run_build_async(
    build_command: str,
    build_directory: Path,
    parallelism_level: int = 1,
    timeout: float = 3600.0,
    **kwargs
) -> Tuple[int, Optional[str], Optional[str]]:
    """
    Run a build command asynchronously and return the results.
    
    Args:
        build_command: The build command to execute
        build_directory: Directory where to run the build
        parallelism_level: Build parallelism level
        timeout: Build timeout in seconds
        **kwargs: Additional arguments for build runner
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    runner = AsyncBuildRunnerFactory.create_runner(
        build_command=build_command,
        build_directory=build_directory,
        parallelism_level=parallelism_level,
        timeout=timeout,
        **kwargs
    )
    
    async with runner:
        await runner.start_build_async()
        return_code = await runner.wait_for_completion_async()
        stdout, stderr = await runner.get_build_output_async()
        
        return return_code, stdout, stderr
