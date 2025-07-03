"""
Build execution management.

This module provides functionality to execute build processes with proper
setup, monitoring coordination, and cleanup.
"""

import logging
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

from ..models.runtime import RunContext
from ..system.commands import run_command, prepare_full_build_command
from ..validation import handle_error, handle_subprocess_error, ErrorSeverity

logger = logging.getLogger(__name__)


class BuildExecutor:
    """
    Manages the execution of build processes.
    
    This class handles the lifecycle of build process execution including
    command preparation, process startup, monitoring coordination, and cleanup.
    """
    
    def __init__(self, run_context: RunContext):
        """
        Initialize the build executor.
        
        Args:
            run_context: Runtime context for the build execution
        """
        self.run_context = run_context
        self.build_process: Optional[subprocess.Popen] = None
        self.build_command: Optional[str] = None
        self.shell_executable: Optional[str] = None
        
    def prepare_build(self, taskset_prefix: str = "") -> None:
        """
        Prepare the build command with all necessary components.
        
        Args:
            taskset_prefix: CPU affinity prefix for build process
        """
        try:
            # Prepare the complete build command
            self.build_command, self.shell_executable = prepare_full_build_command(
                main_command_template=self.run_context.project_config.build_command_template,
                j_level=self.run_context.parallelism_level,
                taskset_prefix=taskset_prefix,
                setup_command=self.run_context.project_config.setup_command_template
            )
            
            logger.info(f"Prepared build command: {self.build_command}")
            
        except Exception as e:
            handle_error(
                error=e,
                context="preparing build command",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger
            )
    
    def start_build(self) -> int:
        """
        Start the build process.
        
        Returns:
            Process ID of the started build process
            
        Raises:
            RuntimeError: If build preparation failed or process couldn't start
        """
        if not self.build_command:
            raise RuntimeError("Build not prepared - call prepare_build() first")
        
        try:
            logger.info("Starting build process...")
            
            # Start the build process
            self.build_process = subprocess.Popen(
                self.build_command,
                cwd=self.run_context.project_dir,
                shell=True,
                executable=self.shell_executable,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            build_pid = self.build_process.pid
            self.run_context.build_process_pid = build_pid
            
            logger.info(f"Build process started with PID {build_pid}")
            return build_pid
            
        except Exception as e:
            handle_subprocess_error(
                error=e,
                command=self.build_command or "unknown",
                reraise=True,
                logger=logger
            )
            raise
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> Tuple[int, str]:
        """
        Wait for the build process to complete and capture output.
        
        Args:
            timeout: Maximum time to wait in seconds (None for no timeout)
            
        Returns:
            Tuple of (return_code, combined_output)
            
        Raises:
            RuntimeError: If no build process is running
            subprocess.TimeoutExpired: If timeout is exceeded
        """
        if not self.build_process:
            raise RuntimeError("No build process is running")
        
        try:
            logger.info("Waiting for build process to complete...")
            
            # Wait for process completion and capture output
            stdout, stderr = self.build_process.communicate(timeout=timeout)
            return_code = self.build_process.returncode
            
            # Combine stdout and stderr (stderr should be empty due to redirection)
            combined_output = stdout or ""
            if stderr:
                combined_output += f"\n--- STDERR ---\n{stderr}"
            
            if return_code == 0:
                logger.info(f"Build completed successfully (exit code: {return_code})")
            else:
                logger.warning(f"Build completed with errors (exit code: {return_code})")
            
            return return_code, combined_output
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"Build process timed out after {timeout} seconds")
            self.terminate_build()
            raise
        except Exception as e:
            handle_error(
                error=e,
                context="waiting for build completion",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger
            )
            raise
    
    def terminate_build(self, grace_period: float = 10.0) -> None:
        """
        Terminate the build process gracefully with fallback to force kill.
        
        Args:
            grace_period: Time to wait for graceful shutdown before force kill
        """
        if not self.build_process:
            logger.debug("No build process to terminate")
            return
        
        try:
            if self.build_process.poll() is None:  # Process is still running
                logger.info(f"Terminating build process {self.build_process.pid}")
                
                # First, try graceful termination
                self.build_process.terminate()
                
                try:
                    # Wait for graceful shutdown
                    self.build_process.wait(timeout=grace_period)
                    logger.info("Build process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination failed
                    logger.warning("Graceful termination failed, force killing build process")
                    self.build_process.kill()
                    self.build_process.wait(timeout=5.0)
                    logger.info("Build process force killed")
            else:
                logger.debug("Build process already terminated")
                
        except Exception as e:
            handle_error(
                error=e,
                context="terminating build process",
                severity=ErrorSeverity.WARNING,
                reraise=False,
                logger=logger
            )
    
    def cleanup(self) -> None:
        """
        Clean up build process resources.
        """
        if self.build_process:
            self.terminate_build()
            self.build_process = None
            self.run_context.build_process_pid = None
        
        logger.debug("Build executor cleanup completed")
    
    def is_running(self) -> bool:
        """
        Check if the build process is currently running.
        
        Returns:
            True if build process is running, False otherwise
        """
        if not self.build_process:
            return False
        return self.build_process.poll() is None
    
    def get_pid(self) -> Optional[int]:
        """
        Get the process ID of the build process.
        
        Returns:
            Build process PID or None if no process is running
        """
        if self.build_process:
            return self.build_process.pid
        return None


class BuildCleaner:
    """
    Handles pre-build and post-build cleanup operations.
    """
    
    def __init__(self, run_context: RunContext):
        """
        Initialize the build cleaner.
        
        Args:
            run_context: Runtime context for the build
        """
        self.run_context = run_context
    
    def pre_clean(self, skip_clean: bool = False) -> bool:
        """
        Perform pre-build cleanup if configured and not skipped.
        
        Args:
            skip_clean: Whether to skip the cleanup step
            
        Returns:
            True if cleanup was successful or skipped, False if cleanup failed
        """
        if skip_clean:
            logger.info("Pre-build cleanup skipped per request")
            return True
        
        clean_command = self.run_context.project_config.clean_command_template
        if not clean_command or not clean_command.strip():
            logger.info("No pre-build cleanup command configured")
            return True
        
        try:
            logger.info(f"Running pre-build cleanup: {clean_command}")
            
            return_code, stdout, stderr = run_command(
                command=clean_command.strip(),
                cwd=self.run_context.project_dir,
                shell=True
            )
            
            if return_code == 0:
                logger.info("Pre-build cleanup completed successfully")
                return True
            else:
                logger.warning(f"Pre-build cleanup failed (exit code: {return_code})")
                if stderr:
                    logger.warning(f"Cleanup stderr: {stderr}")
                return False
                
        except Exception as e:
            handle_error(
                error=e,
                context="pre-build cleanup",
                severity=ErrorSeverity.WARNING,
                reraise=False,
                logger=logger
            )
            return False
