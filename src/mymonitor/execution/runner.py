"""
Build execution management.

This module provides classes for managing build process execution,
including command preparation, process lifecycle management, and cleanup.
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

from ..models.config import ProjectConfig
from ..models.runtime import RunContext
from ..system import prepare_full_build_command, run_command
from ..validation import (
    handle_error, 
    handle_subprocess_error, 
    ErrorSeverity,
    get_error_recovery_strategy,
    with_error_recovery
)

logger = logging.getLogger(__name__)


class BuildExecutor:
    """
    Manages build process execution with enhanced error handling.
    
    This class handles the preparation and execution of build commands,
    including subprocess management and error recovery.
    """
    
    def __init__(self, run_context: RunContext, project_config: ProjectConfig):
        """
        Initialize the build executor.
        
        Args:
            run_context: Runtime context for the build execution
            project_config: Project configuration with build templates
        """
        self.run_context = run_context
        self.project_config = project_config
        self.build_process: Optional[subprocess.Popen] = None
        self.build_command: Optional[str] = None
        self.error_recovery_strategy = get_error_recovery_strategy("process_operations")
        
    def prepare_build(self, build_command_prefix: str) -> None:
        """
        Prepare the build command with error recovery.
        
        Args:
            build_command_prefix: Prefix for the build command (e.g., taskset)
        """
        try:
            self.build_command = self.error_recovery_strategy.execute(
                self._prepare_build_command_internal,
                build_command_prefix,
                error_context="preparing build command",
                logger_instance=logger
            )
        except Exception as e:
            handle_error(
                error=e,
                context="preparing build command",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger
            )
    
    def _prepare_build_command_internal(self, build_command_prefix: str) -> str:
        """Internal method for preparing build command."""
        build_command, _ = prepare_full_build_command(
            main_command_template=self.project_config.build_command_template,
            j_level=self.run_context.parallelism_level,
            taskset_prefix=build_command_prefix,
            setup_command=self.project_config.setup_command_template
        )
        
        # Update run context with actual build command
        self.run_context.actual_build_command = build_command
        
        logger.info(f"Build command prepared: {build_command}")
        return build_command
    
    @with_error_recovery("process_operations", "starting build process")
    def start_build(self) -> int:
        """
        Start the build process with error recovery.
        
        Returns:
            Process ID of the started build process
            
        Raises:
            RuntimeError: If build command is not prepared or process fails to start
        """
        if not self.build_command:
            raise RuntimeError("Build command not prepared - call prepare_build() first")
        
        try:
            # Start the build process
            self.build_process = subprocess.Popen(
                self.build_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.run_context.project_dir,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Update run context with process PID
            self.run_context.build_process_pid = self.build_process.pid
            
            logger.info(f"Build process started with PID {self.build_process.pid}")
            return self.build_process.pid
            
        except Exception as e:
            handle_subprocess_error(
                error=e,
                command=self.build_command,
                reraise=True,
                logger=logger
            )
            raise
    
    def wait_for_completion(self) -> Tuple[int, str]:
        """
        Wait for the build process to complete and capture output.
        
        Returns:
            Tuple of (return_code, output_text)
            
        Raises:
            RuntimeError: If build process is not running
        """
        if not self.build_process:
            raise RuntimeError("Build process not started - call start_build() first")
        
        try:
            # Use the error recovery strategy for the wait operation
            return self.error_recovery_strategy.execute(
                self._wait_for_completion_internal,
                error_context="waiting for build completion",
                logger_instance=logger
            )
        except Exception as e:
            handle_error(
                error=e,
                context="waiting for build process completion",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger
            )
            raise
    
    def _wait_for_completion_internal(self) -> Tuple[int, str]:
        """Internal method for waiting for build completion."""
        # Type assertion since we've already checked that build_process is not None
        assert self.build_process is not None
        
        try:
            # Wait for process to complete and capture output
            stdout, stderr = self.build_process.communicate()
            return_code = self.build_process.returncode
            
            # Combine stdout and stderr
            output = stdout or ""
            if stderr:
                output += f"\n--- STDERR ---\n{stderr}"
            
            logger.info(f"Build process completed with return code {return_code}")
            return return_code, output
            
        except Exception as e:
            # If communication fails, try to terminate the process
            if self.build_process:
                try:
                    self.build_process.terminate()
                    self.build_process.wait(timeout=5)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        self.build_process.kill()
                    except ProcessLookupError:
                        pass
            raise
    
    def cleanup(self) -> None:
        """
        Clean up build process resources with error recovery.
        """
        if self.build_process:
            try:
                self.error_recovery_strategy.execute(
                    self._cleanup_process,
                    error_context="cleaning up build process",
                    logger_instance=logger
                )
            except Exception as e:
                handle_error(
                    error=e,
                    context="cleaning up build process",
                    severity=ErrorSeverity.WARNING,
                    reraise=False,
                    logger=logger
                )
    
    def _cleanup_process(self) -> None:
        """Internal method for cleaning up build process."""
        if self.build_process:
            if self.build_process.poll() is None:
                # Process is still running, terminate it
                logger.info(f"Terminating build process {self.build_process.pid}")
                self.build_process.terminate()
                
                # Wait for graceful termination
                try:
                    self.build_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    logger.warning(f"Force killing build process {self.build_process.pid}")
                    self.build_process.kill()
                    self.build_process.wait(timeout=2)
            
            self.build_process = None
            logger.debug("Build process cleanup completed")


class BuildCleaner:
    """
    Manages build cleanup operations with enhanced error handling.
    
    This class handles pre-build and post-build cleanup operations
    with intelligent error recovery.
    """
    
    def __init__(self, run_context: RunContext, project_config: ProjectConfig):
        """
        Initialize the build cleaner.
        
        Args:
            run_context: Runtime context for the cleanup operations
            project_config: Project configuration with cleanup templates
        """
        self.run_context = run_context
        self.project_config = project_config
        self.error_recovery_strategy = get_error_recovery_strategy("file_operations")
        
    def pre_clean(self, skip_pre_clean: bool = False) -> bool:
        """
        Execute pre-build cleanup with error recovery.
        
        Args:
            skip_pre_clean: Whether to skip the pre-clean step
            
        Returns:
            True if cleanup was successful, False if it failed
        """
        if skip_pre_clean:
            logger.info("Skipping pre-build cleanup as requested")
            return True
        
        if not self.project_config.clean_command_template:
            logger.info("No clean command template defined, skipping pre-build cleanup")
            return True
        
        try:
            return self.error_recovery_strategy.execute(
                self._execute_clean_command,
                error_context="executing pre-build cleanup",
                logger_instance=logger
            )
        except Exception as e:
            handle_error(
                error=e,
                context="pre-build cleanup",
                severity=ErrorSeverity.WARNING,
                reraise=False,
                logger=logger
            )
            return False
    
    def _execute_clean_command(self) -> bool:
        """Internal method for executing clean command."""
        clean_command = self.project_config.clean_command_template
        setup_command = self.project_config.setup_command_template
        
        # Prepare the full clean command
        if setup_command:
            full_command = f"{setup_command} && {clean_command}"
        else:
            full_command = clean_command
        
        logger.info(f"Executing clean command: {full_command}")
        
        # Execute the clean command
        return_code, stdout, stderr = run_command(
            command=full_command,
            cwd=self.run_context.project_dir,
            shell=True
        )
        
        # Log the results
        if return_code == 0:
            logger.info("Clean command executed successfully")
            if stdout:
                logger.debug(f"Clean command stdout: {stdout}")
            return True
        else:
            logger.warning(f"Clean command failed with return code {return_code}")
            if stderr:
                logger.warning(f"Clean command stderr: {stderr}")
            return False
 