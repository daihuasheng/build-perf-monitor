"""
Log management for the orchestration module.

This module handles all log file operations including opening, writing,
and safe cleanup of log files used during the build monitoring process.
"""

import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, IO, List, Tuple

from ..models.results import MonitoringResults
from ..models.runtime import RunContext
from .shared_state import RuntimeState

logger = logging.getLogger(__name__)


class LogManager:
    """
    Handles all log file operations for the build monitoring process.
    
    This class centralizes log file management that was previously scattered
    throughout the BuildRunner class methods.
    """
    
    def __init__(self, state: RuntimeState):
        self.state = state
        self.log_files: Dict[str, IO[Any]] = {}
        
    def open_log_files(self, run_context: RunContext) -> None:
        """
        Open all log files required for the run with robust error handling.
        
        Args:
            run_context: The run context containing path information
            
        Raises:
            IOError: If any required log files cannot be opened
        """
        if not run_context:
            raise ValueError("RunContext is required to open log files")
            
        paths = run_context.paths
        log_dir = paths.output_parquet_file.parent
        
        file_paths = {
            "summary_log": paths.output_summary_log_file,
            "build_stdout": log_dir / "build_stdout.log",
            "build_stderr": log_dir / "build_stderr.log", 
            "clean_log": log_dir / "clean.log",
            "metadata_log": log_dir / "metadata.log",
        }
        
        opened_files = {}
        failed_files = []
        
        try:
            for name, path in file_paths.items():
                try:
                    opened_files[name] = open(path, "w", encoding="utf-8")
                    logger.debug(f"Successfully opened log file: {name} -> {path}")
                except Exception as e:
                    failed_files.append((name, path, str(e)))
                    logger.error(f"Failed to open {name} at {path}: {e}")
                    # Don't break here - try to open remaining files and then cleanup
            
            # If any files failed to open, cleanup and raise
            if failed_files:
                self._safe_close_files(opened_files)
                error_details = "; ".join([f"{name}({path}): {error}" for name, path, error in failed_files])
                raise IOError(f"Failed to open {len(failed_files)} log files: {error_details}")
            
            # All files opened successfully
            self.log_files = opened_files
            logger.info(f"Successfully opened {len(opened_files)} log files")
            
        except Exception as e:
            # Ensure cleanup even if failed_files check or assignment fails
            self._safe_close_files(opened_files)
            logger.error(f"Error in log file opening process: {e}")
            raise

    def close_log_files(self) -> None:
        """Close all opened log files using safe cleanup."""
        if not self.log_files:
            logger.debug("No log files to close")
            return
            
        logger.debug(f"Closing {len(self.log_files)} log files")
        self._safe_close_files(self.log_files)
        self.log_files.clear()

    def get_log_file(self, name: str) -> IO[Any]:
        """
        Get a specific log file handle by name.
        
        Args:
            name: Name of the log file (e.g., 'summary_log', 'build_stdout')
            
        Returns:
            The file handle for the requested log file
            
        Raises:
            KeyError: If the log file name is not found
        """
        if name not in self.log_files:
            raise KeyError(f"Log file '{name}' not found. Available: {list(self.log_files.keys())}")
        return self.log_files[name]

    def write_clean_log(self, command: str, return_code: int, stdout: str, stderr: str) -> None:
        """
        Write clean command execution results to the clean log.
        
        Args:
            command: The executed clean command
            return_code: Exit code of the clean command
            stdout: Standard output from the clean command
            stderr: Standard error from the clean command
        """
        log_file = self.log_files.get("clean_log")
        if log_file:
            log_file.write("--- Clean Command Log ---\n")
            log_file.write(f"Command: {command}\n")
            log_file.write(f"Exit Code: {return_code}\n\n")
            log_file.write("--- STDOUT ---\n")
            log_file.write(stdout)
            log_file.write("\n--- STDERR ---\n")
            log_file.write(stderr)
            log_file.flush()
            
    def write_summary_log(self, build_exit_code: int, results: MonitoringResults) -> None:
        """
        Write monitoring results summary to the summary log.
        
        Args:
            build_exit_code: Exit code from the build process
            results: Monitoring results with memory usage data
        """
        summary_log = self.log_files.get("summary_log")
        if summary_log and results:
            summary_log.write(f"build_exit_code={build_exit_code}\n")
            summary_log.write(f"peak_overall_memory_kb={results.peak_overall_memory_kb}\n")
            summary_log.write(f"peak_overall_memory_epoch={results.peak_overall_memory_epoch}\n\n")
            summary_log.write("--- Category Peak Memory Usage ---\n")
            
            sorted_cats = sorted(
                results.category_peak_sum.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            for cat, peak_mem in sorted_cats:
                num_pids = len(results.category_pid_set.get(cat, set()))
                summary_log.write(f"{cat}: {peak_mem} KB ({num_pids} pids)\n")
            summary_log.flush()

    def log_run_prologue(self, run_context: RunContext) -> None:
        """
        Write metadata and initial configuration to the logs.
        
        Args:
            run_context: The run context containing all metadata
        """
        metadata_log = self.log_files.get("metadata_log")
        if metadata_log:
            for field, value in dataclasses.asdict(run_context).items():
                metadata_log.write(f"{field}: {value}\n")
            metadata_log.flush()
            
        logger.info(f"Logs for this run will be in {run_context.paths.output_parquet_file.parent}")

    def _safe_close_files(self, files_dict: Dict[str, Any]) -> None:
        """
        Safely close a dictionary of file handles with individual error handling.
        
        Args:
            files_dict: Dictionary of file name to file handle mappings
        """
        if not files_dict:
            return
            
        closed_count = 0
        failed_count = 0
        
        for name, file_handle in files_dict.items():
            try:
                if file_handle and hasattr(file_handle, 'close'):
                    file_handle.close()
                    closed_count += 1
                    logger.debug(f"Successfully closed log file: {name}")
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to close log file {name}: {e}")
                # Continue trying to close other files
        
        if closed_count > 0:
            logger.debug(f"Closed {closed_count} log files successfully")
        if failed_count > 0:
            logger.warning(f"Failed to close {failed_count} log files")
