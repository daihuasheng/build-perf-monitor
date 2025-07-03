"""
Process management for the orchestration module.

This module handles process lifecycle management, including starting build processes,
monitoring completion, and comprehensive process termination logic.
"""

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from ..validation import ErrorSeverity, handle_error
from .shared_state import RuntimeState, TimeoutConstants

logger = logging.getLogger(__name__)


class ProcessManager:
    """
    Specialized process lifecycle management and termination logic.
    
    This class extracts all process-related functionality from the original
    BuildRunner class, providing a focused interface for process operations.
    """
    
    def __init__(self, state: RuntimeState):
        self.state = state
        
    def start_build_process(self, command: str, executable: Optional[str], 
                           project_dir: Path, log_files: Dict[str, Any]) -> subprocess.Popen:
        """
        Start the build process with proper logging setup.
        
        Args:
            command: The full build command to execute
            executable: Shell executable (e.g., '/bin/bash') or None
            project_dir: Working directory for the build
            log_files: Dictionary of open log file handles
            
        Returns:
            The started subprocess.Popen object
        """
        logger.info("Starting build process...")
        
        build_process = subprocess.Popen(
            command,
            cwd=project_dir,
            stdout=log_files.get("build_stdout"),
            stderr=log_files.get("build_stderr"),
            shell=True,
            executable=executable,
        )
        
        logger.info(f"Build process started with PID: {build_process.pid} in directory {project_dir}")
        
        # Store in state
        self.state.build_process = build_process
        
        return build_process

    def monitor_build_completion(self) -> int:
        """
        Monitor the build process until completion, handling shutdown requests.
        
        Returns:
            The build process exit code
        """
        if not self.state.build_process:
            raise ValueError("No build process to monitor")
            
        build_exit_code = None
        logger.info("Waiting for build process to complete...")
        
        while build_exit_code is None:
            if self.state.shutdown_requested.is_set():
                logger.warning("Shutdown requested by signal. Terminating build process...")
                self.terminate_process_tree(self.state.build_process.pid, "build process")
                build_exit_code = self.state.build_process.wait()
                logger.info(f"Terminated build process exited with code: {build_exit_code}")
                break

            try:
                build_exit_code = self.state.build_process.wait(timeout=TimeoutConstants.BUILD_WAIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                continue  # Loop again to check shutdown_requested

        logger.info(f"Build process finished with exit code: {build_exit_code}")
        return build_exit_code

    def terminate_process_tree(self, pid: int, name: str) -> None:
        """
        Gracefully terminates a process and all its children with comprehensive cleanup.
        
        This method handles various edge cases including:
        - Process groups and sessions
        - Zombie processes 
        - Race conditions during termination
        - Dynamic child process creation
        """
        if pid <= 0:
            logger.warning(f"Invalid PID {pid} for {name}, skipping termination")
            return
            
        logger.info(f"Starting termination of {name} (PID: {pid}) and its process tree")
        
        try:
            parent = psutil.Process(pid)
            original_parent_status = parent.status()
            logger.debug(f"Parent process {name} status: {original_parent_status}")
            
        except psutil.NoSuchProcess:
            logger.info(f"Process {name} (PID: {pid}) already terminated")
            return
        except psutil.AccessDenied:
            logger.warning(f"Access denied to process {name} (PID: {pid}), attempting force kill")
            self._force_kill_process(pid)
            return
        except Exception as e:
            logger.error(f"Error accessing process {name} (PID: {pid}): {e}")
            return

        # Strategy: Multi-phase termination with escalating force
        phases = [
            {
                "name": "graceful", 
                "signal": "SIGTERM", 
                "timeout": TimeoutConstants.TERMINATION_GRACEFUL_TIMEOUT, 
                "force": False
            },
            {
                "name": "interrupt", 
                "signal": "SIGINT", 
                "timeout": TimeoutConstants.TERMINATION_INTERRUPT_TIMEOUT, 
                "force": False
            }, 
            {
                "name": "force_kill", 
                "signal": "SIGKILL", 
                "timeout": TimeoutConstants.TERMINATION_FORCE_TIMEOUT, 
                "force": True
            },
        ]
        
        for phase_idx, phase in enumerate(phases):
            try:
                # Re-check parent process status before each phase
                if not self._is_process_alive(parent):
                    logger.info(f"Parent process {name} terminated during phase {phase['name']}")
                    break
                    
                # Get current children (they might change between phases)
                children = self._get_process_children(parent)
                all_processes = [parent] + children
                
                if phase_idx == 0:
                    logger.info(f"Phase {phase['name']}: Terminating {name} and {len(children)} children")
                else:
                    logger.info(f"Phase {phase['name']}: {len(all_processes)} processes still running")
                
                # Apply termination signal to all processes
                terminated_pids = self._apply_termination_signal(all_processes, phase)
                
                if not terminated_pids:
                    logger.debug(f"No processes to terminate in phase {phase['name']}")
                    continue
                
                # Wait for processes to terminate
                remaining_processes = self._wait_for_termination(terminated_pids, phase['timeout'])
                
                # Check if termination was successful
                if not remaining_processes:
                    logger.info(f"All processes terminated successfully in phase {phase['name']}")
                    break
                else:
                    logger.warning(f"Phase {phase['name']}: {len(remaining_processes)} processes still alive")
                    if phase_idx == len(phases) - 1:  # Last phase failed
                        self._handle_stubborn_processes(remaining_processes, name)
                        
            except Exception as e:
                logger.error(f"Error in termination phase {phase['name']}: {e}")
                continue
        
        # Final cleanup: handle any remaining zombie processes
        self._cleanup_zombie_processes(pid, name)
        
        # Attempt process group cleanup if the process was a group leader
        self._cleanup_process_group(pid, name)
        
        logger.info(f"Termination process completed for {name} (PID: {pid})")

    def _is_process_alive(self, process: psutil.Process) -> bool:
        """Safely check if a process is still alive and not a zombie."""
        try:
            if not process.is_running():
                return False
            status = process.status()
            return status not in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
        except Exception as e:
            logger.debug(f"Error checking process status: {e}")
            return False

    def _get_process_children(self, parent: psutil.Process) -> List[psutil.Process]:
        """Safely get all children of a process, handling race conditions."""
        children = []
        try:
            # Get children recursively, handling the case where children spawn more children
            for child in parent.children(recursive=True):
                if self._is_process_alive(child):
                    children.append(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Parent or children may have terminated during enumeration
            pass
        except Exception as e:
            logger.warning(f"Error getting process children: {e}")
        
        return children

    def _apply_termination_signal(self, processes: List[psutil.Process], phase: dict) -> List[psutil.Process]:
        """Apply termination signal to a list of processes and return those that were signaled."""
        terminated_pids = []
        signal_name = phase['signal']
        
        for process in processes:
            try:
                if not self._is_process_alive(process):
                    continue
                    
                if phase['force']:
                    process.kill()  # SIGKILL
                else:
                    if signal_name == "SIGTERM":
                        process.terminate()  # SIGTERM
                    elif signal_name == "SIGINT":
                        process.send_signal(signal.SIGINT)  # SIGINT
                    
                terminated_pids.append(process)
                logger.debug(f"Sent {signal_name} to PID {process.pid}")
                
            except psutil.NoSuchProcess:
                # Process already terminated - that's what we want
                continue
            except psutil.AccessDenied:
                logger.warning(f"Access denied sending {signal_name} to PID {process.pid}")
                continue
            except Exception as e:
                logger.warning(f"Error sending {signal_name} to PID {process.pid}: {e}")
                continue
        
        return terminated_pids

    def _wait_for_termination(self, processes: List[psutil.Process], timeout: float) -> List[psutil.Process]:
        """Wait for processes to terminate and return any that are still alive."""
        if not processes:
            return []
            
        try:
            # Use psutil's wait_procs for efficient waiting
            _, still_alive = psutil.wait_procs(processes, timeout=timeout)
            
            # Filter out zombies - they're effectively terminated
            actually_alive = []
            for process in still_alive:
                if self._is_process_alive(process):
                    actually_alive.append(process)
                    
            return actually_alive
            
        except Exception as e:
            logger.warning(f"Error waiting for process termination: {e}")
            # Fallback: manually check each process
            still_alive = []
            for process in processes:
                if self._is_process_alive(process):
                    still_alive.append(process)
            return still_alive

    def _handle_stubborn_processes(self, processes: List[psutil.Process], name: str) -> None:
        """Handle processes that refuse to terminate even after SIGKILL."""
        logger.error(f"Failed to terminate {len(processes)} stubborn processes for {name}")
        
        for process in processes:
            try:
                logger.error(f"Stubborn process: PID {process.pid}, name: {process.name()}, "
                           f"status: {process.status()}, cmdline: {' '.join(process.cmdline()[:3])}")
            except Exception as e:
                logger.error(f"Could not get info for stubborn process PID {process.pid}: {e}")

    def _cleanup_zombie_processes(self, original_pid: int, name: str) -> None:
        """Attempt to clean up any zombie processes related to the terminated process."""
        try:
            # Look for zombie processes that might be children of the original process
            zombie_count = 0
            for proc in psutil.process_iter(['pid', 'ppid', 'status', 'name']):
                try:
                    if proc.info['status'] == psutil.STATUS_ZOMBIE and proc.info['ppid'] == original_pid:
                        zombie_count += 1
                        logger.debug(f"Found zombie child: PID {proc.info['pid']}, name: {proc.info['name']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            if zombie_count > 0:
                logger.info(f"Found {zombie_count} zombie processes related to {name}, they will be cleaned by the OS")
                    
        except Exception as e:
            logger.debug(f"Error during zombie cleanup for {name}: {e}")

    def _cleanup_process_group(self, pid: int, name: str) -> None:
        """Attempt to clean up process group if the terminated process was a group leader."""
        try:
            # Try to kill the process group if it exists
            # This handles cases where the process started its own process group
            os.killpg(pid, signal.SIGKILL)
            logger.debug(f"Sent SIGKILL to process group {pid} for {name}")
        except ProcessLookupError:
            # Process group doesn't exist or already cleaned up
            pass
        except PermissionError:
            logger.debug(f"No permission to kill process group {pid}")
        except Exception as e:
            logger.debug(f"Error cleaning process group {pid}: {e}")

    def _force_kill_process(self, pid: int) -> None:
        """Force kill a single process by PID as last resort."""
        try:
            os.kill(pid, signal.SIGKILL)
            logger.warning(f"Force killed process PID {pid}")
        except ProcessLookupError:
            # Process already gone
            pass
        except Exception as e:
            logger.error(f"Failed to force kill PID {pid}: {e}")