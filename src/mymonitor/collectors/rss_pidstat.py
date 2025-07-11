"""
Memory collector implementation using the 'pidstat' command-line utility.

This module provides the RssPidstatCollector class, which utilizes the `pidstat`
tool (part of the sysstat package) to gather memory metrics like RSS (Resident Set Size)
and VSZ (Virtual Set Size) for processes matching a specified pattern.
"""

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Iterable,
    IO,
    Union,
)

from .base import AbstractMemoryCollector, ProcessMemorySample

logger = logging.getLogger(__name__)


class RssPidstatCollector(AbstractMemoryCollector):
    """
    Collects RSS (Resident Set Size) and VSZ (Virtual Set Size) memory metrics
    using the `pidstat` command.

    This collector launches a `pidstat` subprocess to monitor processes that match
    the provided `process_pattern`. It parses the stdout of `pidstat` to extract
    memory information at specified intervals.

    Attributes:
        PIDSTAT_METRIC_FIELDS: A list of metric field names provided by this collector.
        pidstat_proc: The subprocess.Popen object for the running pidstat command.
        pidstat_stderr_file: Optional path to a file where pidstat's stderr output
                             will be redirected.
        _pidstat_stderr_handle: Optional file handle for the pidstat_stderr_file.
        _stop_lock: Threading lock for safe stop operations.
        _is_stopping: Flag to prevent multiple concurrent stop calls.
        _start_time: Timestamp when the collector was started.
    """

    PIDSTAT_METRIC_FIELDS: List[str] = ["RSS_KB", "VSZ_KB"]
    """Defines the memory metric fields collected by this implementation."""

    def __init__(self, process_pattern: str, monitoring_interval: float, **kwargs):
        """
        Initializes the RssPidstatCollector with enhanced thread safety.

        Args:
            process_pattern: A regex pattern to match processes.
            monitoring_interval: Time interval between pidstat reports.
            **kwargs: Additional keyword arguments, including 'pidstat_stderr_file'.
        """
        super().__init__(process_pattern, monitoring_interval, **kwargs)
        self.pidstat_proc: Optional[subprocess.Popen] = None
        """The `pidstat` subprocess instance, or None if not started."""
        self.pidstat_stderr_file: Optional[Path] = kwargs.get("pidstat_stderr_file")
        """Optional path to redirect `pidstat` stderr output."""
        self._pidstat_stderr_handle: Optional[IO[Any]] = (
            None  # Type hint for file handle
        )
        """File handle for the `pidstat_stderr_file` if provided."""
        
        # Enhanced thread safety and state management
        self._stop_lock = threading.Lock()
        self._is_stopping = False
        self._start_time: Optional[float] = None

    def get_metric_fields(self) -> List[str]:
        """
        Returns the list of memory metric field names provided by this collector.

        Returns:
            A list: ["RSS_KB", "VSZ_KB"].
        """
        return self.PIDSTAT_METRIC_FIELDS

    def get_primary_metric_field(self) -> str:
        """Return the main metric for this collector, which is RSS."""
        return "RSS_KB"

    def start(self) -> None:
        """
        Starts the `pidstat` subprocess to begin collecting memory data.

        Constructs and executes the `pidstat` command with appropriate arguments.
        Sets the `LC_ALL=C` environment variable for `pidstat` to ensure
        consistent output formatting, which is crucial for parsing.

        Raises:
            RuntimeError: If `pidstat` fails to start or is already running.
        """
        with self._stop_lock:
            if self.pidstat_proc is not None:
                raise RuntimeError("RssPidstatCollector is already running")
                
            pidstat_base_cmd: List[str] = [
                "pidstat",
                "-r",
                "-l",
                "-C",
                self.process_pattern,
                str(self.monitoring_interval),
            ]

            pidstat_cmd_final: List[str] = []
            pidstat_prefix_str = ""

            if self.collector_cpu_core is not None and self.taskset_available:
                try:
                    # Validate core ID against available cores if possible, though psutil might not be imported here.
                    # For simplicity, assume core ID is valid if provided.
                    pidstat_cmd_final.extend(
                        ["taskset", "-c", str(self.collector_cpu_core)]
                    )
                    pidstat_prefix_str = f"taskset -c {self.collector_cpu_core} "
                    logger.info(
                        f"Will attempt to pin pidstat to CPU core {self.collector_cpu_core} using taskset."
                    )
                except Exception as e:  # Should not happen with simple string conversion
                    logger.warning(
                        f"Could not form taskset prefix for pidstat: {e}. Pidstat will not be pinned."
                    )

            pidstat_cmd_final.extend(pidstat_base_cmd)

            pidstat_env = os.environ.copy()
            pidstat_env["LC_ALL"] = "C"

            stderr_dest: Optional[Union[int, IO[Any]]] = None  # Default to None
            if self.pidstat_stderr_file:
                try:
                    # Open the stderr log file in write mode.
                    self._pidstat_stderr_handle = open(self.pidstat_stderr_file, "w")
                    stderr_dest = self._pidstat_stderr_handle
                    logger.info(
                        f"pidstat stderr will be redirected to: {self.pidstat_stderr_file}"
                    )
                except IOError as e:
                    logger.error(
                        f"Failed to open pidstat_stderr_file {self.pidstat_stderr_file}: {e}. Stderr will not be captured to file."
                    )
                    stderr_dest = subprocess.DEVNULL  # Fallback if file opening fails
            else:
                # If no specific file is provided, discard stderr to prevent pipe buffer issues.
                stderr_dest = subprocess.DEVNULL

            logger.info(
                f"Starting pidstat collector with command: {' '.join(pidstat_cmd_final)}"
            )
            try:
                self.pidstat_proc = subprocess.Popen(
                    pidstat_cmd_final,  # Use the potentially prefixed command
                    stdout=subprocess.PIPE,  # Capture stdout for parsing.
                    stderr=stderr_dest,  # Redirect stderr as configured.
                    text=True,  # Decode output as text.
                    bufsize=1,  # Line-buffered to get output as it comes.
                    env=pidstat_env,  # Use the modified environment.
                )
                self._start_time = time.monotonic()  # Record start time
                logger.info(
                    f"pidstat process started (PID: {self.pidstat_proc.pid}). Command: '{pidstat_prefix_str}{' '.join(pidstat_base_cmd)}'"
                )
            except Exception as e:
                logger.error(f"Failed to start pidstat: {e}", exc_info=True)
                if self._pidstat_stderr_handle:
                    self._pidstat_stderr_handle.close()
                    self._pidstat_stderr_handle = None  # Reset handle
                # Re-raise as a RuntimeError to indicate a critical failure.
                raise RuntimeError(f"Failed to start pidstat process: {e}") from e

    def stop(self, timeout: float = 10.0) -> bool:
        """
        Stops the `pidstat` subprocess and cleans up resources with enhanced safety.

        Terminates the `pidstat` process gracefully (SIGTERM), with a fallback
        to SIGKILL if it doesn't terminate within a timeout.
        Closes the stderr file handle if it was opened.
        
        Args:
            timeout: Maximum time to wait for pidstat to terminate gracefully.
            
        Returns:
            True if stopped successfully, False if there were issues.
        """
        with self._stop_lock:
            # Prevent multiple concurrent stop calls
            if self._is_stopping:
                logger.debug("Stop already in progress for RssPidstatCollector")
                return True
                
            if self.pidstat_proc is None:
                logger.debug("RssPidstatCollector was not running")
                return True
                
            self._is_stopping = True

        stop_success = True
        stop_time = time.monotonic()

        # Record runtime statistics
        if self._start_time:
            runtime = stop_time - self._start_time
            logger.info(f"RssPidstatCollector ran for {runtime:.2f} seconds")

        try:
            # 首先清理文件句柄，避免在进程终止过程中出现问题
            if self._pidstat_stderr_handle:
                try:
                    self._pidstat_stderr_handle.close()
                    logger.debug(f"Closed pidstat stderr file: {self.pidstat_stderr_file}")
                except IOError as e:
                    logger.error(
                        f"Error closing pidstat stderr file {self.pidstat_stderr_file}: {e}"
                    )
                    stop_success = False
                finally:
                    self._pidstat_stderr_handle = None  # 确保句柄被重置

            # 然后处理进程终止
            if self.pidstat_proc and self.pidstat_proc.poll() is None:  # Check if process is running.
                logger.info(f"Stopping pidstat process (PID: {self.pidstat_proc.pid})...")
                
                try:
                    # 首先尝试优雅终止
                    self.pidstat_proc.terminate()  # Send SIGTERM for graceful shutdown.
                    try:
                        self.pidstat_proc.wait(timeout=timeout)  # Wait up to specified timeout.
                        logger.info(
                            f"pidstat process (PID: {self.pidstat_proc.pid}) terminated gracefully."
                        )
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            f"pidstat process (PID: {self.pidstat_proc.pid}) did not terminate gracefully within {timeout}s, killing..."
                        )
                        # 强制终止
                        self.pidstat_proc.kill()  # Force kill if terminate fails.
                        try:
                            self.pidstat_proc.wait(timeout=2)  # 等待强制终止完成
                            logger.info("pidstat process killed successfully")
                        except subprocess.TimeoutExpired:
                            logger.error(f"Failed to kill pidstat process (PID: {self.pidstat_proc.pid})")
                            stop_success = False
                            
                except Exception as e:  # Catch other potential errors during termination.
                    logger.error(f"Error during pidstat stop/wait: {e}", exc_info=True)
                    stop_success = False
                    # 即使出现异常，也要尝试强制终止
                    try:
                        if self.pidstat_proc.poll() is None:
                            self.pidstat_proc.kill()
                            self.pidstat_proc.wait(timeout=1)
                    except Exception:
                        pass  # 最后的尝试，忽略所有异常
            else:
                logger.debug("pidstat process was not running or already stopped.")

        finally:
            # 确保状态被正确重置
            with self._stop_lock:
                self.pidstat_proc = None  # 确保进程对象被清理
                self._is_stopping = False  # 重置停止标志

        if stop_success:
            logger.info("RssPidstatCollector stopped successfully.")
        else:
            logger.warning("RssPidstatCollector stopped with some issues.")
            
        return stop_success

    def read_samples(self) -> Iterable[List[ProcessMemorySample]]:
        """
        Reads and parses output from the `pidstat` subprocess.

        This generator function iterates over lines from `pidstat`'s stdout.
        It skips header and non-data lines. For each data line representing a process,
        it parses out the PID, VSZ, RSS, and command information, creating a
        `ProcessMemorySample`. Samples collected within one `pidstat` reporting
        interval are grouped and yielded together.

        Yields:
            An iterable of lists, where each inner list contains ProcessMemorySample
            objects collected during one `pidstat` reporting interval.

        Warns:
            If the `pidstat` process is not running or its stdout is unavailable.
            If errors occur while parsing individual lines from `pidstat` output.
        """
        if not self.pidstat_proc or not self.pidstat_proc.stdout:
            logger.warning(
                "pidstat process not running or stdout not available for reading samples."
            )
            return  # Yields nothing, effectively an empty iterable.

        current_interval_samples: List[ProcessMemorySample] = []

        # pidstat outputs a header block, then data lines for each interval.
        # An "Average:" line might appear at the end if pidstat runs for multiple intervals
        # and then exits, or if it's interrupted.
        # The logic here is to accumulate lines that look like data and yield them
        # when a line that is clearly *not* data (header, blank, average) is encountered.

        # 添加非阻塞读取机制
        import select
        import sys
        
        while True:
            # 检查进程是否还在运行
            if self.pidstat_proc.poll() is not None:
                logger.info("pidstat process has terminated, finishing read_samples")
                break
                
            # 使用select进行非阻塞读取（仅在Unix系统上可用）
            if hasattr(select, 'select') and sys.platform != 'win32':
                ready, _, _ = select.select([self.pidstat_proc.stdout], [], [], 0.1)
                if not ready:
                    continue  # 没有数据可读，继续循环
            
            try:
                line = self.pidstat_proc.stdout.readline()
                if not line:  # EOF reached
                    break
            except Exception as e:
                logger.warning(f"Error reading from pidstat stdout: {e}")
                break
                
            line = line.strip()

            # Skip blank lines, header lines, and average lines.
            # "UID" is a common keyword in the header. "Linux" indicates the initial version line.
            if (
                not line
                or "Linux" in line
                or line.startswith("Average:")
                or "UID" in line
            ):
                if current_interval_samples:
                    # If we have accumulated samples and encounter a non-data line,
                    # it signifies the end of an interval's data block.
                    yield current_interval_samples
                    current_interval_samples = []  # Reset for the next interval.
                continue  # Move to the next line.

            # Attempt to parse the line as a data line.
            # Example `pidstat -r -l` output line (column indices can vary slightly):
            # Time   UID   PID    minflt/s majflt/s VSZ    RSS   %MEM Command
            # 00:00:00 0   1234   0.00     0.00     102400 51200 0.5  /usr/bin/some_process -arg
            # We split by whitespace, expecting at least 9 parts for a valid data line.
            # The command (parts[8]) can contain spaces.
            parts = line.split(
                None, 8
            )  # Split up to 8 times, the 9th part is the rest of the line.

            # Check if the line looks like a data line (has enough parts and PID is numeric).
            # Also, `parts[2]` should be the PID. If it's literally "PID", it's a repeated header.
            if len(parts) < 9 or not parts[2].isdigit() or parts[2] == "PID":
                if current_interval_samples:
                    # If a malformed line or unexpected header is encountered after data,
                    # yield what has been collected so far for the current interval.
                    yield current_interval_samples
                    current_interval_samples = []
                logger.debug(f"Skipping non-data or malformed pidstat line: '{line}'")
                continue

            try:
                # Extract relevant fields based on typical `pidstat -r -l` output:
                # parts[2]: PID
                # parts[5]: VSZ (Virtual Set Size in KB)
                # parts[6]: RSS (Resident Set Size in KB)
                # parts[8]: Full command string
                pid_str: str = parts[2]
                vsz_kb_str: str = parts[5]
                rss_kb_str: str = parts[6]
                cmd_full_str: str = parts[8]

                # Extract short command name from the full command string.
                # Takes the first part of the command and then its basename.
                cmd_name_parts = cmd_full_str.split(None, 1)
                cmd_name_short: str = (
                    Path(cmd_name_parts[0]).name if cmd_name_parts else "unknown"
                )

                metrics: Dict[str, Any] = {
                    "RSS_KB": int(rss_kb_str),  # pidstat -r reports RSS in KB.
                    "VSZ_KB": int(vsz_kb_str),  # pidstat -r reports VSZ in KB.
                }

                sample = ProcessMemorySample(
                    pid=pid_str,
                    command_name=cmd_name_short,
                    full_command=cmd_full_str,
                    metrics=metrics,
                )
                current_interval_samples.append(sample)

            except (IndexError, ValueError) as e:
                # Log parsing errors but continue processing other lines.
                logger.warning(
                    f"Error parsing pidstat line: '{line}'. Error: {e}. Parts: {parts}",
                    exc_info=False,  # Keep log concise for frequent parsing issues.
                )
                # If an error occurs, yield any samples collected so far for this interval
                # to avoid losing them, then reset for the next potential block.
                if current_interval_samples:
                    yield current_interval_samples
                    current_interval_samples = []
                continue  # Skip to the next line.

        # After the loop finishes (pidstat process ended or stdout closed),
        # yield any remaining samples that were collected for the last interval.
        if current_interval_samples:
            yield current_interval_samples

        logger.info("Finished reading samples from pidstat stdout.")
