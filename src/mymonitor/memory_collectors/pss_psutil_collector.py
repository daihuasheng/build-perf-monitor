"""
Memory collector implementation using the 'psutil' library.

This module provides the PssPsutilCollector class, which gathers memory metrics
like PSS (Proportional Set Size), USS (Unique Set Size), and RSS (Resident Set Size)
for processes matching a specified pattern.
"""

import logging
import os
import psutil
import re
import time
from typing import List, Dict, Any, Iterable

from .base import AbstractMemoryCollector, ProcessMemorySample

logger = logging.getLogger(__name__)


class PssPsutilCollector(AbstractMemoryCollector):
    """
    Collects PSS, USS, and RSS memory metrics using the psutil library.

    This collector iterates through all running processes, filters them based on a
    regex pattern applied to the command name or full command line, and then
    retrieves memory information using psutil's `memory_full_info()` (for PSS, USS)
    and `memory_info()` (for RSS).

    Attributes:
        PSUTIL_METRIC_FIELDS: A list of metric field names provided by this collector.
        compiled_pattern: The compiled regular expression for matching processes.
        _collecting: A boolean flag indicating if the collector is currently active.
        _stop_event: A boolean flag used to signal the collection loop to stop gracefully.
    """

    PSUTIL_METRIC_FIELDS: List[str] = ["PSS_KB", "USS_KB", "RSS_KB"]
    """Defines the memory metric fields collected by this implementation."""

    def __init__(self, process_pattern: str, monitoring_interval: int, **kwargs):
        """
        Initializes the PssPsutilCollector.

        Args:
            process_pattern: A regex pattern to match against process names or command lines.
            monitoring_interval: The interval in seconds between sampling attempts.
            **kwargs: Additional keyword arguments (currently not used by this collector).

        Raises:
            ValueError: If the provided `process_pattern` is an invalid regular expression.
        """
        super().__init__(process_pattern, monitoring_interval, **kwargs)
        self._collecting: bool = False
        """Flag to indicate if the collector's main loop should be running."""
        self._stop_event: bool = False
        """Event flag to signal the sampling loop to terminate gracefully."""
        try:
            # Compile the regex pattern for efficiency during process iteration.
            self.compiled_pattern: re.Pattern = re.compile(process_pattern)
        except re.error as e:
            logger.error(
                f"Invalid regex pattern for PssPsutilCollector: '{process_pattern}'. Error: {e}"
            )
            # Raise a ValueError to indicate bad configuration.
            raise ValueError(
                f"Invalid regular expression pattern: {process_pattern}"
            ) from e

    def get_metric_fields(self) -> List[str]:
        """
        Returns the list of memory metric field names provided by this collector.

        Returns:
            A list: ["PSS_KB", "USS_KB", "RSS_KB"].
        """
        return self.PSUTIL_METRIC_FIELDS

    def start(self) -> None:
        """
        Starts the memory collection process.

        Sets internal flags to enable the `read_samples` loop.
        """
        logger.info(
            f"Starting PssPsutilCollector (pattern: '{self.process_pattern}', interval: {self.monitoring_interval}s)."
        )
        self._collecting = True
        self._stop_event = False  # Reset stop event in case of restart.

    def stop(self) -> None:
        """
        Stops the memory collection process.

        Sets internal flags to signal the `read_samples` loop to terminate.
        """
        logger.info("Stopping PssPsutilCollector.")
        self._collecting = False  # Indicate that collection should not proceed.
        self._stop_event = True  # Signal the loop in read_samples to exit.

    def read_samples(self) -> Iterable[List[ProcessMemorySample]]:
        """
        Continuously samples memory usage of matching processes until stopped.

        This generator function iterates at `monitoring_interval`. In each iteration,
        it scans all running processes, filters them using `self.compiled_pattern`,
        and collects PSS, USS, and RSS metrics. It yields a list of
        `ProcessMemorySample` objects for each interval.

        The loop can be gracefully stopped by calling the `stop()` method, which
        sets an internal event.

        Yields:
            An iterable of lists, where each inner list contains ProcessMemorySample
            objects for processes matching the pattern during one sampling interval.
            An empty list is yielded if no matching processes are found or if an
            error occurs during an interval.

        Warns:
            If `start()` has not been called before `read_samples()`.
            If sampling takes longer than `monitoring_interval`.
        """
        if not self._collecting:
            logger.warning(
                "PssPsutilCollector not started. Call start() before read_samples()."
            )
            return  # Or: yield from () for an empty iterable

        logger.info("PssPsutilCollector sample reading loop started.")
        while not self._stop_event:  # Main collection loop, check stop event.
            current_interval_samples: List[ProcessMemorySample] = []
            # Record timestamp at the start of the sampling attempt for this interval.
            # This helps in maintaining the monitoring interval more accurately,
            # even if iterating through processes takes some time.
            interval_start_time: float = time.monotonic()

            # Iterate over all running processes, fetching necessary attributes.
            # 'username' and 'create_time' are fetched for potential filtering (e.g., kernel threads).
            for proc in psutil.process_iter(
                ["pid", "name", "cmdline", "username", "create_time"]
            ):
                if (
                    self._stop_event
                ):  # Check stop event frequently within the process loop.
                    logger.debug(
                        "PssPsutilCollector detected stop event during sample collection."
                    )
                    break  # Exit the inner process iteration loop.
                try:
                    # cmdline() can return None or an empty list, ensure it's handled.
                    # Join cmdline list into a string for regex matching.
                    cmdline_str: str = " ".join(proc.info["cmdline"] or [])
                    proc_name: str = (
                        proc.info["name"] or ""
                    )  # name() can return an empty string.

                    # Heuristic to skip kernel processes:
                    # They often have no command line and run as root.
                    # A more definitive check involves reading /proc/[pid]/stat (Linux-specific).
                    if (
                        not cmdline_str and proc.info["pid"] > 0
                    ):  # pid > 0 to skip PID 0 (swapper)
                        try:
                            # Many kernel threads run as root.
                            if proc.info["username"] == "root":
                                # On Linux, check the 9th field (flags) of /proc/[pid]/stat
                                # for the PF_KTHREAD flag (0x00200000).
                                # This is a Linux-specific optimization.
                                if (
                                    os.name == "posix"
                                ):  # Check if on a POSIX system (like Linux)
                                    with open(
                                        f"/proc/{proc.info['pid']}/stat", "r"
                                    ) as f_stat:
                                        stat_parts = f_stat.read().split()
                                        # PF_KTHREAD is bit 21 (0x00200000)
                                        if len(stat_parts) > 8 and (
                                            int(stat_parts[8]) & 0x00200000
                                        ):
                                            continue  # Skip this process, it's a kernel thread.
                        except FileNotFoundError:
                            # Process might have just ended.
                            continue
                        except Exception as stat_e:
                            # Log error if reading stat file fails, but continue processing.
                            logger.debug(
                                f"Error reading stat file for PID {proc.info['pid']}: {stat_e}"
                            )
                            # If stat cannot be read, proceed cautiously; it might not be a kernel thread.

                    # Apply the compiled regex pattern to the process name or full command line.
                    if not (
                        self.compiled_pattern.search(proc_name)
                        or (cmdline_str and self.compiled_pattern.search(cmdline_str))
                    ):
                        continue  # Skip if no match.

                    # Get detailed memory information (includes PSS, USS) and basic info (includes RSS).
                    # memory_full_info() might require higher privileges on some systems.
                    mem_full_info = proc.memory_full_info()
                    mem_info = proc.memory_info()  # For RSS.

                    # psutil returns memory in bytes; convert to KB.
                    # Check for attribute existence as some fields might be missing on certain OS
                    # or with older psutil versions. Default to 0 if attribute is missing.
                    pss_kb: float = (
                        mem_full_info.pss / 1024
                        if hasattr(mem_full_info, "pss")
                        else 0.0
                    )
                    uss_kb: float = (
                        mem_full_info.uss / 1024
                        if hasattr(mem_full_info, "uss")
                        else 0.0
                    )
                    rss_kb: float = mem_info.rss / 1024  # 'rss' is generally available.

                    metrics: Dict[str, Any] = {
                        "PSS_KB": int(pss_kb),
                        "USS_KB": int(uss_kb),
                        "RSS_KB": int(rss_kb),
                    }

                    sample = ProcessMemorySample(
                        pid=str(proc.info["pid"]),
                        command_name=proc_name,
                        full_command=cmdline_str,
                        metrics=metrics,
                    )
                    current_interval_samples.append(sample)

                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    # These exceptions are expected when monitoring dynamic processes.
                    # Log at DEBUG level as they are not usually critical errors.
                    pid_val = (
                        proc.pid if hasattr(proc, "pid") else "unknown_pid_on_error"
                    )
                    logger.debug(
                        f"Error processing PID {pid_val}: {type(e).__name__} - {e}"
                    )
                    continue
                except Exception as e:
                    # Catch other unexpected errors during processing of a single process.
                    pid_val = (
                        proc.pid if hasattr(proc, "pid") else "unknown_pid_on_error"
                    )
                    name_val = (
                        proc.info.get("name", "unknown_name")
                        if hasattr(proc, "info") and proc.info
                        else "unknown_name"
                    )
                    # Log as warning, but set exc_info=False to avoid flooding logs with stack traces
                    # for common, non-critical per-process errors.
                    logger.warning(
                        f"Unknown error processing PID {pid_val} ({name_val}): {e}",
                        exc_info=False,
                    )
                    continue

            # If stop event was set during the process iteration and no samples were collected,
            # it's cleaner to break before yielding an empty list due to stop.
            if self._stop_event and not current_interval_samples:
                logger.debug(
                    "PssPsutilCollector stopped; no samples to yield for this interval."
                )
                break  # Exit the main while loop.

            yield current_interval_samples  # Yield samples for this interval (can be empty).

            if self._stop_event:  # Check stop event again after yielding.
                logger.debug(
                    "PssPsutilCollector detected stop event after yielding samples."
                )
                break  # Exit the main while loop.

            # Calculate elapsed time for this sampling interval and sleep for the remainder.
            elapsed_time: float = time.monotonic() - interval_start_time
            sleep_time: float = self.monitoring_interval - elapsed_time

            if sleep_time > 0:
                # Make the sleep interruptible by checking _stop_event in smaller chunks.
                # This allows for a more responsive stop.
                chunk_sleep: float = 0.1  # Sleep in 100ms chunks.
                while sleep_time > 0 and not self._stop_event:
                    actual_sleep: float = min(sleep_time, chunk_sleep)
                    time.sleep(actual_sleep)
                    sleep_time -= actual_sleep
            elif sleep_time < 0:
                # Log if sampling took longer than the configured interval.
                logger.warning(
                    f"PssPsutilCollector sampling took {elapsed_time:.2f}s, "
                    f"which is longer than the monitoring interval of {self.monitoring_interval}s."
                )

        logger.info("PssPsutilCollector sample reading loop has finished.")
