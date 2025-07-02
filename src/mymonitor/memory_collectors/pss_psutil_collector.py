"""
Memory collector implementation using the 'psutil' library.

This module provides the PssPsutilCollector class, which gathers memory metrics
like PSS (Proportional Set Size), USS (Unique Set Size), and RSS (Resident Set Size)
for processes matching a specified pattern.
"""

import itertools
import logging
import os
import psutil
import re
import time
from typing import List, Iterable, Optional

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

    def __init__(
        self,
        process_pattern: str,
        monitoring_interval: float,
        mode: str = "full_scan",  # Add mode parameter
        **kwargs,
    ):
        """
        Initializes the PssPsutilCollector.

        Args:
            process_pattern: A regex pattern to match against process names or command lines.
            monitoring_interval: The interval in seconds between sampling attempts.
            mode: The mode of operation for the collector, either 'full_scan' or 'descendants_only'.
            **kwargs: Additional keyword arguments (currently not used by this collector).

        Raises:
            ValueError: If the provided `process_pattern` is an invalid regular expression.
        """
        super().__init__(process_pattern, monitoring_interval, **kwargs)
        # This list tells psutil.process_iter which process attributes to pre-fetch
        # for performance. These are the attributes needed by _get_sample_for_process.
        self._iter_attrs = ["pid", "name", "cmdline"]

        # Store the main build process PID for the optimization.
        self.build_process_pid: Optional[str] = kwargs.get("build_process_pid")
        if self.build_process_pid is not None and isinstance(self.build_process_pid, int):
            self.build_process_pid = str(self.build_process_pid)  # 转换为字符串
        self._collecting: bool = False
        """Flag to indicate if the collector's main loop should be running."""
        self._stop_event: bool = False
        """Event flag to signal the sampling loop to terminate gracefully."""
        self.mode = mode
        if self.mode not in ["full_scan", "descendants_only"]:
            raise ValueError(f"Invalid PssPsutilCollector mode: {self.mode}")
        logger.info(f"PssPsutilCollector initialized in '{self.mode}' mode.")

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

    def get_primary_metric_field(self) -> str:
        """Return the main metric for this collector, which is PSS."""
        return "PSS_KB"

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

    def _get_sample_for_process(
        self, proc: psutil.Process
    ) -> Optional[ProcessMemorySample]:
        """
        Checks a single process against all filters and returns a memory sample.

        This helper function encapsulates the logic for checking one process.
        It returns a ProcessMemorySample if the process is a match, otherwise None.

        Args:
            proc: A psutil.Process object.

        Returns:
            A ProcessMemorySample object if the process matches all criteria,
            otherwise None.
        """
        try:
            # Instead of accessing the .info property directly, which can fail
            # in race conditions with short-lived processes, we explicitly call .as_dict()
            # inside the try block. This is more robust and ensures the required
            # attributes are fetched on-demand for each process.
            proc_info = proc.as_dict(attrs=["pid", "name", "cmdline"])
            cmdline_str: str = " ".join(proc_info["cmdline"] or [])
            proc_name: str = proc_info["name"] or ""

            # OPTIMIZED: Kernel thread check
            if os.name == "posix":
                try:
                    with open(f"/proc/{proc_info['pid']}/stat", "r") as f_stat:
                        stat_parts = f_stat.read().split()
                        if len(stat_parts) > 8 and (int(stat_parts[8]) & 0x00200000):
                            return None  # It's a kernel thread, skip.
                except (FileNotFoundError, IOError, ValueError, IndexError):
                    return None  # Cannot check, safer to skip.

            # Regex pattern match
            pattern_matches = (
                self.compiled_pattern.search(proc_name)
                or (cmdline_str and self.compiled_pattern.search(cmdline_str))
            )
            
            if not pattern_matches:
                return None  # Does not match project's process pattern, skip.
            
            # Log when we find a matching process
            logger.debug(f"Found matching process: PID={proc_info['pid']}, name={proc_name}, cmdline={cmdline_str[:100]}...")

            # If all checks pass, get memory info and create the sample.
            mem_full_info = proc.memory_full_info()
            mem_info = proc.memory_info()
            pss_kb = (mem_full_info.pss / 1024) if hasattr(mem_full_info, "pss") else 0
            uss_kb = (mem_full_info.uss / 1024) if hasattr(mem_full_info, "uss") else 0
            rss_kb = mem_info.rss / 1024

            return ProcessMemorySample(
                pid=str(proc_info["pid"]),
                command_name=proc_name,
                full_command=cmdline_str,
                metrics={
                    "PSS_KB": int(pss_kb),
                    "USS_KB": int(uss_kb),
                    "RSS_KB": int(rss_kb),
                },
            )

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # These are expected and should not be treated as errors.
            return None
        except Exception as e:
            # Catch any other unexpected errors for a single process.
            pid_val = proc.pid if hasattr(proc, "pid") else "unknown"
            logger.warning(
                f"Unknown error processing PID {pid_val}: {e}", exc_info=False
            )
            return None

    def read_samples(self) -> Iterable[List[ProcessMemorySample]]:
        """
        Continuously samples memory usage using a "descendants-first" optimization.
        This generator yields lists of samples at each monitoring interval. It is
        designed to be robust against race conditions during shutdown.
        """
        if not self._collecting:
            logger.warning("Collector has not been started; read_samples will not run.")
            return

        logger.info("PssPsutilCollector sample reading loop started.")
        iteration_count = 0
        # The main loop continues as long as the collector is active.
        # The stop() method will set _stop_event to True, causing a graceful exit.
        while True:
            interval_start_time = time.monotonic()
            current_interval_samples: List[ProcessMemorySample] = []
            iteration_count += 1
            
            logger.debug(f"Starting sampling iteration {iteration_count}")

            # --- OPTIMIZATION: Execute logic based on the configured mode ---
            if self.mode == "descendants_only" and self.build_process_pid:
                # --- FAST PATH: Only scan descendants of the main build process ---
                logger.debug(f"Using descendants_only mode for PID {self.build_process_pid}")
                try:
                    # 转换字符串PID为整数用于psutil.Process
                    parent_proc = psutil.Process(int(self.build_process_pid))
                    descendants = list(parent_proc.children(recursive=True))
                    logger.debug(f"Found {len(descendants)} descendants of build process")
                    
                    for p in itertools.chain([parent_proc], descendants):
                        sample = self._get_sample_for_process(p)
                        if sample:
                            current_interval_samples.append(sample)
                except psutil.NoSuchProcess:
                    logger.warning(
                        f"Main build process PID {self.build_process_pid} disappeared. "
                        "Consider using 'full_scan' mode if processes are missed."
                    )
                    self.build_process_pid = None  # Fallback to full scan next time
                except Exception as e:
                    logger.warning(f"Error scanning descendants: {e}", exc_info=False)
            else:
                # --- SAFE PATH (DEFAULT): Scan all system processes ---
                if self.mode == "descendants_only" and not self.build_process_pid:
                    logger.debug(
                        "descendants_only mode is active, but no build PID is set. Performing full scan."
                    )

                processes_scanned = 0
                for proc in psutil.process_iter(self._iter_attrs):
                    if self._stop_event:
                        break
                    processes_scanned += 1
                    sample = self._get_sample_for_process(proc)
                    if sample:
                        current_interval_samples.append(sample)
                
                logger.debug(f"Scanned {processes_scanned} processes in full scan mode")

            logger.debug(f"Iteration {iteration_count}: Found {len(current_interval_samples)} matching samples")
            
            # Log some sample details for debugging
            if current_interval_samples:
                total_memory = sum(sample.metrics.get("PSS_KB", 0) for sample in current_interval_samples)
                logger.debug(f"Total PSS memory in this sample: {total_memory} KB")
                
                # Log first few samples for debugging
                for i, sample in enumerate(current_interval_samples[:3]):
                    logger.debug(f"Sample {i+1}: PID={sample.pid}, cmd={sample.command_name}, PSS={sample.metrics.get('PSS_KB', 0)} KB")

            # *** FIX APPLIED HERE ***
            # Always yield the collected samples for this interval BEFORE checking
            # the main stop condition. This ensures that even on the final loop
            # before shutdown, the collected data is sent to the consumer.
            yield current_interval_samples

            # Now, check if we should exit the main loop AFTER yielding data.
            if self._stop_event:
                logger.info(f"Stop event received, exiting after {iteration_count} iterations")
                break

            # --- Sleep Logic ---
            elapsed_time = time.monotonic() - interval_start_time
            sleep_time = self.monitoring_interval - elapsed_time
            if sleep_time > 0:
                # Use an interruptible sleep that checks the stop event periodically.
                # This allows for a faster shutdown response.
                chunk_sleep = 0.05  # Check for stop event frequently.
                sleep_end_time = time.monotonic() + sleep_time
                while time.monotonic() < sleep_end_time:
                    if self._stop_event:
                        break
                    # Sleep for a short chunk or the remaining time, whichever is smaller.
                    time.sleep(min(chunk_sleep, sleep_end_time - time.monotonic()))
            elif sleep_time < 0:
                logger.warning(
                    f"PssPsutilCollector sampling took {elapsed_time:.2f}s, "
                    f"longer than interval of {self.monitoring_interval}s."
                )

        logger.info(f"PssPsutilCollector sample reading loop has finished after {iteration_count} iterations.")
