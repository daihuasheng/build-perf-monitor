"""
Defines the base structures and abstract class for memory collectors.

This module provides:
- ProcessMemorySample: A dataclass to store memory metrics for a single process.
- AbstractMemoryCollector: An abstract base class (ABC) that defines the
  interface for all memory collector implementations (e.g., using pidstat, psutil).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable

logger = logging.getLogger(__name__)


@dataclass
class ProcessMemorySample:
    """
    Dataclass to hold memory information for a single process at a point in time.

    Attributes:
        pid: Process ID as a string.
        command_name: The base name of the command (e.g., "gcc", "make").
        full_command: The full command line string with arguments.
        metrics: A dictionary storing specific memory metrics.
                 Example: {"RSS_KB": 1234, "PSS_KB": 1000}
    """

    pid: str
    command_name: str
    full_command: str
    # Specific memory metrics will be stored here, e.g., {"RSS_KB": 123, "VSZ_KB": 456}
    metrics: Dict[str, Any]


class AbstractMemoryCollector(ABC):
    """
    Abstract base class for memory collectors.

    This class defines the common interface that all memory collector
    implementations must adhere to. Subclasses are responsible for
    implementing the specific logic to collect memory data from processes
    matching a given pattern.
    """

    def __init__(self, process_pattern: str, monitoring_interval: int, **kwargs):
        """
        Initializes the AbstractMemoryCollector.

        Args:
            process_pattern: A string (often a regex pattern) used to identify
                             processes to monitor.
            monitoring_interval: The interval in seconds at which the collector
                                 should aim to gather samples. For some collectors
                                 (like pidstat), this is passed directly to the tool.
                                 For others (like psutil), it dictates the polling frequency.
            **kwargs: Additional keyword arguments specific to a collector implementation.
                      For example, 'pidstat_stderr_file' for RssPidstatCollector.
        """
        self.process_pattern = process_pattern
        self.monitoring_interval = monitoring_interval
        self.collector_kwargs = kwargs  # Store any additional arguments
        logger.info(
            f"Initializing {self.__class__.__name__} with pattern: '{process_pattern}', "
            f"interval: {monitoring_interval}s, extra_args: {kwargs}"
        )

    @abstractmethod
    def start(self) -> None:
        """
        Starts the memory collection process.

        This method should handle any setup required to begin collecting data,
        such as launching a subprocess (e.g., pidstat) or initializing internal
        state for polling (e.g., for psutil-based collectors).
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stops the memory collection process and cleans up any resources.

        This method should ensure that any running subprocesses are terminated
        and any open resources (like file handles) are closed.
        """
        pass

    @abstractmethod
    def get_metric_fields(self) -> List[str]:
        """
        Returns a list of memory metric field names that this collector provides.

        The order of metric names in the list should correspond to the order
        in which their values will appear in the CSV output and potentially
        in the `metrics` dictionary of `ProcessMemorySample`.

        Returns:
            A list of strings, where each string is a metric name (e.g., "RSS_KB").
            Example: ["RSS_KB", "VSZ_KB"] or ["PSS_KB", "USS_KB", "RSS_KB"]
        """
        pass

    @abstractmethod
    def read_samples(self) -> Iterable[List[ProcessMemorySample]]:
        """
        A generator that yields lists of ProcessMemorySample objects.

        Each list yielded represents all monitored processes captured at a single
        sampling point or interval. The implementation should handle the timing
        of samples:
        - If internally driven (e.g., pidstat outputting at its own interval),
          this method parses and yields data as it becomes available.
        - If on-demand (e.g., psutil iterating through processes), this method
          should perform one round of sampling and then yield, potentially
          managing its own sleep/timing to respect `monitoring_interval`.

        Yields:
            An iterable of lists, where each inner list contains ProcessMemorySample
            objects collected during one sampling interval.
        """
        pass
