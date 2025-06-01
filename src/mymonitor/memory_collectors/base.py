import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProcessMemorySample:
    """Dataclass to hold memory information for a single process at a point in time."""
    pid: str
    command_name: str
    full_command: str
    # Specific memory metrics will be stored here, e.g., {"RSS_KB": 123, "VSZ_KB": 456}
    metrics: Dict[str, Any]


class AbstractMemoryCollector(ABC):
    """Abstract base class for memory collectors."""

    def __init__(self, process_pattern: str, monitoring_interval: int, **kwargs):
        self.process_pattern = process_pattern
        self.monitoring_interval = monitoring_interval
        self.collector_kwargs = kwargs
        logger.info(f"Initializing {self.__class__.__name__} with pattern: '{process_pattern}', interval: {monitoring_interval}s")

    @abstractmethod
    def start(self):
        """Starts the memory collection process (e.g., launches a subprocess)."""
        pass

    @abstractmethod
    def stop(self):
        """Stops the memory collection process and cleans up resources."""
        pass

    @abstractmethod
    def get_metric_fields(self) -> List[str]:
        """
        Returns a list of memory metric field names that this collector provides.
        Example: ["RSS_KB", "VSZ_KB"] or ["PSS_KB", "USS_KB", "RSS_KB"]
        """
        pass

    @abstractmethod
    def read_samples(self) -> Iterable[List[ProcessMemorySample]]:
        """
        A generator that yields a list of ProcessMemorySample objects.
        Each list represents all monitored processes captured at a single sampling point/interval.
        This method should handle the timing of samples if it's internally driven (like pidstat),
        or expect to be called periodically if it's an on-demand sampler.
        """
        pass