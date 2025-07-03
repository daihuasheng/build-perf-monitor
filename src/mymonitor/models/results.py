"""
Monitoring results data models.

This module contains data structures for storing and organizing
monitoring results and collected memory data.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass
class MonitoringResults:
    """
    Holds all the aggregated results from a completed monitoring loop.
    """

    # A list of dictionaries, where each dict is a raw data row for the Parquet file.
    all_samples_data: List[Dict[str, Any]]
    # A dictionary holding peak memory stats for individual processes, keyed by category.
    category_stats: Dict[str, Dict[str, Any]]
    # The peak sum of memory across all monitored processes at any single interval.
    peak_overall_memory_kb: int
    # The timestamp (epoch seconds) when the overall peak memory was observed.
    peak_overall_memory_epoch: int
    # A dictionary holding the peak summed memory for each category, keyed by category.
    category_peak_sum: Dict[str, int]
    # A dictionary holding the set of unique PIDs observed for each category.
    category_pid_set: Dict[str, Set[str]]
