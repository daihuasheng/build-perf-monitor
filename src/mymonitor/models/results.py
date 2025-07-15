"""
Monitoring results data models and aggregation structures.

This module defines the data structures used to store and organize the complete
results of a monitoring session. It provides a comprehensive container for all
collected data, statistics, and analysis results that can be easily serialized
and stored using the storage system.

The models are designed to support:
- Raw sample data for detailed analysis
- Aggregated statistics by process categories
- Peak memory usage tracking and timestamps
- Process identification and categorization
- Efficient serialization to storage formats
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass
class MonitoringResults:
    """
    Comprehensive container for all monitoring results and analysis data.

    This class serves as the central data structure for storing all collected
    memory samples, statistical analysis, and categorized results from a complete
    monitoring session. It provides a structured way to organize and access
    the monitoring data for storage, reporting, and visualization.

    The data is organized into several key components:
    - Raw sample data for all monitored processes
    - Category-based statistics and aggregations
    - Peak memory usage tracking across the entire monitoring session
    - Process categorization and identification

    This structure is designed to be easily serializable to the storage system
    and provides all necessary data for generating reports and visualizations.
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
