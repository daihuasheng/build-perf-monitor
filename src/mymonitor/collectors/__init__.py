"""
Memory collectors package for process memory monitoring.

This package provides a comprehensive framework for collecting memory usage data
from processes during build operations. It includes:

- Abstract interfaces defining the collector contract
- Multiple collector implementations using different backends:
  - PSS-based collection using psutil (recommended)
  - RSS-based collection using pidstat (legacy)
- Factory pattern for runtime collector selection
- Robust error handling and resource management

The collectors are designed to work with the monitoring orchestration system,
providing real-time memory usage data for analysis and visualization. They support
different memory metrics (PSS, RSS, VSZ) and can be configured for various
sampling intervals and process filtering patterns.
"""

from .base import AbstractMemoryCollector, ProcessMemorySample
from .pss_psutil import PssPsutilCollector
from .rss_pidstat import RssPidstatCollector
from .factory import CollectorFactory

__all__ = [
    "AbstractMemoryCollector",
    "ProcessMemorySample",
    "PssPsutilCollector",
    "RssPidstatCollector",
    "CollectorFactory",
]
