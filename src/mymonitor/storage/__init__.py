"""
Storage module for high-performance monitoring data management.

This module provides a comprehensive storage system for monitoring data with:
- High-efficiency Parquet format with columnar storage and compression
- Flexible storage backends with a common interface
- Support for various compression algorithms (Snappy, Gzip, Brotli, LZ4, Zstd)
- Backward compatibility with legacy formats
- High-level data management API for saving and loading monitoring results

The storage system is designed to minimize disk space usage while providing
fast data access for analysis and visualization. It uses Polars for efficient
DataFrame operations and supports column pruning for optimized performance.
"""

from .base import DataStorage
from .parquet_storage import ParquetStorage
from .factory import create_storage

__all__ = ["DataStorage", "ParquetStorage", "create_storage"]
