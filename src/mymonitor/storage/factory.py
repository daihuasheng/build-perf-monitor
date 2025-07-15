"""
Factory for creating storage instances.
"""

import logging
from typing import Literal

from .base import DataStorage
from .parquet_storage import ParquetStorage

logger = logging.getLogger(__name__)


def create_storage(
    format_type: Literal["parquet", "json"] = "parquet",
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "snappy",
) -> DataStorage:
    """
    Create a storage instance based on the specified format type.

    Args:
        format_type: Storage format type ('parquet' or 'json')
        compression: Compression algorithm (for Parquet only)

    Returns:
        DataStorage instance

    Raises:
        ValueError: If an unsupported format type is specified
    """
    if format_type == "parquet":
        logger.debug(f"Creating ParquetStorage with compression: {compression}")
        return ParquetStorage(compression=compression)
    elif format_type == "json":
        logger.debug("Creating ParquetStorage for JSON mode (uses JSON for metadata)")
        return ParquetStorage(
            compression=compression
        )  # ParquetStorage handles both formats
    else:
        raise ValueError(f"Unsupported storage format: {format_type}")
