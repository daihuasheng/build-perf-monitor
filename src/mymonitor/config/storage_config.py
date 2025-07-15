"""
Storage configuration model and validation.

This module defines the StorageConfig dataclass which encapsulates all
storage-related configuration options including format selection, compression
settings, and backward compatibility options. It provides validation methods
to ensure configuration consistency and type safety.
"""

from typing import Literal, Dict, Any
from dataclasses import dataclass


@dataclass
class StorageConfig:
    """
    Configuration model for data storage settings.

    This class encapsulates all storage-related configuration options including
    the primary storage format, compression settings, and backward compatibility
    options. It provides validation and serialization methods for configuration
    management.

    Attributes:
        format: Primary storage format type ('parquet' recommended, 'json' for compatibility)
            - 'parquet': High-performance columnar format with compression
            - 'json': Human-readable format for small datasets and metadata
        compression: Compression algorithm for Parquet format
            - 'snappy': Fast compression/decompression (default)
            - 'gzip': Higher compression ratio, slower
            - 'brotli': Very high compression ratio
            - 'lz4': Very fast compression
            - 'zstd': Modern balanced compression
        generate_legacy_formats: Whether to generate CSV files for backward compatibility
            When True, both the primary format and CSV will be generated

    Note:
        Compression setting only applies to Parquet format. JSON format
        is always stored uncompressed for human readability.
    """

    format: Literal["parquet", "json"] = "parquet"
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "snappy"
    generate_legacy_formats: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StorageConfig":
        """
        Create a StorageConfig instance from a dictionary.

        Args:
            config_dict: Dictionary containing storage configuration

        Returns:
            StorageConfig instance

        Raises:
            ValueError: If invalid configuration values are provided
        """
        # Extract values with defaults
        format_type = config_dict.get("format", "parquet")
        compression = config_dict.get("compression", "snappy")
        generate_legacy = config_dict.get("generate_legacy_formats", False)

        # Validate format
        if format_type not in ("parquet", "json"):
            raise ValueError(f"Unsupported storage format: {format_type}")

        # Validate compression (only for Parquet)
        if format_type == "parquet" and compression not in (
            "snappy",
            "gzip",
            "brotli",
            "lz4",
            "zstd",
        ):
            raise ValueError(f"Unsupported compression algorithm: {compression}")

        return cls(
            format=format_type,
            compression=compression,
            generate_legacy_formats=generate_legacy,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the StorageConfig to a dictionary.

        Returns:
            Dictionary representation of the StorageConfig
        """
        return {
            "format": self.format,
            "compression": self.compression,
            "generate_legacy_formats": self.generate_legacy_formats,
        }
