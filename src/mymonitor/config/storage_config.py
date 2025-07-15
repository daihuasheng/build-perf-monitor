"""
Storage configuration model.
"""

from typing import Literal, Dict, Any
from dataclasses import dataclass


@dataclass
class StorageConfig:
    """
    Storage configuration model.
    
    Attributes:
        format: Storage format type ('parquet' or 'json')
        compression: Compression algorithm (for Parquet only)
        generate_legacy_formats: Whether to generate legacy formats for backward compatibility
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
        if format_type == "parquet" and compression not in ("snappy", "gzip", "brotli", "lz4", "zstd"):
            raise ValueError(f"Unsupported compression algorithm: {compression}")
        
        return cls(
            format=format_type,
            compression=compression,
            generate_legacy_formats=generate_legacy
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
            "generate_legacy_formats": self.generate_legacy_formats
        }
