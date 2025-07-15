"""
Unit tests for storage configuration.
"""

import pytest

from src.mymonitor.config.storage_config import StorageConfig


class TestStorageConfig:
    """Test cases for StorageConfig class."""
    
    def test_default_values(self):
        """Test default values."""
        config = StorageConfig()
        assert config.format == "parquet"
        assert config.compression == "snappy"
        assert config.generate_legacy_formats is False
    
    def test_custom_values(self):
        """Test custom values."""
        config = StorageConfig(
            format="parquet",
            compression="gzip",
            generate_legacy_formats=True
        )
        assert config.format == "parquet"
        assert config.compression == "gzip"
        assert config.generate_legacy_formats is True
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        config_dict = {
            "format": "parquet",
            "compression": "gzip",
            "generate_legacy_formats": True
        }
        config = StorageConfig.from_dict(config_dict)
        
        assert config.format == "parquet"
        assert config.compression == "gzip"
        assert config.generate_legacy_formats is True
    
    def test_from_dict_defaults(self):
        """Test creating from dictionary with defaults."""
        config_dict = {}
        config = StorageConfig.from_dict(config_dict)
        
        assert config.format == "parquet"
        assert config.compression == "snappy"
        assert config.generate_legacy_formats is False
    
    def test_from_dict_partial(self):
        """Test creating from dictionary with partial values."""
        config_dict = {"format": "parquet"}
        config = StorageConfig.from_dict(config_dict)
        
        assert config.format == "parquet"
        assert config.compression == "snappy"
        assert config.generate_legacy_formats is False
    
    def test_from_dict_invalid_format(self):
        """Test creating from dictionary with invalid format."""
        config_dict = {"format": "invalid"}
        
        with pytest.raises(ValueError) as excinfo:
            StorageConfig.from_dict(config_dict)
        
        assert "Unsupported storage format" in str(excinfo.value)
    
    def test_from_dict_invalid_compression(self):
        """Test creating from dictionary with invalid compression."""
        config_dict = {"format": "parquet", "compression": "invalid"}
        
        with pytest.raises(ValueError) as excinfo:
            StorageConfig.from_dict(config_dict)
        
        assert "Unsupported compression algorithm" in str(excinfo.value)
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        config = StorageConfig(
            format="parquet",
            compression="gzip",
            generate_legacy_formats=True
        )
        config_dict = config.to_dict()
        
        assert config_dict == {
            "format": "parquet",
            "compression": "gzip",
            "generate_legacy_formats": True
        }
