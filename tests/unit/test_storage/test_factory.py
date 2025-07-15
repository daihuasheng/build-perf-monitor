"""
Unit tests for storage factory.
"""

import pytest

from src.mymonitor.storage.factory import create_storage
from src.mymonitor.storage.parquet_storage import ParquetStorage


class TestStorageFactory:
    """Test cases for storage factory."""
    
    def test_create_parquet_storage(self):
        """Test creating ParquetStorage."""
        storage = create_storage("parquet")
        assert isinstance(storage, ParquetStorage)
        assert storage.compression == "snappy"
        
        storage = create_storage("parquet", "gzip")
        assert isinstance(storage, ParquetStorage)
        assert storage.compression == "gzip"
    
    def test_create_storage_unsupported_format(self):
        """Test creating storage with unsupported format."""
        with pytest.raises(ValueError) as excinfo:
            create_storage("unsupported")
        
        assert "Unsupported storage format" in str(excinfo.value)
