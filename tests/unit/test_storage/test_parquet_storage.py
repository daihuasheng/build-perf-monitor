"""
Unit tests for Parquet storage implementation.
"""

import os
import tempfile
from pathlib import Path
import pytest
import polars as pl

from src.mymonitor.storage.parquet_storage import ParquetStorage


class TestParquetStorage:
    """Test cases for ParquetStorage class."""

    def test_initialization(self):
        """Test ParquetStorage initialization."""
        storage = ParquetStorage()
        assert storage.compression == "snappy"

        storage = ParquetStorage(compression="gzip")
        assert storage.compression == "gzip"

    def test_save_load_dataframe(self):
        """Test saving and loading a DataFrame."""
        storage = ParquetStorage()

        # Create test data
        df = pl.DataFrame(
            {
                "timestamp": [1650123456.789, 1650123457.789],
                "pid": [12345, 12346],
                "process_name": ["gcc", "ld"],
                "rss_kb": [102400, 51200],
                "vms_kb": [204800, 102400],
                "pss_kb": [98304, 49152],
                "category": ["compiler", "linker"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.parquet"

            # Save DataFrame
            storage.save_dataframe(df, str(file_path))

            # Verify file exists
            assert file_path.exists()

            # Load DataFrame
            loaded_df = storage.load_dataframe(str(file_path))

            # Verify data
            assert len(loaded_df) == 2
            assert loaded_df.columns == df.columns
            assert loaded_df["pid"].to_list() == [12345, 12346]
            assert loaded_df["process_name"].to_list() == ["gcc", "ld"]

    def test_load_dataframe_with_column_pruning(self):
        """Test loading a DataFrame with column pruning."""
        storage = ParquetStorage()

        # Create test data
        df = pl.DataFrame(
            {
                "timestamp": [1650123456.789, 1650123457.789],
                "pid": [12345, 12346],
                "process_name": ["gcc", "ld"],
                "rss_kb": [102400, 51200],
                "vms_kb": [204800, 102400],
                "pss_kb": [98304, 49152],
                "category": ["compiler", "linker"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.parquet"

            # Save DataFrame
            storage.save_dataframe(df, str(file_path))

            # Load only specific columns
            loaded_df = storage.load_dataframe(
                str(file_path), columns=["pid", "process_name"]
            )

            # Verify data
            assert len(loaded_df) == 2
            assert loaded_df.columns == ["pid", "process_name"]
            assert "rss_kb" not in loaded_df.columns
            assert loaded_df["pid"].to_list() == [12345, 12346]
            assert loaded_df["process_name"].to_list() == ["gcc", "ld"]

    def test_save_load_dict(self):
        """Test saving and loading a dictionary."""
        storage = ParquetStorage()

        # Create test data
        data = {
            "project_name": "test_project",
            "peak_memory_kb": 98304,
            "process_count": 2,
            "categories": ["compiler", "linker"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.json"

            # Save dictionary
            storage.save_dict(data, str(file_path))

            # Verify file exists
            assert file_path.exists()

            # Load dictionary
            loaded_data = storage.load_dict(str(file_path))

            # Verify data
            assert loaded_data == data
            assert loaded_data["project_name"] == "test_project"
            assert loaded_data["peak_memory_kb"] == 98304
            assert loaded_data["categories"] == ["compiler", "linker"]

    def test_append_dataframe(self):
        """Test appending to an existing DataFrame."""
        storage = ParquetStorage()

        # Create initial data
        df1 = pl.DataFrame(
            {
                "timestamp": [1650123456.789],
                "pid": [12345],
                "process_name": ["gcc"],
                "category": ["compiler"],
            }
        )

        # Create data to append
        df2 = pl.DataFrame(
            {
                "timestamp": [1650123457.789],
                "pid": [12346],
                "process_name": ["ld"],
                "category": ["linker"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.parquet"

            # Save initial DataFrame
            storage.save_dataframe(df1, str(file_path))

            # Append second DataFrame
            storage.append_dataframe(df2, str(file_path))

            # Load combined DataFrame
            loaded_df = storage.load_dataframe(str(file_path))

            # Verify data
            assert len(loaded_df) == 2
            assert loaded_df["pid"].to_list() == [12345, 12346]
            assert loaded_df["process_name"].to_list() == ["gcc", "ld"]
            assert loaded_df["category"].to_list() == ["compiler", "linker"]

    def test_append_to_nonexistent_file(self):
        """Test appending to a non-existent file."""
        storage = ParquetStorage()

        # Create data
        df = pl.DataFrame(
            {
                "timestamp": [1650123456.789],
                "pid": [12345],
                "process_name": ["gcc"],
                "category": ["compiler"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "nonexistent.parquet"

            # Append to non-existent file (should create it)
            storage.append_dataframe(df, str(file_path))

            # Verify file exists
            assert file_path.exists()

            # Load DataFrame
            loaded_df = storage.load_dataframe(str(file_path))

            # Verify data
            assert len(loaded_df) == 1
            assert loaded_df["pid"].to_list() == [12345]
            assert loaded_df["process_name"].to_list() == ["gcc"]

    def test_file_exists(self):
        """Test file_exists method."""
        storage = ParquetStorage()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.parquet"

            # File should not exist initially
            assert not storage.file_exists(str(file_path))

            # Create file
            with open(file_path, "w") as f:
                f.write("test")

            # File should exist now
            assert storage.file_exists(str(file_path))

    def test_get_file_size(self):
        """Test get_file_size method."""
        storage = ParquetStorage()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"

            # Create file with known content
            with open(file_path, "w") as f:
                f.write("test content")

            # Get file size
            size = storage.get_file_size(str(file_path))

            # Verify size
            assert size == len("test content")

    def test_get_file_size_nonexistent(self):
        """Test get_file_size for non-existent file."""
        storage = ParquetStorage()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "nonexistent.txt"

            # Get file size for non-existent file
            size = storage.get_file_size(str(file_path))

            # Should return 0
            assert size == 0

    def test_compression_efficiency(self):
        """Test that Parquet compression is efficient."""
        storage = ParquetStorage()

        # Create test data with some repetition for better compression
        rows = []
        for i in range(100):
            rows.append(
                {
                    "timestamp": 1650123456.0 + i,
                    "pid": 12345 + (i % 5),
                    "process_name": f"process_{i % 5}",
                    "rss_kb": 100000 + i * 1000,
                    "category": "compiler" if i % 2 == 0 else "linker",
                }
            )

        df = pl.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save as Parquet
            parquet_path = Path(tmpdir) / "test.parquet"
            storage.save_dataframe(df, str(parquet_path))

            # Save as CSV for comparison
            csv_path = Path(tmpdir) / "test.csv"
            df.write_csv(csv_path)

            # Get file sizes
            parquet_size = os.path.getsize(parquet_path)
            csv_size = os.path.getsize(csv_path)

            # Parquet should be smaller than CSV
            # For small test datasets, the compression ratio might not be as dramatic
            # as with large real-world datasets, so we use a more reasonable threshold
            assert (
                parquet_size < csv_size * 0.9
            ), f"Parquet size ({parquet_size}) not smaller than CSV ({csv_size})"
