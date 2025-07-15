"""
Abstract base class for data storage implementations.

This module defines the DataStorage abstract base class which serves as the
interface for all storage backend implementations. It establishes a contract
for the required functionality that any storage implementation must provide.

The interface includes methods for:
- Saving and loading DataFrames with optional column pruning
- Saving and loading dictionary data (for metadata and configuration)
- Appending data to existing files
- Checking file existence and retrieving file sizes

By using this abstract interface, the monitoring system can work with different
storage backends interchangeably, allowing for flexibility in storage formats
and future extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import polars as pl


class DataStorage(ABC):
    """Abstract base class for data storage implementations."""

    @abstractmethod
    def save_dataframe(self, df: pl.DataFrame, path: str) -> None:
        """
        Save a Polars DataFrame to the specified path.

        Args:
            df: Polars DataFrame to save
            path: File path to save to
        """
        pass

    @abstractmethod
    def load_dataframe(
        self, path: str, columns: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Load a Polars DataFrame from the specified path.

        Args:
            path: File path to load from
            columns: Optional list of columns to load (for column pruning)

        Returns:
            Loaded Polars DataFrame
        """
        pass

    @abstractmethod
    def save_dict(self, data: Dict[str, Any], path: str) -> None:
        """
        Save dictionary data to the specified path.

        Args:
            data: Dictionary data to save
            path: File path to save to
        """
        pass

    @abstractmethod
    def load_dict(self, path: str) -> Dict[str, Any]:
        """
        Load dictionary data from the specified path.

        Args:
            path: File path to load from

        Returns:
            Loaded dictionary data
        """
        pass

    @abstractmethod
    def append_dataframe(self, df: pl.DataFrame, path: str) -> None:
        """
        Append a Polars DataFrame to an existing file.

        Args:
            df: Polars DataFrame to append
            path: File path to append to
        """
        pass

    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists at the specified path.

        Args:
            path: File path to check

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abstractmethod
    def get_file_size(self, path: str) -> int:
        """
        Get the size of a file in bytes.

        Args:
            path: File path to check

        Returns:
            File size in bytes
        """
        pass
