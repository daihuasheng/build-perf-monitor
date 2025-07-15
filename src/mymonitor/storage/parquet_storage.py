"""
Parquet storage implementation using Polars for high-performance data operations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
import polars as pl

from .base import DataStorage

logger = logging.getLogger(__name__)


class ParquetStorage(DataStorage):
    """
    High-performance Parquet storage implementation using Polars.
    
    This implementation provides:
    - Efficient columnar storage with compression
    - Fast column-wise operations
    - Automatic schema inference and validation
    - Support for various compression algorithms
    """
    
    def __init__(self, compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "snappy"):
        """
        Initialize Parquet storage with specified compression.
        
        Args:
            compression: Compression algorithm to use
        """
        self.compression = compression
        logger.debug(f"Initialized ParquetStorage with compression: {compression}")
    
    def save_dataframe(self, df: pl.DataFrame, path: str) -> None:
        """
        Save a Polars DataFrame to Parquet format.
        
        Args:
            df: Polars DataFrame to save
            path: File path to save to
        """
        try:
            # Ensure parent directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write to Parquet with compression
            df.write_parquet(path, compression=self.compression)
            
            logger.debug(f"Saved DataFrame with {len(df)} rows to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save DataFrame to {path}: {e}")
            raise
    
    def load_dataframe(self, path: str, columns: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Load a Polars DataFrame from Parquet format.
        
        Args:
            path: File path to load from
            columns: Optional list of columns to load (for column pruning)
            
        Returns:
            Loaded Polars DataFrame
        """
        try:
            if columns:
                # Use column pruning for better performance
                df = pl.read_parquet(path, columns=columns)
                logger.debug(f"Loaded DataFrame with columns {columns} from {path}")
            else:
                df = pl.read_parquet(path)
                logger.debug(f"Loaded DataFrame with {len(df)} rows from {path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load DataFrame from {path}: {e}")
            raise
    
    def save_dict(self, data: Dict[str, Any], path: str) -> None:
        """
        Save dictionary data to JSON format.
        
        Note: For small metadata files, JSON is more appropriate than Parquet.
        
        Args:
            data: Dictionary data to save
            path: File path to save to
        """
        try:
            # Ensure parent directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON for better readability of metadata
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved dictionary data to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save dictionary to {path}: {e}")
            raise
    
    def load_dict(self, path: str) -> Dict[str, Any]:
        """
        Load dictionary data from JSON format.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded dictionary data
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"Loaded dictionary data from {path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load dictionary from {path}: {e}")
            raise
    
    def append_dataframe(self, df: pl.DataFrame, path: str) -> None:
        """
        Append a Polars DataFrame to an existing Parquet file.
        
        Args:
            df: Polars DataFrame to append
            path: File path to append to
        """
        try:
            if self.file_exists(path):
                # Load existing data and concatenate
                existing_df = self.load_dataframe(path)
                combined_df = pl.concat([existing_df, df])
                self.save_dataframe(combined_df, path)
                logger.debug(f"Appended {len(df)} rows to existing file {path}")
            else:
                # File doesn't exist, just save the new data
                self.save_dataframe(df, path)
                logger.debug(f"Created new file {path} with {len(df)} rows")
                
        except Exception as e:
            logger.error(f"Failed to append DataFrame to {path}: {e}")
            raise
    
    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists at the specified path.
        
        Args:
            path: File path to check
            
        Returns:
            True if file exists, False otherwise
        """
        return Path(path).exists()
    
    def get_file_size(self, path: str) -> int:
        """
        Get the size of a file in bytes.
        
        Args:
            path: File path to check
            
        Returns:
            File size in bytes
        """
        try:
            return Path(path).stat().st_size
        except FileNotFoundError:
            return 0
