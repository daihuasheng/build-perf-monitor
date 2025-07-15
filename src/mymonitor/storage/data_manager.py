"""
Data storage manager for monitoring results.

This module provides a high-level interface for saving and loading
monitoring data using the configured storage format.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import polars as pl

from .factory import create_storage
from ..config import get_config
from ..models.results import MonitoringResults

logger = logging.getLogger(__name__)


class DataStorageManager:
    """
    High-level data storage manager for monitoring results.

    This class provides a unified interface for saving and loading
    monitoring data, automatically using the configured storage format
    and handling data conversion between different formats.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the data storage manager.

        Args:
            output_dir: Directory where data files will be stored
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get storage configuration
        config = get_config()
        storage_config = config.monitor.storage

        self.storage_format = storage_config.format
        self.compression = storage_config.compression
        self.generate_legacy = storage_config.generate_legacy_formats

        # Create storage instance
        self.storage = create_storage(self.storage_format, self.compression)

        # Create legacy storage if needed
        if self.generate_legacy:
            self.legacy_storage = create_storage("json")

        logger.debug(
            f"Initialized DataStorageManager with format: {self.storage_format}"
        )

    def save_monitoring_results(
        self, results: MonitoringResults, run_context: Any
    ) -> None:
        """
        Save complete monitoring results to storage.

        This method orchestrates the saving of all monitoring data including:
        - Memory samples in the configured format (Parquet/JSON)
        - Runtime metadata (build configuration, system info)
        - Analysis results (statistics, peak memory usage)
        - Human-readable summary logs

        Args:
            results: MonitoringResults object containing all collected data
            run_context: Runtime context with build and system metadata

        Raises:
            Exception: If any storage operation fails
        """
        try:
            logger.info("Saving monitoring results...")

            if not results or not results.all_samples_data:
                logger.warning("No monitoring data to save")
                return

            # Save memory samples
            self._save_memory_samples(results.all_samples_data)

            # Save metadata
            self._save_metadata(run_context)

            # Save analysis results
            self._save_analysis_results(results)

            # Save summary log (always in text format for readability)
            self._save_summary_log(results, run_context)

            logger.info(f"Successfully saved monitoring results to: {self.output_dir}")

        except Exception as e:
            logger.error(f"Error saving monitoring results: {e}", exc_info=True)
            raise

    def _save_memory_samples(self, samples_data: List[Dict[str, Any]]) -> None:
        """
        Save memory samples data to the configured storage format.

        This method converts the raw memory samples to a Polars DataFrame
        and saves it using the configured storage format (Parquet or JSON).
        If legacy format generation is enabled, it will also save in CSV format.

        Args:
            samples_data: List of dictionaries containing memory sample data
                Each dictionary represents one memory measurement with keys like
                'timestamp', 'pid', 'process_name', 'rss_kb', etc.

        Raises:
            Exception: If the save operation fails
        """
        if not samples_data:
            logger.warning("No memory samples to save")
            return

        # Convert to Polars DataFrame
        df = pl.DataFrame(samples_data)

        # Determine file path and extension
        if self.storage_format == "parquet":
            file_path = self.output_dir / "memory_samples.parquet"
        else:
            file_path = self.output_dir / "memory_samples.json"

        # Save using configured storage
        if self.storage_format == "parquet":
            self.storage.save_dataframe(df, str(file_path))
        else:
            # For JSON, convert DataFrame to list of dicts
            data_list = df.to_dicts()
            self.storage.save_dict({"samples": data_list}, str(file_path))

        logger.info(f"Saved {len(samples_data)} memory samples to: {file_path}")

        # Save legacy format if requested
        if self.generate_legacy and self.storage_format != "json":
            legacy_path = self.output_dir / "memory_samples.csv"
            df.write_csv(legacy_path)
            logger.info(f"Saved legacy CSV format to: {legacy_path}")

    def _save_metadata(self, run_context: Any) -> None:
        """
        Save runtime metadata to JSON format.

        This method extracts relevant metadata from the run context and saves
        it as a JSON file for later analysis and debugging. The metadata includes
        build configuration, system settings, and monitoring parameters.

        Args:
            run_context: Runtime context object containing build and system metadata
                Expected attributes include project_name, build_command, CPU settings, etc.

        Note:
            Metadata is always saved in JSON format for human readability,
            regardless of the configured storage format for data files.
        """
        metadata = {
            "project_name": getattr(run_context, "project_name", "unknown"),
            "project_dir": str(getattr(run_context, "project_dir", "")),
            "process_pattern": getattr(run_context, "process_pattern", ""),
            "actual_build_command": getattr(run_context, "actual_build_command", ""),
            "parallelism_level": getattr(run_context, "parallelism_level", 0),
            "monitoring_interval": getattr(run_context, "monitoring_interval", 0.0),
            "collector_type": getattr(run_context, "collector_type", ""),
            "current_timestamp_str": getattr(run_context, "current_timestamp_str", ""),
            "taskset_available": getattr(run_context, "taskset_available", False),
            "build_cores_target_str": getattr(
                run_context, "build_cores_target_str", ""
            ),
            "monitor_script_pinned_to_core_info": getattr(
                run_context, "monitor_script_pinned_to_core_info", ""
            ),
            "monitor_core_id": getattr(run_context, "monitor_core_id", None),
        }

        metadata_path = self.output_dir / "metadata.json"
        self.storage.save_dict(metadata, str(metadata_path))
        logger.debug(f"Saved metadata to: {metadata_path}")

    def _save_analysis_results(self, results: MonitoringResults) -> None:
        """
        Save processed analysis results to JSON format.

        This method extracts key statistics and analysis results from the
        MonitoringResults object and saves them in a structured JSON format
        for easy consumption by analysis tools and scripts.

        Args:
            results: MonitoringResults object containing processed statistics
                including peak memory usage, category breakdowns, and sample counts

        Note:
            Analysis results are always saved in JSON format for structured
            access by downstream analysis tools.
        """
        analysis_data = {
            "peak_overall_memory_kb": results.peak_overall_memory_kb,
            "peak_overall_memory_epoch": results.peak_overall_memory_epoch,
            "category_peak_sum": results.category_peak_sum,
            "category_stats": results.category_stats,
            "total_samples": len(results.all_samples_data),
        }

        analysis_path = self.output_dir / "analysis_results.json"
        self.storage.save_dict(analysis_data, str(analysis_path))
        logger.debug(f"Saved analysis results to: {analysis_path}")

    def _save_summary_log(self, results: MonitoringResults, run_context: Any) -> None:
        """Save human-readable summary log."""
        summary_path = self.output_dir / "summary.log"

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Build Monitoring Summary\n")
            f.write("=======================\n\n")
            f.write(f"Project: {getattr(run_context, 'project_name', 'unknown')}\n")
            f.write(f"Parallelism: -j{getattr(run_context, 'parallelism_level', 0)}\n")

            # Format peak memory in GB
            peak_memory_gb = results.peak_overall_memory_kb / 1024 / 1024
            f.write(f"Peak Overall Memory (PSS_KB): {peak_memory_gb:.2f} GB\n")
            f.write(f"Samples Collected: {len(results.all_samples_data)}\n\n")

            # Write category statistics
            if results.category_stats:
                f.write("--- Category Peak Memory Usage ---\n")

                # Group by major category
                major_categories = {}
                for category, stats in results.category_stats.items():
                    if ":" in category:
                        major_cat = category.split(":")[0]
                    else:
                        major_cat = category

                    if major_cat not in major_categories:
                        major_categories[major_cat] = {}
                    major_categories[major_cat][category] = stats

                # Write grouped statistics
                for major_cat, minor_cats in major_categories.items():
                    total_peak_kb = sum(
                        stats["peak_sum_kb"] for stats in minor_cats.values()
                    )
                    total_pids = sum(
                        stats["process_count"] for stats in minor_cats.values()
                    )

                    f.write(f"{major_cat}:\n")
                    f.write(
                        f"  Total Peak Memory: {total_peak_kb} KB ({total_pids} total pids)\n"
                    )

                    # Write minor categories
                    for minor_cat, stats in minor_cats.items():
                        peak_kb = stats["peak_sum_kb"]
                        process_count = stats["process_count"]
                        single_peak_kb = int(stats["average_peak_kb"])
                        f.write(
                            f"    {minor_cat}: {peak_kb} KB (total, {process_count} pids), "
                            f"single process peak: {single_peak_kb} KB\n"
                        )

                    f.write("\n")

        logger.info(f"Saved summary log to: {summary_path}")

    def load_memory_samples(self, columns: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Load memory samples from storage with optional column pruning.

        This method loads the memory samples data from the configured storage format.
        It supports column pruning for better performance when only specific columns
        are needed for analysis.

        Args:
            columns: Optional list of columns to load (for performance optimization)
                When specified, only these columns will be loaded from storage,
                significantly reducing memory usage and improving performance
                for large datasets. Common columns to select include:
                ["timestamp", "pid", "process_name", "rss_kb", "category"]

        Returns:
            Polars DataFrame with memory samples containing requested columns

        Raises:
            FileNotFoundError: If the memory samples file doesn't exist
            Exception: If any other error occurs during loading

        Examples:
            # Load all columns
            df = storage_manager.load_memory_samples()

            # Load only specific columns for better performance
            df = storage_manager.load_memory_samples(
                columns=["timestamp", "pid", "rss_kb"]
            )
        """
        if self.storage_format == "parquet":
            file_path = self.output_dir / "memory_samples.parquet"
            if self.storage.file_exists(str(file_path)):
                return self.storage.load_dataframe(str(file_path), columns)
        else:
            file_path = self.output_dir / "memory_samples.json"
            if self.storage.file_exists(str(file_path)):
                data = self.storage.load_dict(str(file_path))
                return pl.DataFrame(data.get("samples", []))

        raise FileNotFoundError(f"No memory samples found in {self.output_dir}")

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about stored files and storage configuration.

        This method provides detailed information about the storage setup and
        existing files, useful for debugging, monitoring, and analysis tools.

        Returns:
            Dictionary containing storage information with the following structure:
            {
                "storage_format": str,      # Current storage format (parquet/json)
                "compression": str,         # Compression algorithm used
                "output_dir": str,          # Output directory path
                "files": {                  # Information about existing files
                    "filename": {
                        "size_bytes": int,  # File size in bytes
                        "exists": bool      # Whether file exists
                    }
                }
            }

        Note:
            This method checks for all possible output files including main data files
            and legacy formats, providing a complete overview of the storage state.
        """
        info = {
            "storage_format": self.storage_format,
            "compression": self.compression,
            "output_dir": str(self.output_dir),
            "files": {},
        }

        # Check for main data files
        for filename in [
            "memory_samples.parquet",
            "memory_samples.json",
            "memory_samples.csv",
        ]:
            file_path = self.output_dir / filename
            if file_path.exists():
                info["files"][filename] = {
                    "size_bytes": self.storage.get_file_size(str(file_path)),
                    "exists": True,
                }

        return info
