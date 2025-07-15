#!/usr/bin/env python3
"""
Storage format conversion tool.

This tool converts monitoring data between different storage formats,
particularly useful for migrating from CSV/JSON to Parquet format.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import polars as pl

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mymonitor.storage.factory import create_storage

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration for the conversion tool.

    Configures the logging system with appropriate format and level.
    When verbose mode is enabled, DEBUG level messages are shown,
    otherwise only INFO and above are displayed.

    Args:
        verbose: Whether to enable verbose (DEBUG) logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def convert_file(
    input_path: Path,
    output_path: Path,
    input_format: str,
    output_format: str,
    compression: str = "snappy",
) -> bool:
    """
    Convert a single file from one storage format to another.

    This function handles the conversion between different data storage formats,
    automatically detecting and handling different JSON structures. It provides
    detailed logging of the conversion process including file size changes.

    Args:
        input_path: Path to the input file to convert
        output_path: Path where the converted file will be saved
        input_format: Input format ('csv', 'json', 'parquet')
        output_format: Output format ('csv', 'json', 'parquet')
        compression: Compression algorithm for Parquet output
            Options: 'snappy', 'gzip', 'brotli', 'lz4', 'zstd'

    Returns:
        True if conversion completed successfully, False if any error occurred

    Note:
        For JSON input, the function handles both list-of-records format and
        dictionary format with a 'samples' key. The output directory is
        automatically created if it doesn't exist.
    """
    try:
        logger.info(f"Converting {input_path} -> {output_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load data based on input format
        if input_format == "csv":
            df = pl.read_csv(input_path)
        elif input_format == "parquet":
            df = pl.read_parquet(input_path)
        elif input_format == "json":
            # For JSON files, try to load as a list of records
            import json

            with open(input_path, "r") as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                df = pl.DataFrame(data)
            elif isinstance(data, dict) and "samples" in data:
                df = pl.DataFrame(data["samples"])
            else:
                logger.error(f"Unsupported JSON structure in {input_path}")
                return False
        else:
            logger.error(f"Unsupported input format: {input_format}")
            return False

        # Save data based on output format
        if output_format == "csv":
            df.write_csv(output_path)
        elif output_format == "parquet":
            df.write_parquet(output_path, compression=compression)
        elif output_format == "json":
            # Save as list of records
            data = df.to_dicts()
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            logger.error(f"Unsupported output format: {output_format}")
            return False

        # Report file sizes
        input_size = input_path.stat().st_size
        output_size = output_path.stat().st_size
        compression_ratio = (
            (1 - output_size / input_size) * 100 if input_size > 0 else 0
        )

        logger.info(
            f"Conversion complete: {input_size:,} bytes -> {output_size:,} bytes "
            f"({compression_ratio:+.1f}% size change)"
        )

        return True

    except Exception as e:
        logger.error(f"Error converting {input_path}: {e}")
        return False


def find_data_files(directory: Path, pattern: str) -> List[Path]:
    """
    Find data files in a directory matching a pattern.

    Args:
        directory: Directory to search
        pattern: File pattern (e.g., "*.csv", "*.parquet")

    Returns:
        List of matching file paths
    """
    return list(directory.glob(pattern))


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    input_format: str,
    output_format: str,
    compression: str = "snappy",
    recursive: bool = False,
) -> int:
    """
    Convert all data files in a directory from one format to another.

    This function scans the input directory for files matching the specified
    input format and converts each file to the output format, preserving the
    directory structure in the output directory. It provides progress reporting
    and summary statistics of the conversion process.

    Args:
        input_dir: Source directory containing files to convert
        output_dir: Destination directory for converted files
        input_format: Input file format extension ('csv', 'json', 'parquet')
        output_format: Output file format extension ('csv', 'json', 'parquet')
        compression: Compression algorithm for Parquet output
            Options: 'snappy', 'gzip', 'brotli', 'lz4', 'zstd'
        recursive: Whether to search subdirectories recursively
            When True, the full directory structure is preserved in the output

    Returns:
        Number of files successfully converted (0 if no files were found or converted)

    Note:
        The output directory structure mirrors the input directory structure,
        with file extensions changed to match the output format.
    """
    # Determine file patterns
    input_pattern = f"*.{input_format}"
    output_ext = output_format

    # Find input files
    if recursive:
        input_files = list(input_dir.rglob(input_pattern))
    else:
        input_files = list(input_dir.glob(input_pattern))

    if not input_files:
        logger.warning(f"No {input_format} files found in {input_dir}")
        return 0

    logger.info(f"Found {len(input_files)} {input_format} files to convert")

    converted_count = 0

    for input_file in input_files:
        # Calculate relative path for maintaining directory structure
        rel_path = input_file.relative_to(input_dir)
        output_file = output_dir / rel_path.with_suffix(f".{output_ext}")

        if convert_file(
            input_file, output_file, input_format, output_format, compression
        ):
            converted_count += 1

    logger.info(f"Successfully converted {converted_count}/{len(input_files)} files")
    return converted_count


def main():
    """
    Main entry point for the storage format conversion tool.

    This function sets up the command-line argument parser, validates inputs,
    and orchestrates the conversion process. It handles both single file and
    directory conversions with comprehensive error handling and logging.

    The tool supports conversion between CSV, JSON, and Parquet formats with
    various compression options for Parquet output. It provides detailed
    progress reporting and error messages for troubleshooting.

    Exit codes:
        0: Successful conversion
        1: Conversion failed or no files processed

    Examples:
        # Convert single file
        python tools/convert_storage.py data.csv data.parquet

        # Convert directory with custom compression
        python tools/convert_storage.py logs/ logs_parquet/ --compression gzip --recursive
    """
    parser = argparse.ArgumentParser(
        description="Convert monitoring data between storage formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single CSV file to Parquet
  python tools/convert_storage.py data.csv data.parquet --input-format csv --output-format parquet
  
  # Convert all CSV files in a directory to Parquet
  python tools/convert_storage.py logs/old/ logs/new/ --input-format csv --output-format parquet
  
  # Convert with different compression
  python tools/convert_storage.py data.csv data.parquet --compression gzip
  
  # Convert recursively
  python tools/convert_storage.py logs/ logs_parquet/ --recursive
        """,
    )

    parser.add_argument("input", help="Input file or directory path")
    parser.add_argument("output", help="Output file or directory path")
    parser.add_argument(
        "--input-format",
        choices=["csv", "json", "parquet"],
        default="csv",
        help="Input format (default: csv)",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json", "parquet"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--compression",
        choices=["snappy", "gzip", "brotli", "lz4", "zstd"],
        default="snappy",
        help="Compression algorithm for Parquet (default: snappy)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for files recursively in directories",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # Perform conversion
    try:
        if input_path.is_file():
            # Single file conversion
            success = convert_file(
                input_path,
                output_path,
                args.input_format,
                args.output_format,
                args.compression,
            )
            sys.exit(0 if success else 1)
        else:
            # Directory conversion
            converted_count = convert_directory(
                input_path,
                output_path,
                args.input_format,
                args.output_format,
                args.compression,
                args.recursive,
            )
            sys.exit(0 if converted_count > 0 else 1)

    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
