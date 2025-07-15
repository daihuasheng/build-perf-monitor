# Storage Formats

> **Languages**: [English](storage_formats.md) | [中文](storage_formats.zh-CN.md)

## Overview

The monitoring system now supports efficient Parquet format storage, providing significant performance and storage advantages compared to traditional CSV/JSON formats.

## Supported Formats

### Parquet Format (Recommended)

**Advantages:**
- **High Storage Efficiency**: 75-80% space reduction compared to CSV/JSON
- **Excellent Query Performance**: Columnar storage, read only needed columns
- **Built-in Compression**: Supports multiple compression algorithms
- **Schema Preservation**: Retains data type information
- **Analytics-Friendly**: Seamless integration with Pandas, Polars, Spark, etc.

**Compression Algorithms:**
- `snappy` (default): Balanced compression ratio and speed
- `gzip`: Higher compression ratio, slower
- `brotli`: High compression ratio
- `lz4`: Fast compression
- `zstd`: Modern efficient compression

### JSON Format

Used for metadata and small configuration files, maintaining human readability.

## Configuration

Configure storage format in `conf/config.toml`:

```toml
[monitor.storage]
# Data storage format: "parquet" (recommended) or "json"
format = "parquet"

# Compression algorithm (for parquet only): "snappy", "gzip", "brotli", "lz4", "zstd"
compression = "snappy"

# Whether to generate legacy formats (backward compatibility)
generate_legacy_formats = false
```

## Generated File Structure

After monitoring completes, the following files are generated in the logs directory:

```text
logs/
└── <project_name>_<timestamp>/
    ├── memory_samples.parquet      # Main monitoring data (Parquet format)
    ├── metadata.json               # Run metadata
    ├── analysis_results.json       # Analysis results
    ├── summary.log                 # Human-readable summary
    └── memory_samples.csv          # Legacy format (only when enabled)
```

## Performance Comparison

The following is a performance comparison of different formats (based on real monitoring data):

| Data Type | CSV Size | Parquet Size | Space Savings | Query Performance Gain |
|-----------|----------|--------------|---------------|------------------------|
| Memory samples (100k rows) | 25 MB | 5 MB | 80% | 3-5x |
| CPU samples (100k rows) | 18 MB | 4 MB | 78% | 3-5x |
| Process info (1k processes) | 2 MB | 0.4 MB | 80% | 2-3x |

## Data Conversion

### Using Conversion Tool

The system provides a built-in conversion tool to migrate existing data:

```bash
# Convert single file
uv run python tools/convert_storage.py \
    old_data.csv new_data.parquet \
    --input-format csv --output-format parquet

# Convert entire directory
uv run python tools/convert_storage.py \
    logs/old_format/ logs/parquet_format/ \
    --input-format csv --output-format parquet --recursive

# Use different compression algorithm
uv run python tools/convert_storage.py \
    data.csv data.parquet --compression gzip
```

### Conversion Tool Options

- `--input-format`: Input format (csv, json, parquet)
- `--output-format`: Output format (csv, json, parquet)
- `--compression`: Compression algorithm (snappy, gzip, brotli, lz4, zstd)
- `--recursive`: Process subdirectories recursively
- `--verbose`: Verbose output

## Data Access

### Using Polars (Recommended)

```python
import polars as pl

# Load complete data
df = pl.read_parquet("memory_samples.parquet")

# Load only specific columns (better performance)
df = pl.read_parquet("memory_samples.parquet", columns=["timestamp", "pid", "rss_kb"])

# Filter data
df = pl.read_parquet("memory_samples.parquet").filter(
    pl.col("category") == "compiler"
)
```

### Using Pandas

```python
import pandas as pd

# Load data
df = pd.read_parquet("memory_samples.parquet")

# Load only specific columns
df = pd.read_parquet("memory_samples.parquet", columns=["timestamp", "pid", "rss_kb"])
```

### Using Storage Manager

```python
from mymonitor.storage.data_manager import DataStorageManager
from pathlib import Path

# Create storage manager
manager = DataStorageManager(Path("logs/project_20230416/"))

# Load memory samples
df = manager.load_memory_samples()

# Load only specific columns
df = manager.load_memory_samples(columns=["pid", "process_name", "rss_kb"])

# Get storage information
info = manager.get_storage_info()
print(f"Storage format: {info['storage_format']}")
print(f"File sizes: {info['files']}")
```

## Backward Compatibility

### Enable Legacy Formats

If you need to generate CSV format simultaneously for backward compatibility:

```toml
[monitor.storage]
format = "parquet"
generate_legacy_formats = true  # Also generate CSV files
```

### Migration Strategies

1. **Gradual Migration**:
   - Enable `generate_legacy_formats = true`
   - Verify Parquet data correctness
   - Update analysis scripts to use Parquet
   - Disable legacy format generation

2. **One-time Migration**:
   - Use conversion tool to convert existing data
   - Update configuration to use Parquet
   - Update analysis scripts

## Best Practices

1. **Use Parquet Format**: Get optimal performance and storage efficiency
2. **Choose Appropriate Compression Algorithm**:
   - General purpose: `snappy` (default)
   - Storage priority: `gzip` or `brotli`
   - Speed priority: `lz4`
3. **Column-wise Queries**: Load only needed columns for better performance
4. **Batch Processing**: Use batch reading for large datasets
5. **Regular Cleanup**: Delete unnecessary legacy format files

## Troubleshooting

### Common Issues

**Q: Cannot open Parquet files**
A: Ensure `polars-lts-cpu[pandas, pyarrow]` dependencies are installed

**Q: Data doesn't match after conversion**
A: Check data types and encoding, some special characters may need handling

**Q: Parquet files larger than CSV**
A: This is normal for small files, Parquet contains metadata. Large files will be significantly smaller.

**Q: Cannot read old CSV files**
A: Use conversion tool to migrate to Parquet format

### Performance Tuning

1. **Column Pruning**: Read only needed columns
2. **Predicate Pushdown**: Apply filter conditions during reading
3. **Batch Processing**: Use chunked reading for large datasets
4. **Appropriate Compression**: Choose compression algorithm based on use case

## Technical Details

### Data Schema

Parquet files preserve the following data schema:

```text
memory_samples.parquet:
├── timestamp: TIMESTAMP
├── pid: INT64
├── process_name: STRING
├── rss_kb: INT64
├── vms_kb: INT64
├── pss_kb: INT64
└── category: STRING
```

### Compression Performance

Characteristics of different compression algorithms:

| Algorithm | Compression Ratio | Compression Speed | Decompression Speed | Use Case |
|-----------|------------------|-------------------|---------------------|----------|
| snappy | Medium | Fast | Fast | General purpose, default choice |
| gzip | High | Slow | Medium | Storage priority |
| brotli | Very High | Slow | Medium | Long-term storage |
| lz4 | Low | Very Fast | Very Fast | Real-time processing |
| zstd | High | Fast | Fast | Modern applications |
