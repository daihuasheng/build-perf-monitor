# API Reference

This document provides detailed API reference for MyMonitor's core modules and classes.

## Core Classes

### DataStorageManager

High-level interface for managing monitoring data storage.

```python
from mymonitor.storage.data_manager import DataStorageManager
from pathlib import Path

# Initialize storage manager
manager = DataStorageManager(Path("output_directory"))

# Save monitoring results
manager.save_monitoring_results(results, run_context)

# Load memory samples
df = manager.load_memory_samples()

# Load specific columns only
df = manager.load_memory_samples(columns=["pid", "process_name", "rss_kb"])

# Get storage information
info = manager.get_storage_info()
```

**Methods:**

- `save_monitoring_results(results: MonitoringResults, run_context: Any) -> None`
  - Saves complete monitoring results including memory samples, metadata, and analysis
  - Creates multiple output files: Parquet data, JSON metadata, summary logs

- `load_memory_samples(columns: Optional[List[str]] = None) -> pl.DataFrame`
  - Loads memory sample data from storage
  - Supports column pruning for better performance
  - Returns Polars DataFrame for efficient data manipulation

- `get_storage_info() -> Dict[str, Any]`
  - Returns information about storage format, compression, and file details
  - Useful for debugging and monitoring storage usage

### AbstractMemoryCollector

Base class for memory data collection implementations.

```python
from mymonitor.collectors.base import AbstractMemoryCollector

class CustomCollector(AbstractMemoryCollector):
    def collect_single_sample(self) -> List[ProcessMemorySample]:
        # Implementation for collecting memory data
        pass
```

**Abstract Methods:**

- `collect_single_sample() -> List[ProcessMemorySample]`
  - Collects memory data for all monitored processes
  - Returns list of ProcessMemorySample objects
  - Must be implemented by concrete collector classes

### ProcessClassifier

Engine for categorizing build processes based on configurable rules.

```python
from mymonitor.classification.classifier import get_process_category

# Classify a process
category, subcategory = get_process_category(
    cmd_name="gcc",
    full_cmd="/usr/bin/gcc -O2 main.c",
    rules=classification_rules
)
```

**Functions:**

- `get_process_category(cmd_name: str, full_cmd: str, rules: List[RuleConfig]) -> Tuple[str, str]`
  - Classifies process based on command name and full command line
  - Returns tuple of (major_category, minor_category)
  - Uses priority-based rule matching

## Configuration Classes

### StorageConfig

Configuration for data storage options.

```python
from mymonitor.config.storage_config import StorageConfig

# Create from dictionary
config = StorageConfig.from_dict({
    "format": "parquet",
    "compression": "snappy",
    "generate_legacy_formats": False
})

# Access properties
print(config.format)        # "parquet"
print(config.compression)   # "snappy"
```

**Properties:**

- `format: str` - Storage format ("parquet" or "json")
- `compression: str` - Compression algorithm for Parquet
- `generate_legacy_formats: bool` - Whether to generate CSV files

### MonitorConfig

Main monitoring configuration.

```python
from mymonitor.config.validators import validate_monitor_config

# Validate configuration
validated_config = validate_monitor_config(config_dict)
```

## Data Models

### MonitoringResults

Container for complete monitoring session results.

```python
from mymonitor.models.results import MonitoringResults

results = MonitoringResults(
    all_samples_data=memory_samples,
    category_stats=category_statistics,
    peak_overall_memory_kb=max_memory,
    peak_overall_memory_epoch=peak_timestamp,
    category_peak_sum=peak_by_category,
    category_pid_set=processes_by_category
)
```

**Properties:**

- `all_samples_data: List[Dict]` - Raw memory sample data
- `category_stats: Dict` - Statistics grouped by process category
- `peak_overall_memory_kb: int` - Maximum total memory usage
- `peak_overall_memory_epoch: float` - Timestamp of peak memory usage
- `category_peak_sum: Dict` - Peak memory usage by category
- `category_pid_set: Dict` - Process IDs grouped by category

### ProcessMemorySample

Individual memory measurement for a process.

```python
from mymonitor.models.memory import ProcessMemorySample

sample = ProcessMemorySample(
    timestamp=time.time(),
    pid=12345,
    process_name="gcc",
    rss_kb=102400,
    vms_kb=204800,
    pss_kb=98304,
    category="compiler"
)
```

**Properties:**

- `timestamp: float` - Unix timestamp of measurement
- `pid: int` - Process ID
- `process_name: str` - Process executable name
- `rss_kb: int` - Resident Set Size in KB
- `vms_kb: int` - Virtual Memory Size in KB
- `pss_kb: int` - Proportional Set Size in KB
- `category: str` - Process classification category

## Utility Functions

### Storage Factory

```python
from mymonitor.storage.factory import create_storage

# Create Parquet storage
storage = create_storage("parquet", compression="snappy")

# Create JSON storage
storage = create_storage("json")
```

### CPU Management

```python
from mymonitor.system.cpu_manager import CPUManager

cpu_manager = CPUManager()
allocation_plan = cpu_manager.allocate_cores(
    strategy="adaptive",
    parallelism_level=8,
    enable_affinity=True
)
```

## Error Handling

All API functions may raise the following exceptions:

- `ConfigurationError` - Invalid configuration parameters
- `StorageError` - Storage operation failures
- `CollectionError` - Memory collection failures
- `ValidationError` - Data validation failures

## Thread Safety

- `DataStorageManager` - Thread-safe for concurrent read operations
- `ProcessClassifier` - Thread-safe with internal caching
- `AbstractMemoryCollector` - Implementation-dependent
- Configuration classes - Immutable after creation

## Performance Considerations

- Use column pruning when loading large datasets: `load_memory_samples(columns=["pid", "rss_kb"])`
- Parquet format provides 3-5x faster queries compared to CSV
- Memory collectors should be used with appropriate CPU affinity settings
- Classification rules are cached for better performance
