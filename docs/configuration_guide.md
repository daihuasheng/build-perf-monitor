# Configuration Guide

MyMonitor uses TOML configuration files to customize monitoring behavior. This guide covers all configuration options and best practices.

## Configuration Files Overview

MyMonitor uses three main configuration files:

- `conf/config.toml` - Main monitoring settings
- `conf/projects.toml` - Project-specific build configurations
- `conf/rules.toml` - Process classification rules

## Main Configuration (`conf/config.toml`)

### General Settings

```toml
[monitor.general]
# Default parallelism levels to test
default_jobs = [4, 8, 16]

# Skip plot generation after monitoring
skip_plots = false

# Enable detailed logging
verbose = false
```

**Options:**
- `default_jobs`: List of `-j` values to test automatically
- `skip_plots`: Set to `true` to disable automatic plot generation
- `verbose`: Enable debug-level logging

### Collection Settings

```toml
[monitor.collection]
# Sampling interval in seconds
interval_seconds = 0.05

# Memory collector type
metric_type = "pss_psutil"

# Process scanning mode
pss_collector_mode = "full_scan"

# Enable descendants-only mode for better performance
descendants_only = false
```

**Collector Types:**
- `pss_psutil`: PSS memory using psutil (recommended)
- `rss_psutil`: RSS memory using psutil
- `hybrid`: Hybrid producer-consumer model

**Scanning Modes:**
- `full_scan`: Scan all system processes (reliable)
- `descendants_only`: Only scan build process descendants (faster)

### Storage Settings

```toml
[monitor.storage]
# Storage format: "parquet" (recommended) or "json"
format = "parquet"

# Compression algorithm for Parquet
compression = "snappy"

# Generate legacy CSV files for backward compatibility
generate_legacy_formats = false
```

**Compression Options:**
- `snappy`: Fast compression/decompression (default)
- `gzip`: Higher compression ratio, slower
- `brotli`: High compression ratio
- `lz4`: Very fast compression
- `zstd`: Modern balanced compression

### CPU Scheduling

```toml
[monitor.scheduling]
# CPU allocation strategy
scheduling_policy = "adaptive"

# Enable CPU affinity pinning
enable_cpu_affinity = true

# Specific core for monitoring process
monitor_core = 0

# Manual core allocation (for manual policy)
build_cores = "0-7"
monitoring_cores = "8-11"
```

**Scheduling Policies:**
- `adaptive`: Automatically allocate cores based on system capacity
- `manual`: Use explicitly specified core ranges

### Thread Pool Configuration

```toml
[monitor.thread_pools.monitoring]
# Number of monitoring threads
max_workers = 4

# Enable CPU affinity for monitoring threads
enable_affinity = true

[monitor.thread_pools.build]
# Build execution threads
max_workers = 1
enable_affinity = false

[monitor.thread_pools.io]
# I/O operation threads
max_workers = 2
enable_affinity = false
```

## Project Configuration (`conf/projects.toml`)

Define build projects and their specific settings:

```toml
[[projects]]
name = "qemu"
dir = "/host/qemu/build"
build_command_template = "make -j<N>"
clean_command_template = "make clean"
process_pattern = "make|gcc|g\\+\\+|ld|ar|ranlib"

[[projects]]
name = "linux_kernel"
dir = "/usr/src/linux"
build_command_template = "make -j<N> bzImage"
clean_command_template = "make mrproper"
process_pattern = "make|gcc|ld|scripts"
```

**Required Fields:**
- `name`: Unique project identifier
- `dir`: Build directory path
- `build_command_template`: Build command with `<N>` placeholder for parallelism
- `clean_command_template`: Command to clean build artifacts
- `process_pattern`: Regex pattern to match relevant processes

## Classification Rules (`conf/rules.toml`)

Define how processes are categorized:

```toml
[[rules]]
major_category = "Compiler"
minor_category = "CPP_Compile"
priority = 1
match_type = "regex"
pattern = "g\\+\\+|clang\\+\\+"

[[rules]]
major_category = "Compiler"
minor_category = "C_Compile"
priority = 2
match_type = "exact"
pattern = "gcc"

[[rules]]
major_category = "Linker"
minor_category = "Static_Link"
priority = 3
match_type = "contains"
pattern = "ar"
```

**Rule Fields:**
- `major_category`: Primary classification (e.g., "Compiler", "Linker")
- `minor_category`: Subcategory (e.g., "CPP_Compile", "Static_Link")
- `priority`: Lower numbers = higher priority
- `match_type`: How to match the pattern
- `pattern`: Pattern to match against process names

**Match Types:**
- `exact`: Exact string match
- `regex`: Regular expression match
- `contains`: Substring match
- `in_list`: Match against comma-separated list

## Environment Variables

Override configuration with environment variables:

```bash
# Override sampling interval
export MYMONITOR_INTERVAL=0.1

# Override storage format
export MYMONITOR_STORAGE_FORMAT=json

# Override compression
export MYMONITOR_COMPRESSION=gzip

# Enable verbose logging
export MYMONITOR_VERBOSE=true
```

## Best Practices

### Performance Optimization

1. **Sampling Interval**: Use 0.05s for detailed analysis, 0.1s for general monitoring
2. **Storage Format**: Always use Parquet for better performance
3. **Compression**: Use `snappy` for balanced performance, `gzip` for storage savings
4. **CPU Affinity**: Enable for consistent performance measurements

### Resource Management

1. **Thread Pool Sizing**: 
   - Monitoring threads: `min(4, parallelism_level)`
   - Build threads: Always 1
   - I/O threads: 2-4 depending on storage speed

2. **Memory Usage**:
   - Enable `descendants_only` for large systems
   - Use column pruning when loading data
   - Clean up old log files regularly

### Reliability

1. **Error Handling**: Enable verbose logging for troubleshooting
2. **Backup**: Keep backup copies of working configurations
3. **Validation**: Test configuration changes with small builds first

## Configuration Validation

MyMonitor validates all configuration files on startup. Common validation errors:

### Invalid Core Ranges
```
Error: Invalid core range '0-99' - only 8 cores available
```
**Solution**: Adjust core ranges to match system capacity

### Missing Project Directory
```
Error: Project directory '/invalid/path' does not exist
```
**Solution**: Ensure project directories exist and are accessible

### Invalid Regex Pattern
```
Error: Invalid regex pattern 'gcc[' in classification rule
```
**Solution**: Fix regex syntax in rules.toml

## Advanced Configuration

### Custom Collectors

Create custom memory collectors by extending `AbstractMemoryCollector`:

```python
from mymonitor.collectors.base import AbstractMemoryCollector

class CustomCollector(AbstractMemoryCollector):
    def collect_single_sample(self):
        # Custom implementation
        pass
```

Register in configuration:
```toml
[monitor.collection]
metric_type = "custom"
custom_collector_class = "mymodule.CustomCollector"
```

### Dynamic Configuration

Load configuration programmatically:

```python
from mymonitor.config.manager import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config()

# Override specific settings
config.monitor.collection.interval_seconds = 0.1
```

## Troubleshooting

### Common Issues

1. **High CPU Usage**: Increase sampling interval or enable descendants_only
2. **Missing Processes**: Adjust process_pattern in projects.toml
3. **Storage Errors**: Check disk space and permissions
4. **Classification Issues**: Review and update rules.toml priorities

### Debug Mode

Enable comprehensive debugging:

```toml
[monitor.general]
verbose = true

[monitor.collection]
debug_process_discovery = true
debug_memory_collection = true
```

This enables detailed logging for process discovery and memory collection phases.
