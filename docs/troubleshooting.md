# Troubleshooting Guide

This guide helps diagnose and resolve common issues with MyMonitor.

## Common Issues

### 1. Installation Problems

#### UV Installation Issues

**Problem**: `uv: command not found`

**Solution**:
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

#### Dependency Conflicts

**Problem**: Package dependency conflicts during installation

**Solution**:
```bash
# Clean install with UV
uv pip install --force-reinstall -e ".[dev]"

# Or create fresh virtual environment
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Configuration Issues

#### Invalid Configuration Files

**Problem**: `ConfigurationError: Invalid TOML syntax`

**Solution**:
1. Validate TOML syntax using online validator
2. Check for missing quotes around strings
3. Ensure proper array formatting

```toml
# Correct
default_jobs = [4, 8, 16]
process_pattern = "gcc|g\\+\\+"

# Incorrect
default_jobs = 4, 8, 16  # Missing brackets
process_pattern = gcc|g++  # Missing quotes
```

#### Missing Project Directory

**Problem**: `Project directory '/path/to/project' does not exist`

**Solution**:
```bash
# Verify directory exists
ls -la /path/to/project

# Update projects.toml with correct path
[[projects]]
name = "myproject"
dir = "/correct/path/to/project"
```

#### Invalid Core Ranges

**Problem**: `Invalid core range '0-15' - only 8 cores available`

**Solution**:
```bash
# Check available cores
nproc

# Update configuration
[monitor.scheduling]
build_cores = "0-7"  # Adjust to available cores
monitoring_cores = "8-11"  # May need to overlap or reduce
```

### 3. Permission Issues

#### Process Access Denied

**Problem**: `PermissionError: [Errno 13] Permission denied: '/proc/1234/stat'`

**Solution**:
```bash
# Run with appropriate permissions
sudo python -m mymonitor monitor -p myproject

# Or adjust process discovery scope
[monitor.collection]
descendants_only = true  # Only monitor build descendants
```

#### File System Permissions

**Problem**: Cannot write to output directory

**Solution**:
```bash
# Check directory permissions
ls -la logs/

# Fix permissions
chmod 755 logs/
chown $USER:$USER logs/

# Or specify different output directory
python -m mymonitor monitor -p myproject --output-dir /tmp/mymonitor_logs
```

### 4. Memory Collection Issues

#### No Processes Found

**Problem**: `No processes found matching pattern 'gcc|make'`

**Diagnosis**:
```bash
# Check if build processes are running
ps aux | grep -E "gcc|make"

# Verify process pattern in projects.toml
cat conf/projects.toml | grep process_pattern
```

**Solution**:
1. Update process pattern to match actual build tools
2. Use broader pattern initially: `process_pattern = ".*"`
3. Enable debug logging to see discovered processes

```toml
[monitor.general]
verbose = true

[monitor.collection]
debug_process_discovery = true
```

#### High CPU Usage

**Problem**: MyMonitor consuming excessive CPU

**Solution**:
```toml
# Increase sampling interval
[monitor.collection]
interval_seconds = 0.1  # Reduce from 0.05

# Enable descendants-only mode
descendants_only = true

# Reduce thread pool size
[monitor.thread_pools.monitoring]
max_workers = 2  # Reduce from 4
```

#### Memory Collection Errors

**Problem**: `psutil.NoSuchProcess: process no longer exists`

**Solution**: This is normal for short-lived processes. Enable error tolerance:

```toml
[monitor.collection]
ignore_process_errors = true
max_error_rate = 0.1  # Allow 10% error rate
```

### 5. Storage Issues

#### Parquet File Corruption

**Problem**: `ParquetError: Invalid parquet file`

**Diagnosis**:
```bash
# Check file integrity
python -c "import polars as pl; df = pl.read_parquet('memory_samples.parquet')"
```

**Solution**:
```bash
# Enable legacy format generation for backup
[monitor.storage]
generate_legacy_formats = true

# Or use different compression
compression = "gzip"  # More robust than snappy
```

#### Disk Space Issues

**Problem**: `OSError: [Errno 28] No space left on device`

**Solution**:
```bash
# Check disk space
df -h

# Clean old logs
find logs/ -name "*.parquet" -mtime +7 -delete

# Use higher compression
[monitor.storage]
compression = "brotli"  # Higher compression ratio
```

### 6. Performance Issues

#### Slow Data Loading

**Problem**: Loading Parquet files takes too long

**Solution**:
```python
# Use column pruning
df = manager.load_memory_samples(columns=["timestamp", "pid", "rss_kb"])

# Or load in chunks for large files
import polars as pl
df = pl.scan_parquet("memory_samples.parquet").select(["timestamp", "rss_kb"]).collect()
```

#### Build Monitoring Overhead

**Problem**: Monitoring significantly slows down build

**Solution**:
```toml
# Reduce monitoring frequency
[monitor.collection]
interval_seconds = 0.2

# Use fewer monitoring threads
[monitor.thread_pools.monitoring]
max_workers = 1

# Enable CPU affinity to isolate monitoring
[monitor.scheduling]
enable_cpu_affinity = true
```

### 7. Visualization Issues

#### Plot Generation Fails

**Problem**: `ImportError: No module named 'plotly'`

**Solution**:
```bash
# Install visualization dependencies
uv pip install -e ".[export]"

# Or skip plot generation
[monitor.general]
skip_plots = true
```

#### Empty or Incorrect Plots

**Problem**: Plots show no data or incorrect data

**Diagnosis**:
```python
# Check data availability
from mymonitor.storage.data_manager import DataStorageManager
manager = DataStorageManager(Path("logs/run_20231201_120000"))
df = manager.load_memory_samples()
print(f"Loaded {len(df)} samples")
print(df.head())
```

**Solution**:
1. Verify data was collected during monitoring
2. Check process classification rules
3. Ensure sufficient monitoring duration

### 8. System-Specific Issues

#### macOS Issues

**Problem**: `OSError: [Errno 1] Operation not permitted`

**Solution**:
```bash
# Grant Terminal full disk access in System Preferences
# Or use sudo for system process access
sudo python -m mymonitor monitor -p myproject
```

#### Windows Issues

**Problem**: Process monitoring not working on Windows

**Solution**:
```toml
# Use Windows-compatible collector
[monitor.collection]
metric_type = "rss_psutil"  # More reliable on Windows

# Adjust process patterns for Windows
process_pattern = "cl\\.exe|link\\.exe|msbuild"
```

#### Container/Docker Issues

**Problem**: Cannot access host processes from container

**Solution**:
```bash
# Mount /proc when running in container
docker run -v /proc:/host/proc mymonitor

# Or run on host system instead of container
```

## Debugging Techniques

### 1. Enable Verbose Logging

```toml
[monitor.general]
verbose = true

[monitor.collection]
debug_process_discovery = true
debug_memory_collection = true
```

### 2. Test Individual Components

```python
# Test configuration loading
from mymonitor.config.manager import ConfigManager
config = ConfigManager().load_config()
print(config)

# Test memory collection
from mymonitor.collectors.factory import CollectorFactory
collector = CollectorFactory().create_collector("pss_psutil")
samples = collector.collect_single_sample()
print(f"Collected {len(samples)} samples")

# Test storage
from mymonitor.storage.data_manager import DataStorageManager
manager = DataStorageManager(Path("/tmp/test"))
info = manager.get_storage_info()
print(info)
```

### 3. Monitor System Resources

```bash
# Monitor CPU usage
top -p $(pgrep -f mymonitor)

# Monitor memory usage
ps -o pid,vsz,rss,comm -p $(pgrep -f mymonitor)

# Monitor file descriptors
lsof -p $(pgrep -f mymonitor)
```

### 4. Validate Data Integrity

```python
# Check for data consistency
import polars as pl
df = pl.read_parquet("memory_samples.parquet")

# Verify timestamps are sequential
assert df["timestamp"].is_sorted()

# Check for missing data
assert df.null_count().sum() == 0

# Verify memory values are reasonable
assert (df["rss_kb"] > 0).all()
```

## Getting Help

### 1. Log Analysis

When reporting issues, include relevant log excerpts:

```bash
# Enable verbose logging and capture output
python -m mymonitor monitor -p myproject --verbose 2>&1 | tee mymonitor.log

# Extract error messages
grep -i error mymonitor.log
grep -i exception mymonitor.log
```

### 2. System Information

Provide system details:

```bash
# Python version
python --version

# UV version
uv --version

# System information
uname -a

# Available cores
nproc

# Memory information
free -h
```

### 3. Configuration Validation

Validate and share configuration:

```bash
# Test configuration loading
python -c "
from mymonitor.config.manager import ConfigManager
try:
    config = ConfigManager().load_config()
    print('Configuration loaded successfully')
except Exception as e:
    print(f'Configuration error: {e}')
"
```

### 4. Minimal Reproduction

Create minimal test case:

```bash
# Test with simple project
mkdir /tmp/test_project
cd /tmp/test_project
echo 'int main() { return 0; }' > test.c

# Create minimal configuration
cat > conf/projects.toml << EOF
[[projects]]
name = "test"
dir = "/tmp/test_project"
build_command_template = "gcc -o test test.c"
clean_command_template = "rm -f test"
process_pattern = "gcc"
EOF

# Run monitoring
python -m mymonitor monitor -p test -j 1
```

## Performance Optimization

### 1. Monitoring Overhead Reduction

```toml
# Optimize for minimal overhead
[monitor.collection]
interval_seconds = 0.1
descendants_only = true
metric_type = "rss_psutil"  # Faster than PSS

[monitor.thread_pools.monitoring]
max_workers = 1

[monitor.scheduling]
enable_cpu_affinity = true
monitor_core = 0  # Isolate monitoring to specific core
```

### 2. Storage Optimization

```toml
# Optimize storage performance
[monitor.storage]
format = "parquet"
compression = "snappy"  # Fast compression
generate_legacy_formats = false
```

### 3. Memory Usage Optimization

```python
# Use column pruning when loading data
df = manager.load_memory_samples(columns=["timestamp", "pid", "rss_kb"])

# Process data in chunks for large datasets
chunk_size = 10000
for chunk in df.iter_slices(chunk_size):
    process_chunk(chunk)
```
