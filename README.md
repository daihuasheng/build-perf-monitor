# MyMonitor - Build Performance Monitoring Tool

> **Languages**: [English](README.md) | [中文](README.zh-CN.md)

MyMonitor is a comprehensive build performance monitoring tool that tracks memory usage during software compilation processes. It provides detailed insights into memory consumption patterns across different process categories and parallelism levels.

## 🚀 Features

- **Real-time Memory Monitoring**: Track PSS/RSS memory usage during build processes
- **Process Classification**: Automatically categorize build processes (compilers, linkers, scripts, etc.)
- **Multi-parallelism Analysis**: Compare performance across different `-j` levels
- **Interactive Visualizations**: Generate detailed time-series and summary plots
- **Modular Architecture**: Clean, maintainable codebase with specialized modules
- **Advanced Error Handling**: Robust retry mechanisms and circuit breaker patterns
- **Comprehensive Reporting**: Hierarchical category statistics and build summaries

## 📁 Architecture Overview

MyMonitor follows a modular architecture with clear separation of concerns:

```
src/mymonitor/
├── cli/                    # Command-line interface and orchestration
├── config/                 # Configuration management (TOML files)
├── models/                 # Data models and structures
├── validation/             # Input validation and error handling strategies
├── system/                 # System interaction (CPU allocation, commands)
├── classification/         # Process categorization engine
├── collectors/             # Memory data collection (PSS/RSS)
├── monitoring/             # Monitoring coordination
└── executor/               # Build process execution and thread pool management
```

## 🛠️ Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd mymonitor

# Install with uv
uv pip install -e .

# For development with testing dependencies
uv pip install -e ".[dev]"

# For PNG export capabilities
uv pip install -e ".[export]"
```

### Using Pip

```bash
pip install -e .
```

## ⚙️ Configuration

MyMonitor uses TOML configuration files located in the `conf/` directory:

### Main Configuration (`conf/config.toml`)
```toml
[monitor.general]
default_jobs = [4, 8, 16]           # Default parallelism levels
skip_plots = false                   # Generate plots after monitoring
log_root_dir = "logs"               # Output directory

[monitor.collection]
interval_seconds = 0.05             # Sampling interval
metric_type = "pss_psutil"          # Memory collector type
pss_collector_mode = "full_scan"    # Process scanning mode

[monitor.scheduling]
scheduling_policy = "adaptive"       # CPU scheduling strategy
monitor_core = 0                    # Core for monitoring processes
```

### Projects Configuration (`conf/projects.toml`)
```toml
[[projects]]
name = "qemu"
dir = "/host/qemu/build"
build_command_template = "make -j<N>"
process_pattern = "make|gcc|clang|ld|..."
clean_command_template = "make clean"
```

### Classification Rules (`conf/rules.toml`)
Defines process categorization rules with major and minor categories.

## 🖥️ Command Line Usage

### Basic Usage

```bash
# Monitor default projects with default parallelism levels
mymonitor

# Monitor specific project
mymonitor -p qemu

# Monitor with custom parallelism levels
mymonitor -p qemu -j 8,4,2,1

# Skip pre-build cleanup
mymonitor -p qemu --no-pre-clean
```

### Command Line Options

- `-p, --project PROJECT`: Specify project to monitor
- `-j, --jobs JOBS`: Comma-separated parallelism levels (e.g., "8,16")
- `--no-pre-clean`: Skip pre-build cleanup step
- `-h, --help`: Show help message

### Available Projects

Current configured projects:
- `qemu`: QEMU virtualization platform
- `aosp`: Android Open Source Project
- `chromium`: Chromium web browser

## 📊 Output Structure

MyMonitor generates organized output in the `logs/` directory:

```
logs/
└── run_20250703_143052/              # Timestamped run directory
    ├── qemu_j8_pss_psutil_20250703_143052/   # Per-parallelism data
    │   ├── summary.log               # Enhanced summary with category stats
    │   ├── memory_samples.parquet    # Raw memory data
    │   ├── build_stdout.log          # Build output
    │   ├── metadata.log              # Run metadata
    │   ├── qemu_j8_PSS_KB_lines_plot.html     # Time-series line plot
    │   └── qemu_j8_PSS_KB_stacked_plot.html   # Stacked area plot
    ├── qemu_j4_pss_psutil_20250703_143053/   # Additional parallelism levels
    │   └── ...
    └── qemu_build_summary_plot.html   # Cross-parallelism comparison
```

### Enhanced Summary Format

The new summary.log format provides comprehensive build analysis:

```
Project: qemu
Parallelism: -j8
Total Build & Monitoring Duration: 125.3s (125.30 seconds)
Peak Overall Memory (PSS_KB): 15000000 KB
Samples Collected: 2506
Build Exit Code: 0

--- Category Peak Memory Usage ---

CPP_Compile:
  Total Peak Memory: 12000000 KB (800 total pids)
    GCCInternalCompiler: 10000000 KB (600 pids)
    Driver_Compile: 2000000 KB (200 pids)

CPP_Link:
  Total Peak Memory: 2500000 KB (45 total pids)
    DirectLinker: 2500000 KB (45 pids)
...
```

## 📈 Visualization

MyMonitor automatically generates interactive plots using Plotly:

### Summary Plots
- **Cross-parallelism comparison**: Build time vs. memory usage across different `-j` levels
- **Dual-axis visualization**: Memory (bars) and duration (line) on the same chart

### Detailed Plots
- **Time-series line plots**: Memory usage trends over time by category
- **Stacked area plots**: Total memory composition by process category
- **Interactive features**: Zoom, pan, hover details, category filtering

### Plot Generation

Plots are generated automatically after monitoring, or manually:

```bash
# Generate all plots for a run
python tools/plotter.py --log-dir logs/run_20250703_143052

# Generate only summary plot
python tools/plotter.py --log-dir logs/run_20250703_143052 --summary-plot

# Generate plots for specific parallelism
python tools/plotter.py --log-dir logs/run_20250703_143052 --jobs 8

# Filter by category or top-N
python tools/plotter.py --log-dir logs/run_20250703_143052 --category CPP_Compile
python tools/plotter.py --log-dir logs/run_20250703_143052 --top-n 5
```

## 🔧 Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/test_basic_monitoring.py
uv run pytest tests/test_plotter_tool.py

# Run with verbose output
uv run pytest -v
```

### Test Coverage

MyMonitor includes comprehensive test coverage:
- **Basic monitoring**: End-to-end workflow validation
- **Process classification**: Rule engine and categorization
- **Plotting tools**: Chart generation and format validation
- **Configuration**: TOML parsing and validation
- **System utilities**: CPU allocation and process management

### Code Quality

The codebase follows modern Python practices:
- **Type hints**: Full type annotation coverage
- **Error handling**: Comprehensive validation and error recovery
- **Modular design**: Clear separation of concerns
- **Documentation**: Detailed docstrings and comments

## 🆕 Recent Improvements

### Major Refactoring (v2.0)
- **Modular Architecture**: Replaced monolithic files with specialized modules
- **Enhanced Error Handling**: Added retry mechanisms, circuit breakers, and recovery strategies
- **Improved CLI**: Added `-p` alias and better argument validation
- **Better Testing**: Expanded test coverage with 60+ test cases

### Summary and Visualization Enhancements
- **Hierarchical Statistics**: Major/minor category grouping in summaries
- **Plotter Integration**: Improved format compatibility and plot organization
- **Build Timing**: Added comprehensive duration tracking
- **Memory Metrics**: Clear PSS_KB/RSS_KB labeling

### Performance Optimizations
- **Collectors Consolidation**: Unified memory_collectors and collectors directories
- **BuildRunner Improvements**: Enhanced data aggregation and result formatting
- **File Organization**: Logical grouping of plots by parallelism level

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run the test suite: `uv run pytest`
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🐛 Troubleshooting

### Common Issues

**Process pattern not matching**: Update the `process_pattern` in `projects.toml` to include relevant build tools.

**Permission errors**: Ensure the monitoring user has read access to `/proc` filesystem for memory data collection.

**Missing dependencies**: Install system dependencies like `pidstat` for RSS collection:
```bash
sudo apt-get install sysstat  # Ubuntu/Debian
```

**Plot generation fails**: Install export dependencies for PNG output:
```bash
uv pip install mymonitor[export]
```

For more help, check the logs in the monitoring output directory or run with verbose logging.

---

> **Languages**: [English](README.md) | [中文](README.zh-CN.md)
