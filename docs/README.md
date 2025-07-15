# MyMonitor Documentation

> **Languages**: [English](README.md) | [中文](README.zh-CN.md)

Welcome to the MyMonitor documentation! This comprehensive guide covers everything you need to know about using and developing with MyMonitor.

## Quick Start

- **New Users**: Start with the [Configuration Guide](configuration_guide.md)
- **Developers**: Check out the [Developer Guide](developer_guide.md)
- **Troubleshooting**: See the [Troubleshooting Guide](troubleshooting.md)

## Documentation Structure

### User Guides

#### [Configuration Guide](configuration_guide.md)
Complete guide to configuring MyMonitor for your projects, including:
- Main configuration files (`config.toml`, `projects.toml`, `rules.toml`)
- Storage settings and compression options
- CPU scheduling and thread pool configuration
- Environment variables and best practices

#### [Storage Formats](storage_formats.md)
Comprehensive guide to data storage formats:
- Parquet vs CSV/JSON performance comparison
- Configuration options and compression algorithms
- Data conversion tools and migration strategies
- Best practices for data access and analysis

#### [Troubleshooting Guide](troubleshooting.md)
Solutions to common issues:
- Installation and dependency problems
- Configuration errors and validation issues
- Performance optimization techniques
- System-specific troubleshooting (macOS, Windows, containers)

### Developer Resources

#### [API Reference](api_reference.md)
Complete API documentation covering:
- Core classes (`DataStorageManager`, `AbstractMemoryCollector`, `ProcessClassifier`)
- Configuration classes and data models
- Utility functions and error handling
- Thread safety and performance considerations

#### [Developer Guide](developer_guide.md)
Essential information for contributors:
- Development environment setup
- Coding standards and style guidelines
- Testing practices and frameworks
- Adding new features (storage backends, collectors)
- Performance considerations and debugging techniques

## Feature Highlights

### High-Performance Storage
- **Parquet Format**: 75-80% space savings compared to CSV/JSON
- **Fast Queries**: 3-5x performance improvement with column pruning
- **Multiple Compression**: Snappy, Gzip, Brotli, LZ4, Zstd options
- **Backward Compatibility**: Optional legacy format generation

### Advanced Monitoring
- **Real-time Collection**: PSS/RSS memory monitoring with configurable intervals
- **Process Classification**: Automatic categorization with rule-based engine
- **CPU Affinity**: Intelligent core allocation for consistent measurements
- **Thread Pool Management**: Optimized parallel processing

### Developer-Friendly
- **Modular Architecture**: Clean separation of concerns
- **Comprehensive Testing**: 135+ test cases with full coverage
- **Type Safety**: Complete type hints throughout codebase
- **Extensible Design**: Easy to add new collectors and storage backends

## Getting Help

### Documentation Issues
If you find errors or missing information in the documentation:
1. Check the [Troubleshooting Guide](troubleshooting.md) first
2. Search existing issues in the repository
3. Create a new issue with the `documentation` label

### Technical Support
For technical questions and support:
1. Review the [API Reference](api_reference.md) for detailed function documentation
2. Check the [Configuration Guide](configuration_guide.md) for setup issues
3. Enable verbose logging for debugging information

### Contributing
Interested in contributing? See the [Developer Guide](developer_guide.md) for:
- Development environment setup
- Coding standards and guidelines
- Testing requirements
- Pull request process

## Version Information

This documentation is for MyMonitor v2.1+, which includes:
- Parquet storage system
- Enhanced configuration management
- Improved error handling
- Comprehensive test coverage

For older versions, please refer to the appropriate git tags.

## License

MyMonitor is released under the MIT License. See the LICENSE file for details.

---

**Quick Navigation:**
- [Configuration Guide](configuration_guide.md) - Set up MyMonitor for your projects
- [Storage Formats](storage_formats.md) - Understand data storage options
- [API Reference](api_reference.md) - Complete function and class documentation
- [Developer Guide](developer_guide.md) - Contribute to MyMonitor development
- [Troubleshooting](troubleshooting.md) - Solve common issues
