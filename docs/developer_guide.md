# Developer Guide

This guide covers development setup, coding standards, testing practices, and contribution guidelines for MyMonitor.

## Development Environment Setup

### Prerequisites

- Python 3.11 or higher
- UV package manager (recommended)
- Git for version control

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mymonitor

# Install with development dependencies
uv pip install -e ".[dev]"

# Install with all optional dependencies
uv pip install -e ".[dev,export]"
```

### Development Dependencies

The `[dev]` extra includes:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-asyncio` - Async testing support
- `black` - Code formatting
- `isort` - Import sorting
- `flake8` - Linting
- `mypy` - Type checking

## Project Structure

```
src/mymonitor/
├── cli/                    # Command-line interface
│   ├── __init__.py
│   ├── main.py            # Entry point
│   └── orchestrator.py    # Main orchestration logic
├── config/                 # Configuration management
│   ├── __init__.py
│   ├── loader.py          # TOML file loading
│   ├── manager.py         # Configuration management
│   ├── storage_config.py  # Storage configuration models
│   └── validators.py      # Configuration validation
├── models/                 # Data models
│   ├── __init__.py
│   ├── config.py          # Configuration models
│   ├── memory.py          # Memory data models
│   ├── results.py         # Result models
│   └── runtime.py         # Runtime models
├── storage/                # Data storage layer
│   ├── __init__.py
│   ├── base.py            # Storage interface
│   ├── data_manager.py    # High-level storage manager
│   ├── factory.py         # Storage factory
│   └── parquet_storage.py # Parquet implementation
├── collectors/             # Memory data collection
│   ├── __init__.py
│   ├── base.py            # Abstract collector
│   ├── factory.py         # Collector factory
│   └── pss_psutil.py      # PSS collector implementation
├── classification/         # Process classification
│   ├── __init__.py
│   └── classifier.py      # Classification engine
├── system/                 # System interaction
│   ├── __init__.py
│   └── cpu_manager.py     # CPU management
├── executor/               # Thread pool management
│   ├── __init__.py
│   └── thread_pool.py     # Thread pool implementation
└── validation/             # Input validation
    ├── __init__.py
    └── strategies.py       # Validation strategies
```

## Coding Standards

### Code Style

MyMonitor follows PEP 8 with these specific guidelines:

1. **Line Length**: Maximum 88 characters (Black default)
2. **Imports**: Use `isort` for consistent import ordering
3. **Type Hints**: All public functions must have type hints
4. **Docstrings**: Use Google-style docstrings

### Example Function

```python
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def process_memory_samples(
    samples: List[ProcessMemorySample],
    filter_category: Optional[str] = None
) -> List[ProcessMemorySample]:
    """Process and filter memory samples.
    
    Args:
        samples: List of memory samples to process
        filter_category: Optional category filter
        
    Returns:
        Filtered list of memory samples
        
    Raises:
        ValueError: If samples list is empty
    """
    if not samples:
        raise ValueError("Samples list cannot be empty")
    
    logger.debug(f"Processing {len(samples)} memory samples")
    
    if filter_category:
        samples = [s for s in samples if s.category == filter_category]
        logger.debug(f"Filtered to {len(samples)} samples for category {filter_category}")
    
    return samples
```

### Type Hints

Use comprehensive type hints:

```python
from typing import Dict, List, Optional, Union, Protocol
from pathlib import Path

# Use Protocol for interface definitions
class DataStorage(Protocol):
    def save_dataframe(self, df: pl.DataFrame, path: str) -> None: ...
    def load_dataframe(self, path: str) -> pl.DataFrame: ...

# Use Union for multiple types
def load_config(path: Union[str, Path]) -> Dict[str, Any]: ...

# Use Optional for nullable values
def get_process_category(name: str, rules: Optional[List[Rule]] = None) -> str: ...
```

## Testing

### Test Structure

```
tests/
├── unit/                   # Unit tests
│   ├── test_classification/
│   ├── test_config/
│   ├── test_storage/
│   └── ...
├── integration/            # Integration tests
├── performance/            # Performance tests
└── e2e/                   # End-to-end tests
```

### Writing Tests

#### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch
from mymonitor.storage.data_manager import DataStorageManager

class TestDataStorageManager:
    """Test cases for DataStorageManager."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        with patch("mymonitor.storage.data_manager.get_config") as mock:
            mock_config = Mock()
            mock_config.monitor.storage.format = "parquet"
            mock.return_value = mock_config
            yield mock
    
    def test_initialization(self, mock_config, tmp_path):
        """Test DataStorageManager initialization."""
        manager = DataStorageManager(tmp_path)
        assert manager.storage_format == "parquet"
        assert manager.output_dir == tmp_path
    
    def test_save_monitoring_results(self, mock_config, tmp_path):
        """Test saving monitoring results."""
        manager = DataStorageManager(tmp_path)
        # Test implementation
```

#### Integration Tests

```python
import tempfile
from pathlib import Path
from mymonitor.config.manager import ConfigManager

def test_config_integration():
    """Test configuration loading and validation integration."""
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Verify configuration is properly loaded and validated
    assert config.monitor.collection.interval_seconds > 0
    assert config.monitor.storage.format in ["parquet", "json"]
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest tests/performance/
uv run pytest tests/e2e/

# Run with coverage
uv run pytest --cov=src/mymonitor --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_storage/test_data_manager.py

# Run with verbose output
uv run pytest -v

# Run tests matching pattern
uv run pytest -k "test_storage"
```

### Test Guidelines

1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external dependencies (filesystem, network, system calls)
3. **Fixtures**: Use pytest fixtures for common test data
4. **Parametrization**: Use `@pytest.mark.parametrize` for multiple test cases
5. **Async Testing**: Use `pytest-asyncio` for async code

## Adding New Features

### 1. Storage Backends

To add a new storage backend:

```python
# 1. Create storage implementation
from mymonitor.storage.base import DataStorage

class NewStorage(DataStorage):
    def save_dataframe(self, df: pl.DataFrame, path: str) -> None:
        # Implementation
        pass
    
    def load_dataframe(self, path: str) -> pl.DataFrame:
        # Implementation
        pass

# 2. Update factory
from mymonitor.storage.factory import create_storage

def create_storage(format_type: str) -> DataStorage:
    if format_type == "new_format":
        return NewStorage()
    # ... existing formats

# 3. Add configuration support
# Update StorageConfig to include new format

# 4. Add tests
class TestNewStorage:
    def test_save_load_cycle(self):
        # Test implementation
```

### 2. Memory Collectors

To add a new memory collector:

```python
# 1. Implement collector
from mymonitor.collectors.base import AbstractMemoryCollector

class NewCollector(AbstractMemoryCollector):
    def collect_single_sample(self) -> List[ProcessMemorySample]:
        # Implementation
        pass

# 2. Update factory
from mymonitor.collectors.factory import CollectorFactory

class CollectorFactory:
    def create_collector(self, collector_type: str) -> AbstractMemoryCollector:
        if collector_type == "new_collector":
            return NewCollector()
        # ... existing collectors

# 3. Add configuration option
# Update configuration validation to include new collector type

# 4. Add tests
class TestNewCollector:
    def test_collect_single_sample(self):
        # Test implementation
```

## Performance Considerations

### Memory Management

1. **Use generators** for large data processing
2. **Implement proper cleanup** in context managers
3. **Monitor memory usage** in long-running operations

```python
def process_large_dataset(data_path: Path) -> Iterator[ProcessedData]:
    """Process large dataset efficiently using generators."""
    with open(data_path) as f:
        for line in f:
            yield process_line(line)

class ResourceManager:
    def __enter__(self):
        self.resource = acquire_resource()
        return self.resource
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.resource.cleanup()
```

### Async Programming

1. **Use async/await** for I/O-bound operations
2. **Avoid blocking calls** in async functions
3. **Use proper exception handling** in async code

```python
import asyncio
from typing import List

async def collect_memory_async(
    collectors: List[AbstractMemoryCollector]
) -> List[ProcessMemorySample]:
    """Collect memory data asynchronously."""
    tasks = [
        asyncio.create_task(collector.collect_async())
        for collector in collectors
    ]
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]
    except Exception as e:
        logger.error(f"Error in async collection: {e}")
        raise
```

## Debugging

### Logging

Use structured logging throughout the codebase:

```python
import logging

logger = logging.getLogger(__name__)

def complex_operation(data: Any) -> Any:
    """Perform complex operation with proper logging."""
    logger.info(f"Starting complex operation with {len(data)} items")
    
    try:
        result = process_data(data)
        logger.debug(f"Processed {len(result)} results")
        return result
    except Exception as e:
        logger.error(f"Error in complex operation: {e}", exc_info=True)
        raise
    finally:
        logger.info("Complex operation completed")
```

### Debug Configuration

Enable debug mode in configuration:

```toml
[monitor.general]
verbose = true

[monitor.collection]
debug_process_discovery = true
debug_memory_collection = true
```

## Release Process

### Version Management

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag
4. Build and test package

```bash
# Update version
uv version patch  # or minor, major

# Run full test suite
uv run pytest

# Build package
uv build

# Create release tag
git tag v$(uv version --short)
git push origin v$(uv version --short)
```

### Documentation Updates

1. Update API documentation for new features
2. Update configuration guide for new options
3. Update README files
4. Generate API docs if needed

## Contributing

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes with tests
4. Run test suite: `uv run pytest`
5. Run linting: `uv run flake8 src/`
6. Format code: `uv run black src/`
7. Sort imports: `uv run isort src/`
8. Commit changes with descriptive message
9. Push branch and create pull request

### Commit Message Format

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(storage): add Parquet compression options`
- `fix(collector): handle process termination gracefully`
- `docs(api): update storage manager documentation`
- `test(integration): add CPU allocation tests`

### Code Review Guidelines

1. **Functionality**: Does the code work as intended?
2. **Tests**: Are there adequate tests for new functionality?
3. **Documentation**: Is the code properly documented?
4. **Performance**: Are there any performance implications?
5. **Security**: Are there any security concerns?
6. **Style**: Does the code follow project conventions?
