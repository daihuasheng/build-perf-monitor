# 开发者指南

本指南涵盖 MyMonitor 的开发环境设置、编码标准、测试实践和贡献指南。

## 开发环境设置

### 前提条件

- Python 3.11 或更高版本
- UV 包管理器（推荐）
- Git 版本控制

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd mymonitor

# 安装开发依赖
uv pip install -e ".[dev]"

# 安装所有可选依赖
uv pip install -e ".[dev,export]"
```

### 开发依赖

`[dev]` 额外包括：
- `pytest` - 测试框架
- `pytest-cov` - 覆盖率报告
- `pytest-asyncio` - 异步测试支持
- `black` - 代码格式化
- `isort` - 导入排序
- `flake8` - 代码检查
- `mypy` - 类型检查

## 项目结构

```
src/mymonitor/
├── cli/                    # 命令行界面
│   ├── __init__.py
│   ├── main.py            # 入口点
│   └── orchestrator.py    # 主编排逻辑
├── config/                 # 配置管理
│   ├── __init__.py
│   ├── loader.py          # TOML 文件加载
│   ├── manager.py         # 配置管理
│   ├── storage_config.py  # 存储配置模型
│   └── validators.py      # 配置验证
├── models/                 # 数据模型
│   ├── __init__.py
│   ├── config.py          # 配置模型
│   ├── memory.py          # 内存数据模型
│   ├── results.py         # 结果模型
│   └── runtime.py         # 运行时模型
├── storage/                # 数据存储层
│   ├── __init__.py
│   ├── base.py            # 存储接口
│   ├── data_manager.py    # 高级存储管理器
│   ├── factory.py         # 存储工厂
│   └── parquet_storage.py # Parquet 实现
├── collectors/             # 内存数据收集
│   ├── __init__.py
│   ├── base.py            # 抽象收集器
│   ├── factory.py         # 收集器工厂
│   └── pss_psutil.py      # PSS 收集器实现
├── classification/         # 进程分类
│   ├── __init__.py
│   └── classifier.py      # 分类引擎
├── system/                 # 系统交互
│   ├── __init__.py
│   └── cpu_manager.py     # CPU 管理
├── executor/               # 线程池管理
│   ├── __init__.py
│   └── thread_pool.py     # 线程池实现
└── validation/             # 输入验证
    ├── __init__.py
    └── strategies.py       # 验证策略
```

## 编码标准

### 代码风格

MyMonitor 遵循 PEP 8，具有以下特定指南：

1. **行长度**：最大 88 个字符（Black 默认值）
2. **导入**：使用 `isort` 进行一致的导入排序
3. **类型提示**：所有公共函数必须有类型提示
4. **文档字符串**：使用 Google 风格的文档字符串

### 示例函数

```python
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def process_memory_samples(
    samples: List[ProcessMemorySample],
    filter_category: Optional[str] = None
) -> List[ProcessMemorySample]:
    """处理和过滤内存样本。
    
    Args:
        samples: 要处理的内存样本列表
        filter_category: 可选的类别过滤器
        
    Returns:
        过滤后的内存样本列表
        
    Raises:
        ValueError: 如果样本列表为空
    """
    if not samples:
        raise ValueError("Samples list cannot be empty")
    
    logger.debug(f"Processing {len(samples)} memory samples")
    
    if filter_category:
        samples = [s for s in samples if s.category == filter_category]
        logger.debug(f"Filtered to {len(samples)} samples for category {filter_category}")
    
    return samples
```

### 类型提示

使用全面的类型提示：

```python
from typing import Dict, List, Optional, Union, Protocol
from pathlib import Path

# 使用 Protocol 进行接口定义
class DataStorage(Protocol):
    def save_dataframe(self, df: pl.DataFrame, path: str) -> None: ...
    def load_dataframe(self, path: str) -> pl.DataFrame: ...

# 使用 Union 表示多种类型
def load_config(path: Union[str, Path]) -> Dict[str, Any]: ...

# 使用 Optional 表示可空值
def get_process_category(name: str, rules: Optional[List[Rule]] = None) -> str: ...
```

## 测试

### 测试结构

```
tests/
├── unit/                   # 单元测试
│   ├── test_classification/
│   ├── test_config/
│   ├── test_storage/
│   └── ...
├── integration/            # 集成测试
├── performance/            # 性能测试
└── e2e/                   # 端到端测试
```

### 编写测试

#### 单元测试

```python
import pytest
from unittest.mock import Mock, patch
from mymonitor.storage.data_manager import DataStorageManager

class TestDataStorageManager:
    """DataStorageManager 的测试用例。"""
    
    @pytest.fixture
    def mock_config(self):
        """创建模拟配置。"""
        with patch("mymonitor.storage.data_manager.get_config") as mock:
            mock_config = Mock()
            mock_config.monitor.storage.format = "parquet"
            mock.return_value = mock_config
            yield mock
    
    def test_initialization(self, mock_config, tmp_path):
        """测试 DataStorageManager 初始化。"""
        manager = DataStorageManager(tmp_path)
        assert manager.storage_format == "parquet"
        assert manager.output_dir == tmp_path
    
    def test_save_monitoring_results(self, mock_config, tmp_path):
        """测试保存监控结果。"""
        manager = DataStorageManager(tmp_path)
        # 测试实现
```

#### 集成测试

```python
import tempfile
from pathlib import Path
from mymonitor.config.manager import ConfigManager

def test_config_integration():
    """测试配置加载和验证集成。"""
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # 验证配置是否正确加载和验证
    assert config.monitor.collection.interval_seconds > 0
    assert config.monitor.storage.format in ["parquet", "json"]
```

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试类别
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest tests/performance/
uv run pytest tests/e2e/

# 运行覆盖率测试
uv run pytest --cov=src/mymonitor --cov-report=html

# 运行特定测试文件
uv run pytest tests/unit/test_storage/test_data_manager.py

# 运行详细输出
uv run pytest -v

# 运行匹配模式的测试
uv run pytest -k "test_storage"
```

### 测试指南

1. **隔离**：每个测试应该是独立的
2. **模拟**：模拟外部依赖（文件系统、网络、系统调用）
3. **夹具**：使用 pytest 夹具进行常见测试数据
4. **参数化**：使用 `@pytest.mark.parametrize` 进行多个测试用例
5. **异步测试**：使用 `pytest-asyncio` 进行异步代码测试

## 添加新功能

### 1. 存储后端

添加新的存储后端：

```python
# 1. 创建存储实现
from mymonitor.storage.base import DataStorage

class NewStorage(DataStorage):
    def save_dataframe(self, df: pl.DataFrame, path: str) -> None:
        # 实现
        pass
    
    def load_dataframe(self, path: str) -> pl.DataFrame:
        # 实现
        pass

# 2. 更新工厂
from mymonitor.storage.factory import create_storage

def create_storage(format_type: str) -> DataStorage:
    if format_type == "new_format":
        return NewStorage()
    # ... 现有格式

# 3. 添加配置支持
# 更新 StorageConfig 以包含新格式

# 4. 添加测试
class TestNewStorage:
    def test_save_load_cycle(self):
        # 测试实现
```

### 2. 内存收集器

添加新的内存收集器：

```python
# 1. 实现收集器
from mymonitor.collectors.base import AbstractMemoryCollector

class NewCollector(AbstractMemoryCollector):
    def collect_single_sample(self) -> List[ProcessMemorySample]:
        # 实现
        pass

# 2. 更新工厂
from mymonitor.collectors.factory import CollectorFactory

class CollectorFactory:
    def create_collector(self, collector_type: str) -> AbstractMemoryCollector:
        if collector_type == "new_collector":
            return NewCollector()
        # ... 现有收集器

# 3. 添加配置选项
# 更新配置验证以包含新收集器类型

# 4. 添加测试
class TestNewCollector:
    def test_collect_single_sample(self):
        # 测试实现
```

## 性能考虑

### 内存管理

1. **使用生成器**进行大数据处理
2. **在上下文管理器中实现适当的清理**
3. **监控长时间运行操作的内存使用情况**

```python
def process_large_dataset(data_path: Path) -> Iterator[ProcessedData]:
    """使用生成器高效处理大型数据集。"""
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

### 异步编程

1. **对 I/O 绑定操作使用 async/await**
2. **在异步函数中避免阻塞调用**
3. **在异步代码中使用适当的异常处理**

```python
import asyncio
from typing import List

async def collect_memory_async(
    collectors: List[AbstractMemoryCollector]
) -> List[ProcessMemorySample]:
    """异步收集内存数据。"""
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

## 调试

### 日志记录

在整个代码库中使用结构化日志记录：

```python
import logging

logger = logging.getLogger(__name__)

def complex_operation(data: Any) -> Any:
    """执行具有适当日志记录的复杂操作。"""
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

### 调试配置

在配置中启用调试模式：

```toml
[monitor.general]
verbose = true

[monitor.collection]
debug_process_discovery = true
debug_memory_collection = true
```

## 发布流程

### 版本管理

1. 更新 `pyproject.toml` 中的版本
2. 更新 CHANGELOG.md
3. 创建 git 标签
4. 构建和测试包

```bash
# 更新版本
uv version patch  # 或 minor, major

# 运行完整测试套件
uv run pytest

# 构建包
uv build

# 创建发布标签
git tag v$(uv version --short)
git push origin v$(uv version --short)
```

### 文档更新

1. 为新功能更新 API 文档
2. 为新选项更新配置指南
3. 更新 README 文件
4. 如果需要，生成 API 文档

## 贡献

### 拉取请求流程

1. Fork 仓库
2. 创建功能分支：`git checkout -b feature-name`
3. 进行更改并添加测试
4. 运行测试套件：`uv run pytest`
5. 运行代码检查：`uv run flake8 src/`
6. 格式化代码：`uv run black src/`
7. 排序导入：`uv run isort src/`
8. 使用描述性消息提交更改
9. 推送分支并创建拉取请求

### 提交消息格式

使用约定式提交格式：

```
type(scope): description

[optional body]

[optional footer]
```

示例：
- `feat(storage): add Parquet compression options`
- `fix(collector): handle process termination gracefully`
- `docs(api): update storage manager documentation`
- `test(integration): add CPU allocation tests`

### 代码审查指南

1. **功能性**：代码是否按预期工作？
2. **测试**：是否有足够的测试覆盖新功能？
3. **文档**：代码是否有适当的文档？
4. **性能**：是否有任何性能影响？
5. **安全性**：是否有任何安全问题？
6. **风格**：代码是否遵循项目约定？
