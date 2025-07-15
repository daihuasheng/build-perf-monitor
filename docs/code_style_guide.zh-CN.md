# 代码风格指南

> **语言**: [English](code_style_guide.md) | [中文](code_style_guide.zh-CN.md)

本文档概述了 MyMonitor 项目的编码标准和文档实践。

## Python 代码风格

### 通用指南

- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 风格指南
- 使用 4 个空格进行缩进（不使用制表符）
- 最大行长度：88 个字符（Black 默认值）
- 所有 Python 文件使用 UTF-8 编码
- 文件以单个换行符结束
- 使用一致的命名约定

### 导入

- 按以下顺序分组导入：
  1. 标准库导入
  2. 相关第三方库导入
  3. 本地应用程序/库特定导入
- 为清晰起见使用绝对导入
- 对包和模块使用 `import` 语句
- 对类和函数使用 `from ... import`
- 避免通配符导入（`from module import *`）

示例：
```python
# 标准库
import os
import sys
from typing import Dict, List, Optional

# 第三方库
import polars as pl
import psutil

# 本地模块
from mymonitor.storage.base import DataStorage
from mymonitor.config import get_config
```

### 命名约定

- 函数、方法、变量和模块使用 `snake_case`
- 类使用 `PascalCase`
- 常量使用 `UPPER_CASE`
- 私有属性前缀加下划线：`_private_var`
- 使用反映用途的描述性名称

### 类型提示

- 为所有函数参数和返回值使用类型提示
- 对可以为 None 的参数使用 `Optional[Type]`
- 对可以是多种类型的参数使用 `Union[Type1, Type2]`
- 谨慎使用 `Any`，仅在真正必要时使用
- 使用 `typing` 模块中的 `List`、`Dict`、`Set` 等

示例：
```python
def process_data(
    input_data: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    max_rows: int = 1000
) -> pl.DataFrame:
    """处理输入数据到 DataFrame。"""
    # 实现
```

## 文档标准

### 模块文档字符串

每个模块应在顶部有一个文档字符串，解释其用途和内容：

```python
"""
模块名称和主要功能。

详细描述模块的功能、关键组件以及它如何融入更大的系统。
如果适当，可能包括使用示例。

主要特性：
- 特性 1：简要描述
- 特性 2：简要描述
"""
```

### 类文档字符串

类应有文档字符串解释其用途、行为和用法：

```python
class ExampleClass:
    """
    类用途的简短描述。
    
    详细解释类的功能、关键特性以及如何使用它。
    可能包括设计模式或架构考虑。
    
    属性：
        attr1: 第一个属性的描述
        attr2: 第二个属性的描述
    
    注意：
        任何特殊考虑或限制
    """
```

### 函数和方法文档字符串

函数和方法应遵循 Google 风格的文档字符串：

```python
def example_function(param1: str, param2: Optional[int] = None) -> bool:
    """
    函数用途的简短描述。
    
    详细解释函数的功能、算法和任何副作用。
    
    Args:
        param1: 第一个参数的描述
        param2: 第二个参数的描述，提及默认值
            如果参数复杂，可以使用缩进的延续
    
    Returns:
        返回值的描述
    
    Raises:
        ExceptionType: 何时以及为什么引发此异常
    
    Examples:
        >>> example_function("test", 42)
        True
    """
```

### 注释

- 谨慎使用注释解释"为什么"而不是"什么"
- 代码更改时保持注释更新
- 使用完整的句子，正确的大写和标点符号
- 避免不增加价值的明显注释

好的注释：
```python
# 跳过空输入的验证以提高性能
if not input_data:
    return default_value
```

差的注释：
```python
# 增加计数器
counter += 1
```

## 代码组织

### 文件结构

- 复杂类一个文件一个类
- 在模块中分组相关函数
- 保持文件专注于单一职责
- 限制文件大小（目标低于 500 行）

### 函数和方法组织

- 保持函数专注于单一任务
- 限制函数长度（目标低于 50 行）
- 使用辅助函数分解复杂逻辑
- 逻辑排序方法：
  1. 特殊方法（`__init__`、`__str__` 等）
  2. 公共方法
  3. 受保护/私有方法

## 错误处理

- 使用特定的异常类型
- 在适当的级别处理异常
- 使用上下文信息记录异常
- 提供有用的错误消息
- 使用上下文管理器进行资源清理

示例：
```python
try:
    result = process_data(input_data)
except ValueError as e:
    logger.error(f"无效的数据格式：{e}")
    raise
except IOError as e:
    logger.error(f"无法读取输入文件：{e}")
    return default_result
finally:
    cleanup_resources()
```

## 日志记录

- 使用标准 `logging` 模块
- 每个模块创建一个日志记录器
- 使用适当的日志级别：
  - `DEBUG`：用于调试的详细信息
  - `INFO`：确认事情正常工作
  - `WARNING`：意外但不是错误的情况
  - `ERROR`：阻止正常操作的错误
  - `CRITICAL`：可能阻止程序继续的严重错误
- 在日志消息中包含上下文

示例：
```python
import logging

logger = logging.getLogger(__name__)

def process_file(file_path: str) -> None:
    """处理文件。"""
    logger.info(f"处理文件：{file_path}")
    try:
        # 处理文件
        logger.debug(f"文件 {file_path} 处理成功")
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错：{e}", exc_info=True)
```

## 测试

- 为所有功能编写单元测试
- 使用 pytest 进行测试
- 争取高测试覆盖率
- 使用描述性测试名称
- 使用 Arrange-Act-Assert 模式构建测试
- 模拟外部依赖

示例：
```python
def test_data_storage_manager_saves_results():
    """测试 DataStorageManager 正确保存监控结果。"""
    # 准备
    manager = DataStorageManager(tmp_path)
    results = create_test_results()
    
    # 执行
    manager.save_monitoring_results(results, mock_context)
    
    # 断言
    assert (tmp_path / "memory_samples.parquet").exists()
    loaded_df = manager.load_memory_samples()
    assert len(loaded_df) == len(results.all_samples_data)
```

## 版本控制

- 编写清晰、简洁的提交消息
- 使用约定式提交格式：
  - `feat(component): 添加新功能`
  - `fix(component): 修复功能中的错误`
  - `docs(component): 更新文档`
  - `refactor(component): 重构代码而不改变行为`
- 保持提交专注于单一更改
- 在提交消息中引用问题编号

## 工具

- 使用 Black 进行代码格式化
- 使用 isort 进行导入排序
- 使用 flake8 进行代码检查
- 使用 mypy 进行类型检查
- 使用 pytest 进行测试

## 结论

遵循这些指南确保整个项目的代码一致性、可读性和可维护性。如有疑问，优先考虑清晰度和可读性，而不是聪明或优化。
