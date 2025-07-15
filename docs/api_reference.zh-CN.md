# API 参考

本文档提供 MyMonitor 核心模块和类的详细 API 参考。

## 核心类

### DataStorageManager

用于管理监控数据存储的高级接口。

```python
from mymonitor.storage.data_manager import DataStorageManager
from pathlib import Path

# 初始化存储管理器
manager = DataStorageManager(Path("output_directory"))

# 保存监控结果
manager.save_monitoring_results(results, run_context)

# 加载内存样本
df = manager.load_memory_samples()

# 仅加载特定列
df = manager.load_memory_samples(columns=["pid", "process_name", "rss_kb"])

# 获取存储信息
info = manager.get_storage_info()
```

**方法：**

- `save_monitoring_results(results: MonitoringResults, run_context: Any) -> None`
  - 保存完整的监控结果，包括内存样本、元数据和分析结果
  - 创建多个输出文件：Parquet 数据、JSON 元数据、摘要日志

- `load_memory_samples(columns: Optional[List[str]] = None) -> pl.DataFrame`
  - 从存储中加载内存样本数据
  - 支持列裁剪以提高性能
  - 返回 Polars DataFrame 用于高效数据操作

- `get_storage_info() -> Dict[str, Any]`
  - 返回有关存储格式、压缩和文件详情的信息
  - 用于调试和监控存储使用情况

### AbstractMemoryCollector

内存数据收集实现的基类。

```python
from mymonitor.collectors.base import AbstractMemoryCollector

class CustomCollector(AbstractMemoryCollector):
    def collect_single_sample(self) -> List[ProcessMemorySample]:
        # 收集内存数据的实现
        pass
```

**抽象方法：**

- `collect_single_sample() -> List[ProcessMemorySample]`
  - 收集所有被监控进程的内存数据
  - 返回 ProcessMemorySample 对象列表
  - 必须由具体收集器类实现

### ProcessClassifier

基于可配置规则对构建进程进行分类的引擎。

```python
from mymonitor.classification.classifier import get_process_category

# 对进程进行分类
category, subcategory = get_process_category(
    cmd_name="gcc",
    full_cmd="/usr/bin/gcc -O2 main.c",
    rules=classification_rules
)
```

**函数：**

- `get_process_category(cmd_name: str, full_cmd: str, rules: List[RuleConfig]) -> Tuple[str, str]`
  - 基于命令名称和完整命令行对进程进行分类
  - 返回 (major_category, minor_category) 元组
  - 使用基于优先级的规则匹配

## 配置类

### StorageConfig

数据存储选项的配置。

```python
from mymonitor.config.storage_config import StorageConfig

# 从字典创建
config = StorageConfig.from_dict({
    "format": "parquet",
    "compression": "snappy",
    "generate_legacy_formats": False
})

# 访问属性
print(config.format)        # "parquet"
print(config.compression)   # "snappy"
```

**属性：**

- `format: str` - 存储格式（"parquet" 或 "json"）
- `compression: str` - Parquet 的压缩算法
- `generate_legacy_formats: bool` - 是否生成 CSV 文件

### MonitorConfig

主要监控配置。

```python
from mymonitor.config.validators import validate_monitor_config

# 验证配置
validated_config = validate_monitor_config(config_dict)
```

## 数据模型

### MonitoringResults

完整监控会话结果的容器。

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

**属性：**

- `all_samples_data: List[Dict]` - 原始内存样本数据
- `category_stats: Dict` - 按进程类别分组的统计信息
- `peak_overall_memory_kb: int` - 最大总内存使用量
- `peak_overall_memory_epoch: float` - 峰值内存使用的时间戳
- `category_peak_sum: Dict` - 按类别的峰值内存使用
- `category_pid_set: Dict` - 按类别分组的进程 ID

### ProcessMemorySample

进程的单个内存测量。

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

**属性：**

- `timestamp: float` - 测量的 Unix 时间戳
- `pid: int` - 进程 ID
- `process_name: str` - 进程可执行文件名
- `rss_kb: int` - 常驻集大小（KB）
- `vms_kb: int` - 虚拟内存大小（KB）
- `pss_kb: int` - 比例集大小（KB）
- `category: str` - 进程分类类别

## 实用函数

### 存储工厂

```python
from mymonitor.storage.factory import create_storage

# 创建 Parquet 存储
storage = create_storage("parquet", compression="snappy")

# 创建 JSON 存储
storage = create_storage("json")
```

### CPU 管理

```python
from mymonitor.system.cpu_manager import CPUManager

cpu_manager = CPUManager()
allocation_plan = cpu_manager.allocate_cores(
    strategy="adaptive",
    parallelism_level=8,
    enable_affinity=True
)
```

## 错误处理

所有 API 函数可能引发以下异常：

- `ConfigurationError` - 无效的配置参数
- `StorageError` - 存储操作失败
- `CollectionError` - 内存收集失败
- `ValidationError` - 数据验证失败

## 线程安全

- `DataStorageManager` - 对并发读取操作线程安全
- `ProcessClassifier` - 具有内部缓存的线程安全
- `AbstractMemoryCollector` - 取决于实现
- 配置类 - 创建后不可变

## 性能考虑

- 加载大型数据集时使用列裁剪：`load_memory_samples(columns=["pid", "rss_kb"])`
- Parquet 格式提供比 CSV 快 3-5 倍的查询速度
- 内存收集器应与适当的 CPU 亲和性设置一起使用
- 分类规则会被缓存以提高性能
