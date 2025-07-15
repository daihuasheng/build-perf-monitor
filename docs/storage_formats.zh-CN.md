# 数据存储格式

> **语言**: [English](storage_formats.md) | [中文](storage_formats.zh-CN.md)

## 概述

监测系统现在支持高效的 Parquet 格式存储，相比传统的 CSV/JSON 格式提供了显著的性能和存储优势。

## 支持的格式

### Parquet 格式（推荐）

**优势：**
- **存储效率高**：相比 CSV/JSON 减少 75-80% 存储空间
- **查询性能好**：列式存储，只读取需要的列
- **内置压缩**：支持多种压缩算法
- **模式保留**：保留数据类型信息
- **分析友好**：与 Pandas、Polars、Spark 等工具无缝集成

**压缩算法：**
- `snappy`（默认）：平衡压缩率和速度
- `gzip`：更高压缩率，较慢
- `brotli`：高压缩率
- `lz4`：快速压缩
- `zstd`：现代高效压缩

### JSON 格式

用于元数据和小型配置文件，保持人类可读性。

## 配置

在 `conf/config.toml` 中配置存储格式：

```toml
[monitor.storage]
# 数据存储格式: "parquet" (推荐) 或 "json"
format = "parquet"

# 压缩算法 (仅用于 parquet): "snappy", "gzip", "brotli", "lz4", "zstd"
compression = "snappy"

# 是否同时生成传统格式 (向后兼容)
generate_legacy_formats = false
```

## 生成的文件结构

监测完成后，在 logs 目录下生成以下文件：

```text
logs/
└── <项目名称>_<时间戳>/
    ├── memory_samples.parquet      # 主要监测数据 (Parquet 格式)
    ├── metadata.json               # 运行元数据
    ├── analysis_results.json       # 分析结果
    ├── summary.log                 # 人类可读的摘要
    └── memory_samples.csv          # 传统格式 (仅在启用时)
```

## 性能对比

以下是不同格式的性能对比（基于真实监测数据）：

| 数据类型 | CSV 大小 | Parquet 大小 | 节省空间 | 查询性能提升 |
|---------|---------|-------------|---------|-------------|
| 内存样本 (10万行) | 25 MB | 5 MB | 80% | 3-5x |
| CPU样本 (10万行) | 18 MB | 4 MB | 78% | 3-5x |
| 进程信息 (1千进程) | 2 MB | 0.4 MB | 80% | 2-3x |

## 数据转换

### 使用转换工具

系统提供了内置的转换工具来迁移现有数据：

```bash
# 转换单个文件
uv run python tools/convert_storage.py \
    old_data.csv new_data.parquet \
    --input-format csv --output-format parquet

# 转换整个目录
uv run python tools/convert_storage.py \
    logs/old_format/ logs/parquet_format/ \
    --input-format csv --output-format parquet --recursive

# 使用不同压缩算法
uv run python tools/convert_storage.py \
    data.csv data.parquet --compression gzip
```

### 转换工具选项

- `--input-format`: 输入格式 (csv, json, parquet)
- `--output-format`: 输出格式 (csv, json, parquet)
- `--compression`: 压缩算法 (snappy, gzip, brotli, lz4, zstd)
- `--recursive`: 递归处理子目录
- `--verbose`: 详细输出

## 数据访问

### 使用 Polars (推荐)

```python
import polars as pl

# 加载完整数据
df = pl.read_parquet("memory_samples.parquet")

# 只加载特定列 (提高性能)
df = pl.read_parquet("memory_samples.parquet", columns=["timestamp", "pid", "rss_kb"])

# 过滤数据
df = pl.read_parquet("memory_samples.parquet").filter(
    pl.col("category") == "compiler"
)
```

### 使用 Pandas

```python
import pandas as pd

# 加载数据
df = pd.read_parquet("memory_samples.parquet")

# 只加载特定列
df = pd.read_parquet("memory_samples.parquet", columns=["timestamp", "pid", "rss_kb"])
```

### 使用存储管理器

```python
from mymonitor.storage.data_manager import DataStorageManager
from pathlib import Path

# 创建存储管理器
manager = DataStorageManager(Path("logs/project_20230416/"))

# 加载内存样本
df = manager.load_memory_samples()

# 只加载特定列
df = manager.load_memory_samples(columns=["pid", "process_name", "rss_kb"])

# 获取存储信息
info = manager.get_storage_info()
print(f"存储格式: {info['storage_format']}")
print(f"文件大小: {info['files']}")
```

## 向后兼容性

### 启用传统格式

如果需要同时生成 CSV 格式以保持向后兼容性：

```toml
[monitor.storage]
format = "parquet"
generate_legacy_formats = true  # 同时生成 CSV 文件
```

### 迁移策略

1. **渐进式迁移**：
   - 启用 `generate_legacy_formats = true`
   - 验证 Parquet 数据正确性
   - 更新分析脚本使用 Parquet
   - 禁用传统格式生成

2. **一次性迁移**：
   - 使用转换工具转换现有数据
   - 更新配置使用 Parquet
   - 更新分析脚本

## 最佳实践

1. **使用 Parquet 格式**：获得最佳性能和存储效率
2. **选择合适的压缩算法**：
   - 一般用途：`snappy`（默认）
   - 存储优先：`gzip` 或 `brotli`
   - 速度优先：`lz4`
3. **列式查询**：只加载需要的列以提高性能
4. **批量处理**：对于大型数据集，使用批量读取
5. **定期清理**：删除不需要的传统格式文件

## 故障排除

### 常见问题

**Q: Parquet 文件无法打开**
A: 确保安装了 `polars-lts-cpu[pandas, pyarrow]` 依赖

**Q: 转换后数据不匹配**
A: 检查数据类型和编码，某些特殊字符可能需要处理

**Q: Parquet 文件比 CSV 大**
A: 对于小文件这是正常的，Parquet 包含元数据。大文件会显著更小。

**Q: 无法读取旧的 CSV 文件**
A: 使用转换工具迁移到 Parquet 格式

### 性能调优

1. **列裁剪**：只读取需要的列
2. **谓词下推**：在读取时应用过滤条件
3. **批量处理**：处理大型数据集时使用分块读取
4. **合适的压缩**：根据使用场景选择压缩算法

## 技术细节

### 数据模式

Parquet 文件保留以下数据模式：

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

### 压缩效果

不同压缩算法的特性：

| 算法 | 压缩率 | 压缩速度 | 解压速度 | 适用场景 |
|------|--------|----------|----------|----------|
| snappy | 中等 | 快 | 快 | 通用，默认选择 |
| gzip | 高 | 慢 | 中等 | 存储优先 |
| brotli | 很高 | 慢 | 中等 | 长期存储 |
| lz4 | 低 | 很快 | 很快 | 实时处理 |
| zstd | 高 | 快 | 快 | 现代应用 |
