# 配置指南

MyMonitor 使用 TOML 配置文件来自定义监控行为。本指南涵盖所有配置选项和最佳实践。

## 配置文件概览

MyMonitor 使用三个主要配置文件：

- `conf/config.toml` - 主要监控设置
- `conf/projects.toml` - 项目特定的构建配置
- `conf/rules.toml` - 进程分类规则

## 主配置文件 (`conf/config.toml`)

### 通用设置

```toml
[monitor.general]
# 要测试的默认并行度级别
default_jobs = [4, 8, 16]

# 监控后跳过图表生成
skip_plots = false

# 启用详细日志记录
verbose = false
```

**选项：**
- `default_jobs`: 自动测试的 `-j` 值列表
- `skip_plots`: 设置为 `true` 禁用自动图表生成
- `verbose`: 启用调试级别日志记录

### 收集设置

```toml
[monitor.collection]
# 采样间隔（秒）
interval_seconds = 0.05

# 内存收集器类型
metric_type = "pss_psutil"

# 进程扫描模式
pss_collector_mode = "full_scan"

# 启用仅后代模式以获得更好的性能
descendants_only = false
```

**收集器类型：**
- `pss_psutil`: 使用 psutil 的 PSS 内存（推荐）
- `rss_psutil`: 使用 psutil 的 RSS 内存
- `hybrid`: 混合生产者-消费者模型

**扫描模式：**
- `full_scan`: 扫描所有系统进程（可靠）
- `descendants_only`: 仅扫描构建进程后代（更快）

### 存储设置

```toml
[monitor.storage]
# 存储格式："parquet"（推荐）或 "json"
format = "parquet"

# Parquet 的压缩算法
compression = "snappy"

# 生成传统 CSV 文件以向后兼容
generate_legacy_formats = false
```

**压缩选项：**
- `snappy`: 快速压缩/解压缩（默认）
- `gzip`: 更高压缩率，较慢
- `brotli`: 高压缩率
- `lz4`: 非常快的压缩
- `zstd`: 现代平衡压缩

### CPU 调度

```toml
[monitor.scheduling]
# CPU 分配策略
scheduling_policy = "adaptive"

# 启用 CPU 亲和性绑定
enable_cpu_affinity = true

# 监控进程的特定核心
monitor_core = 0

# 手动核心分配（用于手动策略）
build_cores = "0-7"
monitoring_cores = "8-11"
```

**调度策略：**
- `adaptive`: 根据系统容量自动分配核心
- `manual`: 使用明确指定的核心范围

### 线程池配置

```toml
[monitor.thread_pools.monitoring]
# 监控线程数
max_workers = 4

# 为监控线程启用 CPU 亲和性
enable_affinity = true

[monitor.thread_pools.build]
# 构建执行线程
max_workers = 1
enable_affinity = false

[monitor.thread_pools.io]
# I/O 操作线程
max_workers = 2
enable_affinity = false
```

## 项目配置 (`conf/projects.toml`)

定义构建项目及其特定设置：

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

**必需字段：**
- `name`: 唯一项目标识符
- `dir`: 构建目录路径
- `build_command_template`: 带有并行度占位符 `<N>` 的构建命令
- `clean_command_template`: 清理构建产物的命令
- `process_pattern`: 匹配相关进程的正则表达式模式

## 分类规则 (`conf/rules.toml`)

定义进程如何分类：

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

**规则字段：**
- `major_category`: 主要分类（例如，"Compiler"、"Linker"）
- `minor_category`: 子类别（例如，"CPP_Compile"、"Static_Link"）
- `priority`: 数字越小 = 优先级越高
- `match_type`: 如何匹配模式
- `pattern`: 与进程名称匹配的模式

**匹配类型：**
- `exact`: 精确字符串匹配
- `regex`: 正则表达式匹配
- `contains`: 子字符串匹配
- `in_list`: 与逗号分隔列表匹配

## 环境变量

使用环境变量覆盖配置：

```bash
# 覆盖采样间隔
export MYMONITOR_INTERVAL=0.1

# 覆盖存储格式
export MYMONITOR_STORAGE_FORMAT=json

# 覆盖压缩
export MYMONITOR_COMPRESSION=gzip

# 启用详细日志记录
export MYMONITOR_VERBOSE=true
```

## 最佳实践

### 性能优化

1. **采样间隔**: 详细分析使用 0.05s，一般监控使用 0.1s
2. **存储格式**: 始终使用 Parquet 以获得更好的性能
3. **压缩**: 使用 `snappy` 平衡性能，使用 `gzip` 节省存储
4. **CPU 亲和性**: 启用以获得一致的性能测量

### 资源管理

1. **线程池大小**：
   - 监控线程：`min(4, parallelism_level)`
   - 构建线程：始终为 1
   - I/O 线程：根据存储速度 2-4

2. **内存使用**：
   - 对大型系统启用 `descendants_only`
   - 加载数据时使用列裁剪
   - 定期清理旧日志文件

### 可靠性

1. **错误处理**: 启用详细日志记录以进行故障排除
2. **备份**: 保留工作配置的备份副本
3. **验证**: 首先用小型构建测试配置更改

## 配置验证

MyMonitor 在启动时验证所有配置文件。常见验证错误：

### 无效的核心范围
```
Error: Invalid core range '0-99' - only 8 cores available
```
**解决方案**: 调整核心范围以匹配系统容量

### 缺少项目目录
```
Error: Project directory '/invalid/path' does not exist
```
**解决方案**: 确保项目目录存在且可访问

### 无效的正则表达式模式
```
Error: Invalid regex pattern 'gcc[' in classification rule
```
**解决方案**: 修复 rules.toml 中的正则表达式语法

## 高级配置

### 自定义收集器

通过扩展 `AbstractMemoryCollector` 创建自定义内存收集器：

```python
from mymonitor.collectors.base import AbstractMemoryCollector

class CustomCollector(AbstractMemoryCollector):
    def collect_single_sample(self):
        # 自定义实现
        pass
```

在配置中注册：
```toml
[monitor.collection]
metric_type = "custom"
custom_collector_class = "mymodule.CustomCollector"
```

### 动态配置

以编程方式加载配置：

```python
from mymonitor.config.manager import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config()

# 覆盖特定设置
config.monitor.collection.interval_seconds = 0.1
```

## 故障排除

### 常见问题

1. **高 CPU 使用率**: 增加采样间隔或启用 descendants_only
2. **缺少进程**: 调整 projects.toml 中的 process_pattern
3. **存储错误**: 检查磁盘空间和权限
4. **分类问题**: 检查并更新 rules.toml 优先级

### 调试模式

启用全面调试：

```toml
[monitor.general]
verbose = true

[monitor.collection]
debug_process_discovery = true
debug_memory_collection = true
```

这将启用进程发现和内存收集阶段的详细日志记录。
