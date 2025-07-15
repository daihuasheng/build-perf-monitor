# 故障排除指南

本指南帮助诊断和解决 MyMonitor 的常见问题。

## 常见问题

### 1. 安装问题

#### UV 安装问题

**问题**: `uv: command not found`

**解决方案**:
```bash
# 安装 UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip
pip install uv
```

#### 依赖冲突

**问题**: 安装过程中包依赖冲突

**解决方案**:
```bash
# 使用 UV 清洁安装
uv pip install --force-reinstall -e ".[dev]"

# 或创建新的虚拟环境
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. 配置问题

#### 无效的配置文件

**问题**: `ConfigurationError: Invalid TOML syntax`

**解决方案**:
1. 使用在线验证器验证 TOML 语法
2. 检查字符串周围是否缺少引号
3. 确保数组格式正确

```toml
# 正确
default_jobs = [4, 8, 16]
process_pattern = "gcc|g\\+\\+"

# 错误
default_jobs = 4, 8, 16  # 缺少方括号
process_pattern = gcc|g++  # 缺少引号
```

#### 缺少项目目录

**问题**: `Project directory '/path/to/project' does not exist`

**解决方案**:
```bash
# 验证目录是否存在
ls -la /path/to/project

# 使用正确路径更新 projects.toml
[[projects]]
name = "myproject"
dir = "/correct/path/to/project"
```

#### 无效的核心范围

**问题**: `Invalid core range '0-15' - only 8 cores available`

**解决方案**:
```bash
# 检查可用核心数
nproc

# 更新配置
[monitor.scheduling]
build_cores = "0-7"  # 调整为可用核心
monitoring_cores = "8-11"  # 可能需要重叠或减少
```

### 3. 权限问题

#### 进程访问被拒绝

**问题**: `PermissionError: [Errno 13] Permission denied: '/proc/1234/stat'`

**解决方案**:
```bash
# 使用适当权限运行
sudo python -m mymonitor monitor -p myproject

# 或调整进程发现范围
[monitor.collection]
descendants_only = true  # 仅监控构建后代
```

#### 文件系统权限

**问题**: 无法写入输出目录

**解决方案**:
```bash
# 检查目录权限
ls -la logs/

# 修复权限
chmod 755 logs/
chown $USER:$USER logs/

# 或指定不同的输出目录
python -m mymonitor monitor -p myproject --output-dir /tmp/mymonitor_logs
```

### 4. 内存收集问题

#### 未找到进程

**问题**: `No processes found matching pattern 'gcc|make'`

**诊断**:
```bash
# 检查构建进程是否正在运行
ps aux | grep -E "gcc|make"

# 验证 projects.toml 中的进程模式
cat conf/projects.toml | grep process_pattern
```

**解决方案**:
1. 更新进程模式以匹配实际构建工具
2. 初始使用更广泛的模式：`process_pattern = ".*"`
3. 启用调试日志记录以查看发现的进程

```toml
[monitor.general]
verbose = true

[monitor.collection]
debug_process_discovery = true
```

#### 高 CPU 使用率

**问题**: MyMonitor 消耗过多 CPU

**解决方案**:
```toml
# 增加采样间隔
[monitor.collection]
interval_seconds = 0.1  # 从 0.05 减少

# 启用仅后代模式
descendants_only = true

# 减少线程池大小
[monitor.thread_pools.monitoring]
max_workers = 2  # 从 4 减少
```

#### 内存收集错误

**问题**: `psutil.NoSuchProcess: process no longer exists`

**解决方案**: 这对于短生命周期进程是正常的。启用错误容忍：

```toml
[monitor.collection]
ignore_process_errors = true
max_error_rate = 0.1  # 允许 10% 错误率
```

### 5. 存储问题

#### Parquet 文件损坏

**问题**: `ParquetError: Invalid parquet file`

**诊断**:
```bash
# 检查文件完整性
python -c "import polars as pl; df = pl.read_parquet('memory_samples.parquet')"
```

**解决方案**:
```bash
# 启用传统格式生成作为备份
[monitor.storage]
generate_legacy_formats = true

# 或使用不同的压缩
compression = "gzip"  # 比 snappy 更稳健
```

#### 磁盘空间问题

**问题**: `OSError: [Errno 28] No space left on device`

**解决方案**:
```bash
# 检查磁盘空间
df -h

# 清理旧日志
find logs/ -name "*.parquet" -mtime +7 -delete

# 使用更高压缩
[monitor.storage]
compression = "brotli"  # 更高压缩率
```

### 6. 性能问题

#### 数据加载缓慢

**问题**: 加载 Parquet 文件耗时过长

**解决方案**:
```python
# 使用列裁剪
df = manager.load_memory_samples(columns=["timestamp", "pid", "rss_kb"])

# 或对大文件分块加载
import polars as pl
df = pl.scan_parquet("memory_samples.parquet").select(["timestamp", "rss_kb"]).collect()
```

#### 构建监控开销

**问题**: 监控显著减慢构建速度

**解决方案**:
```toml
# 减少监控频率
[monitor.collection]
interval_seconds = 0.2

# 使用更少的监控线程
[monitor.thread_pools.monitoring]
max_workers = 1

# 启用 CPU 亲和性以隔离监控
[monitor.scheduling]
enable_cpu_affinity = true
```

### 7. 可视化问题

#### 图表生成失败

**问题**: `ImportError: No module named 'plotly'`

**解决方案**:
```bash
# 安装可视化依赖
uv pip install -e ".[export]"

# 或跳过图表生成
[monitor.general]
skip_plots = true
```

#### 空白或错误的图表

**问题**: 图表显示无数据或错误数据

**诊断**:
```python
# 检查数据可用性
from mymonitor.storage.data_manager import DataStorageManager
manager = DataStorageManager(Path("logs/run_20231201_120000"))
df = manager.load_memory_samples()
print(f"Loaded {len(df)} samples")
print(df.head())
```

**解决方案**:
1. 验证监控期间是否收集了数据
2. 检查进程分类规则
3. 确保监控持续时间足够

### 8. 系统特定问题

#### macOS 问题

**问题**: `OSError: [Errno 1] Operation not permitted`

**解决方案**:
```bash
# 在系统偏好设置中授予终端完全磁盘访问权限
# 或使用 sudo 进行系统进程访问
sudo python -m mymonitor monitor -p myproject
```

#### Windows 问题

**问题**: Windows 上进程监控不工作

**解决方案**:
```toml
# 使用 Windows 兼容的收集器
[monitor.collection]
metric_type = "rss_psutil"  # 在 Windows 上更可靠

# 为 Windows 调整进程模式
process_pattern = "cl\\.exe|link\\.exe|msbuild"
```

#### 容器/Docker 问题

**问题**: 无法从容器访问主机进程

**解决方案**:
```bash
# 在容器中运行时挂载 /proc
docker run -v /proc:/host/proc mymonitor

# 或在主机系统而不是容器中运行
```

## 调试技术

### 1. 启用详细日志记录

```toml
[monitor.general]
verbose = true

[monitor.collection]
debug_process_discovery = true
debug_memory_collection = true
```

### 2. 测试单个组件

```python
# 测试配置加载
from mymonitor.config.manager import ConfigManager
config = ConfigManager().load_config()
print(config)

# 测试内存收集
from mymonitor.collectors.factory import CollectorFactory
collector = CollectorFactory().create_collector("pss_psutil")
samples = collector.collect_single_sample()
print(f"Collected {len(samples)} samples")

# 测试存储
from mymonitor.storage.data_manager import DataStorageManager
manager = DataStorageManager(Path("/tmp/test"))
info = manager.get_storage_info()
print(info)
```

### 3. 监控系统资源

```bash
# 监控 CPU 使用率
top -p $(pgrep -f mymonitor)

# 监控内存使用率
ps -o pid,vsz,rss,comm -p $(pgrep -f mymonitor)

# 监控文件描述符
lsof -p $(pgrep -f mymonitor)
```

### 4. 验证数据完整性

```python
# 检查数据一致性
import polars as pl
df = pl.read_parquet("memory_samples.parquet")

# 验证时间戳是否有序
assert df["timestamp"].is_sorted()

# 检查缺失数据
assert df.null_count().sum() == 0

# 验证内存值是否合理
assert (df["rss_kb"] > 0).all()
```

## 获取帮助

### 1. 日志分析

报告问题时，包含相关日志摘录：

```bash
# 启用详细日志记录并捕获输出
python -m mymonitor monitor -p myproject --verbose 2>&1 | tee mymonitor.log

# 提取错误消息
grep -i error mymonitor.log
grep -i exception mymonitor.log
```

### 2. 系统信息

提供系统详细信息：

```bash
# Python 版本
python --version

# UV 版本
uv --version

# 系统信息
uname -a

# 可用核心
nproc

# 内存信息
free -h
```

### 3. 配置验证

验证并共享配置：

```bash
# 测试配置加载
python -c "
from mymonitor.config.manager import ConfigManager
try:
    config = ConfigManager().load_config()
    print('Configuration loaded successfully')
except Exception as e:
    print(f'Configuration error: {e}')
"
```

### 4. 最小重现

创建最小测试用例：

```bash
# 使用简单项目测试
mkdir /tmp/test_project
cd /tmp/test_project
echo 'int main() { return 0; }' > test.c

# 创建最小配置
cat > conf/projects.toml << EOF
[[projects]]
name = "test"
dir = "/tmp/test_project"
build_command_template = "gcc -o test test.c"
clean_command_template = "rm -f test"
process_pattern = "gcc"
EOF

# 运行监控
python -m mymonitor monitor -p test -j 1
```

## 性能优化

### 1. 监控开销减少

```toml
# 优化以获得最小开销
[monitor.collection]
interval_seconds = 0.1
descendants_only = true
metric_type = "rss_psutil"  # 比 PSS 更快

[monitor.thread_pools.monitoring]
max_workers = 1

[monitor.scheduling]
enable_cpu_affinity = true
monitor_core = 0  # 将监控隔离到特定核心
```

### 2. 存储优化

```toml
# 优化存储性能
[monitor.storage]
format = "parquet"
compression = "snappy"  # 快速压缩
generate_legacy_formats = false
```

### 3. 内存使用优化

```python
# 加载数据时使用列裁剪
df = manager.load_memory_samples(columns=["timestamp", "pid", "rss_kb"])

# 对大型数据集分块处理数据
chunk_size = 10000
for chunk in df.iter_slices(chunk_size):
    process_chunk(chunk)
```
