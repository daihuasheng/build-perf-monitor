# MyMonitor - 构建性能监控工具

MyMonitor 是一个综合性的构建性能监控工具，用于追踪软件编译过程中的内存使用情况。它提供了详细的内存消耗模式分析，支持不同进程类别和并行度级别的对比。

## 🚀 主要特性

- **实时内存监控**: 追踪构建过程中的PSS/RSS内存使用情况
- **进程分类**: 自动分类构建进程（编译器、链接器、脚本等）
- **多并行度分析**: 比较不同 `-j` 级别的性能表现
- **交互式可视化**: 生成详细的时间序列和汇总图表
- **模块化架构**: 清晰、可维护的代码库，专门模块化设计
- **高级错误处理**: 强大的重试机制和熔断器模式
- **综合报告**: 分层的类别统计和构建摘要

## 📁 架构概览

MyMonitor 采用模块化架构，明确分离各个关注点：

```
src/mymonitor/
├── cli/                    # 命令行界面
├── config/                 # 配置管理（TOML文件）
├── models/                 # 数据模型和结构
├── validation/             # 输入验证和错误处理策略
├── system/                 # 系统交互（CPU分配、命令执行）
├── classification/         # 进程分类引擎
├── collectors/             # 内存数据收集（PSS/RSS）
├── monitoring/             # 监控协调和编排
├── execution/              # 构建过程执行和清理
├── runner/                 # 主要监控工作流编排
└── orchestration/          # 高级组件协调
```

## 🛠️ 安装说明

### 使用 UV（推荐）

```bash
# 克隆仓库
git clone <repository-url>
cd mymonitor

# 使用 uv 安装
uv pip install -e .

# 开发环境，包含测试依赖
uv pip install -e ".[dev]"

# 支持PNG导出功能
uv pip install -e ".[export]"
```

### 使用 Pip

```bash
pip install -e .
```

## ⚙️ 配置说明

MyMonitor 使用位于 `conf/` 目录的TOML配置文件：

### 主配置文件 (`conf/config.toml`)
```toml
[monitor.general]
default_jobs = [4, 8, 16]           # 默认并行度级别
skip_plots = false                   # 监控后生成图表
log_root_dir = "logs"               # 输出目录

[monitor.collection]
interval_seconds = 0.05             # 采样间隔
metric_type = "pss_psutil"          # 内存收集器类型
pss_collector_mode = "full_scan"    # 进程扫描模式

[monitor.scheduling]
scheduling_policy = "adaptive"       # CPU调度策略
monitor_core = 0                    # 监控进程使用的核心
```

### 项目配置 (`conf/projects.toml`)
```toml
[[projects]]
name = "qemu"
dir = "/host/qemu/build"
build_command_template = "make -j<N>"
process_pattern = "make|gcc|clang|ld|..."
clean_command_template = "make clean"
```

### 分类规则 (`conf/rules.toml`)
定义进程分类规则，包含主要和次要类别。

## 🖥️ 命令行使用

### 基本使用

```bash
# 使用默认并行度监控默认项目
mymonitor

# 监控特定项目
mymonitor -p qemu

# 使用自定义并行度级别
mymonitor -p qemu -j 8,4,2,1

# 跳过预构建清理
mymonitor -p qemu --no-pre-clean
```

### 命令行选项

- `-p, --project PROJECT`: 指定要监控的项目
- `-j, --jobs JOBS`: 逗号分隔的并行度级别（如 "8,16"）
- `--no-pre-clean`: 跳过预构建清理步骤
- `-h, --help`: 显示帮助信息

### 可用项目

当前配置的项目：
- `qemu`: QEMU虚拟化平台
- `aosp`: Android开源项目
- `chromium`: Chromium网络浏览器

## 📊 输出结构

MyMonitor 在 `logs/` 目录中生成有组织的输出：

```
logs/
└── run_20250703_143052/              # 时间戳运行目录
    ├── qemu_j8_pss_psutil_20250703_143052/   # 每个并行度的数据
    │   ├── summary.log               # 增强摘要，包含类别统计
    │   ├── memory_samples.parquet    # 原始内存数据
    │   ├── build_stdout.log          # 构建输出
    │   ├── metadata.log              # 运行元数据
    │   ├── qemu_j8_PSS_KB_lines_plot.html     # 时间序列线图
    │   └── qemu_j8_PSS_KB_stacked_plot.html   # 堆叠面积图
    ├── qemu_j4_pss_psutil_20250703_143053/   # 其他并行度级别
    │   └── ...
    └── qemu_build_summary_plot.html   # 跨并行度比较
```

### 增强摘要格式

新的 summary.log 格式提供了综合的构建分析：

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

## 📈 可视化功能

MyMonitor 使用 Plotly 自动生成交互式图表：

### 汇总图表
- **跨并行度比较**: 不同 `-j` 级别的构建时间与内存使用对比
- **双轴可视化**: 在同一图表上显示内存（柱状图）和持续时间（线图）

### 详细图表
- **时间序列线图**: 按类别显示内存使用趋势
- **堆叠面积图**: 按进程类别显示总内存组成
- **交互功能**: 缩放、平移、悬停详情、类别过滤

### 图表生成

图表在监控后自动生成，或手动生成：

```bash
# 为运行生成所有图表
python tools/plotter.py --log-dir logs/run_20250703_143052

# 仅生成汇总图表
python tools/plotter.py --log-dir logs/run_20250703_143052 --summary-plot

# 为特定并行度生成图表
python tools/plotter.py --log-dir logs/run_20250703_143052 --jobs 8

# 按类别或前N个过滤
python tools/plotter.py --log-dir logs/run_20250703_143052 --category CPP_Compile
python tools/plotter.py --log-dir logs/run_20250703_143052 --top-n 5
```

## 🔧 开发指南

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试模块
uv run pytest tests/test_basic_monitoring.py
uv run pytest tests/test_plotter_tool.py

# 详细输出运行
uv run pytest -v
```

### 测试覆盖

MyMonitor 包含全面的测试覆盖：
- **基础监控**: 端到端工作流验证
- **进程分类**: 规则引擎和分类
- **绘图工具**: 图表生成和格式验证
- **配置**: TOML解析和验证
- **系统实用程序**: CPU分配和进程管理

### 代码质量

代码库遵循现代Python实践：
- **类型提示**: 完整的类型注解覆盖
- **错误处理**: 全面的验证和错误恢复
- **模块化设计**: 清晰的关注点分离
- **文档**: 详细的文档字符串和注释

## 🆕 最新改进

### 主要重构 (v2.0)
- **模块化架构**: 将单体文件替换为专门模块
- **增强错误处理**: 添加重试机制、熔断器和恢复策略
- **改进CLI**: 添加 `-p` 别名和更好的参数验证
- **更好的测试**: 扩展测试覆盖，包含60+测试用例

### 摘要和可视化增强
- **分层统计**: 摘要中的主要/次要类别分组
- **绘图集成**: 改进格式兼容性和图表组织
- **构建计时**: 添加全面的持续时间跟踪
- **内存指标**: 清晰的 PSS_KB/RSS_KB 标记

### 性能优化
- **收集器整合**: 统一 memory_collectors 和 collectors 目录
- **BuildRunner改进**: 增强数据聚合和结果格式化
- **文件组织**: 按并行度级别逻辑分组图表

## 🤝 贡献指南

1. Fork 仓库
2. 创建功能分支: `git checkout -b feature-name`
3. 进行更改并添加测试
4. 运行测试套件: `uv run pytest`
5. 提交拉取请求

## 📄 许可证

[在此添加您的许可证信息]

## 🐛 故障排除

### 常见问题

**进程模式不匹配**: 更新 `projects.toml` 中的 `process_pattern` 以包含相关构建工具。

**权限错误**: 确保监控用户对 `/proc` 文件系统有读取权限以收集内存数据。

**缺少依赖**: 安装系统依赖，如用于RSS收集的 `pidstat`：
```bash
sudo apt-get install sysstat  # Ubuntu/Debian
```

**图表生成失败**: 安装PNG输出的导出依赖：
```bash
uv pip install mymonitor[export]
```

如需更多帮助，请查看监控输出目录中的日志或使用详细日志记录运行。 