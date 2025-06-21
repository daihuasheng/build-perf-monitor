# MyMonitor

MyMonitor 是一个用于监控和分析复杂构建过程（如编译大型 C/C++ 项目）内存使用情况的命令行工具。它能够详细记录构建过程中各个子进程的内存占用，并根据可配置的规则对其进行分类，最终生成详细的数据报告和可视化图表。

MyMonitor is a command-line tool for monitoring and analyzing the memory usage of complex build processes, such as compiling large C/C++ projects. It can record detailed memory consumption of individual subprocesses during a build, categorize them based on configurable rules, and generate detailed data reports and visualizations.

---

## 目录 (Table of Contents)

- [MyMonitor](#mymonitor)
  - [目录 (Table of Contents)](#目录-table-of-contents)
  - [特性 (Features)](#特性-features)
  - [安装与配置 (Installation \& Setup)](#安装与配置-installation--setup)
  - [核心概念：配置文件 (Core Concept: Configuration Files)](#核心概念配置文件-core-concept-configuration-files)
    - [主配置文件: `config.toml`](#主配置文件-configtoml)
    - [项目定义: `projects.toml`](#项目定义-projectstoml)
    - [分类规则: `rules.toml`](#分类规则-rulestoml)
  - [如何... (How To...)](#如何-how-to)
    - [如何配置工具初次使用 (How to Configure the Tool for First Use)](#如何配置工具初次使用-how-to-configure-the-tool-for-first-use)
    - [如何添加一个新的监控项目 (How to Add a New Project to Monitor)](#如何添加一个新的监控项目-how-to-add-a-new-project-to-monitor)
    - [如何修改或添加分类规则 (How to Modify or Add Classification Rules)](#如何修改或添加分类规则-how-to-modify-or-add-classification-rules)
  - [使用方法 (Usage)](#使用方法-usage)
  - [输出结果 (Output)](#输出结果-output)
  - [工作原理 (How It Works)](#工作原理-how-it-works)

---

## 特性 (Features)

- **详细的内存追踪 (Detailed Memory Tracking)**: 可使用 `psutil` (追踪 PSS/USS/RSS) 或 `pidstat` (追踪 RSS/VSZ) 作为后端，精确采样内存数据。
- **自动化进程分类 (Automated Process Categorization)**: 通过高度可定制的规则文件 (`rules.toml`)，自动将进程（如 `gcc`, `clang`, `ld`, `ninja`）归类到不同类别（如 `Compiler`, `Linker`, `BuildSystem`）。
- **支持多项目和多并发度 (Multi-Project & Multi-Parallelism Support)**: 可在一次运行中，针对多个项目，使用不同的并发度（`-j` 值）进行监控。
- **CPU 亲和性控制 (CPU Affinity Control)**: 可将监控脚本和构建进程绑定到指定的 CPU核心，减少相互干扰。
- **丰富的报告和可视化 (Rich Reporting & Visualization)**:
  - 为每次运行生成详细的摘要日志 (`_summary.log`)。
  - 将所有原始采样数据保存为高效的 Parquet 文件 (`.parquet`)，便于后续分析。
  - 自动生成交互式 HTML 图表，直观展示内存使用趋势。

---

## 安装与配置 (Installation & Setup)

**1. 系统依赖 (Prerequisites)**

- Python 3.12+
- (可选，若使用 `rss_pidstat` 收集器) `sysstat` 包。在基于 Debian/Ubuntu 的系统上，可以通过以下命令安装：

  ```bash
  sudo apt-get update && sudo apt-get install sysstat
  ```

**2. 克隆仓库 (Clone the Repository)**

```bash
git clone <your-repository-url>
cd mymonitor
```

**3. 创建虚拟环境并安装依赖 (Create a Virtual Environment & Install Dependencies)**

建议使用 `uv` 或 `pip` 在虚拟环境中安装依赖。

```bash
# 创建虚拟环境 (Create a virtual environment)
python3 -m venv .venv
source .venv/bin/activate

# 使用 uv 安装 (Using uv)
# uv pip install -e ".[dev,export]"

# 或使用 pip 安装 (Or using pip)
pip install -e ".[dev,export]"
```

> `[dev]` 会安装测试依赖（如 `pytest`）。
> `[export]` 会安装 `kaleido`，用于将图表导出为静态图片（如 `.png`）。

---

## 核心概念：配置文件 (Core Concept: Configuration Files)

所有配置都位于 `conf/` 目录下，采用 TOML 格式。

All configuration resides in the `conf/` directory in TOML format.

### 主配置文件: `config.toml`

这是应用的全局入口点。它定义了监控器的默认行为和指向其他配置文件的路径。

This is the global entry point for the application. It defines the monitor's default behavior and points to other configuration files.

### 项目定义: `projects.toml`

此文件定义了所有可以被监控的目标项目。每个项目都是一个独立的构建任务。

This file defines all target projects that can be monitored. Each project represents a distinct build task.

### 分类规则: `rules.toml`

这是分类引擎的核心。此文件包含一系列规则，用于将监控到的进程名称和命令行映射到具体的类别。

This is the core of the classification engine. This file contains a set of rules used to map monitored process names and command lines to specific categories.

---

## 如何... (How To...)

### 如何配置工具初次使用 (How to Configure the Tool for First Use)

1.  **打开 `conf/config.toml`**:
    - 检查 `log_root_dir`，确保日志输出目录符合您的期望。默认为 `./logs`。
    - 根据您的需求调整 `default_jobs`（默认的并发级别列表）。
    - 选择您偏好的 `metric_type` (`pss_psutil` 或 `rss_pidstat`)。

2.  **打开 `conf/projects.toml`**:
    - 查看示例项目（如 `qemu`），并根据您的环境修改 `dir` 字段，使其指向您本地的项目源代码路径。

3.  **运行一次测试 (Run a Test)**:
    ```bash
    mymonitor -p qemu -j 4
    ```
    如果一切顺利，您应该会在 `logs/` 目录下看到一个新生成的 `run_<timestamp>` 文件夹，其中包含日志和数据文件。

### 如何添加一个新的监控项目 (How to Add a New Project to Monitor)

1.  **打开 `conf/projects.toml` 文件。**
2.  在文件末尾添加一个新的 `[[projects]]` 部分。
3.  填充所有必需的字段。

**示例**: 假设您要添加一个名为 `my-kernel` 的项目。

```toml
# filepath: conf/projects.toml
# ... existing projects ...

[[projects]]
# 项目的唯一名称
name = "my-kernel"

# 项目的根目录，所有命令将在此执行
dir = "/path/to/your/linux-kernel"

# (可选) 构建前需要执行的设置命令
setup_command_template = "export ARCH=arm64 && export CROSS_COMPILE=aarch64-linux-gnu-"

# 构建命令模板，<N> 会被并发级别替换
build_command_template = "make -j<N> Image.gz"

# 用于识别相关进程的正则表达式。这是关键！
# 需要包含所有可能出现的编译器、链接器、脚本等进程名。
process_pattern = "make|gcc|aarch64-linux-gnu-gcc|ld|as|python[0-9._-]*"

# (可选) 清理构建产物的命令
clean_command_template = "make mrproper"
```

### 如何修改或添加分类规则 (How to Modify or Add Classification Rules)

分类引擎根据 `conf/rules.toml` 中的规则进行工作。规则按 `priority` 字段降序处理，**第一个匹配的规则生效**。

1.  **打开 `conf/rules.toml` 文件。**
2.  要修改现有规则，只需找到对应的 `[[rules]]` 部分并编辑其字段。
3.  要添加新规则，请在文件中添加一个新的 `[[rules]]` 部分。

**关键字段解释**:

-   `major_category`: 宽泛的类别 (e.g., `CPP_Compile`, `BuildSystem`).
-   `category`: 具体的子类别 (e.g., `GCCInternalCompiler`, `Ninja`).
-   `priority`: **最重要的字段**。高优先级的规则会先被检查。这允许您为通用命令（如 `gcc`）创建精细的、基于参数的规则，同时也有一个低优先级的通用后备规则。
-   `match_field`: 要匹配的进程属性 (`current_cmd_name`, `current_cmd_full`, `orig_cmd_name`, `orig_cmd_full`).
-   `match_type`: 匹配方式 (`exact`, `contains`, `startswith`, `endswith`, `regex`, `in_list`).
-   `pattern` / `patterns`: 用于匹配的字符串或字符串列表。

**示例**: 假设您的构建系统使用一个名为 `super-linker` 的特殊链接器，您想将其分类。

```toml
# filepath: conf/rules.toml
# ... existing rules ...

[[rules]]
major_category = "CPP_Link"
category = "SuperLinker"
priority = 156 # 比通用的 'ld' (155) 优先级稍高
match_field = "current_cmd_name"
match_type = "exact"
pattern = "super-linker"
comment = "Categorizes our custom high-performance linker."
```

---

## 使用方法 (Usage)

该工具通过 `mymonitor` 命令运行。

The tool is run via the `mymonitor` command.

**基本用法 (Basic Usage)**:

```bash
# 运行所有在 projects.toml 中定义的项目，使用 config.toml 中的默认并发度
mymonitor

# 仅运行 'qemu' 和 'chromium' 项目
mymonitor -p qemu chromium

# 运行 'aosp' 项目，并覆盖并发度为 -j8 和 -j16
mymonitor -p aosp -j 8 16

# 运行并覆盖内存收集器类型
mymonitor --metric-type rss_pidstat

# 仅运行清理命令，不进行监控
mymonitor -p qemu --clean-only

# 查看所有可用选项
mymonitor --help
```

---

## 输出结果 (Output)

每次运行都会在 `log_root_dir` (默认为 `logs/`) 中创建一个唯一的带时间戳的目录，例如 `logs/run_20250621_143000/`。

Each run creates a unique, timestamped directory inside `log_root_dir` (default: `logs/`), for example `logs/run_20250621_143000/`.

该目录包含：

-   **`*_summary.log`**: 人类可读的摘要文件，包含运行配置、构建命令的输出、峰值内存统计和分类摘要。
-   **`*.parquet`**: 高性能的列式数据文件，包含每次采样间隔收集到的所有原始数据。可用于 `pandas` 或 `polars` 进行深入分析。
-   **`*.html`**: 交互式的 Plotly 图表，可视化内存使用情况。
-   **`*.png`** (如果安装了 `kaleido`): 图表的静态图片版本。

---

## 工作原理 (How It Works)

1.  **初始化 (Initialization)**: `main.py` 解析命令行参数，并加载 `config.py` 中的配置。
2.  **编排 (Orchestration)**: `main.py` 遍历所有选定的项目和并发级别。对于每一次组合，它都会调用 `monitor_utils.py` 中的核心函数 `run_and_monitor_build`。
3.  **执行与监控 (Execution & Monitoring)**:
    -   `run_and_monitor_build` 启动构建命令作为一个子进程。
    -   同时，它根据配置（`pss_psutil` 或 `rss_pidstat`）启动一个内存收集器。
    -   主监控循环从收集器获取内存样本，并使用 `process_utils.get_process_category` 对每个进程进行分类。
    -   所有数据被聚合并保存在内存中。
4.  **报告 (Reporting)**: 构建完成后，聚合的结果被写入 `.parquet` 文件和 `_summary.log` 文件。
5.  **可视化 (Visualization)**: 最后，`plotter.py` 被调用，读取 `.parquet` 文件并生成 HTML/PNG 图表。
