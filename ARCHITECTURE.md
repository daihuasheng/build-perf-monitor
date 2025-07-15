# MyMonitor 架构文档

> **文档版本**: v2.1
> **最后更新**: 2025-07-15
> **适用版本**: MyMonitor v2.1+ (测试完善版)

## 📋 目录

- [概述](#概述)
- [设计原则](#设计原则)
- [架构概览](#架构概览)
- [核心模块](#核心模块)
- [数据流](#数据流)
- [配置系统](#配置系统)
- [监控流程](#监控流程)
- [错误处理策略](#错误处理策略)
- [性能优化](#性能优化)
- [测试架构](#测试架构)
- [扩展性设计](#扩展性设计)

---

## 概述

MyMonitor 是一个专业的构建性能监控工具，采用现代化的异步架构设计，用于实时监控软件编译过程中的内存使用情况。项目经过 v2.0 重大重构，从单体架构转换为模块化设计，提供了更好的可维护性、扩展性和性能。

### 核心功能

- **实时内存监控**: 使用 PSS/RSS 指标追踪构建过程内存消耗
- **智能进程分类**: 基于规则引擎自动分类编译器、链接器、脚本等进程
- **多并行度分析**: 支持跨不同 `-j` 级别的性能对比分析
- **交互式可视化**: 生成详细的时间序列图表和汇总统计
- **异步架构**: 基于 AsyncIO 的高性能监控系统
- **容错处理**: 完善的错误恢复机制和资源管理

---

## 设计原则

### 1. 模块化分离 (Modular Separation)

- **单一职责原则**: 每个模块专注于特定功能域
- **清晰边界**: 模块间通过明确接口通信
- **低耦合**: 最小化模块间依赖

### 2. 混合异步架构 (Hybrid Async Architecture)

- **主线程异步循环**: 使用 AsyncIO 协程进行任务调度和协调
- **同步工作线程**:
  - 监控线程：使用线程池处理内存采样（唯一需要线程池的地方）
  - 构建线程：单线程执行构建任务
  - I/O线程：单线程处理文件I/O操作
- **性能优化**: 异步分派 + 同步执行的混合模式，兼顾性能和简洁性
- **职责分离**: 线程管理模块只负责任务调度，具体收集由collector模块负责

### 3. 配置驱动 (Configuration-Driven)

- **声明式配置**: 通过 TOML 文件定义行为
- **分层配置**: 全局、项目、规则三层配置体系
- **运行时验证**: 配置加载时的完整性检查

### 4. 健壮性优先 (Robustness-First)

- **优雅降级**: 部分失败时继续运行
- **资源清理**: 确保资源正确释放
- **错误恢复**: 自动重试和熔断机制

### 5. 可观测性 (Observability)

- **结构化日志**: 分级日志记录
- **监控指标**: 性能和健康状态监控
- **调试支持**: 详细的错误信息和堆栈跟踪

---

## 架构概览

```text
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                            │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   main.py       │    │  orchestrator.py │                │
│  │ (Entry Point)   │    │ (BuildRunner)   │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Configuration Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ config.toml │  │projects.toml│  │ rules.toml  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │           Configuration Management                       ││
│  │         (Validation & Loading)                          ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Execution Layer                          │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ BuildProcess    │    │ Thread Pool     │                │
│  │ Manager         │    │ Management      │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring Layer                         │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ HybridArchitect │    │ Collector       │                │
│  │ ure (Producer-  │    │ Factory         │                │
  │ Consumer)       │    │                 │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Collection Layer                         │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Memory          │    │ Async Sample    │                │
│  │ Collectors      │    │ Processing      │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Classification Layer                     │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Process         │    │ Rule Engine     │                │
│  │ Classifier      │    │ (Cached)        │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                      Storage Layer                          │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Parquet Files   │    │ Summary Logs    │                │
│  │ (Raw Data)      │    │ (Statistics)    │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Visualization Layer                      │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Plotly Charts   │    │ External        │                │
│  │ (Interactive)   │    │ Plotter Tool    │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心模块

### 1. CLI 层 (`cli/`)

**职责**: 命令行接口和顶层编排

- **`main.py`**: 程序入口点，处理命令行参数，信号处理，全局生命周期管理
- **`orchestrator.py`**: `BuildRunner` 类，协调整个监控流程的异步执行

**核心类**:

```python
class BuildRunner:
    """主要的异步构建监控器，协调构建执行和监控"""
    
    async def run_async(self) -> bool:
        """运行完整的构建监控流程 - 直接使用 HybridArchitecture"""
        
    def run(self) -> bool:
        """同步包装器，为 CLI 使用"""
```

**重构变化**:

- **简化架构**: CLI 直接使用 HybridArchitecture，无中间抽象层
- **统一工厂创建**: CollectorFactory 创建逻辑集中在 RunContext.create_collector_factory()
- **专用线程池**: 线程池管理专注于监控任务，构建和I/O任务使用单线程异步处理
- **依赖注入**: HybridArchitecture 通过构造函数接收线程池管理器，职责分离清晰
- **分类逻辑分离**: 进程分类从采样阶段移至聚合阶段，提高性能
- **更清晰的依赖**: BuildRunner → HybridArchitecture → CollectorFactory → Collectors

**设计特点**:

- 信号处理确保优雅关闭
- 异步/同步接口兼容
- 全局错误处理和日志记录

### 2. 配置层 (`config/`)

**职责**: 配置管理和验证

**模块组成**:

- **加载器**: TOML 文件解析和缓存
- **验证器**: 配置完整性检查
- **模型**: 强类型配置数据结构

**配置层次**:

```text
config.toml     ← 全局监控配置
projects.toml   ← 项目定义
rules.toml      ← 分类规则
```

**核心接口**:

```python
def get_config() -> AppConfig:
    """获取应用配置，带缓存机制"""

def validate_monitor_config(config: MonitorConfig) -> None:
    """验证监控配置的合法性"""
```

### 3. 数据模型层 (`models/`)

**职责**: 定义应用中使用的所有数据结构

**模型分类**:

- **配置模型** (`config.py`): `AppConfig`, `MonitorConfig`, `ProjectConfig`, `RuleConfig`
- **运行时模型** (`runtime.py`): `RunContext`, `RunPaths`, `CpuAllocationPlan`
- **结果模型** (`results.py`): `MonitoringResults`

**设计特点**:

- 使用 `@dataclass` 减少样板代码
- 类型注解确保类型安全
- 后处理方法处理兼容性逻辑

### 4. 执行层 (`executor/`)

**职责**: 构建进程管理和监控专用线程池管理

**核心组件**:

- **`BuildProcessManager`**: 异步构建进程执行器
- **`ThreadPoolManager`**: 监控专用线程池管理
- **`ThreadPoolConfig`**: 线程池配置

**关键特性**:

```python
class BuildProcessManager:
    """异步构建进程管理器"""
    
    async def start_build_async(self) -> int:
        """异步启动构建进程"""
        
    async def wait_for_completion_async(self) -> int:
        """等待构建完成，支持超时和取消"""
        
    async def cancel_build_async(self) -> None:
        """取消运行中的构建"""
```

**线程池特性**:

- 专注于监控任务的线程池管理
- 构建和I/O任务使用单线程异步处理
- CPU亲和性绑定确保监控性能
- 支持取消和超时的线程池任务

### 5. 监控层 (`monitoring/`)

**职责**: 混合架构异步监控

**核心类**:

```python
class HybridArchitecture:
    """混合监控架构 - 主线程异步循环 + 多工作线程处理"""
    
    async def start_monitoring(self, build_pid: int) -> None:
        """开始监控指定进程"""
        
    async def stop_monitoring(self) -> None:
        """停止监控并收集结果"""
        
    async def get_results(self) -> MonitoringResults:
        """获取监控结果"""
```

**架构特点**:

- **生产者-消费者模式**: 发现工作者 + 采样工作者
- **异步协调**: 主线程使用 asyncio 事件循环
- **专用线程池**: 监控任务使用专门的线程池处理
- **统一工厂管理**: CollectorFactory 创建逻辑集中化

### 6. 收集器层 (`collectors/`)

**职责**: 内存数据收集的具体实现

**重构变化**:

- **CollectorFactory**: 位于 `collectors/factory.py`，专注于收集器实例创建
- **更清晰的模块边界**: 收集器相关逻辑集中在 collectors 模块

**收集器类型**:

- **`PssPsutilCollector`**: 基于 psutil 的 PSS 内存收集
- **`RssPidstatCollector`**: 基于 pidstat 的 RSS 内存收集

**设计模式**:

```python
class AbstractMemoryCollector(ABC):
    """内存收集器抽象基类"""
    
    def collect_single_sample(self) -> List[ProcessMemorySample]:
        """收集单次内存数据样本 - 同步方法，由架构层调用"""
```

**工厂模式**:

```python
class CollectorFactory:
    """收集器工厂 - 根据配置创建收集器实例"""
    
    def create_collector(self, collector_cpu_core: int) -> AbstractMemoryCollector:
        """创建内存收集器实例"""
```
    
    def collect_single_sample(self) -> List[ProcessMemorySample]:
        """收集单次内存数据样本"""
```

**优化特性**:

- 可选的 `descendants_only` 模式提升性能
- 智能进程发现算法
- 批量数据处理

### 7. 存储层 (`storage/`)

**职责**: 高效存储和管理监控数据

**核心功能**:

```python
class DataStorage(ABC):
    """数据存储抽象基类"""

    def save_dataframe(self, df: pl.DataFrame, path: str) -> None:
        """保存数据帧到指定路径"""

    def load_dataframe(self, path: str, columns: Optional[List[str]] = None) -> pl.DataFrame:
        """从指定路径加载数据帧，支持列裁剪"""
```

**存储格式**:

- **Parquet 格式** (推荐): 列式存储，高压缩率，支持列裁剪
- **JSON 格式**: 用于元数据和小型配置文件

**存储管理器**:

```python
class DataStorageManager:
    """高级数据存储管理器"""

    def save_monitoring_results(self, results: MonitoringResults, run_context: Any) -> None:
        """保存完整监控结果"""

    def load_memory_samples(self, columns: Optional[List[str]] = None) -> pl.DataFrame:
        """加载内存样本数据"""
```

**优化特性**:

- 高压缩率 (减少 75-80% 存储空间)
- 列式存储 (提高查询性能 3-5 倍)
- 批量处理 (减少 I/O 操作)
- 数据类型保留 (提高数据质量)

### 8. 分类层 (`classification/`)

**职责**: 进程分类和规则引擎

**核心功能**:

```python
def get_process_category(
    cmd_name: str, 
    full_cmd: str, 
    rules: List[RuleConfig]
) -> Tuple[str, str]:
    """根据规则对进程进行分类，返回 (major_category, minor_category)"""
```

**分类机制**:

- 基于优先级的规则匹配
- 支持多种匹配模式：exact, regex, contains, in_list
- LRU 缓存提升性能

**分类层次**:

```text
Major Category (主类别)
├── CPP_Compile
│   ├── GCCInternalCompiler
│   └── Driver_Compile
├── CPP_Link  
│   ├── DirectLinker
│   └── Driver_Link
└── ...
```

### 8. 系统层 (`system/`)

**职责**: 系统交互和资源管理

**主要功能**:

- **CPU 分配策略**: 自适应和手动 CPU 核心分配
- **命令执行**: 跨平台命令运行工具
- **系统检查**: 依赖项验证和系统兼容性

**CPU 调度算法**:

```python
def plan_cpu_allocation(
    cores_policy: str,
    parallelism_level: int,
    monitoring_workers: int
) -> CpuAllocationPlan:
    """计划 CPU 核心分配"""
```

### 9. 验证层 (`validation/`)

**职责**: 输入验证和错误处理

**错误处理策略**:

- **重试机制**: 带指数退避的自动重试
- **熔断器**: 防止级联故障
- **错误恢复**: 优雅降级和资源清理

**验证器类型**:

- 数据类型验证
- 范围检查
- 格式验证
- 业务规则验证

---

## 数据流

### 1. 配置流

```text
TOML 文件 → 解析器 → 验证器 → 配置模型 → 缓存
```

### 2. 监控流

```text
构建启动 → 进程发现 → 任务队列 → 内存采样 → 结果队列 → 数据聚合 → 进程分类 → 结果存储
    ↓           ↓           ↓           ↓           ↓           ↓           ↓           ↓
构建进程    发现Worker   异步队列    采样Workers   异步队列    结果Worker   分类器     MonitoringResults
(单线程)   (1个协程)    (缓冲)     (N个协程)    (缓冲)     (1个协程)   (缓存)      (持久化)
```

### 3. 可视化流

```text
原始数据 → 统计计算 → 图表生成 → HTML 输出
```

### 详细数据流程

#### 启动阶段

1. **配置加载**: 解析 TOML 文件，验证配置项
2. **环境准备**: 创建输出目录，设置日志
3. **资源分配**: CPU 核心分配，线程池初始化

#### 执行阶段

1. **构建启动**: 在指定目录执行构建命令
2. **监控启动**: 创建监控任务，开始数据收集
3. **数据流转**: 收集 → 分类 → 聚合 → 存储

#### 结果阶段

1. **数据处理**: 计算统计信息，生成摘要
2. **文件输出**: 使用 Parquet 格式保存监控数据，生成 JSON 元数据和人类可读的摘要日志
3. **可视化**: 生成交互式图表

#### 存储优化流程

1. **数据收集**: 使用 Polars DataFrame 进行高效数据处理
2. **批量写入**: 减少 I/O 操作，提高写入性能
3. **压缩存储**: 使用 Snappy 压缩算法，平衡压缩率和速度
4. **列式存储**: 支持列裁剪，提高查询性能

---

## 配置系统

### 配置文件层次

#### 1. 主配置 (`config.toml`)

```toml
[monitor.general]
default_jobs = [4, 8, 16]      # 默认并行度
skip_plots = false             # 跳过图表生成
log_root_dir = "logs"          # 输出目录

[monitor.collection]
interval_seconds = 0.05        # 采样间隔
metric_type = "pss_psutil"     # 收集器类型
pss_collector_mode = "full_scan"  # 扫描模式

[monitor.scheduling]
scheduling_policy = "adaptive"  # CPU 调度策略
monitor_core = 0               # 监控进程核心
enable_cpu_affinity = true     # 启用 CPU 亲和性

[monitor.hybrid]
hybrid_discovery_interval = 0.01     # 发现间隔
hybrid_sampling_workers = 4          # 采样工作线程数
hybrid_task_queue_size = 1000        # 任务队列大小
hybrid_enable_prioritization = true  # 任务优先级
```

#### 2. 项目配置 (`projects.toml`)

```toml
[[projects]]
name = "qemu"
dir = "/host/qemu/build"
build_command_template = "make -j<N>"
process_pattern = "make|gcc|clang|ld"
clean_command_template = "make clean"
setup_command_template = ""
```

#### 3. 分类规则 (`rules.toml`)

```toml
[[rules]]
priority = 100
major_category = "CPP_Compile"
category = "GCCInternalCompiler"
match_field = "current_cmd_name"
match_type = "in_list"
patterns = ["cc1", "cc1plus", "collect2"]
```

### 配置验证策略

1. **类型检查**: 确保配置项类型正确
2. **范围验证**: 数值在合理范围内
3. **依赖检查**: 验证配置项间的依赖关系
4. **资源验证**: 检查文件路径和系统资源

---

## 监控流程

### 混合异步监控架构

MyMonitor 采用**混合异步架构**，结合异步主线程和同步工作线程的优势：

#### 线程分工和数量配置

**1. 主线程 (1个)**
- **类型**: 异步事件循环线程
- **职责**: 任务调度协调、异步队列管理、生命周期控制、信号处理

**2. 监控线程池 (可配置，默认4个)**
- **类型**: 同步工作线程池 (`ManagedThreadPoolExecutor`)
- **职责**: 内存数据采样、进程信息收集、CPU亲和性绑定
- **数量**: `min(4, parallelism_level)` 或配置指定

**3. 构建线程 (1个)**
- **类型**: 单线程异步执行器
- **职责**: 执行构建命令、构建进程管理、构建状态监控

**4. I/O线程 (1个)**
- **类型**: 单线程异步执行器
- **职责**: 文件读写操作、日志输出、结果序列化

#### CPU核心分配策略

**Adaptive策略 (自适应分配)**:

1. **构建任务核心分配**:
   - 基础分配: `max(1.25 * parallelism_level, parallelism_level + 4)`
   - 目标: 为构建任务提供充足的CPU资源

2. **监控任务核心分配**:
   - 优先级: 在构建任务分配后，从剩余核心中分配
   - 上限: 最多16个核心
   - 策略: 有剩余核心时独立分配，无剩余时与构建任务共享

3. **资源不足处理**:
   - 警告条件: 构建任务分配核心数 < parallelism_level
   - 处理方式: 输出警告但继续执行

**Manual策略 (手动分配)**:
- 根据配置文件中的 `manual_build_cores` 和 `manual_monitoring_cores` 分配

### 详细执行流程

#### 阶段 1: 初始化阶段

```text
BuildRunner.run_async → 创建RunContext → 初始化线程池 → CPU核心分配规划 → 创建HybridArchitecture → 设置监控基础设施
```

1. **配置加载**: 加载TOML配置文件
2. **线程池初始化**: 创建监控专用线程池
3. **CPU分配规划**: 根据策略分配CPU核心
4. **监控架构创建**: 通过依赖注入创建HybridArchitecture

#### 阶段 2: 监控设置阶段

```text
setup_monitoring → 创建发现收集器 → 创建采样收集器 → 初始化异步队列 → 创建停止事件
```

1. **收集器创建**: 创建发现收集器(1个)和采样收集器(多个)
2. **队列初始化**: 创建任务队列和结果队列
3. **事件初始化**: 创建停止事件用于优雅关闭

#### 阶段 3: 构建执行阶段

**构建进程启动**:
```python
# 单线程异步启动构建
self.process, self.build_pid = await loop.run_in_executor(
    None, self._start_build_process
)
```

**监控Workers启动**:

- **发现Worker (1个异步协程)**: 周期性发现进程，放入任务队列
- **采样Workers (N个异步协程)**: 从任务队列获取任务，执行内存采样
- **结果处理Worker (1个异步协程)**: 从结果队列收集采样结果

#### 阶段 4: 同步异步交互机制

**异步到同步的调用**:
```python
# 在监控线程池中执行同步收集
samples = await loop.run_in_executor(
    monitoring_pool.executor,
    collector.collect_single_sample
)
```

**线程池中的CPU亲和性设置**:
- 每个监控线程绑定到特定CPU核心
- 避免线程迁移开销，提高缓存局部性

#### 阶段 5: 结果聚合和清理

```text
构建完成 → 停止监控 → 等待所有Workers完成 → 聚合采样结果 → 进程分类处理 → 生成MonitoringResults → 清理资源
```

1. **停止信号**: 设置停止事件
2. **等待Workers**: 等待所有异步任务完成
3. **结果聚合**: 在聚合阶段进行进程分类(性能优化)
4. **资源清理**: 确保所有资源正确释放

### 异步监控架构

```python
async def monitoring_workflow():
    """异步监控工作流"""
    
    # 1. 设置阶段
    await setup_run_context()
    await setup_monitoring_infrastructure()
    
    # 2. 执行阶段
    build_task = create_task(start_build_process())
    monitor_task = create_task(start_monitoring())
    
    # 3. 等待完成
    await gather(build_task, monitor_task)
    
    # 4. 清理阶段
    await cleanup_resources()
```

### 监控生命周期

#### 1. 初始化 (Initialization)

- 解析配置文件
- 验证系统依赖
- 分配计算资源

#### 2. 准备 (Preparation)

- 创建运行上下文
- 计划 CPU 分配
- 初始化收集器

#### 3. 执行 (Execution)

- 启动构建进程
- 开始监控循环
- 实时数据收集

#### 4. 收集 (Collection)

```python
async def monitoring_loop():
    """监控循环"""
    while not stop_event.is_set():
        # 收集内存数据
        memory_data = await collector.collect_memory_async()
        
        # 分类进程
        for process in memory_data:
            category = get_process_category(
                process.cmd_name, 
                process.full_cmd, 
                rules
            )
            process.category = category
        
        # 存储数据
        await store_data(memory_data)
        
        # 等待下次采样
        await asyncio.sleep(interval)
```

#### 5. 终止 (Termination)

- 停止监控任务
- 等待构建完成
- 生成最终报告

### 数据收集策略

#### 全扫描模式 (Full Scan)

- **优点**: 可靠，不遗漏进程
- **缺点**: 开销较大
- **适用**: 复杂构建系统

#### 后代模式 (Descendants Only)

- **优点**: 高性能，低开销
- **缺点**: 可能遗漏分离进程
- **适用**: 简单线性构建

---

## 错误处理策略

### 分层错误处理

#### 1. 系统级错误

- **配置错误**: 立即退出，显示详细错误信息
- **依赖缺失**: 提供安装建议，优雅退出
- **权限错误**: 检查 `/proc` 访问权限

#### 2. 运行时错误

- **进程失败**: 记录错误，继续监控其他进程
- **收集器错误**: 重试机制，降级到其他收集器
- **存储错误**: 缓存数据，定期重试写入

#### 3. 资源错误

- **内存不足**: 清理缓存，减少并发度
- **磁盘空间**: 压缩输出，清理临时文件
- **网络错误**: 本地存储，禁用外部依赖

### 错误恢复机制

#### 重试策略

```python
@with_retry(
    max_attempts=3,
    backoff=ExponentialBackoff(initial=1.0, maximum=10.0)
)
async def resilient_operation():
    """带重试的操作"""
    pass
```

#### 熔断器模式

```python
@circuit_breaker(
    failure_threshold=5,
    recovery_timeout=30.0
)
async def protected_operation():
    """受熔断器保护的操作"""
    pass
```

#### 优雅降级

- **监控失败**: 继续构建，记录警告
- **分类失败**: 使用默认分类
- **可视化失败**: 保留原始数据

---

## 性能优化

### 1. 异步并发

- **协程调度**: 使用 `asyncio` 实现高效并发
- **线程池**: CPU 密集型任务使用线程池
- **非阻塞 I/O**: 所有 I/O 操作异步化

### 2. 缓存策略

- **配置缓存**: 避免重复解析 TOML 文件
- **分类缓存**: LRU 缓存常见进程分类
- **数据缓存**: 批量处理内存数据

### 3. 资源管理

- **内存优化**: 流式处理大型数据集
- **CPU 优化**: 智能 CPU 核心分配
- **I/O 优化**: 批量写入，异步刷新

### 4. 存储优化

- **Parquet 格式**: 列式存储，减少 75-80% 存储空间
- **压缩算法**: 支持多种压缩算法 (Snappy, Gzip, Brotli, LZ4, Zstd)
- **列裁剪**: 只读取需要的列，提高查询性能 3-5 倍
- **批量处理**: 减少 I/O 操作，提高写入性能
- **数据类型保留**: 避免类型转换问题，提高数据质量

### 4. 算法优化

- **进程发现**: 高效的进程树遍历
- **模式匹配**: 编译正则表达式缓存
- **数据聚合**: 增量统计计算

### 性能监控

#### 关键指标

- **采样率**: 每秒采样次数
- **内存使用**: 监控进程内存消耗
- **CPU 使用**: 监控进程 CPU 占用
- **I/O 吞吐**: 数据写入速度

#### 性能调优

```toml
# 配置示例
[monitor.collection]
interval_seconds = 0.05  # 更高频率采样

[monitor.scheduling]
max_concurrent_monitors = 8  # 增加并发度

[monitor.hybrid]
hybrid_sampling_workers = 8  # 增加采样线程数
hybrid_enable_prioritization = true  # 启用优化
```

---

## 测试架构

MyMonitor 采用分层测试策略，确保代码质量和系统稳定性。

### 1. 测试金字塔

```
    /\     端到端测试 (E2E)
   /  \    ├─ 完整工作流测试
  /____\   ├─ 多场景验证
 /      \  └─ 真实环境模拟
/________\
   集成测试 (Integration)
   ├─ 模块间协作测试
   ├─ 配置验证测试
   └─ 资源管理测试

单元测试 (Unit) - 基础层
├─ 核心逻辑测试
├─ 边界条件测试
└─ 错误处理测试
```

### 2. 测试分类

#### 单元测试 (81个)
- **位置**: `tests/unit/`
- **覆盖**: 核心模块的独立功能
- **特点**: 快速执行、隔离测试、Mock外部依赖

主要测试模块：
- `test_classification/` - 进程分类逻辑
- `test_config/` - 配置验证和加载
- `test_executor/` - 线程池和任务执行
- `test_system/` - CPU管理和资源分配
- `test_storage/` - 存储系统和数据管理

#### 集成测试 (8个)
- **位置**: `tests/integration/`
- **覆盖**: 模块间协作和配置集成
- **特点**: 验证组件协作、真实配置测试

主要测试场景：
- CPU分配策略集成
- 线程池与资源管理集成
- 配置验证集成测试

#### 性能测试 (9个)
- **位置**: `tests/performance/`
- **覆盖**: 系统性能特征验证
- **特点**: 性能基准测试、资源使用监控

主要测试内容：
- CPU分配算法性能
- 进程分类缓存性能
- 并发处理能力测试

#### 端到端测试 (6个)
- **位置**: `tests/e2e/`
- **覆盖**: 完整监控工作流
- **特点**: 真实场景模拟、完整功能验证

主要测试场景：
- 完整监控工作流
- 资源受限环境测试
- 手动CPU分配测试
- 构建失败处理测试
- 进程分类集成测试
- 优雅关闭测试

### 3. 测试基础设施

#### 测试配置管理
```python
@pytest.fixture
def config_files(temp_dir):
    """创建临时配置文件"""
    # 自动生成测试配置
    # 支持多种测试场景
```

#### Mock策略
- **外部依赖Mock**: 系统调用、文件操作、网络请求
- **进程Mock**: 模拟构建进程和监控目标
- **资源Mock**: CPU核心、内存信息、系统状态

#### 测试数据管理
- **Fixture系统**: 可重用的测试数据和环境
- **临时文件**: 自动清理的测试文件系统
- **配置模板**: 标准化的测试配置

### 4. 关键修复

在测试完善过程中发现并修复了以下关键问题：

#### 混合监控架构Bug修复
```python
# 问题：采样收集器未启动
# 位置：src/mymonitor/monitoring/architectures.py
# 修复：为采样收集器添加start()调用

for i, collector in enumerate(self.sampling_collectors):
    collector.build_process_pid = build_process_pid
    collector.start()  # 关键修复
```

#### 测试Mock完善
- **文件系统Mock**: 处理`/proc/{pid}/stat`文件读取
- **进程Mock**: 使用namedtuple确保数据类型正确
- **上下文管理器**: 支持subprocess的with语句

### 5. 测试执行

#### 运行所有测试
```bash
uv run pytest                    # 运行全部104个测试
uv run pytest tests/unit/       # 运行单元测试
uv run pytest tests/e2e/        # 运行端到端测试
```

#### 测试覆盖率
```bash
uv run pytest --cov=src/mymonitor --cov-report=html
```

#### 性能测试
```bash
uv run pytest -m performance    # 运行性能测试
```

### 6. 持续集成

测试架构支持CI/CD集成：
- **快速反馈**: 单元测试优先执行
- **分层验证**: 按测试层级逐步验证
- **环境隔离**: 每个测试独立的临时环境
- **资源清理**: 自动清理测试产生的临时文件

---

## 扩展性设计

### 1. 插件化架构

#### 收集器扩展

```python
class CustomMemoryCollector(BaseMemoryCollector):
    """自定义内存收集器"""
    
    async def collect_memory_async(self) -> List[ProcessInfo]:
        """实现自定义收集逻辑"""
        pass
```

#### 分类器扩展

```python
def custom_categorization_rule(process: ProcessInfo) -> Tuple[str, str]:
    """自定义分类规则"""
    return ("Custom", "CustomCategory")
```

### 2. 配置扩展

#### 自定义配置节

```toml
[custom.extension]
enable_feature = true
custom_parameter = "value"
```

#### 动态配置加载

```python
def load_custom_config(config_path: Path) -> Dict[str, Any]:
    """加载自定义配置"""
    pass
```

### 3. 输出格式扩展

#### 新输出格式

```python
class JSONExporter:
    """JSON 格式导出器"""
    
    def export(self, results: MonitoringResults) -> None:
        """导出为 JSON 格式"""
        pass
```

#### 可视化扩展

```python
class CustomVisualizer:
    """自定义可视化器"""
    
    def generate_plot(self, data: pd.DataFrame) -> str:
        """生成自定义图表"""
        pass
```

### 4. 平台支持

#### 跨平台兼容

- **Windows**: PowerShell 命令支持
- **macOS**: BSD 工具链支持
- **Linux**: 完整功能支持

#### 容器化支持

- **Docker**: 容器内监控能力
- **Kubernetes**: 集群级监控
- **CI/CD**: 集成到构建流水线

---

## 总结

MyMonitor v2.1 采用了现代化的异步架构设计和完善的测试体系，实现了以下核心优势：

1. **高性能**: 异步 I/O 和智能并发控制
2. **高可靠**: 完善的错误处理和资源管理
3. **高质量**: 104个测试用例确保代码质量
4. **高扩展**: 模块化设计和插件化架构
5. **易维护**: 清晰的代码结构和完整的文档

### 版本亮点

- **测试完善**: 建立了完整的测试金字塔（单元→集成→性能→端到端）
- **Bug修复**: 发现并修复了混合监控架构中的关键问题
- **质量保证**: 100%测试通过率，确保系统稳定性
- **文档更新**: 完善的架构文档和测试说明

这种架构设计和测试体系确保了 MyMonitor 能够在各种复杂的构建环境中稳定运行，同时为未来的功能扩展和性能优化提供了坚实的基础。

---

> **更多信息**:  

> - [README.md](README.md) - 用户指南  
> - [README.zh-CN.md](README.zh-CN.md) - 中文用户指南  
> - [TODO.md](TODO.md) - 开发计划  
> - [REFACTORING_REPORT.md](REFACTORING_REPORT.md) - 重构报告
