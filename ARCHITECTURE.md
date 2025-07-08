# MyMonitor 架构文档

> **文档版本**: v1.0  
> **最后更新**: 2025-07-08  
> **适用版本**: MyMonitor v2.0+  

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

### 2. 异步优先 (Async-First)

- **AsyncIO 架构**: 核心监控使用协程实现
- **非阻塞 I/O**: 所有 I/O 操作异步化
- **并发安全**: 使用适当的同步原语

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
│  │ Async Monitoring│    │ Memory          │                │
│  │ Coordinator     │    │ Collectors      │                │
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
        """运行完整的构建监控流程"""
        
    def run(self) -> bool:
        """同步包装器，为 CLI 使用"""
```

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

**职责**: 构建进程管理和线程池管理

**核心组件**:

- **`BuildProcessManager`**: 异步构建进程执行器
- **`ThreadPoolManager`**: 全局线程池管理
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

**异步特性**:

- 非阻塞进程启动和等待
- 支持取消和超时
- 线程池中执行 CPU 密集型操作

### 5. 监控层 (`monitoring/`)

**职责**: 异步内存监控协调

**核心类**:

```python
class AsyncMonitoringCoordinator:
    """异步监控协调器"""
    
    async def setup_monitoring(self, monitoring_cores: List[int]) -> None:
        """设置监控基础设施"""
        
    async def start_monitoring(self, build_pid: int) -> None:
        """开始监控指定进程"""
        
    async def stop_monitoring(self) -> None:
        """停止监控并收集结果"""
```

**监控机制**:

- 基于 `asyncio.create_task()` 的后台监控
- 可配置的采样间隔
- 支持多种内存收集器

### 6. 收集器层 (`collectors/`)

**职责**: 内存数据收集的具体实现

**收集器类型**:

- **`PSS_PSUtil`**: 基于 psutil 的 PSS 内存收集
- **`RSS_PidStat`**: 基于 pidstat 的 RSS 内存收集
- **`AsyncMemoryCollector`**: 异步收集器包装

**设计模式**:

```python
class BaseMemoryCollector(ABC):
    """内存收集器基类"""
    
    @abstractmethod
    async def collect_memory_async(self) -> List[ProcessInfo]:
        """异步收集内存数据"""
```

**优化特性**:

- 可选的 `descendants_only` 模式提升性能
- 智能进程发现算法
- 批量数据处理

### 7. 分类层 (`classification/`)

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
构建启动 → 进程发现 → 内存收集 → 进程分类 → 数据聚合 → 结果存储
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
2. **文件输出**: 保存 Parquet 文件和日志
3. **可视化**: 生成交互式图表

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

[monitor.async_settings]
enable_thread_pool_optimization = true  # 线程池优化
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

[monitor.async_settings]
enable_thread_pool_optimization = true  # 启用优化
```

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

MyMonitor v2.0 采用了现代化的异步架构设计，实现了以下核心优势：

1. **高性能**: 异步 I/O 和智能并发控制
2. **高可靠**: 完善的错误处理和资源管理
3. **高扩展**: 模块化设计和插件化架构
4. **易维护**: 清晰的代码结构和完整的文档

这种架构设计确保了 MyMonitor 能够在各种复杂的构建环境中稳定运行，同时为未来的功能扩展和性能优化提供了坚实的基础。

---

> **更多信息**:  

> - [README.md](README.md) - 用户指南  
> - [README.zh-CN.md](README.zh-CN.md) - 中文用户指南  
> - [TODO.md](TODO.md) - 开发计划  
> - [REFACTORING_REPORT.md](REFACTORING_REPORT.md) - 重构报告
