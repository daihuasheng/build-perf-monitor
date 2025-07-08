"""
混合监控架构的数据模型。

该模块定义了基于生产者-消费者模式的混合监控架构中使用的核心数据结构，
包括任务模型、结果模型和配置模型。
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

# 注意：ProcessMemorySample在collectors.base中定义
# 为了避免循环导入，我们在这里使用类型提示的字符串形式


@dataclass
class ProcessTask:
    """
    发现Worker生成的采样任务。
    
    每个ProcessTask代表一个需要进行内存采样的进程。发现Worker将这些任务
    放入任务队列，供采样Worker消费。
    
    Attributes:
        pid: 进程ID
        command_name: 进程名称（如 "gcc", "clang"）
        full_command: 完整命令行
        discovery_timestamp: 发现该进程的时间戳
        priority: 任务优先级（0=最高优先级）
        retry_count: 重试次数
        pattern_match_info: 模式匹配的额外信息
    """
    pid: int
    command_name: str
    full_command: str
    discovery_timestamp: float = field(default_factory=time.time)
    priority: int = 0
    retry_count: int = 0
    pattern_match_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后处理，确保数据一致性。"""
        if self.discovery_timestamp <= 0:
            self.discovery_timestamp = time.time()
    
    def __lt__(self, other: 'ProcessTask') -> bool:
        """支持优先级队列排序，优先级数值越小越优先。"""
        if not isinstance(other, ProcessTask):
            return NotImplemented
        return self.priority < other.priority
    
    def create_retry_task(self) -> 'ProcessTask':
        """创建重试任务。"""
        return ProcessTask(
            pid=self.pid,
            command_name=self.command_name,
            full_command=self.full_command,
            discovery_timestamp=time.time(),  # 更新发现时间
            priority=self.priority + 1,  # 降低优先级
            retry_count=self.retry_count + 1,
            pattern_match_info=self.pattern_match_info.copy()
        )


@dataclass 
class SampleResult:
    """
    采样Worker生成的结果。
    
    每个SampleResult包含一个进程的内存采样数据以及相关的元数据。
    
    Attributes:
        pid: 进程ID
        sample: 内存采样数据
        timestamp: 采样时间戳
        worker_id: 执行采样的Worker ID
        category: 进程分类结果 (major_category, minor_category)
        sampling_duration: 采样耗时（毫秒）
        error_info: 错误信息（如果采样失败）
    """
    pid: int
    sample: Any  # ProcessMemorySample对象，使用Any避免循环导入
    timestamp: float = field(default_factory=time.time)
    worker_id: int = 0
    category: Optional[Tuple[str, str]] = None
    sampling_duration: float = 0.0  # 毫秒
    error_info: Optional[str] = None
    
    def __post_init__(self):
        """后处理，确保数据一致性。"""
        if self.timestamp <= 0:
            self.timestamp = time.time()
    
    @property
    def is_successful(self) -> bool:
        """判断采样是否成功。"""
        return self.error_info is None and self.sample is not None
    
    @property
    def memory_kb(self) -> int:
        """获取主要内存指标（PSS优先，否则RSS）。"""
        if not self.sample:
            return 0
        return self.sample.metrics.get('PSS_KB', self.sample.metrics.get('RSS_KB', 0))


@dataclass
class HybridCollectorConfig:
    """
    混合收集器配置。
    
    该配置类定义了混合监控架构的所有可调参数。
    
    Attributes:
        discovery_interval: 发现Worker扫描间隔（秒）
        sampling_workers: 采样Worker数量
        task_queue_size: 任务队列最大大小
        result_queue_size: 结果队列最大大小
        enable_prioritization: 是否启用任务优先级
        max_retry_count: 最大重试次数
        queue_timeout: 队列操作超时时间（秒）
        worker_timeout: Worker操作超时时间（秒）
        enable_queue_monitoring: 是否启用队列监控
        batch_result_size: 结果批处理大小
    """
    discovery_interval: float = 0.01  # 10ms发现间隔
    sampling_workers: int = 4
    task_queue_size: int = 1000
    result_queue_size: int = 2000
    enable_prioritization: bool = True
    max_retry_count: int = 3
    queue_timeout: float = 0.1  # 100ms
    worker_timeout: float = 5.0  # 5秒
    enable_queue_monitoring: bool = True
    batch_result_size: int = 50
    
    def __post_init__(self):
        """后处理，验证配置参数。"""
        if self.discovery_interval <= 0:
            raise ValueError("discovery_interval must be positive")
        if self.sampling_workers < 1:
            raise ValueError("sampling_workers must be at least 1")
        if self.task_queue_size < 10:
            raise ValueError("task_queue_size must be at least 10")
        if self.result_queue_size < 10:
            raise ValueError("result_queue_size must be at least 10")
        if self.max_retry_count < 0:
            raise ValueError("max_retry_count must be non-negative")
        if self.queue_timeout <= 0:
            raise ValueError("queue_timeout must be positive")
        if self.worker_timeout <= 0:
            raise ValueError("worker_timeout must be positive")
        if self.batch_result_size < 1:
            raise ValueError("batch_result_size must be at least 1")
    
    @classmethod
    def create_performance_optimized(cls) -> 'HybridCollectorConfig':
        """创建性能优化的配置。"""
        return cls(
            discovery_interval=0.005,  # 5ms高频发现
            sampling_workers=6,        # 更多采样Worker
            task_queue_size=2000,      # 更大队列
            result_queue_size=4000,
            enable_prioritization=True,
            max_retry_count=2,         # 减少重试
            queue_timeout=0.05,        # 更短超时
            batch_result_size=100      # 更大批处理
        )
    
    @classmethod
    def create_resource_conservative(cls) -> 'HybridCollectorConfig':
        """创建资源保守的配置。"""
        return cls(
            discovery_interval=0.02,   # 20ms较低频率
            sampling_workers=2,        # 较少Worker
            task_queue_size=500,       # 较小队列
            result_queue_size=1000,
            enable_prioritization=False,
            max_retry_count=5,         # 更多重试
            queue_timeout=0.2,         # 更长超时
            batch_result_size=20       # 较小批处理
        )


@dataclass
class HybridCollectorStats:
    """
    混合收集器运行时统计。
    
    该类用于跟踪混合监控架构的性能指标和运行状态。
    
    Attributes:
        start_time: 开始时间
        total_processes_discovered: 发现的进程总数
        total_tasks_generated: 生成的任务总数
        total_samples_completed: 完成的采样总数
        total_samples_failed: 失败的采样总数
        average_discovery_interval: 平均发现间隔
        average_sampling_duration: 平均采样耗时
        peak_task_queue_size: 任务队列峰值大小
        peak_result_queue_size: 结果队列峰值大小
        worker_performance: 各Worker性能统计
    """
    start_time: float = field(default_factory=time.time)
    total_processes_discovered: int = 0
    total_tasks_generated: int = 0
    total_samples_completed: int = 0
    total_samples_failed: int = 0
    average_discovery_interval: float = 0.0
    average_sampling_duration: float = 0.0
    peak_task_queue_size: int = 0
    peak_result_queue_size: int = 0
    worker_performance: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    @property
    def uptime_seconds(self) -> float:
        """运行时间（秒）。"""
        return time.time() - self.start_time
    
    @property
    def success_rate(self) -> float:
        """采样成功率（百分比）。"""
        total = self.total_samples_completed + self.total_samples_failed
        if total == 0:
            return 0.0
        return (self.total_samples_completed / total) * 100.0
    
    @property
    def discovery_rate(self) -> float:
        """进程发现率（进程/秒）。"""
        if self.uptime_seconds <= 0:
            return 0.0
        return self.total_processes_discovered / self.uptime_seconds
    
    @property
    def sampling_rate(self) -> float:
        """采样完成率（采样/秒）。"""
        if self.uptime_seconds <= 0:
            return 0.0
        return self.total_samples_completed / self.uptime_seconds
    
    def get_worker_stats(self, worker_id: int) -> Dict[str, Any]:
        """获取指定Worker的统计信息。"""
        return self.worker_performance.get(worker_id, {
            'samples_completed': 0,
            'samples_failed': 0,
            'average_duration': 0.0,
            'last_activity': 0.0
        })
    
    def update_worker_stats(self, worker_id: int, completed: bool, duration: float):
        """更新Worker统计信息。"""
        if worker_id not in self.worker_performance:
            self.worker_performance[worker_id] = {
                'samples_completed': 0,
                'samples_failed': 0,
                'total_duration': 0.0,
                'last_activity': time.time()
            }
        
        stats = self.worker_performance[worker_id]
        if completed:
            stats['samples_completed'] += 1
        else:
            stats['samples_failed'] += 1
        
        stats['total_duration'] += duration
        stats['last_activity'] = time.time()
        
        # 计算平均耗时
        total_samples = stats['samples_completed'] + stats['samples_failed']
        if total_samples > 0:
            stats['average_duration'] = stats['total_duration'] / total_samples


# 便于导入的类型别名
ProcessTaskQueue = 'queue.Queue[ProcessTask]'
SampleResultQueue = 'queue.Queue[SampleResult]'
