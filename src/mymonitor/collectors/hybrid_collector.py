"""
混合内存收集器实现。

该模块实现了基于生产者-消费者模式的混合内存收集器，
协调进程发现Worker和采样Worker的工作。
"""

import logging
import queue
import threading
import time
from typing import List, Iterable, Optional, Dict, Any
from .base import AbstractMemoryCollector, ProcessMemorySample
from .discovery_worker import ProcessDiscoveryWorker
from .sampling_worker import SamplingWorker
from ..models.results import MonitoringResults  # 只导入这个避免循环导入
from ..models.hybrid_monitoring import (
    ProcessTask,
    SampleResult,
    HybridCollectorConfig,
    HybridCollectorStats
)

logger = logging.getLogger(__name__)


class HybridMemoryCollector(AbstractMemoryCollector):
    """
    混合内存收集器 - 协调发现和采样Worker。
    
    该收集器实现了高效的生产者-消费者模式：
    - 1个发现Worker负责快速进程发现
    - N个采样Worker负责并行内存采样
    - 任务队列和结果队列协调工作
    
    特性：
    1. 高效的并行处理
    2. 智能去重和重试机制
    3. 详细的性能统计
    4. 灵活的配置选项
    5. 优雅的启停控制
    """
    
    def __init__(
        self,
        process_pattern: str,
        monitoring_interval: float,
        config: Optional[HybridCollectorConfig] = None,
        **kwargs
    ):
        """
        初始化混合内存收集器。
        
        Args:
            process_pattern: 进程匹配正则表达式
            monitoring_interval: 监控间隔（秒）
            config: 混合收集器配置
            **kwargs: 其他参数（兼容基类）
        """
        super().__init__(process_pattern, monitoring_interval, **kwargs)
        
        # 配置
        self.config = config or HybridCollectorConfig()
        
        # 队列
        if self.config.enable_prioritization:
            # 使用优先级队列
            self.task_queue: 'queue.Queue[ProcessTask]' = queue.PriorityQueue(
                maxsize=self.config.task_queue_size
            )
        else:
            # 使用普通队列
            self.task_queue = queue.Queue(maxsize=self.config.task_queue_size)
        
        self.result_queue: 'queue.Queue[SampleResult]' = queue.Queue(
            maxsize=self.config.result_queue_size
        )
        
        # Worker管理
        self.discovery_worker: Optional[ProcessDiscoveryWorker] = None
        self.sampling_workers: List[SamplingWorker] = []
        
        # 状态管理
        self.collecting = False
        self.stop_event = threading.Event()
        self.stats = HybridCollectorStats()
        
        # 结果处理
        self.current_batch: List[ProcessMemorySample] = []
        self.last_batch_time = 0.0
        
        logger.info(f"HybridMemoryCollector initialized with {self.config.sampling_workers} workers")
    
    def start(self) -> None:
        """启动混合收集器。"""
        if self.collecting:
            logger.warning("HybridMemoryCollector already started")
            return
        
        logger.info("Starting HybridMemoryCollector...")
        
        self.collecting = True
        self.stop_event.clear()
        self.stats = HybridCollectorStats()  # 重置统计
        
        try:
            # 启动发现Worker
            self.discovery_worker = ProcessDiscoveryWorker(
                process_pattern=self.process_pattern,
                task_queue=self.task_queue,
                config=self.config,
                stats=self.stats
            )
            self.discovery_worker.start()
            
            # 启动采样Worker
            for i in range(self.config.sampling_workers):
                worker = SamplingWorker(
                    worker_id=i,
                    task_queue=self.task_queue,
                    result_queue=self.result_queue,
                    config=self.config,
                    stats=self.stats
                )
                worker.start()
                self.sampling_workers.append(worker)
            
            logger.info(
                f"HybridMemoryCollector started with 1 discovery worker "
                f"and {len(self.sampling_workers)} sampling workers"
            )
            
        except Exception as e:
            logger.error(f"Failed to start HybridMemoryCollector: {e}", exc_info=True)
            self.stop()
            raise
    
    def stop(self) -> None:
        """停止混合收集器。"""
        if not self.collecting:
            return
        
        logger.info("Stopping HybridMemoryCollector...")
        
        self.collecting = False
        self.stop_event.set()
        
        # 停止发现Worker
        if self.discovery_worker:
            self.discovery_worker.stop(timeout=5.0)
            self.discovery_worker = None
        
        # 停止采样Worker
        for worker in self.sampling_workers:
            worker.stop(timeout=5.0)
        self.sampling_workers.clear()
        
        # 清空队列
        self._drain_queues()
        
        logger.info("HybridMemoryCollector stopped")
    
    def read_samples(self) -> Iterable[List[ProcessMemorySample]]:
        """
        读取采样结果。
        
        该方法从结果队列中读取采样数据，按时间间隔批量返回。
        
        Yields:
            ProcessMemorySample列表
        """
        logger.info("HybridMemoryCollector sample reading started")
        
        self.last_batch_time = time.time()
        
        try:
            while self.collecting:
                current_batch = []
                batch_start = time.time()
                
                # 从结果队列收集样本，直到达到批处理大小或超时
                while (len(current_batch) < self.config.batch_result_size and
                       time.time() - batch_start < self.monitoring_interval):
                    
                    try:
                        result = self.result_queue.get(
                            timeout=min(0.1, self.monitoring_interval / 10)
                        )
                        
                        if result and result.sample:
                            current_batch.append(result.sample)
                            # 更新队列峰值统计
                            self._update_queue_stats()
                            
                    except queue.Empty:
                        # 没有新结果，继续等待
                        continue
                
                # 检查是否应该yield批次
                now = time.time()
                if (current_batch or 
                    now - self.last_batch_time >= self.monitoring_interval):
                    
                    if current_batch:
                        logger.debug(
                            f"Yielding batch of {len(current_batch)} samples "
                            f"after {now - batch_start:.3f}s"
                        )
                    
                    yield current_batch
                    self.last_batch_time = now
                    
                    # 更新统计
                    if current_batch:
                        self._log_performance_info()
                
                # 检查停止条件
                if self.stop_event.is_set():
                    break
            
            # 收集剩余的结果
            final_batch = []
            try:
                while True:
                    result = self.result_queue.get_nowait()
                    if result and result.sample:
                        final_batch.append(result.sample)
            except queue.Empty:
                pass
            
            if final_batch:
                logger.info(f"Yielding final batch of {len(final_batch)} samples")
                yield final_batch
        
        except Exception as e:
            logger.error(f"Error in read_samples: {e}", exc_info=True)
        
        finally:
            logger.info("HybridMemoryCollector sample reading finished")
    
    def get_metric_fields(self) -> List[str]:
        """获取指标字段列表。"""
        return ["PSS_KB", "RSS_KB", "USS_KB", "VMS_KB"]
    
    def get_primary_metric_field(self) -> str:
        """获取主要指标字段。"""
        return "PSS_KB"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取详细的统计信息。
        
        Returns:
            包含所有统计信息的字典
        """
        # 基本统计
        stats = {
            'collector_type': 'hybrid',
            'config': {
                'discovery_interval': self.config.discovery_interval,
                'sampling_workers': self.config.sampling_workers,
                'task_queue_size': self.config.task_queue_size,
                'result_queue_size': self.config.result_queue_size,
                'enable_prioritization': self.config.enable_prioritization,
                'max_retry_count': self.config.max_retry_count,
            },
            'runtime': {
                'uptime_seconds': self.stats.uptime_seconds,
                'collecting': self.collecting,
            },
            'performance': {
                'discovery_rate': self.stats.discovery_rate,
                'sampling_rate': self.stats.sampling_rate,
                'success_rate': self.stats.success_rate,
                'average_discovery_interval': self.stats.average_discovery_interval,
                'average_sampling_duration': self.stats.average_sampling_duration,
            },
            'counters': {
                'total_processes_discovered': self.stats.total_processes_discovered,
                'total_tasks_generated': self.stats.total_tasks_generated,
                'total_samples_completed': self.stats.total_samples_completed,
                'total_samples_failed': self.stats.total_samples_failed,
            },
            'queues': {
                'peak_task_queue_size': self.stats.peak_task_queue_size,
                'peak_result_queue_size': self.stats.peak_result_queue_size,
                'current_task_queue_size': self._get_queue_size(self.task_queue),
                'current_result_queue_size': self._get_queue_size(self.result_queue),
            }
        }
        
        # 发现Worker统计
        if self.discovery_worker:
            stats['discovery_worker'] = self.discovery_worker.get_performance_info()
        
        # 采样Worker统计
        stats['sampling_workers'] = []
        for worker in self.sampling_workers:
            stats['sampling_workers'].append(worker.get_performance_info())
        
        return stats
    
    def _drain_queues(self) -> None:
        """清空所有队列。"""
        try:
            while True:
                self.task_queue.get_nowait()
        except queue.Empty:
            pass
        
        try:
            while True:
                self.result_queue.get_nowait()
        except queue.Empty:
            pass
    
    def _update_queue_stats(self) -> None:
        """更新队列统计信息。"""
        try:
            result_queue_size = self.result_queue.qsize()
            if result_queue_size > self.stats.peak_result_queue_size:
                self.stats.peak_result_queue_size = result_queue_size
        except Exception:
            # qsize()在某些平台上可能不可用
            pass
    
    def _get_queue_size(self, q: queue.Queue) -> int:
        """安全地获取队列大小。"""
        try:
            return q.qsize()
        except Exception:
            return -1  # 表示不可用
    
    def _log_performance_info(self) -> None:
        """定期记录性能信息。"""
        # 每30秒记录一次详细信息
        now = time.time()
        if hasattr(self, '_last_perf_log'):
            if now - self._last_perf_log < 30:
                return
        self._last_perf_log = now
        
        stats = self.get_stats()
        perf = stats['performance']
        counters = stats['counters']
        
        logger.info(
            f"HybridCollector Performance: "
            f"discovery_rate={perf['discovery_rate']:.1f}/s, "
            f"sampling_rate={perf['sampling_rate']:.1f}/s, "
            f"success_rate={perf['success_rate']:.1f}%, "
            f"completed={counters['total_samples_completed']}, "
            f"failed={counters['total_samples_failed']}"
        )
