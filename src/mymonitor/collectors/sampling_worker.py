"""
采样Worker实现。

该模块实现了混合监控架构中的采样Worker，负责从任务队列获取任务
并执行内存采样。
"""

import logging
import queue
import threading
import time
from typing import Optional
import psutil

from ..models.hybrid_monitoring import (
    ProcessTask, 
    SampleResult, 
    HybridCollectorConfig,
    HybridCollectorStats
)
from .base import ProcessMemorySample
from ..classification import get_process_category

logger = logging.getLogger(__name__)


class SamplingWorker:
    """
    采样Worker - 负责从队列获取任务并执行内存采样。
    
    该Worker负责：
    1. 从任务队列获取ProcessTask
    2. 执行内存采样
    3. 进程分类
    4. 将结果放入结果队列
    
    设计特点：
    - 支持超时机制，避免阻塞
    - 自动重试失败的任务
    - 详细的性能统计
    - 线程安全，支持优雅停止
    """
    
    def __init__(
        self,
        worker_id: int,
        task_queue: 'queue.Queue[ProcessTask]',
        result_queue: 'queue.Queue[SampleResult]',
        config: HybridCollectorConfig,
        stats: Optional[HybridCollectorStats] = None
    ):
        """
        初始化采样Worker。
        
        Args:
            worker_id: Worker唯一标识
            task_queue: 任务队列
            result_queue: 结果队列
            config: 混合收集器配置
            stats: 统计信息对象（可选）
        """
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.config = config
        self.stats = stats
        
        # 状态管理
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.running = False
        
        # 性能统计
        self.samples_completed = 0
        self.samples_failed = 0
        self.total_sampling_time = 0.0
        self.last_activity_time = 0.0
        
        logger.debug(f"SamplingWorker {worker_id} initialized")
    
    def start(self) -> None:
        """启动采样Worker。"""
        if self.running:
            logger.warning(f"SamplingWorker {self.worker_id} already running")
            return
        
        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self.sampling_loop,
            name=f"SamplingWorker-{self.worker_id}",
            daemon=True
        )
        self.thread.start()
        logger.info(f"SamplingWorker {self.worker_id} started")
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        停止采样Worker。
        
        Args:
            timeout: 等待停止的超时时间（秒）
        """
        if not self.running:
            return
        
        logger.info(f"Stopping SamplingWorker {self.worker_id}...")
        self.running = False
        self.stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                logger.warning(
                    f"SamplingWorker {self.worker_id} did not stop within timeout"
                )
            else:
                logger.info(f"SamplingWorker {self.worker_id} stopped successfully")
    
    def sampling_loop(self) -> None:
        """主要的采样循环。"""
        logger.info(f"SamplingWorker {self.worker_id} loop started")
        
        try:
            while not self.stop_event.is_set():
                try:
                    # 从队列获取任务
                    task = self.task_queue.get(timeout=self.config.queue_timeout)
                    
                    # 执行采样
                    result = self._sample_process(task)
                    
                    # 处理结果
                    if result:
                        self._enqueue_result(result)
                        self.samples_completed += 1
                    else:
                        self.samples_failed += 1
                        self._handle_failed_task(task)
                    
                    # 标记任务完成
                    self.task_queue.task_done()
                    
                    # 更新活动时间
                    self.last_activity_time = time.time()
                    
                except queue.Empty:
                    # 没有任务，继续循环
                    continue
                    
                except Exception as e:
                    logger.error(
                        f"Error in SamplingWorker {self.worker_id} loop: {e}",
                        exc_info=True
                    )
                    self.samples_failed += 1
                    
        except Exception as e:
            logger.error(
                f"Fatal error in SamplingWorker {self.worker_id}: {e}",
                exc_info=True
            )
        finally:
            logger.info(f"SamplingWorker {self.worker_id} loop finished")
    
    def _sample_process(self, task: ProcessTask) -> Optional[SampleResult]:
        """
        执行进程采样。
        
        Args:
            task: 采样任务
            
        Returns:
            采样结果或None（如果失败）
        """
        sampling_start = time.time()
        
        try:
            # 获取进程对象
            proc = psutil.Process(task.pid)
            
            # 执行内存采样
            memory_info = proc.memory_full_info()
            
            # 创建采样数据
            sample = ProcessMemorySample(
                pid=str(task.pid),
                command_name=task.command_name,
                full_command=task.full_command,
                metrics={
                    'PSS_KB': int(memory_info.pss / 1024) if hasattr(memory_info, 'pss') else 0,
                    'RSS_KB': int(memory_info.rss / 1024),
                    'USS_KB': int(memory_info.uss / 1024) if hasattr(memory_info, 'uss') else 0,
                    'VMS_KB': int(memory_info.vms / 1024) if hasattr(memory_info, 'vms') else 0,
                }
            )
            
            # 进程分类
            category = get_process_category(task.command_name, task.full_command)
            
            # 计算采样耗时
            sampling_duration = (time.time() - sampling_start) * 1000  # 毫秒
            self.total_sampling_time += sampling_duration
            
            # 创建结果
            result = SampleResult(
                pid=task.pid,
                sample=sample,
                timestamp=time.time(),
                worker_id=self.worker_id,
                category=category,
                sampling_duration=sampling_duration
            )
            
            logger.debug(
                f"Worker {self.worker_id} sampled PID {task.pid}: "
                f"{sample.metrics.get('PSS_KB', 0)} KB PSS in {sampling_duration:.1f}ms"
            )
            
            # 更新统计信息
            self._update_stats(True, sampling_duration)
            
            return result
            
        except psutil.NoSuchProcess:
            # 进程已死亡，这是正常情况
            logger.debug(f"Process {task.pid} died before sampling")
            sampling_duration = (time.time() - sampling_start) * 1000
            self._update_stats(False, sampling_duration)
            return None
            
        except psutil.AccessDenied:
            # 权限不足，记录但不重试
            logger.debug(f"Access denied for process {task.pid}")
            sampling_duration = (time.time() - sampling_start) * 1000
            self._update_stats(False, sampling_duration)
            return None
            
        except Exception as e:
            # 其他错误
            sampling_duration = (time.time() - sampling_start) * 1000
            logger.warning(
                f"Error sampling process {task.pid}: {e}",
                exc_info=False
            )
            self._update_stats(False, sampling_duration)
            return None
    
    def _enqueue_result(self, result: SampleResult) -> None:
        """
        将结果放入结果队列。
        
        Args:
            result: 采样结果
        """
        try:
            self.result_queue.put_nowait(result)
        except queue.Full:
            logger.warning(
                f"Result queue full, dropping result for PID {result.pid}"
            )
            # 可以考虑实现更智能的处理策略，比如丢弃最旧的结果
    
    def _handle_failed_task(self, task: ProcessTask) -> None:
        """
        处理失败的任务。
        
        Args:
            task: 失败的任务
        """
        if task.retry_count < self.config.max_retry_count:
            # 创建重试任务
            retry_task = task.create_retry_task()
            try:
                self.task_queue.put_nowait(retry_task)
                logger.debug(f"Retrying task for PID {task.pid} (attempt {retry_task.retry_count})")
            except queue.Full:
                logger.warning(f"Cannot retry task for PID {task.pid}: queue full")
        else:
            logger.debug(f"Max retries exceeded for PID {task.pid}")
    
    def _update_stats(self, success: bool, duration: float) -> None:
        """
        更新统计信息。
        
        Args:
            success: 是否成功
            duration: 耗时（毫秒）
        """
        if self.stats:
            self.stats.update_worker_stats(self.worker_id, success, duration)
            
            if success:
                self.stats.total_samples_completed += 1
            else:
                self.stats.total_samples_failed += 1
            
            # 更新平均采样耗时
            total_samples = self.stats.total_samples_completed + self.stats.total_samples_failed
            if total_samples > 0:
                # 指数移动平均
                alpha = 0.1
                if self.stats.average_sampling_duration == 0:
                    self.stats.average_sampling_duration = duration
                else:
                    self.stats.average_sampling_duration = (
                        alpha * duration + 
                        (1 - alpha) * self.stats.average_sampling_duration
                    )
    
    def get_performance_info(self) -> dict:
        """
        获取性能信息。
        
        Returns:
            包含性能统计的字典
        """
        total_samples = self.samples_completed + self.samples_failed
        avg_duration = 0.0
        if total_samples > 0:
            avg_duration = self.total_sampling_time / total_samples
        
        success_rate = 0.0
        if total_samples > 0:
            success_rate = (self.samples_completed / total_samples) * 100.0
        
        return {
            'worker_id': self.worker_id,
            'running': self.running,
            'samples_completed': self.samples_completed,
            'samples_failed': self.samples_failed,
            'success_rate': success_rate,
            'average_duration_ms': avg_duration,
            'last_activity': self.last_activity_time,
            'task_queue_size': (
                self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else -1
            ),
            'result_queue_size': (
                self.result_queue.qsize() if hasattr(self.result_queue, 'qsize') else -1
            )
        }
