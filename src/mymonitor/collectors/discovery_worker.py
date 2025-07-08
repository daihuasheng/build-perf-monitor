"""
进程发现Worker实现。

该模块实现了混合监控架构中的进程发现Worker，负责高频扫描系统进程
并生成采样任务。
"""

import logging
import queue
import re
import threading
import time
from typing import Set, List, Optional, Pattern
import psutil

from ..models.hybrid_monitoring import ProcessTask, HybridCollectorConfig, HybridCollectorStats

logger = logging.getLogger(__name__)


class ProcessDiscoveryWorker:
    """
    进程发现Worker - 负责快速扫描系统进程。
    
    该Worker负责：
    1. 高频扫描系统进程（默认10ms间隔）
    2. 检测新进程、持续进程、消亡进程
    3. 模式匹配和去重
    4. 生成采样任务到队列
    
    设计特点：
    - 使用集合去重，避免重复扫描相同进程
    - 自动清理死亡进程，减少内存占用
    - 支持优先级分配，重要进程优先采样
    - 线程安全，支持优雅停止
    """
    
    def __init__(
        self,
        process_pattern: str,
        task_queue: 'queue.Queue[ProcessTask]',
        config: HybridCollectorConfig,
        stats: Optional[HybridCollectorStats] = None
    ):
        """
        初始化进程发现Worker。
        
        Args:
            process_pattern: 进程匹配正则表达式
            task_queue: 任务队列
            config: 混合收集器配置
            stats: 统计信息对象（可选）
        """
        self.process_pattern: Pattern[str] = re.compile(process_pattern)
        self.task_queue = task_queue
        self.config = config
        self.stats = stats
        
        # 状态管理
        self.known_pids: Set[int] = set()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.running = False
        
        # 性能监控
        self.last_scan_time = 0.0
        self.scan_count = 0
        self.last_stats_update = time.time()
        
        logger.debug(f"ProcessDiscoveryWorker initialized with pattern: {process_pattern}")
    
    def start(self) -> None:
        """启动发现Worker。"""
        if self.running:
            logger.warning("ProcessDiscoveryWorker already running")
            return
        
        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self.discovery_loop,
            name="ProcessDiscovery",
            daemon=True
        )
        self.thread.start()
        logger.info("ProcessDiscoveryWorker started")
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        停止发现Worker。
        
        Args:
            timeout: 等待停止的超时时间（秒）
        """
        if not self.running:
            return
        
        logger.info("Stopping ProcessDiscoveryWorker...")
        self.running = False
        self.stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                logger.warning("ProcessDiscoveryWorker did not stop within timeout")
            else:
                logger.info("ProcessDiscoveryWorker stopped successfully")
    
    def discovery_loop(self) -> None:
        """主要的进程发现循环。"""
        logger.info("ProcessDiscoveryWorker loop started")
        
        try:
            while not self.stop_event.is_set():
                loop_start = time.time()
                
                # 执行一次扫描
                new_tasks = self._scan_and_enqueue()
                
                # 更新统计信息
                self._update_stats(new_tasks)
                
                # 计算睡眠时间
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.config.discovery_interval - elapsed)
                
                if sleep_time > 0:
                    # 可中断的睡眠
                    if self.stop_event.wait(timeout=sleep_time):
                        break
                else:
                    # 扫描时间超过间隔，记录警告
                    logger.warning(
                        f"Process discovery scan took {elapsed:.3f}s, "
                        f"longer than interval {self.config.discovery_interval:.3f}s"
                    )
                
        except Exception as e:
            logger.error(f"Error in process discovery loop: {e}", exc_info=True)
        finally:
            logger.info("ProcessDiscoveryWorker loop finished")
    
    def _scan_and_enqueue(self) -> List[ProcessTask]:
        """
        扫描进程并入队新任务。
        
        Returns:
            新创建的任务列表
        """
        scan_start = time.time()
        current_pids = set()
        new_tasks = []
        processes_scanned = 0
        
        try:
            # 快速扫描所有进程
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if self.stop_event.is_set():
                    break
                
                processes_scanned += 1
                pid = proc.info['pid']
                current_pids.add(pid)
                
                # 只处理新发现的进程
                if pid not in self.known_pids:
                    task = self._create_task_if_matches(proc.info)
                    if task:
                        new_tasks.append(task)
            
            # 更新已知进程集合（自动清理死亡进程）
            old_count = len(self.known_pids)
            self.known_pids = current_pids
            new_count = len(self.known_pids)
            
            # 记录进程变化
            if old_count != new_count:
                logger.debug(f"Process count changed: {old_count} -> {new_count}")
            
            # 将新任务加入队列
            queued_count = 0
            for task in new_tasks:
                try:
                    if self.config.enable_prioritization:
                        # 如果队列支持优先级，使用put_nowait
                        self.task_queue.put_nowait(task)
                    else:
                        self.task_queue.put_nowait(task)
                    queued_count += 1
                except queue.Full:
                    logger.warning(f"Task queue full, dropping task for PID {task.pid}")
                    break
            
            scan_duration = time.time() - scan_start
            
            if new_tasks:
                logger.debug(
                    f"Discovery scan: {processes_scanned} processes scanned, "
                    f"{len(new_tasks)} new tasks generated, {queued_count} queued, "
                    f"took {scan_duration:.3f}s"
                )
            
            # 更新性能统计
            self.last_scan_time = scan_duration
            self.scan_count += 1
            
        except Exception as e:
            logger.error(f"Error during process scan: {e}", exc_info=True)
        
        return new_tasks
    
    def _create_task_if_matches(self, proc_info: dict) -> Optional[ProcessTask]:
        """
        根据进程信息创建任务（如果匹配模式）。
        
        Args:
            proc_info: psutil进程信息字典
            
        Returns:
            ProcessTask对象或None（如果不匹配）
        """
        try:
            pid = proc_info['pid']
            name = proc_info['name'] or ''
            cmdline = proc_info['cmdline'] or []
            full_command = ' '.join(cmdline) if cmdline else name
            
            # 检查是否匹配模式
            if not (self.process_pattern.search(name) or 
                    self.process_pattern.search(full_command)):
                return None
            
            # 计算优先级（可选的智能优先级分配）
            priority = self._calculate_priority(name, full_command)
            
            task = ProcessTask(
                pid=pid,
                command_name=name,
                full_command=full_command,
                priority=priority,
                pattern_match_info={
                    'matched_field': 'name' if self.process_pattern.search(name) else 'cmdline',
                    'discovery_scan': self.scan_count
                }
            )
            
            logger.debug(f"Created task for PID {pid}: {name}")
            return task
            
        except Exception as e:
            logger.warning(f"Error creating task for process {proc_info}: {e}")
            return None
    
    def _calculate_priority(self, name: str, full_command: str) -> int:
        """
        计算任务优先级。
        
        Args:
            name: 进程名称
            full_command: 完整命令行
            
        Returns:
            优先级值（0=最高优先级）
        """
        if not self.config.enable_prioritization:
            return 0
        
        # 简单的优先级规则
        # 编译器相关进程优先级较高
        if any(compiler in name.lower() for compiler in ['gcc', 'clang', 'cc1', 'ld', 'ar']):
            return 0  # 最高优先级
        
        # 构建工具优先级中等
        if any(tool in name.lower() for tool in ['make', 'ninja', 'cmake']):
            return 1
        
        # 其他进程优先级较低
        return 2
    
    def _update_stats(self, new_tasks: List[ProcessTask]) -> None:
        """
        更新统计信息。
        
        Args:
            new_tasks: 新创建的任务列表
        """
        if not self.stats:
            return
        
        now = time.time()
        
        # 基本统计
        self.stats.total_processes_discovered += len(self.known_pids)
        self.stats.total_tasks_generated += len(new_tasks)
        
        # 计算平均发现间隔
        if self.last_stats_update > 0:
            interval = now - self.last_stats_update
            if self.stats.average_discovery_interval == 0:
                self.stats.average_discovery_interval = interval
            else:
                # 指数移动平均
                alpha = 0.1
                self.stats.average_discovery_interval = (
                    alpha * interval + 
                    (1 - alpha) * self.stats.average_discovery_interval
                )
        
        self.last_stats_update = now
        
        # 更新队列大小峰值
        try:
            current_task_queue_size = self.task_queue.qsize()
            if current_task_queue_size > self.stats.peak_task_queue_size:
                self.stats.peak_task_queue_size = current_task_queue_size
        except Exception:
            # qsize()在某些平台上可能不可用
            pass
    
    def get_performance_info(self) -> dict:
        """
        获取性能信息。
        
        Returns:
            包含性能统计的字典
        """
        return {
            'running': self.running,
            'known_processes': len(self.known_pids),
            'scan_count': self.scan_count,
            'last_scan_duration': self.last_scan_time,
            'average_scan_interval': (
                self.stats.average_discovery_interval if self.stats else 0.0
            ),
            'queue_size': (
                self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else -1
            )
        }
