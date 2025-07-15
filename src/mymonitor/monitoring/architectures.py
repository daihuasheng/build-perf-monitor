"""
混合监控架构实现。

该模块定义了基于生产者-消费者模式的混合监控架构，
一个发现Worker负责进程发现，多个采样Worker负责内存采样，
通过队列协调工作，实现高效的内存监控。
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from ..models.results import MonitoringResults
from ..collectors.base import AbstractMemoryCollector
from ..collectors.factory import CollectorFactory

# 进程分类逻辑已移至结果聚合阶段
from ..executor.thread_pool import ThreadPoolConfig

logger = logging.getLogger(__name__)


class HybridArchitecture:
    """
    混合监控架构实现。

    使用生产者-消费者模式，通过异步队列协调进程发现和内存采样：
    - 发现Worker：周期性发现进程并放入任务队列
    - 采样Workers：从任务队列获取任务并执行内存采样
    - 结果处理Worker：收集采样结果并进行聚合

    特点：
    - 共享线程池避免频繁创建销毁
    - 集成进程分类功能
    - 优雅的资源管理和错误处理
    - 高效的批量结果处理
    """

    def __init__(
        self, collector_factory: CollectorFactory, config, thread_pool_manager=None
    ):
        """
        初始化混合监控架构。

        Args:
            collector_factory: 收集器工厂实例
            config: 混合架构配置 (HybridCollectorConfig 或兼容字典)
            thread_pool_manager: 线程池管理器实例（可选，用于依赖注入）
        """
        self.collector_factory = collector_factory
        self.config = config
        self.thread_pool_manager = thread_pool_manager

        # 监控组件
        self.discovery_collector: Optional[AbstractMemoryCollector] = None
        self.sampling_collectors: List[AbstractMemoryCollector] = []

        # 异步组件
        self.task_queue: Optional[asyncio.Queue] = None
        self.result_queue: Optional[asyncio.Queue] = None
        self.stop_event: Optional[asyncio.Event] = None

        # 异步任务
        self.discovery_task: Optional[asyncio.Task] = None
        self.sampling_tasks: List[asyncio.Task] = []
        self.result_processor_task: Optional[asyncio.Task] = None

        # 结果存储
        self.collected_samples: List[Dict[str, Any]] = []
        self.total_samples_collected: int = 0
        self.results: Optional[MonitoringResults] = None

        logger.info("HybridArchitecture initialized")

    def _get_config_value(self, key: str, default=None):
        """
        从配置中获取值，支持属性访问和字典访问。

        Args:
            key: 配置键名
            default: 默认值

        Returns:
            配置值
        """
        if hasattr(self.config, key):
            # 如果是对象（如HybridCollectorConfig），使用属性访问
            return getattr(self.config, key)
        elif hasattr(self.config, "get"):
            # 如果是字典，使用字典访问
            return self.config.get(key, default)
        else:
            # 回退到默认值
            return default

    async def setup_monitoring(self, monitoring_cores: List[int]) -> None:
        """设置混合监控基础设施。线程池已在外部初始化。"""
        logger.info(f"Setting up hybrid monitoring on cores: {monitoring_cores}")

        try:
            # 创建异步队列
            task_queue_size = self._get_config_value("task_queue_size", 1000)
            result_queue_size = self._get_config_value("result_queue_size", 2000)

            self.task_queue = asyncio.Queue(maxsize=task_queue_size)
            self.result_queue = asyncio.Queue(maxsize=result_queue_size)
            self.stop_event = asyncio.Event()

            # 为发现Worker创建收集器（使用第一个核心）
            discovery_core = monitoring_cores[0] if monitoring_cores else 0
            self.discovery_collector = self.collector_factory.create_collector(
                discovery_core
            )

            # 为采样Worker创建收集器
            num_sampling_workers = min(
                self._get_config_value("sampling_workers", 4), len(monitoring_cores)
            )

            self.sampling_collectors = []
            for i in range(num_sampling_workers):
                core_id = monitoring_cores[i % len(monitoring_cores)]
                collector = self.collector_factory.create_collector(core_id)
                self.sampling_collectors.append(collector)

            logger.info(
                f"Hybrid monitoring setup complete: 1 discovery worker, {len(self.sampling_collectors)} sampling workers"
            )

        except Exception as e:
            logger.error(f"Failed to setup hybrid monitoring: {e}")
            # 清理已创建的资源
            await self._cleanup_resources()
            raise

    async def start_monitoring(self, build_process_pid: int) -> None:
        """启动混合监控。"""
        logger.info(f"Starting hybrid monitoring for PID {build_process_pid}")

        try:
            # 重置状态
            self.stop_event.clear()
            self.collected_samples = []
            self.total_samples_collected = 0

            # 启动发现Worker
            self.discovery_task = asyncio.create_task(
                self._discovery_worker(build_process_pid), name="hybrid-discovery"
            )

            # 启动采样Workers
            self.sampling_tasks = []
            for i, collector in enumerate(self.sampling_collectors):
                # 启动采样收集器
                collector.build_process_pid = build_process_pid
                collector.start()

                task = asyncio.create_task(
                    self._sampling_worker(collector, i), name=f"hybrid-sampling-{i}"
                )
                self.sampling_tasks.append(task)

            # 启动结果处理Worker
            self.result_processor_task = asyncio.create_task(
                self._result_processor(), name="hybrid-result-processor"
            )

            logger.info("Hybrid monitoring started")

        except Exception as e:
            logger.error(f"Failed to start hybrid monitoring: {e}")
            await self._cleanup_resources()
            raise

    async def stop_monitoring(self) -> Optional[MonitoringResults]:
        """停止混合监控并收集结果。"""
        logger.info("Stopping hybrid monitoring")

        try:
            # 发出停止信号
            if self.stop_event:
                self.stop_event.set()

            # 等待所有任务完成
            tasks_to_wait = []
            if self.discovery_task:
                tasks_to_wait.append(self.discovery_task)
            if self.sampling_tasks:
                tasks_to_wait.extend(self.sampling_tasks)
            if self.result_processor_task:
                tasks_to_wait.append(self.result_processor_task)

            if tasks_to_wait:
                await asyncio.gather(*tasks_to_wait, return_exceptions=True)

            # 停止所有收集器
            await self._stop_collectors()

            # 汇总结果
            if self.collected_samples:
                self.results = self._aggregate_results(self.collected_samples)
                logger.info(
                    f"Hybrid monitoring stopped, collected {len(self.collected_samples)} samples"
                )
            else:
                logger.warning("No samples collected during hybrid monitoring")
                self.results = None

            return self.results

        except Exception as e:
            logger.error(f"Error stopping hybrid monitoring: {e}")
            return None
        finally:
            # 清理资源
            await self._cleanup_resources()

    async def _cleanup_resources(self) -> None:
        """清理所有资源。"""
        try:
            logger.debug("Resource cleanup completed for hybrid architecture")

        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")

    async def _stop_collectors(self) -> None:
        """清理收集器资源。"""
        # 由于我们使用 collect_single_sample() 而不是收集器的内置线程，
        # 这里只需要清理引用即可
        logger.debug("Cleaning up collector references")

        # 清理收集器引用
        self.discovery_collector = None
        self.sampling_collectors = []

    def get_results(self) -> Optional[MonitoringResults]:
        """获取混合监控结果。"""
        return self.results

    async def _discovery_worker(self, build_pid: int) -> None:
        """
        发现Worker：周期性发现进程并将PID放入任务队列。
        """
        discovery_interval = self._get_config_value("discovery_interval", 0.01)
        logger.debug(f"Discovery worker started with interval {discovery_interval}s")

        try:
            # 设置发现收集器
            self.discovery_collector.build_process_pid = build_pid
            self.discovery_collector.start()

            while not self.stop_event.is_set():
                try:
                    # 发现进程（使用单样本收集方法）
                    discovered_samples = await self._discover_processes_async()

                    # 将发现的进程放入任务队列
                    for sample in discovered_samples:
                        try:
                            task_item = {
                                "pid": sample.pid,
                                "cmd_name": sample.command_name,
                                "full_cmd": sample.full_command,
                                "discovery_time": asyncio.get_event_loop().time(),
                                "priority": self._get_task_priority(
                                    sample.command_name
                                ),
                            }

                            # 非阻塞放入队列，避免队列满时阻塞发现
                            queue_timeout = self._get_config_value("queue_timeout", 0.1)
                            await asyncio.wait_for(
                                self.task_queue.put(task_item), timeout=queue_timeout
                            )

                        except asyncio.TimeoutError:
                            # 队列满，跳过这个进程
                            logger.debug(
                                f"Task queue full, skipping process {sample.pid}"
                            )
                            continue
                        except Exception as e:
                            logger.warning(f"Error adding task to queue: {e}")

                    # 等待下次发现间隔
                    try:
                        await asyncio.wait_for(
                            self.stop_event.wait(), timeout=discovery_interval
                        )
                        # 如果等到了stop事件，退出循环
                        break
                    except asyncio.TimeoutError:
                        # 超时是正常的，继续下次发现
                        continue

                except Exception as e:
                    logger.warning(f"Error in discovery worker: {e}")
                    await asyncio.sleep(discovery_interval)

        except Exception as e:
            logger.error(f"Discovery worker failed: {e}")
        finally:
            logger.debug("Discovery worker stopped")

    async def _sampling_worker(
        self, collector: AbstractMemoryCollector, worker_id: int
    ) -> None:
        """
        采样Worker：从任务队列获取PID，执行内存采样。
        """
        logger.debug(f"Sampling worker {worker_id} started")
        queue_timeout = self._get_config_value("queue_timeout", 0.1)

        try:
            while not self.stop_event.is_set():
                try:
                    # 从任务队列获取采样任务
                    task_item = await asyncio.wait_for(
                        self.task_queue.get(), timeout=queue_timeout
                    )

                    # 执行采样
                    sample_result = await self._sample_process_async(
                        collector, task_item, worker_id
                    )

                    if sample_result:
                        # 将结果放入结果队列
                        try:
                            await asyncio.wait_for(
                                self.result_queue.put(sample_result),
                                timeout=queue_timeout,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"Result queue full, dropping sample from worker {worker_id}"
                            )

                    # 标记任务完成
                    self.task_queue.task_done()

                except asyncio.TimeoutError:
                    # 队列空，继续等待
                    continue
                except Exception as e:
                    logger.warning(f"Error in sampling worker {worker_id}: {e}")

        except Exception as e:
            logger.error(f"Sampling worker {worker_id} failed: {e}")
        finally:
            logger.debug(f"Sampling worker {worker_id} stopped")

    async def _result_processor(self) -> None:
        """
        结果处理Worker：从结果队列收集采样结果。
        """
        logger.debug("Result processor started")
        queue_timeout = self._get_config_value("queue_timeout", 0.1)
        batch_size = self._get_config_value("batch_result_size", 50)
        batch_buffer = []

        try:
            while not self.stop_event.is_set():
                try:
                    # 从结果队列获取结果
                    result = await asyncio.wait_for(
                        self.result_queue.get(), timeout=queue_timeout
                    )

                    batch_buffer.append(result)

                    # 批量处理结果
                    if len(batch_buffer) >= batch_size:
                        self.collected_samples.extend(batch_buffer)
                        self.total_samples_collected += len(batch_buffer)
                        batch_buffer = []

                    self.result_queue.task_done()

                except asyncio.TimeoutError:
                    # 队列空，处理缓存的结果
                    if batch_buffer:
                        self.collected_samples.extend(batch_buffer)
                        self.total_samples_collected += len(batch_buffer)
                        batch_buffer = []
                    continue
                except Exception as e:
                    logger.warning(f"Error in result processor: {e}")

            # 处理剩余的缓存结果
            if batch_buffer:
                self.collected_samples.extend(batch_buffer)
                self.total_samples_collected += len(batch_buffer)

        except Exception as e:
            logger.error(f"Result processor failed: {e}")
        finally:
            logger.debug("Result processor stopped")

    async def _discover_processes_async(self) -> List:
        """使用线程池异步发现进程。"""
        if not self.discovery_collector:
            return []

        try:
            # 使用注入的线程池管理器
            loop = asyncio.get_event_loop()
            if not self.thread_pool_manager:
                logger.warning(
                    "No thread pool manager provided, using default executor"
                )
                monitoring_pool = None
            else:
                monitoring_pool = self.thread_pool_manager.get_pool("monitoring")

            def sync_discover():
                try:
                    # 使用新的单样本收集方法
                    return self.discovery_collector.collect_single_sample()
                except Exception as e:
                    logger.warning(f"Error in sync process discovery: {e}")
                    return []

            # 在线程池中执行同步操作
            if monitoring_pool and monitoring_pool.executor:
                executor = monitoring_pool.executor
            else:
                executor = None  # Use default executor

            samples = await loop.run_in_executor(executor, sync_discover)
            return samples

        except Exception as e:
            logger.warning(f"Error discovering processes: {e}")
            return []

    async def _sample_process_async(
        self, collector: AbstractMemoryCollector, task_item: Dict, worker_id: int
    ) -> Optional[Dict]:
        """使用线程池异步采样进程内存，并集成进程分类。"""
        if not self.thread_pool_manager:
            logger.warning("No thread pool manager provided for sampling")
            return None

        monitoring_pool = self.thread_pool_manager.get_pool("monitoring")
        if not monitoring_pool:
            logger.warning("No monitoring thread pool available")
            return None

        try:
            # 使用线程池执行同步的收集器操作
            loop = asyncio.get_event_loop()

            def sync_sample():
                try:
                    # 设置收集器的目标PID（如果支持）
                    if hasattr(collector, "build_process_pid"):
                        collector.build_process_pid = task_item.get("pid", 0)

                    # 使用新的单样本收集方法
                    samples = collector.collect_single_sample()

                    # 查找匹配的进程样本
                    target_pid = str(task_item.get("pid", ""))
                    for sample in samples:
                        if str(sample.pid) == target_pid:
                            # 获取主要内存指标
                            primary_metric = collector.get_primary_metric_field()
                            memory_kb = sample.metrics.get(primary_metric, 0)

                            return {
                                "timestamp": time.time(),
                                "pid": sample.pid,
                                "command_name": sample.command_name,
                                "full_command": sample.full_command,
                                "metrics": sample.metrics.copy(),
                                "memory_kb": memory_kb,
                                "worker_id": worker_id,
                                # 分类信息将在结果聚合阶段添加
                            }

                    # 如果没有找到匹配的PID，但有其他样本，返回第一个
                    if samples:
                        sample = samples[0]
                        primary_metric = collector.get_primary_metric_field()
                        memory_kb = sample.metrics.get(primary_metric, 0)

                        return {
                            "timestamp": time.time(),
                            "pid": sample.pid,
                            "command_name": sample.command_name,
                            "full_command": sample.full_command,
                            "metrics": sample.metrics.copy(),
                            "memory_kb": memory_kb,
                            "worker_id": worker_id,
                            # 分类信息将在结果聚合阶段添加
                        }

                    return None

                except Exception as e:
                    logger.warning(f"Error in sync sampling: {e}")
                    return None

            # 在线程池中执行同步操作
            if monitoring_pool.executor:
                executor = monitoring_pool.executor
            else:
                executor = None

            result = await loop.run_in_executor(executor, sync_sample)
            return result

        except Exception as e:
            logger.warning(f"Error sampling process: {e}")
            return None

    def _get_task_priority(self, cmd_name: str) -> int:
        """
        获取任务优先级（编译器进程优先级更高）。
        """
        enable_prioritization = self._get_config_value("enable_prioritization", True)
        if not enable_prioritization:
            return 0

        # 编译器进程优先级更高
        compiler_commands = {"gcc", "g++", "clang", "clang++", "cc1", "cc1plus"}
        if cmd_name.lower() in compiler_commands:
            return 10

        # 链接器进程中等优先级
        linker_commands = {"ld", "gold", "lld"}
        if cmd_name.lower() in linker_commands:
            return 5

        # 其他进程低优先级
        return 1

    def _aggregate_results(
        self, samples: List[Dict[str, Any]]
    ) -> Optional[MonitoringResults]:
        """聚合采样结果为MonitoringResults，并在此阶段进行进程分类。"""
        if not samples:
            logger.warning("No samples to aggregate")
            return None

        logger.info(f"Aggregating {len(samples)} samples into MonitoringResults")

        # 导入分类函数（延迟导入避免循环依赖）
        from ..classification.classifier import get_process_category

        # 初始化聚合数据结构
        all_samples_data = []
        category_stats = {}
        category_peak_sum = {}
        category_pid_set = {}
        peak_overall_memory_kb = 0
        peak_overall_memory_epoch = 0

        # 处理每个样本，在此阶段进行分类
        interval_memory_sums = {}  # timestamp -> total_memory

        for sample in samples:
            timestamp = sample.get("timestamp", 0)
            pid = str(sample.get("pid", ""))
            memory_kb = sample.get("memory_kb", 0)
            command_name = sample.get("command_name", "")
            full_command = sample.get("full_command", "")

            # 在聚合阶段进行进程分类
            major_category, minor_category = get_process_category(
                command_name, full_command
            )

            # 创建包含分类信息的样本副本
            classified_sample = sample.copy()
            classified_sample["category"] = minor_category
            classified_sample["major_category"] = major_category
            all_samples_data.append(classified_sample)

            # 使用分类后的类别进行统计
            category = minor_category

            # 更新类别统计
            if category not in category_stats:
                category_stats[category] = {}
                category_pid_set[category] = set()
                category_peak_sum[category] = 0

            # 收集PID
            if pid:
                category_pid_set[category].add(pid)

            # 更新类别峰值
            if memory_kb > category_peak_sum[category]:
                category_peak_sum[category] = memory_kb

            # 计算时间点总内存使用
            if timestamp not in interval_memory_sums:
                interval_memory_sums[timestamp] = 0
            interval_memory_sums[timestamp] += memory_kb

        # 找到整体峰值内存和对应时间戳
        if interval_memory_sums:
            peak_overall_memory_epoch = max(
                interval_memory_sums.keys(), key=lambda t: interval_memory_sums[t]
            )
            peak_overall_memory_kb = interval_memory_sums[peak_overall_memory_epoch]

        # 计算每个类别的详细统计
        for category in category_stats:
            category_samples = [
                s for s in all_samples_data if s.get("category") == category
            ]
            if category_samples:
                memory_values = [s.get("memory_kb", 0) for s in category_samples]
                category_stats[category] = {
                    "peak_memory_kb": max(memory_values),
                    "avg_memory_kb": sum(memory_values) / len(memory_values),
                    "sample_count": len(category_samples),
                    "unique_pids": len(category_pid_set[category]),
                }

        result = MonitoringResults(
            all_samples_data=all_samples_data,
            category_stats=category_stats,
            peak_overall_memory_kb=peak_overall_memory_kb,
            peak_overall_memory_epoch=peak_overall_memory_epoch,
            category_peak_sum=category_peak_sum,
            category_pid_set=category_pid_set,
        )

        logger.info(
            f"Aggregation complete: {len(all_samples_data)} samples, "
            f"peak memory: {peak_overall_memory_kb} KB at {peak_overall_memory_epoch}"
        )

        return result

    async def start(self, build_pid: int = 12345) -> None:
        """
        Convenience method for testing: setup and start monitoring.

        Args:
            build_pid: Process ID to monitor (defaults to test PID)
        """
        # Use default monitoring cores for testing
        monitoring_cores = [0, 1, 2, 3]
        await self.setup_monitoring(monitoring_cores)
        await self.start_monitoring(build_pid)

    async def stop(self) -> Optional[MonitoringResults]:
        """
        Convenience method for testing: stop monitoring and return results.

        Returns:
            Monitoring results if available
        """
        return await self.stop_monitoring()
