"""
Asynchronous monitoring coordination and orchestration.

This module provides the AsyncMonitoringCoordinator that manages the collection
of memory data from build processes using AsyncIO and ThreadPoolExecutor,
replacing the complex multiprocessing approach.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

from ..classification import get_process_category
from ..models.results import MonitoringResults
from ..models.runtime import RunContext
from ..collectors.base import AbstractMemoryCollector, ProcessMemorySample
from ..validation import handle_error, ErrorSeverity
from ..config import get_config

logger = logging.getLogger(__name__)


class AsyncMonitoringCoordinator:
    """
    Asynchronous coordinator for memory monitoring during build execution.
    
    This class manages monitoring using AsyncIO and ThreadPoolExecutor,
    providing better resource utilization and simpler error handling
    than the multiprocessing approach.
    """
    
    def __init__(self, run_context: RunContext):
        """
        Initialize the async monitoring coordinator.
        
        Args:
            run_context: Runtime context for the monitoring run
        """
        self.run_context = run_context
        self.executor: Optional[ThreadPoolExecutor] = None
        self.monitoring_tasks: List[asyncio.Task] = []
        self.collectors: List[AbstractMemoryCollector] = []
        self.results: Optional[MonitoringResults] = None
        self.is_monitoring = False
        self.samples_collected: List[Dict[str, Any]] = []
        self._shutdown_event = asyncio.Event()
        
        # Get configuration for async settings
        config = get_config()
        self.monitoring_timeout = config.monitor.monitoring_timeout
        self.graceful_shutdown_timeout = config.monitor.graceful_shutdown_timeout
        self.max_concurrent_monitors = config.monitor.max_concurrent_monitors
        
    async def setup_monitoring(self, monitoring_cores: List[int]) -> None:
        """
        Set up monitoring infrastructure including thread pool and collectors.
        
        Args:
            monitoring_cores: List of CPU core IDs for monitoring workers
        """
        try:
            # Create ThreadPoolExecutor with configured max workers
            max_workers = min(len(monitoring_cores), self.max_concurrent_monitors)
            self.executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="AsyncMonitor"
            )
            
            # Create collectors for each monitoring core
            self.collectors = []
            for core_id in monitoring_cores:
                try:
                    # Create collector synchronously to avoid executor issues
                    collector = self._create_collector_sync(core_id)
                    self.collectors.append(collector)
                except Exception as e:
                    logger.warning(f"Failed to create collector for core {core_id}: {e}")
                    
            if not self.collectors:
                raise RuntimeError("No collectors could be created")
                
            logger.info(f"Set up {len(self.collectors)} async monitoring collectors on cores {monitoring_cores}")
            
        except Exception as e:
            handle_error(
                error=e,
                context="setting up async monitoring infrastructure",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger
            )
    
    async def start_monitoring(self, build_process_pid: int) -> None:
        """
        Start the monitoring process for the given build PID.
        
        Args:
            build_process_pid: Process ID of the build process to monitor
        """
        if not self.executor or not self.collectors:
            raise RuntimeError("Monitoring not set up - call setup_monitoring() first")
            
        try:
            self.is_monitoring = True
            self.samples_collected = []
            self._shutdown_event.clear()
            
            # Start monitoring tasks for each collector with timeout support
            for i, collector in enumerate(self.collectors):
                task = asyncio.create_task(
                    self._monitor_process_with_timeout(collector, build_process_pid, i),
                    name=f"monitor-{i}"
                )
                self.monitoring_tasks.append(task)
                
            logger.info(f"Started async monitoring for build PID {build_process_pid} with {len(self.monitoring_tasks)} tasks (timeout: {self.monitoring_timeout}s)")
            
        except Exception as e:
            handle_error(
                error=e,
                context=f"starting async monitoring for PID {build_process_pid}",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger
            )
    
    async def _monitor_process_with_timeout(self, collector: AbstractMemoryCollector, build_process_pid: int, worker_id: int) -> None:
        """
        Monitor a process with timeout support.
        
        Args:
            collector: Memory collector instance
            build_process_pid: Process ID of the build process 
            worker_id: Worker identifier
        """
        try:
            # Run monitoring with timeout
            await asyncio.wait_for(
                self._monitor_process_async(collector, build_process_pid, worker_id),
                timeout=self.monitoring_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Monitoring worker {worker_id} timed out after {self.monitoring_timeout}s")
        except Exception as e:
            logger.error(f"Monitoring worker {worker_id} failed: {e}")
    
    async def stop_monitoring(self) -> None:
        """
        Stop all monitoring tasks and collect final results.
        """
        if not self.is_monitoring:
            logger.warning("Monitoring not active - nothing to stop")
            return
            
        try:
            # Signal shutdown to all monitoring tasks
            self._shutdown_event.set()
            self.is_monitoring = False
            
            # Wait for all monitoring tasks to complete
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
                
            # Stop all collectors
            for collector in self.collectors:
                try:
                    collector.stop(timeout=self.graceful_shutdown_timeout)
                except Exception as e:
                    logger.warning(f"Error stopping collector: {e}")
            
            # Shutdown executor with timeout to prevent hanging
            if self.executor:
                try:
                    # Cancel all pending futures first
                    self.executor.shutdown(wait=False, cancel_futures=True)
                    # Give it a brief moment to clean up
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Error shutting down executor: {e}")
                finally:
                    self.executor = None
                
            # Aggregate results
            if self.samples_collected:
                self.results = self._aggregate_monitoring_data(self.samples_collected)
                logger.info(f"Async monitoring stopped, collected {len(self.samples_collected)} samples")
            else:
                logger.warning("No monitoring data collected")
                self.results = None
                
            # Clean up
            self.monitoring_tasks.clear()
            self.collectors.clear()
                
        except Exception as e:
            handle_error(
                error=e,
                context="stopping async monitoring",
                severity=ErrorSeverity.WARNING,
                reraise=False,
                logger=logger
            )
    
    async def _monitor_process_async(self, collector: AbstractMemoryCollector, build_pid: int, worker_id: int) -> None:
        """
        Asynchronous monitoring worker that collects memory samples.
        
        Args:
            collector: Memory collector instance
            build_pid: Build process PID to monitor
            worker_id: Unique worker identifier
        """
        worker_logger = logging.getLogger(f"{__name__}.worker-{worker_id}")
        
        try:
            # Set build PID and start collector
            collector.build_process_pid = build_pid
            
            # Start collector directly (it's not blocking)
            collector.start()
            
            worker_logger.debug(f"Worker {worker_id}: Started monitoring")
            
            # Main monitoring loop
            while not self._shutdown_event.is_set():
                try:
                    # Wait for either shutdown event or interval timeout
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.run_context.monitoring_interval
                    )
                    # If we reach here, shutdown was signaled
                    break
                except asyncio.TimeoutError:
                    # Timeout expired, collect sample
                    await self._collect_sample_async(collector, worker_id, worker_logger)
                    
        except Exception as e:
            worker_logger.error(f"Worker {worker_id}: Fatal error in monitoring loop: {e}", exc_info=True)
        finally:
            worker_logger.debug(f"Worker {worker_id}: Exiting")
    
    async def _collect_sample_async(self, collector: AbstractMemoryCollector, worker_id: int, worker_logger: logging.Logger) -> None:
        """
        Collect a memory sample asynchronously.
        
        Args:
            collector: Memory collector instance
            worker_id: Worker identifier
            worker_logger: Logger for this worker
        """
        try:
            # Run sample collection in thread pool - get only ONE sample, not the whole generator
            loop = asyncio.get_event_loop()
            sample_batch = await loop.run_in_executor(
                self.executor,
                self._get_next_sample_sync,
                collector
            )
            
            # Process collected samples
            if sample_batch:
                epoch = time.time()
                for sample in sample_batch:
                    major_cat, minor_cat = get_process_category(
                        sample.command_name, sample.full_command
                    )
                    sample_dict = {
                        'epoch': epoch,
                        'pid': sample.pid,
                        'command_name': sample.command_name,
                        'full_command': sample.full_command,
                        'major_category': major_cat,
                        'minor_category': minor_cat,
                        'timestamp': time.time(),
                        'worker_id': worker_id,
                        **sample.metrics
                    }
                    self.samples_collected.append(sample_dict)
                
                worker_logger.debug(f"Worker {worker_id}: Collected {len(sample_batch)} samples")
                
        except Exception as e:
            worker_logger.warning(f"Worker {worker_id}: Data collection error: {e}")
    
    async def _create_collector_async(self, core_id: int) -> AbstractMemoryCollector:
        """
        Create a memory collector instance asynchronously.
        
        Args:
            core_id: CPU core ID for this collector
            
        Returns:
            Memory collector instance
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._create_collector_sync,
            core_id
        )
    
    def _create_collector_sync(self, core_id: int) -> AbstractMemoryCollector:
        """
        Create a memory collector instance (synchronous version).
        
        Args:
            core_id: CPU core ID for this collector
            
        Returns:
            Memory collector instance
        """
        # Get app config for collector-specific settings
        try:
            app_config = get_config()
            pss_collector_mode = getattr(app_config.monitor, 'pss_collector_mode', 'full_scan')
        except Exception:
            pss_collector_mode = 'full_scan'
        
        # Common kwargs for all collectors
        collector_kwargs = {
            'collector_cpu_core': core_id,
            'taskset_available': self.run_context.taskset_available,
        }
        
        collector_type = self.run_context.collector_type
        
        if collector_type == "pss_psutil":
            from ..collectors.pss_psutil import PssPsutilCollector
            collector_kwargs['mode'] = pss_collector_mode
            return PssPsutilCollector(
                self.run_context.process_pattern,
                self.run_context.monitoring_interval,
                **collector_kwargs
            )
        elif collector_type == "rss_pidstat":
            from ..collectors.rss_pidstat import RssPidstatCollector
            collector_kwargs['pidstat_stderr_file'] = self.run_context.paths.collector_aux_log_file
            return RssPidstatCollector(
                self.run_context.process_pattern,
                self.run_context.monitoring_interval,
                **collector_kwargs
            )
        else:
            raise ValueError(f"Unknown collector type: {collector_type}")
    
    def get_results(self) -> Optional[MonitoringResults]:
        """
        Get the monitoring results after monitoring has completed.
        
        Returns:
            MonitoringResults instance or None if no data was collected
        """
        return self.results
    
    def _aggregate_monitoring_data(self, all_samples: List[Dict[str, Any]]) -> MonitoringResults:
        """
        Aggregate raw monitoring samples into structured results.
        
        Args:
            all_samples: List of raw sample dictionaries from collectors
            
        Returns:
            Aggregated MonitoringResults
        """
        if not all_samples:
            return MonitoringResults(
                all_samples_data=[],
                category_stats={},
                peak_overall_memory_kb=0,
                peak_overall_memory_epoch=0,
                category_peak_sum={},
                category_pid_set={}
            )
        
        # Determine primary metric field (prefer PSS_KB, fall back to RSS_KB)
        primary_metric = "PSS_KB" if "PSS_KB" in all_samples[0] else "RSS_KB"
        
        # Group samples by epoch for aggregation
        epoch_groups = {}
        for sample in all_samples:
            epoch = sample['epoch']
            if epoch not in epoch_groups:
                epoch_groups[epoch] = []
            epoch_groups[epoch].append(sample)
        
        # Find peak overall memory and calculate category statistics
        peak_overall_memory_kb = 0
        peak_overall_memory_epoch = 0
        category_peak_sum = {}
        category_pid_set = {}
        
        # Track peak memory for each individual process
        process_peak_memory = {}  # pid -> peak_memory
        process_category_map = {}  # pid -> category
        
        for epoch, samples in epoch_groups.items():
            # Calculate total memory for this epoch
            epoch_total_memory = sum(sample.get(primary_metric, 0) for sample in samples)
            
            if epoch_total_memory > peak_overall_memory_kb:
                peak_overall_memory_kb = epoch_total_memory
                peak_overall_memory_epoch = int(epoch)
            
            # Group by category for this epoch and track individual process peaks
            category_memory = {}
            for sample in samples:
                major_cat = sample.get('major_category', 'Unknown')
                minor_cat = sample.get('minor_category', 'Unknown')
                category = f"{major_cat}:{minor_cat}"
                
                memory_val = sample.get(primary_metric, 0)
                pid = sample['pid']
                
                # Track individual process peak memory
                if pid not in process_peak_memory or memory_val > process_peak_memory[pid]:
                    process_peak_memory[pid] = memory_val
                process_category_map[pid] = category
                
                # Track category total memory for this epoch
                if category not in category_memory:
                    category_memory[category] = 0
                category_memory[category] += memory_val
                
                # Track PIDs for each category
                if category not in category_pid_set:
                    category_pid_set[category] = set()
                category_pid_set[category].add(sample['pid'])
            
            # Update peak values for each category
            for category, memory in category_memory.items():
                if category not in category_peak_sum or memory > category_peak_sum[category]:
                    category_peak_sum[category] = memory
        
        # Calculate category statistics using individual process peaks
        category_stats = {}
        for pid, peak_memory in process_peak_memory.items():
            category = process_category_map[pid]
            if category not in category_stats:
                category_stats[category] = {
                    'peak_sum_kb': 0,
                    'process_count': 0,
                    'average_peak_kb': 0
                }
            
            category_stats[category]['peak_sum_kb'] += peak_memory
            category_stats[category]['process_count'] += 1
        
        # Calculate averages
        for category, stats in category_stats.items():
            if stats['process_count'] > 0:
                stats['average_peak_kb'] = stats['peak_sum_kb'] / stats['process_count']
        
        # Convert sets to lists for JSON serialization
        category_pid_set_lists = {k: list(v) for k, v in category_pid_set.items()}
        
        return MonitoringResults(
            all_samples_data=all_samples,
            category_stats=category_stats,
            peak_overall_memory_kb=peak_overall_memory_kb,
            peak_overall_memory_epoch=peak_overall_memory_epoch,
            category_peak_sum=category_peak_sum,
            category_pid_set=category_pid_set_lists
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_monitoring()

    def _get_next_sample_sync(self, collector: AbstractMemoryCollector) -> Optional[List[ProcessMemorySample]]:
        """
        Get the next sample from the collector synchronously.
        
        Args:
            collector: Memory collector instance
            
        Returns:
            List of ProcessMemorySample or None if no samples available
        """
        try:
            # For psutil-based collectors, use direct sampling instead of the infinite generator
            if hasattr(collector, '_collect_single_sample'):
                return collector._collect_single_sample()
            else:
                # Fallback: get one sample from the generator with limited iteration
                sample_iter = collector.read_samples()
                return next(sample_iter)
                
        except StopIteration:
            return None
        except Exception as e:
            logger.warning(f"Error getting sample from collector: {e}")
            return None
