"""
Monitoring coordination and orchestration.

This module provides the main monitoring coordinator that manages the collection
of memory data from build processes using the configured collector strategy.
"""

import logging
import multiprocessing
import queue
import time
from pathlib import Path
from typing import List, Optional

from ..classification import get_process_category
from ..models.results import MonitoringResults
from ..models.runtime import RunContext
from ..collectors.base import AbstractMemoryCollector
from ..validation import handle_error, ErrorSeverity
from ..config import get_config

logger = logging.getLogger(__name__)


class MonitoringCoordinator:
    """
    Coordinates memory monitoring during build execution.
    
    This class manages the setup and coordination of memory collection workers,
    data aggregation, and result compilation for build monitoring runs.
    """
    
    def __init__(self, run_context: RunContext):
        """
        Initialize the monitoring coordinator.
        
        Args:
            run_context: Runtime context for the monitoring run
        """
        self.run_context = run_context
        self.input_queue: Optional[multiprocessing.Queue] = None
        self.output_queue: Optional[multiprocessing.Queue] = None
        self.monitoring_processes: List[multiprocessing.Process] = []
        self.results: Optional[MonitoringResults] = None
        
    def setup_monitoring(self, monitoring_cores: List[int]) -> None:
        """
        Set up monitoring infrastructure including queues and worker processes.
        
        Args:
            monitoring_cores: List of CPU core IDs for monitoring workers
        """
        try:
            # Create multiprocessing queues for communication
            self.input_queue = multiprocessing.Queue()
            self.output_queue = multiprocessing.Queue()
            
            # Start monitoring worker processes
            for i, core_id in enumerate(monitoring_cores):
                worker_process = multiprocessing.Process(
                    target=_monitoring_worker_static,
                    args=(core_id, self.input_queue, self.output_queue, self.run_context),
                    name=f"MemoryMonitor-{i}-Core{core_id}"
                )
                worker_process.start()
                self.monitoring_processes.append(worker_process)
                
            logger.info(f"Started {len(self.monitoring_processes)} monitoring workers on cores {monitoring_cores}")
            
        except Exception as e:
            handle_error(
                error=e,
                context="setting up monitoring infrastructure",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger
            )
    
    def start_monitoring(self, build_process_pid: int) -> None:
        """
        Start the monitoring process for the given build PID.
        
        Args:
            build_process_pid: Process ID of the build process to monitor
        """
        if not self.input_queue:
            raise RuntimeError("Monitoring not set up - call setup_monitoring() first")
            
        try:
            # Send start signal to monitoring workers
            for _ in self.monitoring_processes:
                self.input_queue.put({
                    'action': 'start',
                    'build_pid': build_process_pid,
                    'interval': self.run_context.monitoring_interval,
                    'process_pattern': self.run_context.process_pattern,
                    'collector_type': self.run_context.collector_type
                })
                
            logger.info(f"Started monitoring for build PID {build_process_pid}")
            
        except Exception as e:
            handle_error(
                error=e,
                context=f"starting monitoring for PID {build_process_pid}",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger
            )
    
    def stop_monitoring(self) -> None:
        """
        Stop all monitoring workers and collect final results.
        """
        if not self.input_queue or not self.output_queue:
            logger.warning("Monitoring not properly initialized - nothing to stop")
            return
            
        try:
            # Send stop signal to all workers
            for _ in self.monitoring_processes:
                self.input_queue.put({'action': 'stop'})
            
            # Collect results from workers
            all_samples = []
            timeout_seconds = 10.0
            
            for i in range(len(self.monitoring_processes)):
                try:
                    worker_results = self.output_queue.get(timeout=timeout_seconds)
                    if worker_results and 'samples' in worker_results:
                        all_samples.extend(worker_results['samples'])
                except queue.Empty:
                    logger.warning(f"Timeout waiting for results from worker {i}")
            
            # Wait for worker processes to finish
            for process in self.monitoring_processes:
                process.join(timeout=5.0)
                if process.is_alive():
                    logger.warning(f"Force terminating monitoring process {process.name}")
                    process.terminate()
                    process.join(timeout=2.0)
            
            self.monitoring_processes.clear()
            
            # Aggregate results
            if all_samples:
                self.results = self._aggregate_monitoring_data(all_samples)
                logger.info(f"Monitoring stopped, collected {len(all_samples)} samples")
            else:
                logger.warning("No monitoring data collected")
                self.results = None
                
        except Exception as e:
            handle_error(
                error=e,
                context="stopping monitoring",
                severity=ErrorSeverity.WARNING,
                reraise=False,
                logger=logger
            )
    
    def get_results(self) -> Optional[MonitoringResults]:
        """
        Get the monitoring results after monitoring has completed.
        
        Returns:
            MonitoringResults instance or None if no data was collected
        """
        return self.results

    def _aggregate_monitoring_data(self, all_samples: List[dict]) -> MonitoringResults:
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
        
        for epoch, samples in epoch_groups.items():
            # Calculate total memory for this epoch
            epoch_total_memory = sum(sample.get(primary_metric, 0) for sample in samples)
            
            if epoch_total_memory > peak_overall_memory_kb:
                peak_overall_memory_kb = epoch_total_memory
                peak_overall_memory_epoch = int(epoch)
            
            # Group by category for this epoch
            category_memory = {}
            for sample in samples:
                major_cat = sample.get('major_category', 'Unknown')
                minor_cat = sample.get('minor_category', 'Unknown')
                category = f"{major_cat}:{minor_cat}"
                
                memory_val = sample.get(primary_metric, 0)
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
        
        # Create category stats (simplified version)
        category_stats = {}
        for category in category_peak_sum:
            category_stats[category] = {
                'peak_memory_kb': category_peak_sum[category],
                'pid_count': len(category_pid_set[category])
            }
        
        return MonitoringResults(
            all_samples_data=all_samples,
            category_stats=category_stats,
            peak_overall_memory_kb=peak_overall_memory_kb,
            peak_overall_memory_epoch=peak_overall_memory_epoch,
            category_peak_sum=category_peak_sum,
            category_pid_set=category_pid_set
        )


def _monitoring_worker_static(core_id: int, input_queue: multiprocessing.Queue, 
                             output_queue: multiprocessing.Queue, run_context: RunContext) -> None:
    """
    Static worker process function for memory monitoring.
    
    This function is separated from the class to avoid pickle issues with 
    multiprocessing and weakref objects in the class instance.
    
    Args:
        core_id: CPU core ID this worker is assigned to
        input_queue: Queue for receiving control messages
        output_queue: Queue for sending collected data
        run_context: Runtime context for the monitoring run
    """
    # Create a new logger for this worker process
    worker_logger = logging.getLogger(f"mymonitor.monitoring.worker.{core_id}")
    
    try:
        # Set CPU affinity if taskset is available
        if run_context.taskset_available:
            import os
            import subprocess
            pid = os.getpid()
            try:
                subprocess.run(['taskset', '-cp', str(core_id), str(pid)], 
                             check=False, capture_output=True)
            except (FileNotFoundError, subprocess.SubprocessError):
                pass  # Taskset not available or failed, continue without affinity
        
        collector = None
        samples = []
        monitoring_active = False
        
        while True:
            try:
                # Check for control messages
                try:
                    message = input_queue.get_nowait()
                    if message['action'] == 'start':
                        # Initialize and start collector
                        collector = _create_collector_static(
                            message['collector_type'], 
                            message['process_pattern'],
                            run_context
                        )
                        # Set build process PID if provided
                        if 'build_pid' in message:
                            collector.build_process_pid = message['build_pid']
                        
                        # Start the collector - this is critical!
                        try:
                            collector.start()
                            monitoring_active = True
                            worker_logger.debug(f"Worker {core_id}: Started monitoring with collector {message['collector_type']}")
                        except Exception as e:
                            worker_logger.error(f"Worker {core_id}: Failed to start collector: {e}")
                            collector = None
                            monitoring_active = False
                        
                    elif message['action'] == 'stop':
                        monitoring_active = False
                        # Stop the collector properly if it exists
                        if collector:
                            try:
                                collector.stop(timeout=5.0)
                                worker_logger.debug(f"Worker {core_id}: Collector stopped")
                            except Exception as e:
                                worker_logger.warning(f"Worker {core_id}: Error stopping collector: {e}")
                        
                        # Send results back
                        output_queue.put({'samples': samples})
                        worker_logger.debug(f"Worker {core_id}: Stopped monitoring, sent {len(samples)} samples")
                        break
                        
                except queue.Empty:
                    pass  # No control messages, continue monitoring
                
                # Collect data if monitoring is active
                if monitoring_active and collector:
                    try:
                        # Use read_samples which is a generator, get one batch
                        sample_collected = False
                        for sample_batch in collector.read_samples():
                            if sample_batch:
                                # Convert ProcessMemorySample objects to dictionaries with classification
                                epoch = time.time()
                                for sample in sample_batch:
                                    # Classify the process using the classification system
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
                                        **sample.metrics
                                    }
                                    samples.append(sample_dict)
                                sample_collected = True
                                worker_logger.debug(f"Worker {core_id}: Collected {len(sample_batch)} samples")
                            break  # Only get one batch per iteration
                        
                        # If no samples were collected, log this periodically
                        if not sample_collected and len(samples) % 10 == 0:
                            worker_logger.debug(f"Worker {core_id}: No samples collected in this iteration")
                            
                    except Exception as e:
                        worker_logger.warning(f"Worker {core_id}: Data collection error: {e}")
                
                # Sleep for the monitoring interval
                if monitoring_active:
                    time.sleep(run_context.monitoring_interval)
                else:
                    time.sleep(0.1)  # Short sleep when not monitoring
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                worker_logger.error(f"Worker {core_id}: Unexpected error: {e}")
                break
        
    except Exception as e:
        worker_logger.error(f"Worker {core_id}: Fatal error: {e}")
    finally:
        # Ensure collector is properly stopped
        if collector:
            try:
                collector.stop(timeout=2.0)
            except Exception as e:
                worker_logger.warning(f"Worker {core_id}: Error in final collector cleanup: {e}")
        worker_logger.debug(f"Worker {core_id}: Exiting")


def _create_collector_static(collector_type: str, process_pattern: str, run_context: RunContext) -> AbstractMemoryCollector:
    """
    Create a memory collector instance based on the specified type.
    
    This is a static function to avoid pickle issues with class instances.
    
    Args:
        collector_type: Type of collector to create
        process_pattern: Pattern for finding relevant processes
        run_context: Runtime context for collector configuration
        
    Returns:
        Memory collector instance
    """
    # Get app config for collector-specific settings
    try:
        app_config = get_config()
        # Extract collector mode for psutil collectors
        pss_collector_mode = getattr(app_config.monitor, 'pss_collector_mode', 'full_scan')
    except Exception:
        # Fallback to defaults if config is not available
        pss_collector_mode = 'full_scan'
    
    # Common kwargs for all collectors
    collector_kwargs = {
        'collector_cpu_core': run_context.monitor_core_id,
        'taskset_available': run_context.taskset_available,
    }
    
    if collector_type == "pss_psutil":
        from ..collectors.pss_psutil import PssPsutilCollector
        collector_kwargs['mode'] = pss_collector_mode
        return PssPsutilCollector(
            process_pattern, 
            run_context.monitoring_interval,
            **collector_kwargs
        )
    elif collector_type == "rss_pidstat":
        from ..collectors.rss_pidstat import RssPidstatCollector
        # Add pidstat-specific parameters
        collector_kwargs['pidstat_stderr_file'] = run_context.paths.collector_aux_log_file
        return RssPidstatCollector(
            process_pattern, 
            run_context.monitoring_interval,
            **collector_kwargs
        )
    else:
        raise ValueError(f"Unknown collector type: {collector_type}")
 