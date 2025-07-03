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

from ..models.results import MonitoringResults
from ..models.runtime import RunContext
from ..memory_collectors.base import AbstractMemoryCollector
from ..validation import handle_error, ErrorSeverity

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
                    target=self._monitoring_worker,
                    args=(core_id, self.input_queue, self.output_queue),
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
    
    def _monitoring_worker(self, core_id: int, input_queue: multiprocessing.Queue, 
                          output_queue: multiprocessing.Queue) -> None:
        """
        Worker process function for memory monitoring.
        
        Args:
            core_id: CPU core ID this worker is assigned to
            input_queue: Queue for receiving control messages
            output_queue: Queue for sending collected data
        """
        try:
            # Set CPU affinity if taskset is available
            if self.run_context.taskset_available:
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
                            # Initialize collector
                            collector = self._create_collector(
                                message['collector_type'], 
                                message['process_pattern']
                            )
                            monitoring_active = True
                            logger.debug(f"Worker {core_id}: Started monitoring")
                            
                        elif message['action'] == 'stop':
                            monitoring_active = False
                            # Send results back
                            output_queue.put({'samples': samples})
                            logger.debug(f"Worker {core_id}: Stopped monitoring, sent {len(samples)} samples")
                            break
                            
                    except queue.Empty:
                        pass  # No control messages, continue monitoring
                    
                    # Collect data if monitoring is active
                    if monitoring_active and collector:
                        try:
                            # Use read_samples which is a generator, get one batch
                            for sample_batch in collector.read_samples():
                                if sample_batch:
                                    # Convert ProcessMemorySample objects to dictionaries
                                    for sample in sample_batch:
                                        sample_dict = {
                                            'pid': sample.pid,
                                            'command_name': sample.command_name,
                                            'full_command': sample.full_command,
                                            'timestamp': time.time(),
                                            **sample.metrics
                                        }
                                        samples.append(sample_dict)
                                break  # Only get one batch per iteration
                        except Exception as e:
                            logger.warning(f"Worker {core_id}: Data collection error: {e}")
                    
                    # Sleep for the monitoring interval
                    if monitoring_active:
                        time.sleep(self.run_context.monitoring_interval)
                    else:
                        time.sleep(0.1)  # Short sleep when not monitoring
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Worker {core_id}: Unexpected error: {e}")
                    break
            
        except Exception as e:
            logger.error(f"Worker {core_id}: Fatal error: {e}")
        finally:
            logger.debug(f"Worker {core_id}: Exiting")
    
    def _create_collector(self, collector_type: str, process_pattern: str) -> AbstractMemoryCollector:
        """
        Create a memory collector instance based on the specified type.
        
        Args:
            collector_type: Type of collector to create
            process_pattern: Pattern for finding relevant processes
            
        Returns:
            Memory collector instance
        """
        if collector_type == "pss_psutil":
            from ..memory_collectors.pss_psutil_collector import PssPsutilCollector
            return PssPsutilCollector(process_pattern, self.run_context.monitoring_interval)
        elif collector_type == "rss_pidstat":
            from ..memory_collectors.rss_pidstat_collector import RssPidstatCollector  
            return RssPidstatCollector(process_pattern, self.run_context.monitoring_interval)
        else:
            raise ValueError(f"Unknown collector type: {collector_type}")
    
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
        
        # For now, return a basic aggregation
        # In a full implementation, this would do sophisticated analysis
        return MonitoringResults(
            all_samples_data=all_samples,
            category_stats={},
            peak_overall_memory_kb=0,
            peak_overall_memory_epoch=int(time.time()),
            category_peak_sum={},
            category_pid_set={}
        )
 