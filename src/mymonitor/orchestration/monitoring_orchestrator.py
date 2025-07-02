"""
Monitoring orchestration for the orchestration module.

This module coordinates monitoring data collection using a producer-consumer pattern
with multiprocessing workers. It includes the worker entry point function and
result aggregation logic.
"""

import logging
import multiprocessing
import threading
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .. import process_utils
from ..data_models import ErrorSeverity, MonitoringResults, handle_error
from ..memory_collectors.base import AbstractMemoryCollector
from .shared_state import BuildRunnerConfig, RuntimeState, TimeoutConstants

logger = logging.getLogger(__name__)


class MonitoringOrchestrator:
    """
    Coordinates monitoring data collection using producer-consumer pattern.
    
    This class manages the multiprocessing infrastructure for collecting
    and processing memory monitoring data from the build process.
    """
    
    def __init__(self, state: RuntimeState, config: BuildRunnerConfig):
        self.state = state
        self.config = config
        
    def setup_monitoring(self, collector: AbstractMemoryCollector) -> None:
        """
        Set up the monitoring infrastructure including queues and worker processes.
        
        Args:
            collector: The memory collector instance to use for data collection
        """
        if not self.state.run_context:
            raise ValueError("RunContext must be set before setting up monitoring")
            
        # Set the build PID for the collector and start it
        if self.state.build_process:
            collector.build_process_pid = self.state.build_process.pid
            self.state.run_context.build_process_pid = self.state.build_process.pid
            
        self.state.collector = collector
        collector.start()

        # Initialize queues for multiprocessing
        manager = multiprocessing.Manager()
        self.state.input_queue = manager.Queue()
        self.state.output_queue = manager.Queue()

        # Determine the number of worker processes
        self.state.num_workers = len(self.state.monitoring_cores) if self.state.monitoring_cores else 1
        worker_cores = (
            self.state.monitoring_cores
            if self.state.monitoring_cores
            else [None] * self.state.num_workers
        )

        logger.info(f"Starting {self.state.num_workers} monitoring worker process(es).")

        # Create and start the monitoring worker processes
        primary_metric_field = collector.get_primary_metric_field()
        self.state.monitoring_worker_processes = []
        
        for i, core_id in enumerate(worker_cores):
            worker_func = partial(
                self._worker_entry_point, 
                core_id, 
                self.state.input_queue, 
                self.state.output_queue, 
                primary_metric_field
            )
            process = multiprocessing.Process(target=worker_func)
            process.start()
            self.state.monitoring_worker_processes.append(process)
            logger.debug(f"Started monitoring worker process {i+1} (PID: {process.pid}, core: {core_id})")

    def start_producer_thread(self) -> None:
        """Start the producer thread that feeds data to worker processes."""
        if not self.state.collector:
            raise ValueError("Collector must be set up before starting producer thread")
            
        self.state.producer_thread = threading.Thread(target=self._producer_loop)
        self.state.producer_thread.start()

    def shutdown_monitoring(self) -> None:
        """
        Shutdown the monitoring system gracefully.
        
        This stops the collector, waits for the producer thread, and terminates worker processes.
        """
        if self.state.collector:
            logger.info("Stopping collector to signal producer thread to finish...")
            try:
                stop_success = self.state.collector.stop(timeout=TimeoutConstants.COLLECTOR_STOP_TIMEOUT)
                if not stop_success:
                    logger.warning("Collector did not stop cleanly, but continuing with shutdown")
            except Exception as e:
                handle_error(
                    error=e,
                    context="Collector stop during shutdown",
                    severity=ErrorSeverity.ERROR,
                    include_traceback=True,
                    reraise=False,
                    logger=logger
                )

        if self.state.producer_thread:
            logger.info("Waiting for producer thread to finish...")
            self.state.producer_thread.join(timeout=TimeoutConstants.PRODUCER_JOIN_TIMEOUT)
            if self.state.producer_thread.is_alive():
                logger.warning("Producer thread did not finish in time")
            else:
                self.state.producer_finished.set()

        # Signal monitoring workers to terminate and wait for them
        if self.state.monitoring_worker_processes:
            logger.info("All data queued. Signaling monitoring workers to terminate.")
            
            # Send exactly one sentinel per worker to avoid queue flooding
            for i in range(self.state.num_workers):
                try:
                    if self.state.input_queue:
                        self.state.input_queue.put(None, timeout=TimeoutConstants.QUEUE_PUT_TIMEOUT)
                        logger.debug(f"Sent termination signal to worker {i+1}")
                    else:
                        logger.warning("Input queue is None, cannot send termination signal")
                        break
                except Exception as e:
                    handle_error(
                        error=e,
                        context=f"Sending termination signal to worker {i+1}",
                        severity=ErrorSeverity.WARNING,
                        include_traceback=False,
                        reraise=False,
                        logger=logger
                    )
                    break

            # Wait for all worker processes to finish with proper timeout
            for i, process in enumerate(self.state.monitoring_worker_processes):
                try:
                    process.join(timeout=TimeoutConstants.WORKER_JOIN_TIMEOUT)
                    if process.is_alive():
                        logger.warning(f"Worker process {i+1} (PID: {process.pid}) did not finish gracefully.")
                    else:
                        logger.debug(f"Worker process {i+1} finished successfully")
                except Exception as e:
                    handle_error(
                        error=e,
                        context=f"Waiting for worker process {i+1} to finish",
                        severity=ErrorSeverity.WARNING,
                        include_traceback=False,
                        reraise=False,
                        logger=logger
                    )
            
            # Set workers finished event
            self.state.workers_finished.set()
            logger.info("All monitoring workers have terminated.")

    def collect_and_aggregate_results(self) -> Optional[MonitoringResults]:
        """
        Collect and aggregate monitoring results from worker processes.
        
        Returns:
            Aggregated monitoring results or None if no data was collected
        """
        # Process output queue with improved handling
        all_samples_data = []
        final_category_pid_set: Dict[str, set] = {}
        final_category_peak_sum: Dict[str, float] = {}
        group_memory_snapshots = []

        output_items_count = 0
        logger.info("Starting to process output queue...")
        
        # Give workers time to finish writing to output queue
        time.sleep(TimeoutConstants.WORKERS_WRITE_DELAY)
        
        # Process all items in output queue with improved retry logic
        consecutive_timeouts = 0
        
        while consecutive_timeouts < TimeoutConstants.MAX_CONSECUTIVE_TIMEOUTS:
            try:
                if not self.state.output_queue:
                    logger.error("Output queue is None, cannot collect results")
                    break
                worker_result = self.state.output_queue.get(timeout=TimeoutConstants.QUEUE_PROCESS_TIMEOUT)
                consecutive_timeouts = 0  # Reset timeout counter on successful get
                output_items_count += 1
                logger.debug(f"Processing output item {output_items_count}")
                
                samples_in_result = len(worker_result.get("all_samples_data", []))
                logger.debug(f"Output item {output_items_count} contains {samples_in_result} samples")
                
                all_samples_data.extend(worker_result.get("all_samples_data", []))

                # Aggregate category data from the worker
                for cat, pids in worker_result.get("category_pid_set", {}).items():
                    if cat not in final_category_pid_set:
                        final_category_pid_set[cat] = set()
                    final_category_pid_set[cat].update(pids)

                for cat, peak_mem in worker_result.get("category_peak_sum", {}).items():
                    final_category_peak_sum[cat] = max(
                        final_category_peak_sum.get(cat, 0.0), peak_mem
                    )

                # Collect the total memory snapshot for this time point
                if "group_total_memory" in worker_result:
                    group_memory_snapshots.append(
                        {
                            "memory": worker_result["group_total_memory"],
                            "timestamp": worker_result["timestamp"],
                        }
                    )
            except Exception:
                # Count consecutive timeouts to determine when to stop
                consecutive_timeouts += 1
                logger.debug(f"Output queue timeout {consecutive_timeouts}/{TimeoutConstants.MAX_CONSECUTIVE_TIMEOUTS}")

        logger.info(f"Processed {output_items_count} output items from workers")
        logger.info(f"Total samples collected: {len(all_samples_data)}")

        # Post-process aggregated data
        peak_overall_memory_kb = 0
        peak_overall_memory_epoch = 0
        if group_memory_snapshots:
            peak_snapshot = max(group_memory_snapshots, key=lambda x: x["memory"])
            peak_overall_memory_kb = peak_snapshot["memory"]
            peak_overall_memory_epoch = int(peak_snapshot["timestamp"].timestamp())

        if all_samples_data:
            df = pd.DataFrame(all_samples_data)
            # Add epoch timestamp for plotting
            df["epoch"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9

            results = MonitoringResults(
                all_samples_data=df.to_dict("records"),
                category_stats={},
                peak_overall_memory_kb=int(peak_overall_memory_kb),
                peak_overall_memory_epoch=int(peak_overall_memory_epoch),
                category_peak_sum={
                    k: int(v) for k, v in final_category_peak_sum.items()
                },
                category_pid_set=final_category_pid_set,
            )
            
            # Store results in state
            self.state.results = results
            return results
        else:
            # No data collected
            self.state.results = None
            return None

    def _producer_loop(self) -> None:
        """
        The producer loop that runs in a separate thread.
        
        It reads samples from the collector and puts them onto the input queue
        for the worker processes to consume.
        """
        if not self.state.input_queue or not self.state.collector:
            logger.error("Producer: Input queue or collector not available")
            return

        logger.info("Producer loop started. Reading samples from collector.")
        sample_count = 0
        
        try:
            for sample_group in self.state.collector.read_samples():
                # Check for shutdown request
                if self.state.shutdown_requested.is_set():
                    logger.info("Producer: Shutdown requested, stopping sample collection")
                    break
                    
                sample_count += 1
                logger.debug(f"Producer: Received sample group {sample_count} with {len(sample_group)} samples")
                
                # Add a timestamp to the group for later aggregation
                timestamp = pd.Timestamp.now()
                
                try:
                    # Use timeout to avoid blocking indefinitely
                    self.state.input_queue.put((sample_group, timestamp), timeout=5.0)
                    logger.debug(f"Producer: Put sample group {sample_count} into input queue")
                except Exception as e:
                    handle_error(
                        error=e,
                        context=f"Producer queuing sample group {sample_count}",
                        severity=ErrorSeverity.WARNING,
                        include_traceback=False,
                        reraise=False,
                        logger=logger
                    )
                    # Continue trying to process remaining samples
                    continue

        except Exception as e:
            handle_error(
                error=e,
                context="Producer sample collection loop",
                severity=ErrorSeverity.ERROR,
                include_traceback=True,
                reraise=False,
                logger=logger
            )
        finally:
            logger.info(f"Producer loop finished. Processed {sample_count} sample groups. All samples have been queued.")
            # Signal that all data has been queued for processing
            self.state.all_data_queued.set()

    @staticmethod
    def _worker_entry_point(
        core_id: Optional[int], 
        input_queue: Any, 
        output_queue: Any, 
        primary_metric_field: str
    ) -> Dict:
        """
        Entry point for a monitoring worker process.
        
        This is moved from the module-level _monitoring_worker_entry function
        and made into a static method of the orchestrator class.
        """
        # Pin the worker to a specific CPU core if specified
        if core_id is not None:
            try:
                import psutil
                p = psutil.Process()
                p.cpu_affinity([core_id])
                logger.info(f"Monitoring worker (PID:{p.pid}) pinned to core {core_id}.")
            except Exception as e:
                handle_error(
                    error=e,
                    context=f"CPU core pinning to core {core_id}",
                    severity=ErrorSeverity.WARNING,
                    include_traceback=False,
                    reraise=False,
                    logger=logger
                )

        if not input_queue or not output_queue:
            logger.error("Worker: Input or output queue not available")
            return {}

        logger.info(f"Monitoring worker started (core: {core_id})")
        processed_groups = 0
        sentinel_seen = False
        
        # The loop will run until it receives a sentinel value (None)
        while True:
            try:
                queue_item = input_queue.get(timeout=TimeoutConstants.QUEUE_GET_TIMEOUT)
            except Exception as e:
                handle_error(
                    error=e,
                    context="Worker queue get operation",
                    severity=ErrorSeverity.DEBUG,
                    include_traceback=False,
                    reraise=False,
                    logger=logger
                )
                continue
                
            if queue_item is None:  # Sentinel value received
                logger.info(f"Worker: Received sentinel, processed {processed_groups} groups")
                # Only propagate sentinel if this worker hasn't seen it before
                # This prevents multiple workers from flooding the queue with sentinels
                if not sentinel_seen:
                    try:
                        input_queue.put(None, timeout=TimeoutConstants.SENTINEL_PUT_TIMEOUT)
                        logger.debug("Worker: Propagated sentinel to other workers")
                    except Exception as e:
                        handle_error(
                            error=e,
                            context="Worker sentinel propagation",
                            severity=ErrorSeverity.DEBUG,
                            include_traceback=False,
                            reraise=False,
                            logger=logger
                        )
                    sentinel_seen = True
                break

            # Process normal data
            sample_group, timestamp = queue_item
            processed_groups += 1
            logger.debug(f"Worker: Processing group {processed_groups} with {len(sample_group)} samples")

            # Local aggregation for this worker
            local_results = {
                "all_samples_data": [],
                "category_pid_set": {},
                "category_peak_sum": {},
                "group_total_memory": 0.0,
                "timestamp": timestamp,
            }
            group_total_memory = 0.0

            for sample in sample_group:
                major_cat, minor_cat = process_utils.get_process_category(
                    sample.command_name, sample.full_command
                )

                mem_kb = float(sample.metrics.get(primary_metric_field, 0.0))
                group_total_memory += mem_kb

                row_data = {
                    "timestamp": timestamp,  # Carry over the timestamp
                    "major_category": major_cat,
                    "minor_category": minor_cat,
                    "pid": sample.pid,
                    "command_name": sample.command_name,
                }
                row_data.update(sample.metrics)
                local_results["all_samples_data"].append(row_data)

                cat_key = f"{major_cat}:{minor_cat}"

                current_sum = local_results["category_peak_sum"].get(cat_key, 0.0)
                local_results["category_peak_sum"][cat_key] = current_sum + mem_kb

                if cat_key not in local_results["category_pid_set"]:
                    local_results["category_pid_set"][cat_key] = set()
                local_results["category_pid_set"][cat_key].add(sample.pid)

            # Store the total for this snapshot
            local_results["group_total_memory"] = group_total_memory

            # Put the processed chunk of data onto the output queue
            try:
                output_queue.put(local_results, timeout=TimeoutConstants.QUEUE_PUT_TIMEOUT)
                logger.debug(f"Worker: Put processed group {processed_groups} into output queue (total memory: {group_total_memory} KB)")
            except Exception as e:
                handle_error(
                    error=e,
                    context=f"Worker output queue put operation for group {processed_groups}",
                    severity=ErrorSeverity.WARNING,
                    include_traceback=False,
                    reraise=False,
                    logger=logger
                )
                # Continue processing even if one put fails

        logger.info(f"Monitoring worker finished, processed {processed_groups} groups")
        return {}
