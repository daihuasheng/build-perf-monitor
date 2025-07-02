"""
Core utilities for orchestrating the build monitoring process.

This module is the heart of the monitoring application. It defines the main
`run_and_monitor_build` function, which serves as the primary entry point for
executing a build, collecting performance data, and handling the entire
lifecycle of the monitoring process.

The logic is encapsulated within the `BuildRunner` class, which manages state,
handles process setup and teardown, and coordinates the memory collector. This
design improves modularity and makes the monitoring logic more robust and easier
to maintain.
"""

import dataclasses
import logging
import multiprocessing
import multiprocessing.pool
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import IO, Any, Dict, List, Optional

import pandas as pd
import psutil

# Local application imports
from . import process_utils
from .data_models import MonitoringResults, ProjectConfig, RunContext, RunPaths
from .memory_collectors.base import AbstractMemoryCollector

logger = logging.getLogger(__name__)

# --- Signal Handling Infrastructure ---
# Since signal handlers cannot be bound to class instances directly,
# we maintain a registry of active runners and their signal handlers.
_active_runners: Dict[int, "BuildRunner"] = {}
_active_runners_lock = threading.Lock()


def _global_signal_handler(signum: int, frame: Any) -> None:
    """
    Global signal handler that delegates to active BuildRunner instances.
    This is safer than using a single global variable as it can handle
    multiple BuildRunner instances properly.
    """
    logger.warning(f"Signal {signum} received. Notifying all active BuildRunner instances.")
    with _active_runners_lock:
        for runner_id, runner in _active_runners.items():
            logger.info(f"Requesting shutdown for BuildRunner {runner_id}")
            runner.shutdown_requested.set()


# This wrapper is necessary for multiprocessing.Pool to work correctly,
# as it needs a top-level function to pickle and send to worker processes.
def _monitoring_worker_entry(
    core_id: Optional[int], 
    input_queue: Any, 
    output_queue: Any, 
    primary_metric_field: str
) -> Dict:
    """Entry point for a monitoring worker process."""
    # Pin the worker to a specific CPU core if specified
    if core_id is not None:
        try:
            import psutil
            p = psutil.Process()
            p.cpu_affinity([core_id])
            logger.info(
                f"Monitoring worker (PID:{p.pid}) pinned to core {core_id}."
            )
        except Exception as e:
            logger.error(f"Failed to pin worker to core {core_id}: {e}")

    if not input_queue or not output_queue:
        logger.error("Worker: Input or output queue not available")
        return {}

    logger.info(f"Monitoring worker started (core: {core_id})")
    processed_groups = 0
    
    # The loop will run until it receives a sentinel value (None)
    while True:
        try:
            queue_item = input_queue.get(timeout=1.0)  # Add timeout to avoid hanging
        except Exception:
            logger.debug("Worker: Queue get timeout, checking for sentinel")
            continue
            
        if queue_item is None:  # Sentinel value received
            logger.info(f"Worker: Received sentinel, processed {processed_groups} groups")
            input_queue.put(None)  # Propagate sentinel for other workers
            break

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
        output_queue.put(local_results)
        logger.debug(f"Worker: Put processed group {processed_groups} into output queue (total memory: {group_total_memory} KB)")

    logger.info(f"Monitoring worker finished, processed {processed_groups} groups")
    return {}


class BuildRunner:
    """
    Encapsulates the state and logic for a single build and monitor run.
    """

    def __init__(
        self,
        project_config: ProjectConfig,
        parallelism_level: int,
        monitoring_interval: float,
        log_dir: Path,
        collector_type: str,
        skip_pre_clean: bool,
        # --- New Simplified Scheduling Config ---
        scheduling_policy: str,
        manual_build_cores: str,
        manual_monitoring_cores: str,
        monitor_core_id: int,
    ):
        self.project_config = project_config
        self.parallelism_level = parallelism_level
        self.monitoring_interval = monitoring_interval
        self.log_dir = log_dir
        self.collector_type = collector_type
        self.skip_pre_clean = skip_pre_clean
        # --- New Scheduling Params ---
        self.scheduling_policy = scheduling_policy
        self.manual_build_cores = manual_build_cores
        self.manual_monitoring_cores = manual_monitoring_cores
        self.monitor_core_id = monitor_core_id

        # --- Initialize state variables ---
        self.current_timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.taskset_available = shutil.which("taskset") is not None
        self.run_context: Optional[RunContext] = None
        self.collector: Optional[AbstractMemoryCollector] = None
        self.build_process: Optional[subprocess.Popen] = None
        self.log_files: Dict[str, IO[Any]] = {}
        self.shutdown_requested = threading.Event()
        self.build_command_prefix: str = ""

        # --- New: Store prepared command and executable ---
        self.final_build_command: Optional[str] = None
        self.executable_shell: Optional[str] = None

        # --- New state for multi-process monitoring ---
        self.monitoring_worker_processes: List[multiprocessing.Process] = []
        self.producer_thread: Optional[threading.Thread] = None
        self.input_queue: Optional[Any] = None
        self.output_queue: Optional[Any] = None
        self.monitoring_cores: List[int] = []
        self.num_workers = 0
        self.results: Optional[MonitoringResults] = None

        # --- Signal handling state ---
        self._original_sigint_handler = None
        self._original_sigterm_handler = None
        self._signal_handlers_set = False

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for this BuildRunner instance."""
        try:
            # Store original handlers so we can restore them later
            self._original_sigint_handler = signal.signal(signal.SIGINT, _global_signal_handler)
            self._original_sigterm_handler = signal.signal(signal.SIGTERM, _global_signal_handler)
            self._signal_handlers_set = True
            logger.debug("Signal handlers set up for BuildRunner instance")
        except Exception as e:
            logger.warning(f"Failed to set up signal handlers: {e}")

    def _cleanup_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if not self._signal_handlers_set:
            return
        
        try:
            if self._original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            if self._original_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            logger.debug("Signal handlers restored for BuildRunner instance")
        except Exception as e:
            logger.warning(f"Failed to restore signal handlers: {e}")
        finally:
            self._signal_handlers_set = False

    def run(self) -> None:
        """
        Executes the entire build and monitor lifecycle.
        """
        # Register this instance for signal handling
        with _active_runners_lock:
            _active_runners[id(self)] = self
        
        # Set up signal handlers
        self._setup_signal_handlers()
        
        try:
            self.setup()
            if self.run_context:
                self._execute_build_step()
                self._wait_and_report()
        except Exception as e:
            logger.error(f"An error occurred during the run: {e}", exc_info=True)
        finally:
            self.teardown()
            # Clean up signal handlers
            self._cleanup_signal_handlers()
            # Unregister this instance
            with _active_runners_lock:
                _active_runners.pop(id(self), None)

    def setup(self) -> None:
        """
        Prepares the environment for the build and monitoring.
        """
        # --- New Unified CPU Allocation Planning ---
        cpu_plan = process_utils.plan_cpu_allocation(
            policy=self.scheduling_policy,
            j_level=self.parallelism_level,
            manual_build_cores_str=self.manual_build_cores,
            manual_monitor_cores_str=self.manual_monitoring_cores,
            main_monitor_core=self.monitor_core_id,
        )

        self.build_command_prefix = cpu_plan.build_command_prefix
        self.monitoring_cores = cpu_plan.monitoring_cores

        logger.info(f"CPU allocation for build: {cpu_plan.build_cores_desc}")
        logger.info(
            f"CPU allocation for monitoring workers: {cpu_plan.monitoring_cores_desc}"
        )

        # --- Prepare the final build command once ---
        self.final_build_command, self.executable_shell = process_utils.prepare_full_build_command(
            main_command_template=self.project_config.build_command_template,
            j_level=self.parallelism_level,
            taskset_prefix=self.build_command_prefix,
            setup_command=self.project_config.setup_command_template,
        )

        monitor_script_pinned_to_core_info = f"Core {self.monitor_core_id}"

        self.run_context = self._create_run_context(
            cpu_plan.build_cores_desc, monitor_script_pinned_to_core_info
        )
        self._open_log_files()
        self._log_run_prologue()

        if not self.skip_pre_clean:
            self._execute_clean_step()

        self.collector = self._initialize_collector()

        # Initialize queues for multiprocessing
        manager = multiprocessing.Manager()
        self.input_queue = manager.Queue()
        self.output_queue = manager.Queue()

    def teardown(self) -> None:
        """
        Cleans up all resources after the run.
        """
        logger.info("Tearing down monitoring resources...")
        if self.collector:
            self.collector.stop()  # Explicitly stop the collector

        if self.producer_thread and self.producer_thread.is_alive():
            self.producer_thread.join(timeout=5)
            if self.producer_thread.is_alive():
                logger.warning("Producer thread did not terminate gracefully.")

        # Terminate monitoring worker processes
        if self.monitoring_worker_processes:
            logger.info("Terminating monitoring worker processes...")
            for process in self.monitoring_worker_processes:
                if process.is_alive():
                    process.terminate()
            
            # Wait for processes to terminate
            for process in self.monitoring_worker_processes:
                process.join(timeout=5)
                if process.is_alive():
                    logger.warning(f"Worker process {process.pid} did not terminate gracefully, killing it.")
                    process.kill()
                    process.join()

        if self.build_process:
            self._terminate_process_tree(self.build_process.pid, "build process")

        self._close_log_files()
        logger.info("Teardown complete.")

    def _create_run_context(
        self, build_cores_target_str: str, monitor_script_pinned_to_core_info: str
    ) -> RunContext:
        """Creates the RunContext for this monitoring task."""
        if not self.final_build_command:
            raise ValueError("Final build command not prepared. Call setup() first.")

        run_paths = self._generate_run_paths(
            self.log_dir,
            self.project_config.name,
            self.parallelism_level,
            self.collector_type,
            self.current_timestamp_str,
        )

        return RunContext(
            project_name=self.project_config.name,
            project_dir=self.project_config.dir,
            process_pattern=self.project_config.process_pattern,
            actual_build_command=self.final_build_command,
            parallelism_level=self.parallelism_level,
            monitoring_interval=self.monitoring_interval,
            collector_type=self.collector_type,
            current_timestamp_str=self.current_timestamp_str,
            taskset_available=self.taskset_available,
            build_cores_target_str=build_cores_target_str,
            monitor_script_pinned_to_core_info=monitor_script_pinned_to_core_info,
            monitor_core_id=self.monitor_core_id,
            paths=run_paths,
        )

    def _initialize_collector(self) -> AbstractMemoryCollector:
        """Initializes and returns the appropriate memory collector."""
        if not self.run_context:
            raise ValueError("RunContext not initialized.")

        # This is a simplified factory logic. A more robust implementation
        # might use a dictionary mapping collector_type to class.
        if self.collector_type == "pss_psutil":
            from .memory_collectors.pss_psutil_collector import PssPsutilCollector

            collector_class = PssPsutilCollector
        elif self.collector_type == "rss_pidstat":
            from .memory_collectors.rss_pidstat_collector import RssPidstatCollector

            collector_class = RssPidstatCollector
        else:
            raise ValueError(f"Unknown collector type: {self.collector_type}")

        return collector_class(
            process_pattern=self.run_context.process_pattern,
            monitoring_interval=self.run_context.monitoring_interval,
            # Pass collector-specific arguments
            pidstat_stderr_file=self.run_context.paths.collector_aux_log_file,
            collector_cpu_core=self.run_context.monitor_core_id,
            taskset_available=self.run_context.taskset_available,
        )

    def _execute_clean_step(self) -> None:
        """Executes the pre-build clean command if defined."""
        if not self.run_context:
            raise ValueError("RunContext not initialized.")

        clean_command_template = self.project_config.clean_command_template
        setup_command_template = self.project_config.setup_command_template
        if not clean_command_template:
            logger.info("No clean command defined. Skipping pre-clean.")
            return

        logger.info("--- Executing Pre-build Clean Step ---")
        final_clean_command, executable = process_utils.prepare_command_with_setup(
            clean_command_template, setup_command_template
        )

        return_code, stdout, stderr = process_utils.run_command(
            final_clean_command,
            self.project_config.dir,
            shell=True,
            executable_shell=executable,
        )

        log_file = self.log_files.get("clean_log")
        if log_file:
            log_file.write("--- Clean Command Log ---\n")
            log_file.write(f"Command: {final_clean_command}\n")
            log_file.write(f"Exit Code: {return_code}\n\n")
            log_file.write("--- STDOUT ---\n")
            log_file.write(stdout)
            log_file.write("\n--- STDERR ---\n")
            log_file.write(stderr)
            log_file.flush()

        if return_code != 0:
            logger.error("Clean command failed. See clean log for details.")
        else:
            logger.info("Clean command executed successfully.")
        logger.info("--- Finished Pre-build Clean Step ---")

    def _execute_build_step(self) -> None:
        """Starts the build process and the monitoring thread."""
        if not self.run_context or not self.collector:
            raise ValueError("RunContext or Collector not initialized.")
        
        if not self.final_build_command:
            raise ValueError("Final build command not prepared.")

        logger.info("--- Kicking off build process and monitoring thread ---")

        self.build_process = subprocess.Popen(
            self.final_build_command,
            cwd=self.run_context.project_dir,
            stdout=self.log_files["build_stdout"],
            stderr=self.log_files["build_stderr"],
            shell=True,
            executable=self.executable_shell,  # This correctly passes /bin/bash if needed
        )
        logger.info(
            f"Build process started with PID: {self.build_process.pid} in directory {self.run_context.project_dir}"
        )

        # Set the build PID for the collector and start it
        self.collector.build_process_pid = self.build_process.pid
        # Also update the RunContext with the build process PID
        self.run_context.build_process_pid = self.build_process.pid
        self.collector.start()

        # --- Start Producer-Consumer Framework ---
        # Determine the number of worker processes. Default to the number of assigned cores,
        # or 1 if no specific cores are assigned (shared mode).
        self.num_workers = len(self.monitoring_cores) if self.monitoring_cores else 1
        worker_cores = (
            self.monitoring_cores
            if self.monitoring_cores
            else [None] * self.num_workers
        )

        logger.info(f"Starting {self.num_workers} monitoring worker process(es).")

        # Create and start the monitoring worker processes manually
        # We don't use Pool.map_async because we need long-running processes
        from functools import partial
        import multiprocessing

        primary_metric_field = self.collector.get_primary_metric_field()
        self.monitoring_worker_processes = []
        
        for i, core_id in enumerate(worker_cores):
            worker_func = partial(
                _monitoring_worker_entry, 
                core_id, 
                self.input_queue, 
                self.output_queue, 
                primary_metric_field
            )
            process = multiprocessing.Process(target=worker_func)
            process.start()
            self.monitoring_worker_processes.append(process)
            logger.debug(f"Started monitoring worker process {i+1} (PID: {process.pid}, core: {core_id})")

        # The data collection and queuing logic is now in the producer loop,
        # running in a separate thread.
        self.producer_thread = threading.Thread(target=self._producer_loop)
        self.producer_thread.start()

    def _wait_and_report(self) -> None:
        """
        Waits for the build to complete, collects results from workers, and reports.
        """
        if (
            not self.build_process
            or not self.run_context
            or not self.input_queue
            or not self.output_queue
        ):
            raise ValueError("Build process or queues not initialized.")

        build_exit_code = None
        logger.info("Waiting for build process to complete...")
        while build_exit_code is None:
            if self.shutdown_requested.is_set():
                logger.warning(
                    "Shutdown requested by signal. Terminating build process..."
                )
                self._terminate_process_tree(self.build_process.pid, "build process")
                build_exit_code = self.build_process.wait()
                logger.info(
                    f"Terminated build process exited with code: {build_exit_code}"
                )
                break  # Exit the waiting loop

            try:
                build_exit_code = self.build_process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                continue  # Loop again to check shutdown_requested

        logger.info(f"Build process finished with exit code: {build_exit_code}")

        if self.collector:
            logger.info("Stopping collector to signal producer thread to finish...")
            self.collector.stop()

        if self.producer_thread:
            self.producer_thread.join()

        # Signal monitoring workers to terminate and wait for them
        if self.monitoring_worker_processes:
            logger.info("All data queued. Signaling monitoring workers to terminate.")
            # 使用非阻塞方式发送终止信号，避免死锁
            for i in range(self.num_workers):
                try:
                    self.input_queue.put(None, timeout=1.0)  # 使用超时避免无限阻塞
                except Exception as e:
                    logger.warning(f"Failed to send termination signal to worker {i+1}, forcing termination: {e}")
                    break

            # Wait for all worker processes to finish
            for process in self.monitoring_worker_processes:
                process.join(timeout=10)
                if process.is_alive():
                    logger.warning(f"Worker process {process.pid} did not finish gracefully.")
            logger.info("All monitoring workers have terminated.")

        all_samples_data = []
        final_category_pid_set: Dict[str, set] = {}
        final_category_peak_sum: Dict[str, float] = {}
        group_memory_snapshots = []

        # Check output queue - 修复竞态条件
        output_items_count = 0
        logger.info("Starting to process output queue...")
        # 使用更安全的队列处理方式，避免竞态条件
        while True:
            try:
                # 使用非阻塞get，避免empty()和get()之间的竞态条件
                worker_result = self.output_queue.get(timeout=0.1)
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
                # 队列为空或超时，退出循环
                break

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

            self.results = MonitoringResults(
                all_samples_data=df.to_dict("records"),
                category_stats={},
                peak_overall_memory_kb=int(peak_overall_memory_kb),
                peak_overall_memory_epoch=int(peak_overall_memory_epoch),
                category_peak_sum={
                    k: int(v) for k, v in final_category_peak_sum.items()
                },
                category_pid_set=final_category_pid_set,
            )
        else:
            self.results = None  # No data collected

        summary_log = self.log_files.get("summary_log")
        if summary_log and self.results:
            summary_log.write(f"build_exit_code={build_exit_code}\n")
            summary_log.write(
                f"peak_overall_memory_kb={self.results.peak_overall_memory_kb}\n"
            )
            summary_log.write(
                f"peak_overall_memory_epoch={self.results.peak_overall_memory_epoch}\n\n"
            )
            summary_log.write("--- Category Peak Memory Usage ---\n")
            sorted_cats = sorted(
                self.results.category_peak_sum.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            for cat, peak_mem in sorted_cats:
                num_pids = len(self.results.category_pid_set.get(cat, set()))
                summary_log.write(f"{cat}: {peak_mem} KB ({num_pids} pids)\n")
            summary_log.flush()

        if build_exit_code != 0:
            logger.error(
                f"Build failed with exit code {build_exit_code}. Check logs in {self.run_context.paths.output_parquet_file.parent}."
            )

        if self.results and self.run_context:
            try:
                final_df = pd.DataFrame(self.results.all_samples_data)
                if not final_df.empty:
                    final_df.to_parquet(
                        self.run_context.paths.output_parquet_file, index=False
                    )
                    logger.info(
                        f"Memory data saved to {self.run_context.paths.output_parquet_file}"
                    )
            except Exception as e:
                logger.error(f"Failed to save Parquet file: {e}", exc_info=True)

    def _producer_loop(self) -> None:
        """
        The producer loop that runs in a separate thread.
        It reads samples from the collector and puts them onto the input queue
        for the worker processes to consume.
        """
        if not self.input_queue or not self.collector:
            return

        logger.info("Producer loop started. Reading samples from collector.")
        sample_count = 0
        for sample_group in self.collector.read_samples():
            sample_count += 1
            logger.debug(f"Producer: Received sample group {sample_count} with {len(sample_group)} samples")
            # Add a timestamp to the group for later aggregation
            timestamp = pd.Timestamp.now()
            # Pass data as a tuple to avoid modifying the original sample object
            self.input_queue.put((sample_group, timestamp))
            logger.debug(f"Producer: Put sample group {sample_count} into input queue")

        logger.info(f"Producer loop finished. Processed {sample_count} sample groups. All samples have been queued.")



    @staticmethod
    def _generate_run_paths(
        log_dir_base: Path,
        project_name: str,
        parallelism_level: int,
        collector_type: str,
        timestamp: str,
    ) -> RunPaths:
        """Generates all necessary output paths for a given run."""
        sanitized_project_name = "".join(
            c if c.isalnum() else "_" for c in project_name
        )
        run_name = f"{sanitized_project_name}_j{parallelism_level}_{collector_type}_{timestamp}"
        log_dir_for_run = log_dir_base / run_name
        log_dir_for_run.mkdir(parents=True, exist_ok=True)

        return RunPaths(
            output_parquet_file=log_dir_for_run / "memory_samples.parquet",
            output_summary_log_file=log_dir_for_run / "summary.log",
            collector_aux_log_file=log_dir_for_run / "collector_aux.log",
        )

    def _open_log_files(self) -> None:
        """Opens all log files required for the run."""
        if not self.run_context:
            raise ValueError("RunContext not initialized.")
        paths = self.run_context.paths
        log_dir = paths.output_parquet_file.parent
        
        # 使用更安全的文件打开方式，避免文件描述符泄漏
        opened_files = {}
        try:
            file_paths = {
                "summary_log": paths.output_summary_log_file,
                "build_stdout": log_dir / "build_stdout.log",
                "build_stderr": log_dir / "build_stderr.log", 
                "clean_log": log_dir / "clean.log",
                "metadata_log": log_dir / "metadata.log",
            }
            
            for name, path in file_paths.items():
                try:
                    opened_files[name] = open(path, "w", encoding="utf-8")
                except Exception as e:
                    logger.error(f"Failed to open {name} at {path}: {e}")
                    # 关闭已打开的文件
                    for f in opened_files.values():
                        try:
                            f.close()
                        except Exception:
                            pass
                    raise
            
            self.log_files = opened_files
        except Exception:
            # 确保所有已打开的文件都被关闭
            for f in opened_files.values():
                try:
                    f.close()
                except Exception:
                    pass
            raise

    def _close_log_files(self) -> None:
        """Closes all opened log files."""
        for f in self.log_files.values():
            try:
                f.close()
            except Exception as e:
                logger.error(f"Failed to close log file: {e}")
        self.log_files.clear()

    def _log_run_prologue(self) -> None:
        """Writes metadata and initial configuration to the logs."""
        if not self.run_context:
            raise ValueError("RunContext not initialized.")
        metadata_log = self.log_files.get("metadata_log")
        if metadata_log:
            for field, value in dataclasses.asdict(self.run_context).items():
                metadata_log.write(f"{field}: {value}\n")
            metadata_log.flush()
        logger.info(
            f"Logs for this run will be in {self.run_context.paths.output_parquet_file.parent}"
        )

    def _terminate_process_tree(self, pid: int, name: str) -> None:
        """Gracefully terminates a process and all its children."""
        try:
            parent = psutil.Process(pid)
            
            # 多次尝试获取和终止进程树，处理动态产生的子进程
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # 重新获取子进程列表，处理动态产生的子进程
                    children = parent.children(recursive=True)
                    procs_to_terminate = [parent] + children
                    
                    if attempt == 0:
                        logger.info(f"Terminating {name} (PID: {pid}) and its {len(children)} children...")
                    else:
                        logger.info(f"Attempt {attempt + 1}: Found {len(children)} children to terminate")

                    # 首先尝试优雅终止
                    for p in procs_to_terminate:
                        try:
                            if p.is_running():
                                p.terminate()
                        except psutil.NoSuchProcess:
                            continue  # Process already ended

                    # 等待进程终止
                    _, alive = psutil.wait_procs(procs_to_terminate, timeout=2)
                    
                    # 强制杀死仍然存活的进程
                    if alive:
                        logger.warning(f"Force killing {len(alive)} remaining processes")
                        for p in alive:
                            try:
                                if p.is_running():
                                    p.kill()
                            except psutil.NoSuchProcess:
                                continue  # Race condition: process ended before kill
                        
                        # 最后等待
                        psutil.wait_procs(alive, timeout=1)
                    
                    # 检查是否还有新的子进程产生
                    remaining_children = parent.children(recursive=True)
                    if not remaining_children or attempt == max_attempts - 1:
                        break
                        
                except psutil.NoSuchProcess:
                    # 父进程已经终止
                    break

        except psutil.NoSuchProcess:
            logger.info(f"Process {name} (PID: {pid}) already terminated.")
        except Exception as e:
            logger.error(f"Error terminating process tree for {name}: {e}")
