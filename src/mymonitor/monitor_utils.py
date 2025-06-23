"""
Core orchestration logic for monitoring build processes.

This module provides the main `run_and_monitor_build` function which coordinates
the entire process of setting up, executing, monitoring, and cleaning up a build.
It acts as the "engine" of the application, bringing together process management,
data collection, and reporting.

Key responsibilities include:
- Setting up a `RunContext` with all necessary information for a single run.
- Executing pre-build and post-build clean commands.
- Launching the main build command in a subprocess.
- Starting a memory collector in a separate thread or process.
- Reading stdout/stderr from the build process in non-blocking threads.
- Running the main monitoring loop to collect and process memory samples.
- Aggregating results into a `MonitoringResults` object.
- Writing the final summary report and raw data to disk.
- Ensuring all subprocesses are properly terminated, even on error or interruption.
"""

import logging
import polars as pl
import psutil
import queue
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Tuple, Set, Union

# Import local modules
from .data_models import MonitoringResults, ProjectConfig, RunContext, RunPaths
from .memory_collectors.base import AbstractMemoryCollector
from .memory_collectors.pss_psutil_collector import PssPsutilCollector
from .memory_collectors.rss_pidstat_collector import RssPidstatCollector
from .process_utils import (
    determine_build_cpu_affinity,
    get_process_category,
    run_command,
)

logger = logging.getLogger(__name__)

# --- Module Globals ---
# These globals hold references to the currently active subprocesses. They are
# essential for the signal handler (`main.py`) to perform a graceful shutdown
# by terminating these processes from anywhere in the application.
current_build_proc: Optional[subprocess.Popen] = None
active_memory_collector: Optional[AbstractMemoryCollector] = None


# --- Public API ---


def run_and_monitor_build(
    project_config: ProjectConfig,
    parallelism_level: int,
    monitoring_interval: int,
    log_dir: Path,
    collector_type: str,
    skip_pre_clean: bool,
    monitor_core_id_for_collector_and_build_avoidance: Optional[int],
    build_cpu_cores_policy: str,
    specific_build_cores_str: Optional[str],
    monitor_script_pinned_to_core_info: str,
) -> None:
    """
    Main orchestration function for a single build and monitor run.

    This function manages the entire lifecycle for monitoring one project at one
    parallelism level. It ensures that all steps (setup, execution, monitoring,
    reporting, cleanup) are performed in the correct order and that resources
    are properly handled via a `try...finally` block.

    Args:
        project_config: Configuration for the project to build.
        parallelism_level: The number of parallel jobs for the build (e.g., -j value).
        monitoring_interval: The interval in seconds for memory sampling.
        log_dir: The directory to save all output files for this run.
        collector_type: The type of memory collector to use ('pss_psutil' or 'rss_pidstat').
        skip_pre_clean: If True, the pre-build clean step is skipped.
        monitor_core_id_for_collector_and_build_avoidance: The CPU core ID reserved for the monitor.
        build_cpu_cores_policy: The policy for assigning CPU cores to the build process.
        specific_build_cores_str: A string defining specific cores for the build.
        monitor_script_pinned_to_core_info: A descriptive string about monitor pinning.
    """
    global current_build_proc, active_memory_collector

    # --- 1. Setup and Context Creation ---
    project_name = project_config.name
    project_dir = project_config.dir
    build_command_template = project_config.build_command_template
    process_pattern = project_config.process_pattern
    setup_command_template = project_config.setup_command_template
    current_timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    run_paths = _generate_run_paths(
        log_dir, project_name, parallelism_level, collector_type, current_timestamp_str
    )

    base_build_cmd = build_command_template.replace("<N>", str(parallelism_level))
    actual_build_command = (
        f"{setup_command_template} && {base_build_cmd}"
        if setup_command_template
        else base_build_cmd
    )

    taskset_available = "taskset" in run_command("which taskset", Path("."))[1]
    num_total_cores = psutil.cpu_count()

    build_command_prefix, build_cores_target_str = determine_build_cpu_affinity(
        build_cpu_cores_policy,
        specific_build_cores_str,
        monitor_core_id_for_collector_and_build_avoidance,
        taskset_available,
        num_total_cores,
    )

    run_context = RunContext(
        project_name=project_name,
        project_dir=project_dir,
        process_pattern=process_pattern,
        actual_build_command=actual_build_command,
        parallelism_level=parallelism_level,
        monitoring_interval=monitoring_interval,
        collector_type=collector_type,
        current_timestamp_str=current_timestamp_str,
        taskset_available=taskset_available,
        build_cores_target_str=build_cores_target_str,
        monitor_script_pinned_to_core_info=monitor_script_pinned_to_core_info,
        monitor_core_id=monitor_core_id_for_collector_and_build_avoidance,
        paths=run_paths,
    )

    summary_log_header_content = _prepare_initial_summary_log_header(run_context)
    with open(run_context.paths.output_summary_log_file, "w") as f_summary:
        f_summary.write("\n".join(summary_log_header_content) + "\n\n")

    local_active_memory_collector = _create_memory_collector(run_context)
    if not local_active_memory_collector:
        return

    if not skip_pre_clean:
        _execute_clean_step(run_context, project_config, "Pre-Build Clean")
    else:
        logger.info(
            "Skipping pre-build clean step as requested by --no-pre-clean flag."
        )

    # --- 2. Execution and Monitoring ---
    build_exit_code: int = -1
    start_time_seconds: Optional[float] = None
    monitoring_results: Optional[MonitoringResults] = None
    primary_metric_to_track: Optional[str] = None

    # A single thread-safe queue for all log messages from stdout/stderr.
    log_queue: queue.Queue[Optional[Tuple[str, str]]] = queue.Queue()

    # Threads for reading stdout/stderr and a new thread for writing logs.
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None
    log_writer_thread: Optional[threading.Thread] = None

    try:
        metric_fields_header = local_active_memory_collector.get_metric_fields()
        primary_metric_to_track = (
            metric_fields_header[0] if metric_fields_header else None
        )
        local_active_memory_collector.start()

        # Start a dedicated thread to consume from the log queue and write to a file.
        # This serializes all log writes and prevents race conditions.
        log_writer_thread = threading.Thread(
            target=_log_writer,
            args=(log_queue, run_context.paths.output_summary_log_file),
            daemon=True,
        )
        log_writer_thread.start()

        # --- Prepare the final command for execution ---
        # Unify the command execution logic to always use a shell.
        # This is more robust and handles cases where the build_command_template
        # itself is complex (e.g., contains '&&' or 'cd'), not just the setup command.
        # The `taskset` prefix is applied to the shell, which then executes the
        # entire actual_build_command string. This ensures CPU affinity is
        # correctly applied to the entire build process.
        final_command_to_execute = shlex.split(build_command_prefix) + [
            "/bin/bash",
            "-c",
            actual_build_command,
        ]

        # Write build headers via the log queue to prevent file contention.
        log_queue.put(("[MONITOR]", "\n--- Build Command Execution ---"))
        log_queue.put(("[MONITOR]", f"Command: {shlex.join(final_command_to_execute)}"))
        log_queue.put(("[MONITOR]", "--- Build Output ---"))

        start_time_seconds = time.monotonic()
        current_build_proc = subprocess.Popen(
            final_command_to_execute,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,  # shell=False is safer; shell logic is handled above.
            bufsize=1,  # Line-buffered.
        )
        run_context.build_process_pid = current_build_proc.pid

        # Pass the build process PID to the collector if it needs it.
        if local_active_memory_collector and hasattr(
            local_active_memory_collector, "build_process_pid"
        ):
            local_active_memory_collector.build_process_pid = current_build_proc.pid

        active_memory_collector = local_active_memory_collector

        # Start non-blocking reader threads to feed the central log queue.
        stdout_thread = _start_stream_reader_thread(
            current_build_proc.stdout, "[STDOUT]", log_queue
        )
        stderr_thread = _start_stream_reader_thread(
            current_build_proc.stderr, "[STDERR]", log_queue
        )

        # This is the main blocking call that runs the monitoring loop.
        monitoring_results = _perform_monitoring_loop(
            local_active_memory_collector,
            current_build_proc,
            metric_fields_header,
            primary_metric_to_track,
        )

        if current_build_proc:
            build_exit_code = current_build_proc.wait()

    finally:
        # --- 3. Cleanup and Finalization ---
        # This block ensures that all resources are cleaned up, even if errors occur.
        if local_active_memory_collector:
            local_active_memory_collector.stop()
        if current_build_proc and current_build_proc.poll() is None:
            logger.warning("Build process did not exit. Terminating.")
            current_build_proc.terminate()
            current_build_proc.wait()

        end_time_seconds = time.monotonic()
        duration_seconds_float = (
            (end_time_seconds - start_time_seconds) if start_time_seconds else 0
        )
        formatted_duration = time.strftime(
            "%H:%M:%S", time.gmtime(duration_seconds_float)
        )

        # Wait for producer threads (stdout/stderr readers) to finish.
        if stdout_thread:
            stdout_thread.join()
        if stderr_thread:
            stderr_thread.join()

        # Signal the consumer (log writer) that there's no more data and wait for it.
        if log_writer_thread:
            log_queue.put(None)  # Sentinel value to stop the writer.
            log_writer_thread.join()

        if monitoring_results:
            # Compile and write the final text summary.
            _compile_and_write_final_summary_and_aux_logs(
                context=run_context,
                results=monitoring_results,
                build_exit_code=build_exit_code,
                duration_float=duration_seconds_float,
                duration_formatted=formatted_duration,
                primary_metric=primary_metric_to_track,
            )
            # Write the raw collected data to a Parquet file for analysis.
            _write_monitoring_data_to_parquet(
                results=monitoring_results, context=run_context
            )

        _execute_clean_step(run_context, project_config, "Post-Build Clean")


def cleanup_processes() -> None:
    """
    Cleans up any running global subprocesses.

    This function is designed to be called by a signal handler for graceful
    shutdown (e.g., on Ctrl+C). It stops the active memory collector and
    terminates the build process.
    """
    global current_build_proc, active_memory_collector
    if active_memory_collector:
        logger.info("Signal received, stopping memory collector...")
        active_memory_collector.stop()
    if current_build_proc and current_build_proc.poll() is None:
        logger.info("Signal received, terminating build process...")
        current_build_proc.terminate()
        current_build_proc.wait()
    logger.info("Cleanup finished.")


# --- Core Monitoring Logic ---


def _perform_monitoring_loop(
    active_memory_collector: AbstractMemoryCollector,
    current_build_proc: Optional[subprocess.Popen],
    metric_fields_header: List[str],
    primary_metric_to_track: Optional[str],
) -> MonitoringResults:
    """
    The core loop that consumes data from a memory collector and aggregates results.

    This function iterates over the data yielded by the collector, categorizes each
    process sample, and aggregates statistics in real-time. It is designed to be
    robust against shutdown race conditions by ensuring the collector is properly
    stopped and its final output is consumed.

    Args:
        active_memory_collector: The active memory collector instance.
        current_build_proc: The subprocess object for the running build.
        metric_fields_header: A list of metric field names from the collector.
        primary_metric_to_track: The main metric to use for peak calculations.

    Returns:
        A MonitoringResults object containing all collected and aggregated data.
    """
    all_samples_data: List[Dict[str, Any]] = []
    category_stats: Dict[str, Dict[str, Any]] = {}
    peak_overall_memory_kb: int = 0
    peak_overall_memory_epoch: int = 0
    category_peak_sum: Dict[str, int] = {}
    category_pid_set: Dict[str, Set[str]] = {}

    logger.info("Starting monitoring loop...")
    stop_requested = False

    # The loop terminates when the collector's generator is exhausted.
    for samples_in_interval in active_memory_collector.read_samples():
        epoch_now = int(time.time())
        current_interval_sum_kb = 0
        current_category_sums: Dict[str, int] = {}

        # Process each individual process sample from the interval.
        for sample in samples_in_interval:
            major_cat, minor_cat = get_process_category(
                sample.command_name, sample.full_command
            )
            primary_metric_value = (
                sample.metrics.get(primary_metric_to_track, 0)
                if primary_metric_to_track
                else 0
            )
            current_interval_sum_kb += primary_metric_value

            # Create the raw data row for Parquet export with a consistent schema.
            row = {
                "Timestamp_epoch": epoch_now,
                "Record_Type": "PROCESS",
                "PID": sample.pid,
                "Major_Category": major_cat,
                "Minor_Category": minor_cat,
                "Command_Name": sample.command_name,
                "Full_Command": sample.full_command,
                **{k: v for k, v in sample.metrics.items()},
            }
            all_samples_data.append(row)

            # Track the peak memory usage for a single process within each category.
            if primary_metric_to_track:
                category_key = f"{major_cat}_{minor_cat}"
                if (
                    category_key not in category_stats
                    or primary_metric_value
                    > category_stats[category_key].get(primary_metric_to_track, 0)
                ):
                    category_stats[category_key] = {
                        primary_metric_to_track: primary_metric_value,
                        "pid": sample.pid,
                        "cmd": sample.command_name,
                        "full_cmd": sample.full_command,
                    }

            # Track unique PIDs per category.
            category_key_for_pid = f"{major_cat}_{minor_cat}"
            category_pid_set.setdefault(category_key_for_pid, set()).add(sample.pid)

            # Track sum for each category in this interval.
            current_category_sums[category_key_for_pid] = (
                current_category_sums.get(category_key_for_pid, 0)
                + primary_metric_value
            )

        # Update overall and category peak sums for the interval.
        if current_interval_sum_kb > peak_overall_memory_kb:
            peak_overall_memory_kb = current_interval_sum_kb
            peak_overall_memory_epoch = epoch_now

        for cat_key, cat_sum in current_category_sums.items():
            if cat_sum > category_peak_sum.get(cat_key, 0):
                category_peak_sum[cat_key] = cat_sum

        # Add a summary record for the entire interval.
        all_samples_data.append(
            _create_summary_row(
                epoch_now,
                "ALL_SUM",
                "All",
                "All",
                current_interval_sum_kb,
                metric_fields_header,
            )
        )

        # The `read_samples` generator yields one last time after stop() is called.
        # This ensures we consume the final data before the loop terminates.
        if (
            not stop_requested
            and current_build_proc
            and current_build_proc.poll() is not None
        ):
            logger.info(
                "Build process finished. Requesting final sample from collector."
            )
            active_memory_collector.stop()
            stop_requested = True

    logger.info("Monitoring loop finished.")
    return MonitoringResults(
        all_samples_data=all_samples_data,
        category_stats=category_stats,
        peak_overall_memory_kb=peak_overall_memory_kb,
        peak_overall_memory_epoch=peak_overall_memory_epoch,
        category_peak_sum=category_peak_sum,
        category_pid_set=category_pid_set,
    )


# --- Setup & Preparation Helpers ---


def _generate_run_paths(
    log_dir: Path,
    project_name: str,
    parallelism_level: int,
    collector_type: str,
    current_timestamp_str: str,
) -> RunPaths:
    """Generates all necessary output paths for a monitoring run."""
    base_filename_part = f"{project_name}_j{parallelism_level}_mem_{collector_type}_{current_timestamp_str}"
    output_parquet_filename = f"{base_filename_part}.parquet"
    output_summary_log_filename = f"{base_filename_part}_summary.log"
    return RunPaths(
        output_parquet_file=log_dir / output_parquet_filename,
        output_summary_log_file=log_dir / output_summary_log_filename,
        collector_aux_log_file=log_dir / f"{base_filename_part}_collector_aux.log",
    )


def _create_memory_collector(context: RunContext) -> Optional[AbstractMemoryCollector]:
    """
    Initializes and returns a memory collector instance based on the run context.

    Args:
        context: The context object for the current run.

    Returns:
        An initialized memory collector instance, or None if initialization fails.
    """
    global active_memory_collector
    collector_kwargs: Dict[str, Any] = {
        "collector_cpu_core": context.monitor_core_id,
        "taskset_available": context.taskset_available,
        "pidstat_stderr_file": context.paths.collector_aux_log_file,
    }
    try:
        if context.collector_type == "pss_psutil":
            active_memory_collector = PssPsutilCollector(
                context.process_pattern, context.monitoring_interval, **collector_kwargs
            )
        elif context.collector_type == "rss_pidstat":
            active_memory_collector = RssPidstatCollector(
                context.process_pattern, context.monitoring_interval, **collector_kwargs
            )
        else:
            logger.error(f"Unknown memory collector type: {context.collector_type}")
            return None
    except Exception as e:
        logger.error(
            f"Failed to initialize collector '{context.collector_type}': {e}",
            exc_info=True,
        )
        return None
    return active_memory_collector


def _prepare_initial_summary_log_header(context: RunContext) -> List[str]:
    """Prepares the header content for the summary log file."""
    header_content: List[str] = [
        "=" * 80,
        f"Project: {context.project_name}",
        f"Parallelism: -j{context.parallelism_level}",
        f"Memory Metric Collector: {context.collector_type.upper()}",
        f"Source Directory: {context.project_dir.resolve()}",
        f"Actual Build Command: {context.actual_build_command}",
        f"Process Pattern (for collector): {context.process_pattern}",
        f"Monitoring Interval (approx): {context.monitoring_interval} seconds",
        f"Run Timestamp: {context.current_timestamp_str} ({time.strftime('%Y-%m-%d %H:%M:%S')})",
        f"Output Parquet File: {context.paths.output_parquet_file.name}",
        f"Summary Log File: {context.paths.output_summary_log_file.name}",
    ]
    if context.collector_type == "rss_pidstat":
        header_content.append(
            f"Collector Aux Log (temporary, will be merged): {context.paths.collector_aux_log_file.name}"
        )
    header_content.extend(
        [
            f"Monitor Script Pinned to CPU Core: {context.monitor_script_pinned_to_core_info}",
            f"Build Processes CPU Cores Target: {context.build_cores_target_str}",
        ]
    )
    if (
        context.collector_type == "rss_pidstat"
        and context.monitor_core_id is not None
        and context.taskset_available
    ):
        header_content.append(
            f"Pidstat Collector Target CPU Core: {context.monitor_core_id} (via taskset)"
        )
    elif context.collector_type == "rss_pidstat":
        header_content.append(
            "Pidstat Collector Target CPU Core: Not Pinned (monitor_core_id not set or taskset unavailable)"
        )
    header_content.append("-" * 80)
    return header_content


def _prepare_actual_clean_command(
    setup_command_template: Optional[str], clean_command_template: str
) -> Tuple[str, Optional[str]]:
    """Prepares the actual clean command string and the executable shell if needed."""
    executable_shell: Optional[str] = None
    if setup_command_template:
        actual_clean_cmd = f"{setup_command_template} && {clean_command_template}"
        executable_shell = "/bin/bash"
    else:
        actual_clean_cmd = clean_command_template
    return actual_clean_cmd, executable_shell


# --- Execution & I/O Helpers ---


def _log_writer(log_queue: queue.Queue, summary_log_path: Path) -> None:
    """
    Consumes log messages from a queue and writes them to the summary file.

    This function runs in a dedicated thread. It waits for items to appear in
    the queue and writes them to the log file, ensuring all output is sequential
    and thread-safe. It exits when it receives a `None` sentinel value.

    Args:
        log_queue: The thread-safe queue holding log messages.
        summary_log_path: The path to the summary log file to write to.
    """
    try:
        with open(summary_log_path, "a", buffering=1) as f_summary:
            while True:
                item = log_queue.get()
                if item is None:  # Sentinel to indicate end of logging.
                    break
                prefix, line = item
                f_summary.write(f"{prefix} {line}\n")
    except Exception as e:
        logger.error(f"Error in log writer thread: {e}", exc_info=True)


def _start_stream_reader_thread(
    pipe: Optional[IO[str]],
    stream_name_prefix: str,
    log_queue: queue.Queue,
) -> Optional[threading.Thread]:
    """Creates and starts a daemon thread to read from a subprocess stream."""
    if not pipe:
        return None

    thread = threading.Thread(
        target=_stream_output,
        args=(pipe, stream_name_prefix, log_queue),
        daemon=True,
    )
    thread.start()
    return thread


def _stream_output(
    pipe: Optional[IO[str]], prefix: str, log_queue: queue.Queue
) -> None:
    """Reads lines from a subprocess pipe and puts them in a queue."""
    if not pipe:
        return
    try:
        for line in iter(pipe.readline, ""):
            line = line.strip()
            if line:
                logger.info(f"{prefix} {line}")  # Log to console for live feedback.
                # Put message into the queue for the writer thread.
                # Use a timeout to prevent indefinite blocking if the writer thread dies.
                try:
                    log_queue.put((prefix, line), timeout=5.0)
                except queue.Full:
                    logger.error(
                        "Log queue is full, writer thread may have crashed. Stopping log for this stream."
                    )
                    break  # Exit loop to allow this reader thread to terminate.
    except ValueError:
        # This can happen if the pipe is closed while reading.
        logger.warning(f"Stream for {prefix} closed unexpectedly.")
    except Exception as e:
        logger.error(f"Error reading stream for {prefix}: {e}", exc_info=True)
    finally:
        if pipe:
            pipe.close()


def _run_and_log_command_to_summary(
    context: RunContext,
    cmd_str: str,
    cmd_desc: str,
    shell_bool: bool,
    executable_shell: Optional[str] = None,
) -> int:
    """Runs a command and logs its execution details to the summary file."""
    cwd_path = context.project_dir
    summary_file = context.paths.output_summary_log_file
    logger.info(f"Executing {cmd_desc} command: {cmd_str} in {cwd_path}")

    # Ensure the output directory exists before writing the log file.
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, "a") as f_s:
        f_s.write(f"\n--- {cmd_desc} Command ---\n")
        f_s.write(f"Command: {cmd_str}\n")
    exit_code, stdout_val, stderr_val = run_command(
        cmd_str, cwd=cwd_path, shell=shell_bool, executable_shell=executable_shell
    )
    with open(summary_file, "a") as f_s:
        f_s.write(f"Exit Code: {exit_code}\n")
        # Process stdout/stderr line-by-line to add prefixes for consistency.
        if stdout_val:
            f_s.write("--- STDOUT ---\n")
            for line in stdout_val.strip().splitlines():
                f_s.write(f"[STDOUT] {line}\n")
        if stderr_val:
            f_s.write("--- STDERR ---\n")
            for line in stderr_val.strip().splitlines():
                f_s.write(f"[STDERR] {line}\n")
    return exit_code


def _execute_clean_step(
    context: RunContext, project_config: ProjectConfig, description: str
) -> None:
    """Prepares and executes a clean command, logging the process."""
    if not project_config.clean_command_template:
        return
    actual_clean_cmd, executable_shell = _prepare_actual_clean_command(
        project_config.setup_command_template, project_config.clean_command_template
    )
    _run_and_log_command_to_summary(
        context=context,
        cmd_str=actual_clean_cmd,
        cmd_desc=description,
        shell_bool=True,
        executable_shell=executable_shell,
    )


# --- Reporting & Finalization Helpers ---


def _compile_and_write_final_summary_and_aux_logs(
    context: RunContext,
    results: MonitoringResults,
    build_exit_code: int,
    duration_float: float,
    duration_formatted: str,
    primary_metric: Optional[str],
) -> None:
    """Compiles the final summary statistics and writes them to the log file and console."""
    final_summary_log_lines: List[str] = [
        "\n--- Build & Monitoring Summary ---",
        f"Total Build & Monitoring Duration: {duration_formatted} ({duration_float:.2f} seconds)",
        f"Final Build Exit Code: {build_exit_code}",
    ]

    if primary_metric:
        peak_time_str = (
            time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(results.peak_overall_memory_epoch)
            )
            if results.peak_overall_memory_epoch > 0
            else "N/A"
        )
        final_summary_log_lines.append(
            f"Peak Overall Memory ({primary_metric}): {results.peak_overall_memory_kb} KB (at approx. {peak_time_str})"
        )
    else:
        final_summary_log_lines.append(
            "Peak Overall Memory: N/A (no primary metric was tracked for sum)"
        )

    final_summary_log_lines.append(
        f"Peak Single Process Memory Usage (by category, based on {primary_metric or 'N/A'}):"
    )
    if results.category_stats:
        for category, stats in sorted(results.category_stats.items()):
            peak_val = stats.get(primary_metric, "N/A") if primary_metric else "N/A"
            pid_val = stats.get("pid", "N/A")
            cmd_val = stats.get("cmd", "N/A")
            full_cmd_val = stats.get("full_cmd", "N/A")
            final_summary_log_lines.append(
                f"  - {category}: {peak_val} KB (PID: {pid_val}, Command: {cmd_val})"
            )
            final_summary_log_lines.append(f"    Full Command: {full_cmd_val}")
    else:
        final_summary_log_lines.append(
            "  No category-specific single process peak memory data collected."
        )

    final_summary_log_lines.append(
        f"Peak Category Sum Memory Usage (sum of processes in category, based on {primary_metric or 'N/A'}):"
    )
    if results.category_peak_sum:
        for category, peak_sum_val in sorted(results.category_peak_sum.items()):
            final_summary_log_lines.append(f"  - {category}: {peak_sum_val} KB")
    else:
        final_summary_log_lines.append("  No category sum peak memory data collected.")

    final_summary_log_lines.append("Unique Process Counts (by category):")
    if results.category_pid_set:
        for category, pids in sorted(results.category_pid_set.items()):
            final_summary_log_lines.append(
                f"  - {category}: {len(pids)} unique process(es)"
            )
    else:
        final_summary_log_lines.append("  No unique process count data collected.")

    final_summary_log_lines.append("--- End Summary ---")

    with open(context.paths.output_summary_log_file, "a") as f_summary:
        f_summary.write("\n")
        # If the pidstat collector was used, merge its stderr log into the summary.
        if (
            context.collector_type == "rss_pidstat"
            and context.paths.collector_aux_log_file.exists()
            and context.paths.collector_aux_log_file.stat().st_size > 0
        ):
            f_summary.write("\n--- Collector Auxiliary Log Output ---\n")
            try:
                with open(context.paths.collector_aux_log_file, "r") as f_aux:
                    f_summary.write(f_aux.read())
                f_summary.write("--- End Collector Auxiliary Log Output ---\n\n")
                try:
                    context.paths.collector_aux_log_file.unlink()
                except OSError as e_del:
                    logger.warning(f"Could not delete collector aux log: {e_del}")
            except Exception as e_read_aux:
                f_summary.write(f"Error reading collector aux log: {e_read_aux}\n")

        for line in final_summary_log_lines:
            f_summary.write(line + "\n")

    # Also print the summary to the console for immediate feedback.
    for line in final_summary_log_lines:
        logger.info(line)


def _write_monitoring_data_to_parquet(
    results: MonitoringResults, context: RunContext
) -> None:
    """Writes the collected monitoring data to a Parquet file."""
    if not results.all_samples_data:
        logger.info("No data collected to write to Parquet file.")
        return
    try:
        df_to_save = pl.DataFrame(results.all_samples_data)
        output_file = context.paths.output_parquet_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_to_save.write_parquet(output_file)
        logger.info(f"Monitoring data successfully written to {output_file}")
    except Exception as e:
        logger.error(
            f"Failed to write Parquet file {context.paths.output_parquet_file}: {e}",
            exc_info=True,
        )


def _create_summary_row(
    epoch: int,
    record_type: str,
    major_category: str,
    minor_category: str,
    sum_value: Optional[int],
    metric_fields_header: List[str],
) -> Dict[str, Any]:
    """Helper to create a common structure for summary rows in Parquet data."""
    row: Dict[str, Any] = {
        "Timestamp_epoch": epoch,
        "Record_Type": record_type,
        "Major_Category": major_category,
        "Minor_Category": minor_category,
        "PID": None,
        "Command_Name": None,
        "Full_Command": None,
        "Sum_Value": sum_value,
    }
    # Ensure all metric columns exist, even if they are null for summary rows.
    for metric_name in metric_fields_header:
        row[metric_name] = None
    return row
