"""
Core orchestration logic for monitoring build processes.

This module provides the main `run_and_monitor_build` function which coordinates
the entire process of setting up, executing, monitoring, and cleaning up a build.
"""

import logging
import polars as pl  # Polars is only used for writing the final parquet file
import psutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Tuple, Set

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
    """Main orchestration function for a single build and monitor run."""

    global current_build_proc

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

    full_build_command_for_popen = base_build_cmd

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

    build_stdout_lines: List[str] = []
    build_stderr_lines: List[str] = []
    build_exit_code: int = -1
    start_time_seconds: Optional[float] = None
    monitoring_results: Optional[MonitoringResults] = None
    primary_metric_to_track: Optional[str] = None

    try:
        metric_fields_header = local_active_memory_collector.get_metric_fields()
        primary_metric_to_track = (
            metric_fields_header[0] if metric_fields_header else None
        )
        local_active_memory_collector.start()

        cmd_list_for_popen = (
            build_command_prefix.split() if build_command_prefix else []
        ) + full_build_command_for_popen.split()

        with open(run_paths.output_summary_log_file, "a") as f_summary:
            f_summary.write("--- Build Command Execution ---\n")
            f_summary.write(f"Command: {' '.join(cmd_list_for_popen)}\n")

        start_time_seconds = time.monotonic()
        current_build_proc = subprocess.Popen(
            cmd_list_for_popen,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            bufsize=1,
        )

        stdout_thread = _start_stream_reader_thread(
            current_build_proc.stdout,
            "STDOUT",
            run_context,
            build_stdout_lines,
        )
        stderr_thread = _start_stream_reader_thread(
            current_build_proc.stderr,
            "STDERR",
            run_context,
            build_stderr_lines,
        )

        monitoring_results = _perform_monitoring_loop(
            local_active_memory_collector,
            current_build_proc,
            metric_fields_header,
            primary_metric_to_track,
        )

        if current_build_proc:
            build_exit_code = current_build_proc.wait()
            with open(run_paths.output_summary_log_file, "a") as f_summary:
                f_summary.write(
                    f"Build End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f_summary.write(f"Build Exit Code: {build_exit_code}\n")
                if stdout_thread:
                    stdout_thread.join(timeout=1)
                if stderr_thread:
                    stderr_thread.join(timeout=1)
                _write_stream_to_summary(f_summary, "STDOUT", build_stdout_lines)
                _write_stream_to_summary(f_summary, "STDERR", build_stderr_lines)

    finally:
        if local_active_memory_collector:
            local_active_memory_collector.stop()
        if current_build_proc and current_build_proc.poll() is None:
            current_build_proc.terminate()
            try:
                current_build_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                current_build_proc.kill()

        end_time_seconds = time.monotonic()
        duration_seconds_float = (
            (end_time_seconds - start_time_seconds) if start_time_seconds else 0
        )
        formatted_duration = time.strftime(
            "%H:%M:%S", time.gmtime(duration_seconds_float)
        )

        if monitoring_results:
            _compile_and_write_final_summary_and_aux_logs(
                context=run_context,
                results=monitoring_results,
                build_exit_code=build_exit_code,
                duration_float=duration_seconds_float,
                duration_formatted=formatted_duration,
                primary_metric=primary_metric_to_track,
            )
            _write_monitoring_data_to_parquet(monitoring_results, run_context)

        _execute_clean_step(run_context, project_config, "Post-Build Clean")


def cleanup_processes() -> None:
    """Cleans up any running global subprocesses."""
    global current_build_proc, active_memory_collector
    if active_memory_collector:
        active_memory_collector.stop()
    if current_build_proc and current_build_proc.poll() is None:
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
    """Performs the main memory monitoring loop, collecting and processing samples."""
    all_samples_data_loop: List[Dict[str, Any]] = []
    category_stats_loop: Dict[str, Dict[str, Any]] = {}
    peak_overall_memory_kb_loop: int = 0
    peak_overall_memory_epoch_loop: int = 0
    category_peak_sum_loop: Dict[str, int] = {}
    category_pid_set_loop: Dict[str, Set[str]] = {}

    logger.info("Starting monitoring loop...")

    for samples_at_interval in active_memory_collector.read_samples():
        current_epoch = int(time.time())
        category_mem_sum_interval: Dict[str, int] = {}
        all_mem_sum_interval: int = 0
        category_pids_interval: Dict[str, Set[str]] = {}

        for sample in samples_at_interval:
            major_cat, minor_cat = get_process_category(
                sample.command_name, sample.full_command
            )
            category_key = f"{major_cat}_{minor_cat}"

            if major_cat == "Ignored":
                continue

            process_row_dict: Dict[str, Any] = {
                "Timestamp_epoch": current_epoch,
                "Record_Type": "PROCESS",
                "Major_Category": major_cat,
                "Minor_Category": minor_cat,
                "PID": sample.pid,
                "Command_Name": sample.command_name,
                "Full_Command": sample.full_command,
                "Sum_Value": None,
            }
            metric_value_for_sum = 0
            for metric_name in metric_fields_header:
                metric_val = sample.metrics.get(metric_name)
                try:
                    process_row_dict[metric_name] = (
                        int(metric_val) if metric_val is not None else None
                    )
                    if (
                        metric_name == primary_metric_to_track
                        and metric_val is not None
                    ):
                        metric_value_for_sum = int(metric_val)
                except (ValueError, TypeError):
                    process_row_dict[metric_name] = None
            all_samples_data_loop.append(process_row_dict)

            all_mem_sum_interval += metric_value_for_sum
            category_mem_sum_interval.setdefault(category_key, 0)
            category_mem_sum_interval[category_key] += metric_value_for_sum
            category_pids_interval.setdefault(category_key, set()).add(sample.pid)

            if primary_metric_to_track:
                if (
                    category_key not in category_stats_loop
                    or metric_value_for_sum
                    > category_stats_loop[category_key].get(
                        "peak_metric", float("-inf")
                    )
                ):
                    category_stats_loop[category_key] = {
                        "peak_metric": metric_value_for_sum,
                        "pid": sample.pid,
                        "command": sample.command_name,
                        "full_command": sample.full_command,
                    }

        if all_mem_sum_interval > peak_overall_memory_kb_loop:
            peak_overall_memory_kb_loop = all_mem_sum_interval
            peak_overall_memory_epoch_loop = current_epoch

        for cat_tuple_sum, mem_sum_val_sum in category_mem_sum_interval.items():
            # cat_tuple_sum is now a string like "Major_Minor", so we need to split it
            # if we want to write major and minor separately to the Parquet file.
            major_cat_from_key, minor_cat_from_key = cat_tuple_sum.split("_", 1)
            cat_sum_row_dict = _create_summary_row(
                current_epoch,
                "CATEGORY_SUM",
                major_cat_from_key,
                minor_cat_from_key,
                mem_sum_val_sum,
                metric_fields_header,
            )
            all_samples_data_loop.append(cat_sum_row_dict)

        if primary_metric_to_track:
            all_sum_row_dict = _create_summary_row(
                current_epoch,
                "ALL_SUM",
                "ALL",
                "ALL",
                all_mem_sum_interval,
                metric_fields_header,
            )
            all_samples_data_loop.append(all_sum_row_dict)

        for cat, current_cat_sum in category_mem_sum_interval.items():
            category_peak_sum_loop[cat] = max(
                category_peak_sum_loop.get(cat, 0), current_cat_sum
            )

        for cat, pids_in_interval in category_pids_interval.items():
            category_pid_set_loop.setdefault(cat, set()).update(pids_in_interval)

        if current_build_proc and current_build_proc.poll() is not None:
            logger.info("Build process finished. Stopping memory collection.")
            break

    logger.info("Monitoring loop finished.")
    return MonitoringResults(
        all_samples_data=all_samples_data_loop,
        category_stats=category_stats_loop,
        peak_overall_memory_kb=peak_overall_memory_kb_loop,
        peak_overall_memory_epoch=peak_overall_memory_epoch_loop,
        category_peak_sum=category_peak_sum_loop,
        category_pid_set=category_pid_set_loop,
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


def _create_memory_collector(
    context: RunContext,
) -> Optional[AbstractMemoryCollector]:
    """Creates and returns a memory collector instance based on the run context."""
    global active_memory_collector
    collector_kwargs: Dict[str, Any] = {
        "collector_cpu_core": context.monitor_core_id,
        "taskset_available": context.taskset_available,
        "pidstat_stderr_file": context.paths.collector_aux_log_file,
    }
    try:
        if context.collector_type == "rss_pidstat":
            active_memory_collector = RssPidstatCollector(
                context.process_pattern,
                context.monitoring_interval,
                **collector_kwargs,
            )
        elif context.collector_type == "pss_psutil":
            active_memory_collector = PssPsutilCollector(
                context.process_pattern, context.monitoring_interval, **collector_kwargs
            )
        else:
            logger.error(f"Unknown memory collector type: '{context.collector_type}'")
            return None
    except Exception as e:
        logger.error(
            f"Failed to initialize collector '{context.collector_type}': {e}",
            exc_info=True,
        )
        return None
    return active_memory_collector


def _prepare_initial_summary_log_header(context: RunContext) -> List[str]:
    """Prepares the header content for the summary log file using a context object."""
    header_content: List[str] = []
    header_content.append("=" * 80)
    header_content.append(f"Project: {context.project_name}")
    header_content.append(f"Parallelism: -j{context.parallelism_level}")
    header_content.append(f"Memory Metric Collector: {context.collector_type.upper()}")
    header_content.append(f"Source Directory: {context.project_dir.resolve()}")
    header_content.append(f"Actual Build Command: {context.actual_build_command}")
    header_content.append(f"Process Pattern (for collector): {context.process_pattern}")
    header_content.append(
        f"Monitoring Interval (approx): {context.monitoring_interval} seconds"
    )
    header_content.append(
        f"Run Timestamp: {context.current_timestamp_str} ({time.strftime('%Y-%m-%d %H:%M:%S')})"
    )
    header_content.append(
        f"Output Parquet File: {context.paths.output_parquet_file.name}"
    )
    header_content.append(
        f"Summary Log File: {context.paths.output_summary_log_file.name}"
    )
    if context.collector_type == "rss_pidstat":
        header_content.append(
            f"Collector Aux Log (temporary, will be merged): {context.paths.collector_aux_log_file.name}"
        )
    header_content.append(
        f"Monitor Script Pinned to CPU Core: {context.monitor_script_pinned_to_core_info}"
    )
    header_content.append(
        f"Build Processes CPU Cores Target: {context.build_cores_target_str}"
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
            f"Pidstat Collector Target CPU Core: Not Pinned (monitor_core_id not set or taskset unavailable)"
        )
    header_content.append("-" * 80)
    return header_content


def _prepare_actual_clean_command(
    setup_command_template: Optional[str], clean_command_template: str
) -> Tuple[str, Optional[str]]:
    """Prepares the actual clean command string and the executable shell if needed."""
    executable_shell: Optional[str] = None
    actual_clean_cmd: str
    if setup_command_template:
        actual_clean_cmd = f"{setup_command_template} && {clean_command_template}"
        executable_shell = "/bin/bash"
    else:
        actual_clean_cmd = clean_command_template
    return actual_clean_cmd, executable_shell


# --- Execution & I/O Helpers ---


def _start_stream_reader_thread(
    pipe: Optional[IO[str]],
    stream_name: str,
    context: RunContext,
    output_list: List[str],
) -> Optional[threading.Thread]:
    """Creates and starts a daemon thread to read from a subprocess stream."""
    if not pipe:
        return None
    log_prefix = f"[{context.project_name}-j{context.parallelism_level} {stream_name}] "
    thread = threading.Thread(
        target=_stream_output,
        args=(pipe, log_prefix, output_list),
        daemon=True,
    )
    thread.start()
    return thread


def _stream_output(
    pipe: Optional[IO[str]], prefix: str, output_lines_list: List[str]
) -> None:
    """Reads lines from a subprocess pipe, logs them, and collects them."""
    if not pipe:
        return
    try:
        for raw_line_content in iter(pipe.readline, ""):
            stripped_line = raw_line_content.strip()
            logger.debug(f"{prefix}{stripped_line}")
            output_lines_list.append(stripped_line)
    except ValueError:
        logger.debug(f"{prefix}Pipe closed or other ValueError during streaming.")
    except Exception as e:
        logger.error(f"{prefix}Unexpected error in _stream_output: {e}", exc_info=True)
    finally:
        if pipe and not pipe.closed:
            pass


def _run_and_log_command_to_summary(
    context: RunContext,
    cmd_str: str,
    cmd_desc: str,
    shell_bool: bool,
    executable_shell: Optional[str] = None,
) -> int:
    """Runs a command and logs its execution details to the summary file."""
    # Get cwd_path and summary_file from the context object.
    cwd_path = context.project_dir
    summary_file = context.paths.output_summary_log_file

    logger.info(f"Executing {cmd_desc} command: {cmd_str} in {cwd_path}")
    with open(summary_file, "a") as f_s:
        f_s.write(f"\n--- {cmd_desc} Command ---\n")
        f_s.write(f"Command: {cmd_str}\n")
    exit_code, stdout_val, stderr_val = run_command(
        cmd_str, cwd=cwd_path, shell=shell_bool, executable_shell=executable_shell
    )
    with open(summary_file, "a") as f_s:
        f_s.write(f"Exit Code: {exit_code}\n")
        if stdout_val:
            f_s.write(f"--- STDOUT ---\n{stdout_val}\n")
        if stderr_val:
            f_s.write(f"--- STDERR ---\n{stderr_val}\n")
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
    final_summary_log_lines: List[str] = []
    final_summary_log_lines.append("\n--- Build & Monitoring Summary ---")
    final_summary_log_lines.append(
        f"Total Build & Monitoring Duration: {duration_formatted} ({duration_float:.2f} seconds)"
    )
    final_summary_log_lines.append(f"Final Build Exit Code: {build_exit_code}")

    if primary_metric:
        peak_time_str = (
            time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(results.peak_overall_memory_epoch)
            )
            if results.peak_overall_memory_epoch > 0
            else "N/A (no peak recorded or build did not run long enough)"
        )
        final_summary_log_lines.append(
            f"Peak Overall Memory ({primary_metric}): {results.peak_overall_memory_kb} KB (at approx. {peak_time_str})"
        )
    else:
        final_summary_log_lines.append(
            f"Peak Overall Memory: N/A (no primary metric was tracked for sum)"
        )

    final_summary_log_lines.append(
        f"Peak Single Process Memory Usage (by category, based on {primary_metric or 'N/A'}):"
    )
    if results.category_stats:
        for category, stats in sorted(results.category_stats.items()):
            peak_val = stats.get("peak_metric", "N/A")
            pid_val = stats.get("pid", "N/A")
            cmd_val = stats.get("command", "N/A")
            full_cmd_val = stats.get("full_command", "N/A")
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


def _write_stream_to_summary(
    f_handle: IO[Any], stream_name: str, lines: List[str]
) -> None:
    """Writes a list of lines from a stream to the summary log file."""
    if lines:
        f_handle.write(f"\n--- Build Process {stream_name} ---\n")
        for line in lines:
            f_handle.write(line + "\n")
        f_handle.write(f"--- End Build Process {stream_name} ---\n")


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
    for metric_name in metric_fields_header:
        row[metric_name] = None
    return row
