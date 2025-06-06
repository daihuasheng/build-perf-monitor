"""
Utility functions for monitoring build processes.

This module provides functionalities to:
- Stream output from subprocesses.
- Categorize processes based on their command names and arguments.
- Execute shell commands and capture their output.
- Run and monitor a build process, collecting memory usage data using specified collectors.
- Check for the installation of necessary tools (e.g., pidstat).
- Clean up any running subprocesses upon script termination.
"""

import csv
import logging
import os
import re
import subprocess
import time
import threading
from pathlib import Path
from typing import (
    Dict,
    Any,
    Tuple,
    Optional,
    IO,
    List,
)

# Import collectors
from .memory_collectors.base import AbstractMemoryCollector
from .memory_collectors.rss_pidstat_collector import RssPidstatCollector
from .memory_collectors.pss_psutil_collector import PssPsutilCollector


logger = logging.getLogger(__name__)

# --- Module Globals ---
current_build_proc: Optional[subprocess.Popen] = None
"""Global variable to hold the currently running build subprocess. Used for cleanup."""

active_memory_collector: Optional[AbstractMemoryCollector] = None
"""Global variable for the currently active memory collector instance. Used for cleanup."""


def _stream_output(pipe: Optional[IO[str]], prefix: str) -> None:
    """
    Reads lines from a subprocess pipe and logs them with a given prefix.

    This function is designed to be run in a separate thread to allow
    non-blocking reading of a subprocess's stdout or stderr.

    Args:
        pipe: The I/O pipe (e.g., process.stdout) to read from.
              If None, the function returns immediately.
        prefix: A string prefix to prepend to each logged line (e.g., "[STDOUT]").
    """
    if not pipe:
        return
    try:
        for line in iter(pipe.readline, ""):
            # Using print directly for immediate console output, rstrip to remove trailing newline.
            # This is often preferred for build logs for real-time feedback.
            print(f"{prefix}{line.rstrip()}", flush=True)
    except ValueError:  # This can happen if the pipe is closed abruptly.
        logger.debug(f"{prefix} Pipe closed, stream ended.")
    except Exception as e:
        logger.error(f"Error in _stream_output for {prefix}: {e}", exc_info=True)
    finally:
        if hasattr(pipe, "close") and not pipe.closed:
            try:
                pipe.close()
            except Exception as e:
                logger.warning(f"Error closing pipe for {prefix}: {e}")
        logger.debug(f"{prefix} Streaming thread finished.")


def get_process_category(cmd_name: str, cmd_full: str) -> str:
    """
    Categorizes a process based on its command name and full command line.

    This function attempts to identify the role of a process in a build system
    (e.g., compiler, linker, build tool) using a set of predefined rules and
    regular expressions. It also tries to unwrap commands executed via 'sh -c'
    or 'bash -c' to get to the actual underlying command.

    Args:
        cmd_name: The base name of the command (e.g., "gcc", "make").
        cmd_full: The full command line string with arguments.

    Returns:
        A string representing the determined category (e.g., "compile_c_cpp", "link").
        If no specific category is matched, it returns "other_<cmd_name>".
    """
    # Filter out vscode-server processes early as they are usually not part of the build.
    if ".vscode-server" in cmd_full:
        return "ignore_vscode_server"

    orig_cmd_name = cmd_name
    orig_cmd_full = cmd_full

    # Attempt to unwrap commands executed via sh -c "..." or bash -c "..."
    # This helps in identifying the actual command being run.
    sh_bash_pattern = r"^(?:.*/)?(sh|bash)$"
    sh_bash_c_pattern = r"^(?:.*/)?(sh|bash)\s+-c\s+"

    if (
        orig_cmd_name == "sh"
        or orig_cmd_name == "bash"
        or re.search(sh_bash_pattern, orig_cmd_name)
    ) and re.search(sh_bash_c_pattern, orig_cmd_full):
        # Extract the command string after 'sh -c ' or 'bash -c '.
        temp_unwrapped_cmd = re.sub(sh_bash_c_pattern, "", orig_cmd_full, 1)

        # Remove surrounding quotes if present.
        if (
            temp_unwrapped_cmd.startswith('"') and temp_unwrapped_cmd.endswith('"')
        ) or (temp_unwrapped_cmd.startswith("'") and temp_unwrapped_cmd.endswith("'")):
            temp_unwrapped_cmd = temp_unwrapped_cmd[1:-1]

        if temp_unwrapped_cmd:
            cmd_full = temp_unwrapped_cmd
            parts = cmd_full.split(maxsplit=1)
            new_cmd_name_full = parts[0]
            # Update cmd_name to the basename of the unwrapped command.
            cmd_name = os.path.basename(new_cmd_name_full)

    # --- Classification rules operate on potentially unwrapped cmd_name and cmd_full ---
    # cmd_name is now the basename of the (potentially unwrapped) command.
    # cmd_full is the (potentially unwrapped) full command string.

    # C/C++ compilation (cc1, cc1plus, or compiler driver with -c option)
    if (
        cmd_name in ("cc1", "cc1plus")
        or re.search(
            r"(?:^|/)(cc1|cc1plus)\s", cmd_full
        )  # Check if command starts with or is path to cc1/cc1plus
        or (
            (
                cmd_name in ("gcc", "g++", "clang", "clang++", "cc")
                or re.search(r"(?:^|/)(gcc|g\+\+|clang|clang\+\+|cc)\s", cmd_full)
            )
            and re.search(
                r"\s-c(\s|$)", cmd_full
            )  # Presence of '-c' flag indicates compilation
        )
    ):
        return "compile_c_cpp"

    # Linking (ld, lld, collect2, or compiler driver without -c and with -o option)
    if (
        cmd_name in ("ld", "lld", "collect2")
        or re.search(r"(?:^|/)(ld|lld|collect2)\s", cmd_full)
        or (
            (
                cmd_name in ("gcc", "g++", "clang", "clang++", "cc")
                or re.search(r"(?:^|/)(gcc|g\+\+|clang|clang\+\+|cc)\s", cmd_full)
            )
            and not re.search(r"\s-c(\s|$)", cmd_full)  # Absence of '-c'
            and re.search(r"\s-o\s", cmd_full)  # Presence of '-o' (output file)
            and cmd_name
            not in ("cc1", "cc1plus")  # Ensure it's not a compiler frontend
        )
    ):
        return "link"

    # Assembly
    if cmd_name == "as" or re.search(r"(?:^|/)as\s", cmd_full):
        return "assemble"

    # Java compilation
    if cmd_name == "javac" or re.search(r"(?:^|/)javac\s", cmd_full):
        return "compile_java"

    # Android Dexing/D8/R8 tools
    if cmd_name in ("dx", "d8", "r8") or re.search(r"(?:^|/)(dx|d8|r8)\s", cmd_full):
        return "dex_android"

    # Build system processes (make, ninja, etc.)
    build_systems = ["make", "ninja", "meson", "kati", "soong_ui", "siso", "gomacc"]
    if cmd_name in build_systems or re.search(
        r"(?:^|/)(" + "|".join(build_systems) + r")\s", cmd_full
    ):
        return (
            f"build_system_{cmd_name}" if cmd_name in build_systems else "build_system"
        )

    # Python scripts
    if cmd_name.startswith("python") or re.search(
        r"(?:^|/)python[0-9._-]*\s", cmd_full  # Matches python, python3, python3.8 etc.
    ):
        return "script_python"

    # Android resource processing (aapt, aapt2)
    if cmd_name in ("aapt", "aapt2") or re.search(r"(?:^|/)(aapt2?)\s", cmd_full):
        return "android_resource"

    # Android code generation tools (metalava, aidl, hidl-gen)
    android_codegen_tools = ["metalava", "aidl", "hidl-gen"]
    if cmd_name in android_codegen_tools or re.search(
        r"(?:^|/)(" + "|".join(android_codegen_tools) + r")\s", cmd_full
    ):
        return "android_codegen"

    # Chromium specific tools (gn, mojo)
    chromium_tools = ["gn", "mojo"]  # Add more as identified
    if cmd_name in chromium_tools or re.search(
        r"(?:^|/)(" + "|".join(chromium_tools) + r")\s", cmd_full
    ):
        return (
            f"chromium_tool_{cmd_name}"
            if cmd_name in chromium_tools
            else "chromium_tool"
        )

    # QEMU configure script
    if (
        cmd_name == "configure" or re.search(r"(?:^|/)configure\s", cmd_full)
    ) and "qemu" in cmd_full:  # Check for 'qemu' in full command to be specific
        return "qemu_configure"

    # Default category if no specific rule matches
    return f"other_{cmd_name}"


def run_command(command: str, cwd: Path, shell: bool = False) -> Tuple[int, str, str]:
    """
    Executes a given shell command and returns its exit code, stdout, and stderr.

    Args:
        command: The command string to execute.
        cwd: The working directory in which to run the command.
        shell: If True, the command is executed through the shell.
               Set to True if the command includes shell features like pipes or wildcards.
               Defaults to False, where the command is treated as a list of arguments
               if it's not a single executable.

    Returns:
        A tuple containing:
            - exit_code (int): The exit code of the command. -1 if execution failed.
            - stdout (str): The standard output of the command.
            - stderr (str): The standard error of the command.
    """
    logger.info(f"Executing command in {cwd}: {command}")
    try:
        # If shell is False, command should ideally be a list.
        # subprocess.run handles string command as program name if shell=False.
        # For simplicity, if shell=False and command is a string, it's split.
        # If shell=True, command is passed as a string to the shell.
        cmd_to_run = command if shell else command.split()
        process = subprocess.run(
            cmd_to_run,
            cwd=cwd,
            capture_output=True,
            text=True,  # Decodes stdout/stderr as text
            shell=shell,
            check=False,  # Do not raise CalledProcessError, handle returncode manually
        )
        if process.returncode != 0:
            logger.warning(
                f"Command '{command}' failed with exit code {process.returncode} in {cwd}."
            )
            # Log output for failed commands to aid debugging.
            if process.stdout:
                logger.warning(f"Stdout from failed command:\n{process.stdout.strip()}")
            if process.stderr:
                logger.warning(f"Stderr from failed command:\n{process.stderr.strip()}")
        else:
            # Log output for successful commands at DEBUG level.
            if process.stdout:
                logger.debug(
                    f"Stdout from successful command:\n{process.stdout.strip()}"
                )
            if process.stderr:  # Some tools use stderr for informational messages.
                logger.debug(
                    f"Stderr from successful command:\n{process.stderr.strip()}"
                )
            logger.info(
                f"Command '{command}' finished successfully with exit code {process.returncode}."
            )
        return process.returncode, process.stdout, process.stderr
    except Exception as e:
        logger.error(f"Failed to execute command '{command}': {e}", exc_info=True)
        return -1, "", str(e)  # Return a default error tuple


def run_and_monitor_build(
    project_config: Dict[str, Any],
    parallelism_level: int,
    monitoring_interval: int,
    log_dir: Path,
    collector_type: str = "pss_psutil",
) -> None:
    """
    Runs a build for a given project configuration and monitors its memory usage.

    This function orchestrates the build process:
    1. Sets up logging for a specific build run (CSV data and summary log).
    2. Executes optional setup and clean commands.
    3. Starts the specified memory collector.
    4. Executes the main build command, streaming its stdout/stderr.
    5. Collects memory samples at specified intervals while the build is running.
    6. Records collected data, peak memory usage, and other statistics.
    7. Cleans up resources (collector, build process) upon completion or error.

    Args:
        project_config: A dictionary containing project-specific settings like
                        name, directory, build commands, and process patterns.
        parallelism_level: The number of parallel jobs for the build (e.g., -j<N>).
        monitoring_interval: The interval in seconds for polling memory usage.
        log_dir: The run-specific directory where logs (CSV, summary) will be stored.
        collector_type: The type of memory collector to use ("pss_psutil" or "rss_pidstat").
                        Defaults to "pss_psutil".
    """
    global current_build_proc, active_memory_collector  # Allow modification of global vars

    project_name = project_config["NAME"]
    project_dir = Path(project_config["DIR"])
    build_command_template = project_config["BUILD_COMMAND_TEMPLATE"]
    process_pattern = project_config["PROCESS_PATTERN"]
    clean_command_template = project_config.get("CLEAN_COMMAND_TEMPLATE", "")
    setup_command_template = project_config.get("SETUP_COMMAND_TEMPLATE", "")

    current_timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    # Define base filename for logs related to this specific build run.
    base_filename_part = f"{project_name}_j{parallelism_level}_mem_{collector_type}_{current_timestamp_str}"
    output_csv_filename = f"{base_filename_part}.csv"
    output_summary_log_filename = f"{base_filename_part}_summary.log"

    output_csv_file = log_dir / output_csv_filename
    output_summary_log_file = log_dir / output_summary_log_filename

    # Auxiliary log file for collectors like pidstat (stores stderr).
    collector_aux_log_file = (
        log_dir / f"{base_filename_part}_collector_aux.log"
    )  # Generic name

    base_build_cmd = build_command_template.replace("<N>", str(parallelism_level))
    if setup_command_template:
        actual_build_command = f"{setup_command_template} && {base_build_cmd}"
    else:
        actual_build_command = base_build_cmd

    # --- Initial Logging to Summary Log File and Console ---
    summary_log_header_content: List[str] = []  # Explicitly type the list
    summary_log_header_content.append("=" * 80)
    summary_log_header_content.append(f"Project: {project_name}")
    summary_log_header_content.append(f"Parallelism: -j{parallelism_level}")
    summary_log_header_content.append(
        f"Memory Metric Collector: {collector_type.upper()}"
    )
    summary_log_header_content.append(f"Source Directory: {project_dir.resolve()}")
    summary_log_header_content.append(f"Actual Build Command: {actual_build_command}")
    summary_log_header_content.append(
        f"Process Pattern (for collector): {process_pattern}"
    )
    summary_log_header_content.append(
        f"Monitoring Interval (approx): {monitoring_interval} seconds"
    )
    summary_log_header_content.append(
        f"Run Timestamp: {current_timestamp_str} ({time.strftime('%Y-%m-%d %H:%M:%S')})"
    )
    summary_log_header_content.append(f"Output CSV File: {output_csv_file.name}")
    summary_log_header_content.append(
        f"Summary Log File: {output_summary_log_file.name}"
    )
    if collector_type == "rss_pidstat":  # Specific info for pidstat
        summary_log_header_content.append(
            f"Collector Aux Log (temporary, will be merged): {collector_aux_log_file.name}"
        )
    summary_log_header_content.append("-" * 80)

    # Log header to console.
    for line in summary_log_header_content:
        logger.info(line.replace("=", "-"))  # Use '-' for console section breaks

    # Write header to the summary log file.
    with open(output_summary_log_file, "w") as f_summary:
        for line in summary_log_header_content:
            f_summary.write(line + "\n")
        f_summary.write("\n")  # Add a newline for separation.

    if not project_dir.is_dir():
        error_msg = f"Project directory '{project_dir}' does not exist. Skipping build for {project_name} j{parallelism_level}."
        logger.error(error_msg)
        with open(output_summary_log_file, "a") as f_summary:
            f_summary.write(f"ERROR: {error_msg}\n")
        logger.info(
            f"--- End of processing for {project_name} j{parallelism_level} (skipped) ---"
        )
        return

    # Initialize the selected memory collector.
    collector_kwargs: Dict[str, Any] = {}  # Arguments for collector constructor
    if collector_type == "rss_pidstat":
        collector_kwargs["pidstat_stderr_file"] = collector_aux_log_file
        active_memory_collector = RssPidstatCollector(
            process_pattern, monitoring_interval, **collector_kwargs
        )
    elif collector_type == "pss_psutil":
        active_memory_collector = PssPsutilCollector(
            process_pattern,
            monitoring_interval,
            **collector_kwargs,  # Pass empty dict if no specific args
        )
    else:
        error_msg = f"Unsupported collector type: {collector_type}. Cannot monitor {project_name} j{parallelism_level}."
        logger.error(error_msg)
        with open(output_summary_log_file, "a") as f_summary:
            f_summary.write(f"ERROR: {error_msg}\n")
        return

    metric_fields_header = active_memory_collector.get_metric_fields()
    primary_metric_to_track = metric_fields_header[0] if metric_fields_header else None
    csv_header_parts = (
        ["Timestamp_epoch", "Category"]
        + metric_fields_header
        + ["PID", "Command_Name", "Full_Command"]
    )
    csv_header = ",".join(csv_header_parts)

    # Write only header to CSV file (no metadata comments in CSV).
    with open(output_csv_file, "w") as f_csv:
        f_csv.write(csv_header + "\n")

    # --- Helper function to run setup/clean commands and log their output ---
    def _run_and_log_command_to_summary(
        cmd_str: str,
        cmd_desc: str,
        cwd_path: Path,
        shell_bool: bool,
        summary_file: Path,
    ) -> int:
        """Runs a command and logs its execution details to the summary file."""
        logger.info(f"Executing {cmd_desc} command: {cmd_str} in {cwd_path}")
        with open(summary_file, "a") as f_s:
            f_s.write(f"\n--- {cmd_desc} Command ---\n")
            f_s.write(f"Command: {cmd_str}\n")
            f_s.write(f"Directory: {cwd_path}\n")
            f_s.write(f"Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        exit_code, stdout_val, stderr_val = run_command(
            cmd_str, cwd=cwd_path, shell=shell_bool
        )

        with open(summary_file, "a") as f_s:
            f_s.write(f"Exit Code: {exit_code}\n")
            if stdout_val:
                f_s.write(f"Stdout:\n{stdout_val.strip()}\n")
            if stderr_val:
                f_s.write(f"Stderr:\n{stderr_val.strip()}\n")
            f_s.write(f"--- End {cmd_desc} Command ---\n\n")

        if exit_code != 0:
            logger.warning(
                f"{cmd_desc} command failed with exit code {exit_code}. Continuing build process..."
            )
        else:
            logger.info(f"{cmd_desc} command completed successfully.")
        return exit_code

    # Execute clean command if defined.
    if clean_command_template:
        # For projects with setup commands (like AOSP), clean also needs setup
        if setup_command_template:
            actual_clean_command = f"{setup_command_template} && {clean_command_template}"
        else:
            actual_clean_command = clean_command_template
        
        _run_and_log_command_to_summary(
            actual_clean_command,
            "Clean",
            project_dir,
            True,
            output_summary_log_file,
        )

    # --- Build Monitoring ---
    # Statistics to track during the build.
    category_stats: Dict[str, Dict[str, Any]] = (
        {}
    )  # Peak memory for a single process within each category.
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None
    build_exit_code: int = -1  # Default to error
    start_time_seconds: Optional[float] = None
    peak_overall_memory_kb: int = (
        0  # Peak of the sum of primary metric for ALL processes.
    )
    peak_overall_memory_epoch: int = 0  # Timestamp of the overall peak.

    category_peak_sum: Dict[str, int] = (
        {}
    )  # Peak of the sum of primary metric for EACH category.
    category_pid_set: Dict[str, set] = {}  # Set of unique PIDs for EACH category.

    try:
        active_memory_collector.start()
        logger.info(
            f"Starting build command for {project_name} j{parallelism_level}: {actual_build_command}"
        )
        with open(output_summary_log_file, "a") as f_summary:
            f_summary.write(f"--- Build Command Execution ---\n")
            f_summary.write(f"Command: {actual_build_command}\n")
            f_summary.write(f"Directory: {project_dir}\n")
            f_summary.write(f"Build Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        start_time_seconds = time.time()
        build_cmd_to_run_shell = f"bash -c '{actual_build_command}'"
        current_build_proc = subprocess.Popen(
            build_cmd_to_run_shell,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,  # Crucial for 'bash -c' wrapper
            bufsize=1,  # Line-buffered
        )

        # Start threads to stream stdout and stderr from the build process.
        if current_build_proc.stdout:
            stdout_thread = threading.Thread(
                target=_stream_output,
                args=(
                    current_build_proc.stdout,
                    f"[{project_name}-j{parallelism_level} STDOUT] ",
                ),
                daemon=True,  # Threads will exit when main program exits.
            )
            stdout_thread.start()
        if current_build_proc.stderr:
            stderr_thread = threading.Thread(
                target=_stream_output,
                args=(
                    current_build_proc.stderr,
                    f"[{project_name}-j{parallelism_level} STDERR] ",
                ),
                daemon=True,
            )
            stderr_thread.start()

        # Main monitoring loop: collect samples and write to CSV.
        with open(
            output_csv_file, "a", newline=""
        ) as f_csv:  # newline='' recommended for csv module
            csv_writer = csv.writer(f_csv) # Use csv.writer

            for samples_at_interval in active_memory_collector.read_samples():
                current_epoch = int(time.time())
                # Per-interval sums
                category_mem_sum_interval: Dict[str, int] = {}
                all_mem_sum_interval: int = 0
                category_pids_interval: Dict[str, set] = {}

                for sample in samples_at_interval:
                    category = get_process_category(
                        sample.command_name, sample.full_command
                    )
                    if category == "ignore_vscode_server":
                        continue  # Skip ignored categories.

                    # Prepare row data for individual process sample.
                    row_data_individual = [str(current_epoch), category]
                    metric_value_for_sum = 0

                    for metric_name in metric_fields_header:
                        metric_val = sample.metrics.get(metric_name, "")
                        row_data_individual.append(str(metric_val))
                        if metric_name == primary_metric_to_track:
                            try:
                                metric_value_for_sum = int(metric_val) if metric_val else 0
                            except ValueError:
                                logger.warning(
                                    f"Could not convert primary metric '{metric_name}' value '{metric_val}' to int for PID {sample.pid} ({sample.command_name}). Defaulting to 0."
                                )
                                metric_value_for_sum = 0
                    
                    # No need to manually escape full_command if using csv.writer
                    row_data_individual.extend(
                        [sample.pid, sample.command_name, sample.full_command] # Pass raw string
                    )
                    # f_csv.write(",".join(row_data_individual) + "\n") # Old way
                    csv_writer.writerow(row_data_individual) # New way using csv.writer

                    # Accumulate sums for this interval.
                    all_mem_sum_interval += metric_value_for_sum
                    category_mem_sum_interval.setdefault(category, 0)
                    category_mem_sum_interval[category] += metric_value_for_sum

                    # Collect unique PIDs for this category in this interval.
                    category_pids_interval.setdefault(category, set()).add(sample.pid)

                    # Update peak single process metric for each category (overall).
                    if primary_metric_to_track:
                        if (
                            category not in category_stats
                            or metric_value_for_sum
                            > category_stats[category].get("peak_metric", float("-inf"))
                        ):
                            category_stats[category] = {
                                "peak_metric": metric_value_for_sum,
                                "pid": sample.pid,
                                "command": sample.command_name,  # Short command name
                                "full_command": sample.full_command, # Store the full command
                            }

                # After processing all samples in the interval:
                # Update overall peak memory sum (sum of all processes).
                if (
                    primary_metric_to_track
                    and all_mem_sum_interval > peak_overall_memory_kb
                ):
                    peak_overall_memory_kb = all_mem_sum_interval
                    peak_overall_memory_epoch = current_epoch

                # Write CATEGORY_SUM and ALL_SUM aggregate rows to CSV.
                for cat, mem_sum_val in category_mem_sum_interval.items():
                    # f_csv.write(f"{current_epoch},CATEGORY_SUM,{cat},{mem_sum_val}\n") # Old way
                    csv_writer.writerow([str(current_epoch), "CATEGORY_SUM", cat, mem_sum_val]) # New way
                if primary_metric_to_track:
                    # f_csv.write(f"{current_epoch},ALL_SUM,ALL,{all_mem_sum_interval}\n") # Old way
                    csv_writer.writerow([str(current_epoch), "ALL_SUM", "ALL", all_mem_sum_interval]) # New way

                # Update peak sum for each category over the entire build.
                for cat, current_cat_sum in category_mem_sum_interval.items():
                    category_peak_sum[cat] = max(
                        category_peak_sum.get(cat, 0), current_cat_sum
                    )

                # Update unique PID sets for each category over the entire build.
                for cat, pids_in_interval in category_pids_interval.items():
                    category_pid_set.setdefault(cat, set()).update(pids_in_interval)

                f_csv.flush()  # Ensure data is written to disk periodically.

                # Check if the build process has finished.
                if current_build_proc and current_build_proc.poll() is not None:
                    logger.info(
                        f"Build process for {project_name} j{parallelism_level} finished. Stopping memory collection."
                    )
                    break  # Exit monitoring loop.

        # After monitoring loop (build finished or error).
        if current_build_proc:
            build_exit_code = current_build_proc.wait()  # Get final exit code.
            logger.info(
                f"Build process for {project_name} j{parallelism_level} exited with code: {build_exit_code}."
            )
            with open(output_summary_log_file, "a") as f_summary:
                f_summary.write(
                    f"Build End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f_summary.write(f"Build Exit Code: {build_exit_code}\n")
                f_summary.write(f"--- End Build Command Execution ---\n\n")

        # Wait for stdout/stderr streaming threads to finish.
        if stdout_thread and stdout_thread.is_alive():
            stdout_thread.join(timeout=5)  # Timeout to prevent hanging.
        if stderr_thread and stderr_thread.is_alive():
            stderr_thread.join(timeout=5)
        logger.info(
            f"Build process stdout/stderr streaming for {project_name} j{parallelism_level} finished."
        )

    except Exception as e:
        error_msg = f"An critical error occurred during monitoring for {project_name} j{parallelism_level}: {e}"
        logger.error(error_msg, exc_info=True)
        with open(output_summary_log_file, "a") as f_summary:
            f_summary.write(
                f"\nCRITICAL ERROR DURING MONITORING:\n{error_msg}\nDetails: {str(e)}\n"
            )
        # Ensure build process is handled if it's still running after an error.
        if current_build_proc and current_build_proc.poll() is None:
            build_exit_code = current_build_proc.wait()  # Attempt to get exit code.
    finally:
        # --- Cleanup ---
        if active_memory_collector:
            logger.info(
                f"Stopping memory collector ({active_memory_collector.__class__.__name__}) for {project_name} j{parallelism_level}."
            )
            active_memory_collector.stop()
            active_memory_collector = None  # Clear global var

        if current_build_proc and current_build_proc.poll() is None:
            logger.warning(
                f"Build process {current_build_proc.pid} for {project_name} j{parallelism_level} still running in finally block. Attempting to terminate."
            )
            current_build_proc.terminate()
            try:
                current_build_proc.wait(timeout=10)  # Wait for graceful termination.
            except subprocess.TimeoutExpired:
                logger.error(
                    f"Build process {current_build_proc.pid} did not terminate. Killing."
                )
                current_build_proc.kill()  # Force kill if terminate fails.
            current_build_proc = None  # Clear global var

        end_time_seconds = time.time()
        duration_seconds_float = (
            (end_time_seconds - start_time_seconds)
            if start_time_seconds is not None
            else 0
        )
        duration_seconds = int(duration_seconds_float)

        h = duration_seconds // 3600
        m = (duration_seconds % 3600) // 60
        s = duration_seconds % 60
        formatted_duration = f"{h:02d}:{m:02d}:{s:02d}"

        # --- Final Summary Section for Summary Log File ---
        final_summary_log_lines: List[str] = []
        final_summary_log_lines.append("\n--- Build & Monitoring Summary ---")
        final_summary_log_lines.append(
            f"Total Build & Monitoring Duration: {formatted_duration} ({duration_seconds_float:.2f} seconds)"
        )
        final_summary_log_lines.append(f"Final Build Exit Code: {build_exit_code}")

        if primary_metric_to_track:
            peak_time_str = (
                time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(peak_overall_memory_epoch)
                )
                if peak_overall_memory_epoch > 0
                else "N/A (no peak recorded or build did not run long enough)"
            )
            final_summary_log_lines.append(
                f"Peak Overall Memory ({primary_metric_to_track}): {peak_overall_memory_kb} KB (at approx. {peak_time_str})"
            )
        else:
            final_summary_log_lines.append(
                f"Peak Overall Memory: N/A (no primary metric was tracked for sum)"
            )

        final_summary_log_lines.append(
            f"Peak Single Process Memory Usage (by category, based on {primary_metric_to_track or 'N/A'}):"
        )
        if category_stats:
            for category, stats in sorted(category_stats.items()):
                peak_val = stats.get("peak_metric", "N/A")
                pid_val = stats.get("pid", "N/A")
                cmd_val = stats.get("command", "N/A")
                full_cmd_val = stats.get("full_command", "N/A") # Retrieve the full command
                final_summary_log_lines.append(
                    f"  - {category}: {peak_val} KB (PID: {pid_val}, Command: {cmd_val})"
                )
                final_summary_log_lines.append(
                    f"    Full Command: {full_cmd_val}" # Add the full command to the log
                )
        else:
            final_summary_log_lines.append(
                "  No category-specific single process peak memory data collected."
            )

        final_summary_log_lines.append(
            f"Peak Category Sum Memory Usage (sum of processes in category, based on {primary_metric_to_track or 'N/A'}):"
        )
        if category_peak_sum:
            for category, peak_sum_val in sorted(category_peak_sum.items()):
                final_summary_log_lines.append(f"  - {category}: {peak_sum_val} KB")
        else:
            final_summary_log_lines.append(
                "  No category sum peak memory data collected."
            )

        final_summary_log_lines.append("Unique Process Counts (by category):")
        if category_pid_set:
            for category, pids in sorted(category_pid_set.items()):
                final_summary_log_lines.append(
                    f"  - {category}: {len(pids)} unique process(es)"
                )
        else:
            final_summary_log_lines.append("  No unique process count data collected.")

        final_summary_log_lines.append("--- End Summary ---")

        # Append final summary and merge auxiliary logs to the summary log file.
        with open(output_summary_log_file, "a") as f_summary:
            f_summary.write("\n")  # Ensure a blank line before aux log or summary

            # Merge collector's auxiliary log content if it exists (e.g., pidstat stderr).
            if (
                collector_type == "rss_pidstat"  # Check specific collector type for now
                and collector_aux_log_file.exists()
                and collector_aux_log_file.stat().st_size > 0
            ):
                f_summary.write("\n--- Collector Auxiliary Log Output ---\n")
                try:
                    with open(collector_aux_log_file, "r") as f_aux:
                        f_summary.write(f_aux.read())
                    f_summary.write("--- End Collector Auxiliary Log Output ---\n\n")
                    try:  # Attempt to delete the aux file after merging.
                        collector_aux_log_file.unlink()
                        logger.info(
                            f"Merged and deleted collector aux log: {collector_aux_log_file.name}"
                        )
                    except OSError as e_del:
                        logger.warning(
                            f"Could not delete collector aux log {collector_aux_log_file.name}: {e_del}"
                        )
                except Exception as e_read_aux:
                    error_msg_aux = f"Error reading collector aux log {collector_aux_log_file.name}: {e_read_aux}"
                    f_summary.write(f"{error_msg_aux}\n")
                    logger.error(error_msg_aux)

            # Write the final summary lines.
            for line in final_summary_log_lines:
                f_summary.write(line + "\n")

        # Log final summary to console as well.
        for line in final_summary_log_lines:
            logger.info(line)

        logger.info(
            f"Monitoring for {project_name} j{parallelism_level} ({collector_type}) finished."
        )
        logger.info(f"CSV data saved to: {output_csv_file.resolve()}")
        logger.info(f"Summary log saved to: {output_summary_log_file.resolve()}")
        logger.info(
            f"--- End of processing for {project_name} j{parallelism_level} ---"
        )


def check_pidstat_installed() -> bool:
    """
    Checks if the 'pidstat' command is installed and available in the system PATH.

    Returns:
        True if pidstat is installed, False otherwise.
    """
    try:
        # Running 'pidstat -V' typically prints version information.
        # capture_output=True suppresses output to console.
        # check=True will raise CalledProcessError if pidstat returns non-zero.
        subprocess.run(["pidstat", "-V"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # FileNotFoundError if 'pidstat' command doesn't exist.
        # CalledProcessError if 'pidstat -V' fails for some reason.
        return False


def cleanup_processes() -> None:
    """
    Cleans up any running global subprocesses (build process, memory collector).

    This function is intended to be called by a signal handler (e.g., SIGINT, SIGTERM)
    to ensure graceful shutdown of child processes.
    """
    logger.info(
        "Signal received: Cleaning up any running subprocesses and collectors..."
    )
    global current_build_proc, active_memory_collector

    if active_memory_collector:
        logger.info(
            f"Stopping active memory collector ({active_memory_collector.__class__.__name__}) from cleanup handler."
        )
        active_memory_collector.stop()
        active_memory_collector = None  # Reset global variable

    if (
        current_build_proc and current_build_proc.poll() is None
    ):  # Check if process is still running
        logger.info(
            f"Terminating build process (PID {current_build_proc.pid}) from cleanup handler."
        )
        current_build_proc.terminate()  # Send SIGTERM
        try:
            current_build_proc.wait(timeout=10)  # Wait for graceful termination
            logger.info(f"Build process (PID {current_build_proc.pid}) terminated.")
        except subprocess.TimeoutExpired:
            logger.error(
                f"Build process {current_build_proc.pid} did not terminate in cleanup, killing."
            )
            current_build_proc.kill()  # Send SIGKILL if terminate fails
        current_build_proc = None  # Reset global variable
    logger.info("Cleanup finished.")
