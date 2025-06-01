import re
import time
import subprocess
import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, IO
import threading

# Import collectors
from .memory_collectors.base import AbstractMemoryCollector
from .memory_collectors.rss_pidstat_collector import RssPidstatCollector
from .memory_collectors.pss_psutil_collector import PssPsutilCollector


logger = logging.getLogger(__name__)

# Global variable to hold the pidstat process, for cleanup
# current_pidstat_proc: Optional[subprocess.Popen] = None # Now managed by collector
current_build_proc: Optional[subprocess.Popen] = None
active_memory_collector: Optional[AbstractMemoryCollector] = None


def _stream_output(pipe: Optional[IO[str]], prefix: str):
    """
    Reads lines from a pipe and prints them with a prefix.
    Designed to be run in a thread.
    """
    if not pipe:
        return
    try:
        for line in iter(pipe.readline, ""):
            # Using print directly for immediate console output, rstrip to remove trailing newline
            print(f"{prefix}{line.rstrip()}", flush=True)
    except ValueError:  # Pipe closed
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
    This is a Python translation of the awk get_process_category function.
    """
    # Filter out vscode-server processes early
    if ".vscode-server" in cmd_full:
        return "ignore_vscode_server"  # Assign a specific category to ignore later

    orig_cmd_name = cmd_name
    orig_cmd_full = cmd_full

    # Attempt to unwrap commands executed via sh -c "..." or bash -c ...
    sh_bash_pattern = r"^(?:.*/)?(sh|bash)$"
    sh_bash_c_pattern = r"^(?:.*/)?(sh|bash)\s+-c\s+"

    if (
        orig_cmd_name == "sh"
        or orig_cmd_name == "bash"
        or re.search(sh_bash_pattern, orig_cmd_name)
    ) and re.search(sh_bash_c_pattern, orig_cmd_full):

        temp_unwrapped_cmd = re.sub(sh_bash_c_pattern, "", orig_cmd_full, 1)

        # Remove surrounding quotes
        if (
            temp_unwrapped_cmd.startswith('"') and temp_unwrapped_cmd.endswith('"')
        ) or (temp_unwrapped_cmd.startswith("'") and temp_unwrapped_cmd.endswith("'")):
            temp_unwrapped_cmd = temp_unwrapped_cmd[1:-1]

        if temp_unwrapped_cmd:
            cmd_full = temp_unwrapped_cmd
            parts = cmd_full.split(maxsplit=1)
            new_cmd_name_full = parts[0]
            # Extract basename if it's a path
            cmd_name = os.path.basename(new_cmd_name_full)

    # --- Classification rules operate on potentially unwrapped cmd_name and cmd_full ---
    # cmd_name is now the basename of the (potentially unwrapped) command
    # cmd_full is the (potentially unwrapped) full command string

    # C/C++ 编译 (cc1, cc1plus, or compiler driver with -c)
    if (
        cmd_name in ("cc1", "cc1plus")
        or re.search(r"(?:^|/)(cc1|cc1plus)\s", cmd_full)
        or (
            (
                cmd_name in ("gcc", "g++", "clang", "clang++", "cc")
                or re.search(r"(?:^|/)(gcc|g\+\+|clang|clang\+\+|cc)\s", cmd_full)
            )
            and re.search(r"\s-c(\s|$)", cmd_full)
        )
    ):
        return "compile_c_cpp"

    # 链接 (ld, lld, collect2, or compiler driver without -c and with -o)
    if (
        cmd_name in ("ld", "lld", "collect2")
        or re.search(r"(?:^|/)(ld|lld|collect2)\s", cmd_full)
        or (
            (
                cmd_name in ("gcc", "g++", "clang", "clang++", "cc")
                or re.search(r"(?:^|/)(gcc|g\+\+|clang|clang\+\+|cc)\s", cmd_full)
            )
            and not re.search(r"\s-c(\s|$)", cmd_full)
            and re.search(r"\s-o\s", cmd_full)
            and cmd_name not in ("cc1", "cc1plus")
        )
    ):
        return "link"

    # 汇编
    if cmd_name == "as" or re.search(r"(?:^|/)as\s", cmd_full):
        return "assemble"

    # Java 编译
    if cmd_name == "javac" or re.search(r"(?:^|/)javac\s", cmd_full):
        return "compile_java"

    # Android Dexing/D8/R8
    if cmd_name in ("dx", "d8", "r8") or re.search(r"(?:^|/)(dx|d8|r8)\s", cmd_full):
        return "dex_android"

    # 构建系统进程
    build_systems = ["make", "ninja", "meson", "kati", "soong_ui", "siso", "gomacc"]
    if cmd_name in build_systems or re.search(
        r"(?:^|/)(" + "|".join(build_systems) + r")\s", cmd_full
    ):
        return (
            f"build_system_{cmd_name}" if cmd_name in build_systems else "build_system"
        )

    # Python 脚本
    if cmd_name.startswith("python") or re.search(
        r"(?:^|/)python[0-9._-]*\s", cmd_full
    ):
        script_name_match = re.search(r"([a-zA-Z0-9_./-]+\.py)", cmd_full)
        module_match = re.search(r"-m\s+([a-zA-Z0-9_.-]+)", cmd_full)
        script_name_extracted = ""

        if script_name_match:
            script_name_extracted = os.path.basename(script_name_match.group(1))
        elif module_match:
            script_name_extracted = f"module_{module_match.group(1)}"

        if script_name_extracted:
            clean_script_name = script_name_extracted.replace(".py", "")
            # Avoid classifying the main monitor script with its own name if it's run as a .py file
            # For example, if mymonitor itself was a .py file and matched.
            # However, it's usually run as an installed script, so cmd_name might be 'mymonitor'
            # and script_name_extracted might be empty or the script path.
            # If script_name_extracted is the entry point script name like "mymonitor",
            # it's better to classify it as "script_python".
            if clean_script_name == "mymonitor" or clean_script_name.startswith(
                "module_mymonitor"
            ):
                return "script_python"
            return f"script_python_{clean_script_name}"
        return "script_python"  # Fallback for other python invocations

    # Android 资源处理 (aapt, aapt2)
    if cmd_name in ("aapt", "aapt2") or re.search(r"(?:^|/)(aapt2?)\s", cmd_full):
        return "android_resource"

    # Android 代码生成 (metalava, aidl, hidl-gen)
    android_codegen_tools = ["metalava", "aidl", "hidl-gen"]
    if cmd_name in android_codegen_tools or re.search(
        r"(?:^|/)(" + "|".join(android_codegen_tools) + r")\s", cmd_full
    ):
        return "android_codegen"

    # Chromium 特有的一些进程 (gn, mojo)
    chromium_tools = ["gn", "mojo"]
    if cmd_name in chromium_tools or re.search(
        r"(?:^|/)(" + "|".join(chromium_tools) + r")\s", cmd_full
    ):
        return (
            f"chromium_tool_{cmd_name}"
            if cmd_name in chromium_tools
            else "chromium_tool"
        )

    # QEMU configure
    if (
        cmd_name == "configure" or re.search(r"(?:^|/)configure\s", cmd_full)
    ) and "qemu" in cmd_full:
        return "qemu_configure"

    return f"other_{cmd_name}"


def run_command(command: str, cwd: Path, shell: bool = False) -> Tuple[int, str, str]:
    """Executes a command and returns exit code, stdout, stderr."""
    logger.info(f"Executing command in {cwd}: {command}")
    try:
        process = subprocess.run(
            (
                command if shell else command.split()
            ),  # For simple commands not needing shell features
            cwd=cwd,
            capture_output=True,
            text=True,
            shell=shell,  # shell=True if command is a string to be interpreted by the shell
            check=False,
        )
        if process.returncode != 0:
            logger.warning(
                f"Command '{command}' failed with exit code {process.returncode} in {cwd}."
            )
            if process.stdout:
                logger.warning(f"Stdout from failed command:\n{process.stdout.strip()}")
            if process.stderr:
                logger.warning(f"Stderr from failed command:\n{process.stderr.strip()}")
        else:
            if process.stdout:  # Log stdout even for successful commands at DEBUG level
                logger.debug(f"Stdout: {process.stdout.strip()}")
            if (
                process.stderr
            ):  # Log stderr even for successful commands at DEBUG level (some tools use stderr for info)
                logger.debug(f"Stderr: {process.stderr.strip()}")
            logger.info(
                f"Command finished successfully with exit code {process.returncode}"
            )
        return process.returncode, process.stdout, process.stderr
    except Exception as e:
        logger.error(f"Failed to execute command '{command}': {e}", exc_info=True)
        return -1, "", str(e)


def run_and_monitor_build(
    project_config: Dict[str, Any],
    parallelism_level: int,
    monitoring_interval: int,
    log_dir: Path, # This is the run-specific directory from main.py
    collector_type: str = "pss_psutil", 
):
    global current_build_proc, active_memory_collector
    project_name = project_config["NAME"]
    project_dir = Path(project_config["DIR"])
    build_command_template = project_config["BUILD_COMMAND_TEMPLATE"]
    process_pattern = project_config["PROCESS_PATTERN"]
    clean_command_template = project_config.get("CLEAN_COMMAND_TEMPLATE", "")
    setup_command_template = project_config.get("SETUP_COMMAND_TEMPLATE", "")

    current_timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    
    # Define filenames for CSV and the new summary log
    base_filename_part = f"{project_name}_j{parallelism_level}_mem_{collector_type}_{current_timestamp_str}"
    output_csv_filename = f"{base_filename_part}.csv"
    output_summary_log_filename = f"{base_filename_part}_summary.log" # New summary log file name
    
    output_csv_file = log_dir / output_csv_filename
    output_summary_log_file = log_dir / output_summary_log_filename # Path for the new summary log

    # pidstat aux log will be temporary and merged into the summary log
    collector_aux_log_file = log_dir / f"{base_filename_part}_pidstat_aux.log" 

    actual_build_command = build_command_template.replace("<N>", str(parallelism_level))

    # --- Initial Logging to Summary Log File and Console ---
    summary_log_header_content = []
    summary_log_header_content.append("=" * 80)
    summary_log_header_content.append(f"Project: {project_name}")
    summary_log_header_content.append(f"Parallelism: -j{parallelism_level}")
    summary_log_header_content.append(f"Memory Metric: {collector_type.upper()}")
    summary_log_header_content.append(f"Source Directory: {project_dir}")
    summary_log_header_content.append(f"Build Command: {actual_build_command}")
    summary_log_header_content.append(f"Process Pattern (for collector): {process_pattern}")
    summary_log_header_content.append(f"Monitoring Interval (approx): {monitoring_interval} seconds")
    summary_log_header_content.append(f"Log Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_log_header_content.append(f"Output CSV File: {output_csv_file.name}")
    summary_log_header_content.append(f"Summary Log File: {output_summary_log_file.name}")
    if collector_type == "rss_pidstat":
        summary_log_header_content.append(f"Pidstat Aux Log (temporary, will be merged): {collector_aux_log_file.name}")
    summary_log_header_content.append("-" * 80)

    # Write initial info to console logger (using '-' for section breaks to differentiate)
    for line in summary_log_header_content:
        logger.info(line.replace("=", "-")) 

    # Write initial info to the summary log file
    with open(output_summary_log_file, "w") as f_summary:
        for line in summary_log_header_content:
            f_summary.write(line + "\n")
        f_summary.write("\n") # Add a newline for separation before next content

    if not project_dir.is_dir():
        error_msg = f"Project directory '{project_dir}' does not exist. Skipping."
        logger.error(error_msg)
        with open(output_summary_log_file, "a") as f_summary:
            f_summary.write(f"ERROR: {error_msg}\n")
        logger.info("=" * 80) # Console section end
        return

    collector_kwargs = {}
    if collector_type == "rss_pidstat":
        collector_kwargs["pidstat_stderr_file"] = collector_aux_log_file # pidstat still writes to its aux file first
        active_memory_collector = RssPidstatCollector(
            process_pattern, monitoring_interval, **collector_kwargs
        )
    elif collector_type == "pss_psutil":
        active_memory_collector = PssPsutilCollector(
            process_pattern, monitoring_interval, **collector_kwargs
        )
    else:
        error_msg = f"Unsupported collector type: {collector_type}. Exiting."
        logger.error(error_msg)
        with open(output_summary_log_file, "a") as f_summary:
            f_summary.write(f"ERROR: {error_msg}\n")
        return
    
    metric_fields_header = active_memory_collector.get_metric_fields()
    primary_metric_to_track = metric_fields_header[0] if metric_fields_header else None
    csv_header_parts = ["Timestamp_epoch", "Category"] + metric_fields_header + ["PID", "Command_Name", "Full_Command"]
    csv_header = ",".join(csv_header_parts)

    # Write only header to CSV file (no more metadata comments)
    with open(output_csv_file, "w") as f_csv:
        f_csv.write(csv_header + "\n")

    # Helper function to run setup/clean commands and log their output to the summary file
    def _run_and_log_command_to_summary(cmd_str: str, cmd_desc: str, cwd_path: Path, shell_bool: bool, summary_file: Path):
        logger.info(f"Executing {cmd_desc} command: {cmd_str}")
        with open(summary_file, "a") as f_s:
            f_s.write(f"--- {cmd_desc} Command ---\n")
            f_s.write(f"Command: {cmd_str}\n")
            f_s.write(f"Directory: {cwd_path}\n")
        
        exit_code, stdout_val, stderr_val = run_command(cmd_str, cwd=cwd_path, shell=shell_bool)
        
        with open(summary_file, "a") as f_s:
            f_s.write(f"Exit Code: {exit_code}\n")
            if stdout_val: f_s.write(f"Stdout:\n{stdout_val.strip()}\n")
            if stderr_val: f_s.write(f"Stderr:\n{stderr_val.strip()}\n")
            f_s.write(f"--- End {cmd_desc} Command ---\n\n")

        if exit_code != 0:
            logger.warning(f"{cmd_desc} command failed (code {exit_code}). Continuing...")
        else:
            logger.info(f"{cmd_desc} command completed.")
        return exit_code

    if setup_command_template:
        _use_shell_for_setup = True 
        _setup_cmd_to_run = setup_command_template
        if "source " in setup_command_template: # Handle 'source' for shell environment setup
            _setup_cmd_to_run = f"bash -c '. {setup_command_template}'" # Use '.' for sourcing in bash -c
        _run_and_log_command_to_summary(_setup_cmd_to_run, "Setup", project_dir, _use_shell_for_setup, output_summary_log_file)

    if clean_command_template:
        _run_and_log_command_to_summary(clean_command_template, "Clean", project_dir, True, output_summary_log_file)

    category_stats: Dict[str, Dict[str, Any]] = {} 
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None
    build_exit_code = -1
    start_time_seconds: Optional[float] = None
    peak_overall_memory_kb = 0
    peak_overall_memory_epoch = 0

    try:
        active_memory_collector.start()
        logger.info(f"Starting build command: {actual_build_command}")
        with open(output_summary_log_file, "a") as f_summary:
            f_summary.write(f"--- Build Command Execution ---\n")
            f_summary.write(f"Command: {actual_build_command}\n")
            f_summary.write(f"Directory: {project_dir}\n")
            f_summary.write(f"Build Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


        start_time_seconds = time.time()
        build_cmd_to_run = f"bash -c '{actual_build_command}'"
        current_build_proc = subprocess.Popen(
            build_cmd_to_run, cwd=project_dir, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, shell=True, bufsize=1
        )

        if current_build_proc.stdout:
            stdout_thread = threading.Thread(target=_stream_output, args=(current_build_proc.stdout, f"[{project_name}-j{parallelism_level} STDOUT] "), daemon=True)
            stdout_thread.start()
        if current_build_proc.stderr:
            stderr_thread = threading.Thread(target=_stream_output, args=(current_build_proc.stderr, f"[{project_name}-j{parallelism_level} STDERR] "), daemon=True)
            stderr_thread.start()
            
        # Append data to CSV file
        with open(output_csv_file, "a") as f_csv: 
            for samples_at_interval in active_memory_collector.read_samples():
                current_epoch_time = int(time.time()) 
                current_interval_sum_kb = 0

                if not samples_at_interval:
                    if current_build_proc and current_build_proc.poll() is not None:
                        logger.info("Build process ended while collector was active.")
                        break 
                    continue

                for sample in samples_at_interval:
                    category = get_process_category(sample.command_name, sample.full_command)
                    if category == "ignore_vscode_server": 
                        continue

                    row_data = [str(current_epoch_time), category]
                    for metric_name in metric_fields_header: 
                        row_data.append(str(sample.metrics.get(metric_name, ""))) 
                    escaped_full_command = sample.full_command.replace('"', '""')
                    row_data.extend([sample.pid, sample.command_name, f'"{escaped_full_command}"'])
                    
                    f_csv.write(",".join(row_data) + "\n")

                    if primary_metric_to_track:
                        metric_value = sample.metrics.get(primary_metric_to_track)
                        if metric_value is not None:
                            try:
                                current_val = int(metric_value) 
                                if category not in category_stats or current_val > category_stats[category].get("peak_metric", float('-inf')):
                                    category_stats[category] = {
                                        "peak_metric": current_val,
                                        "pid": sample.pid,
                                        "command": sample.command_name
                                    }
                                current_interval_sum_kb += current_val
                            except ValueError:
                                logger.warning(f"Could not convert metric '{primary_metric_to_track}' value '{metric_value}' to int for category stats or sum.")
                
                if primary_metric_to_track and current_interval_sum_kb > peak_overall_memory_kb:
                    peak_overall_memory_kb = current_interval_sum_kb
                    peak_overall_memory_epoch = current_epoch_time
                
                f_csv.flush() 

                if current_build_proc and current_build_proc.poll() is not None:
                    logger.info("Build process finished. Stopping memory collection for this build.")
                    break 
        
        if current_build_proc:
            build_exit_code = current_build_proc.wait()
            logger.info(f"Build process exited with code: {build_exit_code}")
            with open(output_summary_log_file, "a") as f_summary:
                f_summary.write(f"Build End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f_summary.write(f"Build Exit Code: {build_exit_code}\n")
                f_summary.write(f"--- End Build Command Execution ---\n\n")

        if stdout_thread and stdout_thread.is_alive(): stdout_thread.join(timeout=5)
        if stderr_thread and stderr_thread.is_alive(): stderr_thread.join(timeout=5)
        logger.info("Build process stdout/stderr streaming finished.")

    except Exception as e:
        error_msg = f"An error occurred during monitoring for {project_name} j{parallelism_level}: {e}"
        logger.error(error_msg, exc_info=True)
        with open(output_summary_log_file, "a") as f_summary:
            f_summary.write(f"\nCRITICAL ERROR DURING MONITORING:\n{error_msg}\nDetails: {str(e)}\n")
        if current_build_proc and current_build_proc.poll() is None: 
             build_exit_code = current_build_proc.wait() 
    finally:
        if active_memory_collector:
            logger.info(f"Stopping memory collector ({active_memory_collector.__class__.__name__})...")
            active_memory_collector.stop()
            active_memory_collector = None 
        
        if current_build_proc and current_build_proc.poll() is None:
            logger.warning(f"Build process {current_build_proc.pid} still running in finally block, attempting to terminate.")
            current_build_proc.terminate()
            try:
                current_build_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.error(f"Build process {current_build_proc.pid} did not terminate, killing.")
                current_build_proc.kill()
            current_build_proc = None 
        
        end_time_seconds = time.time()
        duration_seconds = int(end_time_seconds - start_time_seconds) if start_time_seconds is not None else 0
        h = duration_seconds // 3600
        m = (duration_seconds % 3600) // 60
        s = duration_seconds % 60
        formatted_duration = f"{h:02d}:{m:02d}:{s:02d}"

        # --- Final Summary Section for Summary Log File ---
        final_summary_log_lines = [] # Use a list to build content for summary log
        final_summary_log_lines.append("--- Build & Monitoring Summary ---")
        final_summary_log_lines.append(f"Build Duration: {formatted_duration} ({duration_seconds} seconds)")
        final_summary_log_lines.append(f"Build Exit Code: {build_exit_code}")
        
        if primary_metric_to_track:
            peak_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(peak_overall_memory_epoch)) if peak_overall_memory_epoch > 0 else "N/A"
            final_summary_log_lines.append(f"Peak Overall Memory ({primary_metric_to_track}): {peak_overall_memory_kb} KB (at {peak_time_str})")
        else:
            final_summary_log_lines.append(f"Peak Overall Memory: N/A (no primary metric identified for sum)")

        final_summary_log_lines.append(f"Peak Memory Usage Summary (by category, based on first metric: {primary_metric_to_track if primary_metric_to_track else 'N/A'}):")
        if category_stats:
            for category, stats in sorted(category_stats.items()):
                peak_val = stats.get('peak_metric', 'N/A')
                pid_val = stats.get('pid', 'N/A')
                cmd_val = stats.get('command', 'N/A')
                final_summary_log_lines.append(f"  - {category}: {peak_val} (PID: {pid_val}, Command: {cmd_val})")
        else:
            final_summary_log_lines.append("  No category-specific peak memory data collected or primary metric not found.")
        final_summary_log_lines.append("--- End Summary ---")

        with open(output_summary_log_file, "a") as f_summary:
            f_summary.write("\n") 
            
            # Merge pidstat aux log content if it exists
            if collector_type == "rss_pidstat" and collector_aux_log_file.exists() and collector_aux_log_file.stat().st_size > 0:
                f_summary.write("\n--- pidstat stderr output ---\n")
                try:
                    with open(collector_aux_log_file, "r") as f_aux:
                        f_summary.write(f_aux.read())
                    f_summary.write("--- End pidstat stderr output ---\n\n")
                    try: # Attempt to delete the aux file after merging
                        collector_aux_log_file.unlink() 
                        logger.info(f"Merged and deleted pidstat aux log: {collector_aux_log_file.name}")
                    except OSError as e_del:
                        logger.warning(f"Could not delete pidstat aux log {collector_aux_log_file.name}: {e_del}")
                except Exception as e_read_aux:
                    error_msg_aux = f"Error reading pidstat aux log {collector_aux_log_file.name}: {e_read_aux}"
                    f_summary.write(f"{error_msg_aux}\n")
                    logger.error(error_msg_aux)
            
            for line in final_summary_log_lines:
                f_summary.write(line + "\n")
        
        # Log final summary to console as well
        for line in final_summary_log_lines:
            logger.info(line)

        logger.info(f"Monitoring for {project_name} j{parallelism_level} ({collector_type}) finished.")
        logger.info(f"CSV data saved to: {output_csv_file}")
        logger.info(f"Summary log saved to: {output_summary_log_file}")
        logger.info("=" * 80)


def check_pidstat_installed():  # This might become check_collector_dependencies()
    # For now, RssPidstatCollector is the only one, so this is fine.
    # If PssPsutilCollector is added, psutil installation should also be checked or handled.
    try:
        subprocess.run(["pidstat", "-V"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def cleanup_processes():
    logger.info("Cleaning up any running subprocesses and collectors...")
    global current_build_proc, active_memory_collector
    if active_memory_collector:
        logger.info(
            f"Stopping active memory collector ({active_memory_collector.__class__.__name__}) from cleanup handler."
        )
        active_memory_collector.stop()
        active_memory_collector = None
    if current_build_proc and current_build_proc.poll() is None:
        logger.info(
            f"Terminating build process (PID {current_build_proc.pid}) from cleanup handler."
        )
        current_build_proc.terminate()
        try:
            current_build_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.error(
                f"Build process {current_build_proc.pid} did not terminate in cleanup, killing."
            )
            current_build_proc.kill()
    logger.info("Cleanup finished.")
