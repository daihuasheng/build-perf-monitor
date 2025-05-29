import re
import time
import subprocess
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, IO

logger = logging.getLogger(__name__)

# Global variable to hold the pidstat process, for cleanup
current_pidstat_proc: Optional[subprocess.Popen] = None
current_build_proc: Optional[subprocess.Popen] = None

def get_process_category(cmd_name: str, cmd_full: str) -> str:
    """
    Categorizes a process based on its command name and full command line.
    This is a Python translation of the awk get_process_category function.
    """
    # Filter out vscode-server processes early
    if ".vscode-server" in cmd_full:
        return "ignore_vscode_server" # Assign a specific category to ignore later

    orig_cmd_name = cmd_name
    orig_cmd_full = cmd_full

    # Attempt to unwrap commands executed via sh -c "..." or bash -c ...
    sh_bash_pattern = r"^(?:.*/)?(sh|bash)$"
    sh_bash_c_pattern = r"^(?:.*/)?(sh|bash)\s+-c\s+"

    if (orig_cmd_name == "sh" or orig_cmd_name == "bash" or
            re.search(sh_bash_pattern, orig_cmd_name)) and \
            re.search(sh_bash_c_pattern, orig_cmd_full):

        temp_unwrapped_cmd = re.sub(sh_bash_c_pattern, "", orig_cmd_full, 1)

        # Remove surrounding quotes
        if (temp_unwrapped_cmd.startswith('"') and temp_unwrapped_cmd.endswith('"')) or \
           (temp_unwrapped_cmd.startswith("'") and temp_unwrapped_cmd.endswith("'")):
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
    if (cmd_name in ("cc1", "cc1plus") or
            re.search(r"(?:^|/)(cc1|cc1plus)\s", cmd_full) or
            ((cmd_name in ("gcc", "g++", "clang", "clang++", "cc") or re.search(r"(?:^|/)(gcc|g\+\+|clang|clang\+\+|cc)\s", cmd_full)) and
             re.search(r"\s-c(\s|$)", cmd_full))):
        return "compile_c_cpp"

    # 链接 (ld, lld, collect2, or compiler driver without -c and with -o)
    if (cmd_name in ("ld", "lld", "collect2") or
            re.search(r"(?:^|/)(ld|lld|collect2)\s", cmd_full) or
            ((cmd_name in ("gcc", "g++", "clang", "clang++", "cc") or re.search(r"(?:^|/)(gcc|g\+\+|clang|clang\+\+|cc)\s", cmd_full)) and
             not re.search(r"\s-c(\s|$)", cmd_full) and
             re.search(r"\s-o\s", cmd_full) and
             cmd_name not in ("cc1", "cc1plus"))):
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
    if cmd_name in build_systems or re.search(r"(?:^|/)(" + "|".join(build_systems) + r")\s", cmd_full):
        return f"build_system_{cmd_name}" if cmd_name in build_systems else "build_system"


    # Python 脚本
    if cmd_name.startswith("python") or re.search(r"(?:^|/)python[0-9._-]*\s", cmd_full):
        script_name_match = re.search(r"([a-zA-Z0-9_./-]+\.py)", cmd_full)
        module_match = re.search(r"-m\s+([a-zA-Z0-9_.-]+)", cmd_full)
        script_name_extracted = ""

        if script_name_match:
            script_name_extracted = os.path.basename(script_name_match.group(1))
        elif module_match:
            script_name_extracted = f"module_{module_match.group(1)}"

        if script_name_extracted:
            # Specific QEMU scripts (add more as needed)
            if script_name_extracted == "qapi-gen.py": return "py_qapi_gen"
            if script_name_extracted == "decodetree.py": return "py_decodetree"
            if script_name_extracted == "feature_to_c.py": return "py_feature_to_c"
            # Generic with script name, removing .py for cleaner category
            clean_script_name = script_name_extracted.replace('.py', '')
            return f"script_python_{clean_script_name}"
        return "script_python" # Fallback if no specific script name extracted


    # Android 资源处理 (aapt, aapt2)
    if cmd_name in ("aapt", "aapt2") or re.search(r"(?:^|/)(aapt2?)\s", cmd_full):
        return "android_resource"

    # Android 代码生成 (metalava, aidl, hidl-gen)
    android_codegen_tools = ["metalava", "aidl", "hidl-gen"]
    if cmd_name in android_codegen_tools or re.search(r"(?:^|/)(" + "|".join(android_codegen_tools) + r")\s", cmd_full):
        return "android_codegen"
    
    # Chromium 特有的一些进程 (gn, mojo)
    chromium_tools = ["gn", "mojo"]
    if cmd_name in chromium_tools or re.search(r"(?:^|/)(" + "|".join(chromium_tools) + r")\s", cmd_full):
        return f"chromium_tool_{cmd_name}" if cmd_name in chromium_tools else "chromium_tool"

    # QEMU configure
    if (cmd_name == "configure" or re.search(r"(?:^|/)configure\s", cmd_full)) and "qemu" in cmd_full:
        return "qemu_configure"

    return f"other_{cmd_name}"


def run_command(command: str, cwd: Path, shell: bool = False) -> Tuple[int, str, str]:
    """Executes a command and returns exit code, stdout, stderr."""
    logger.info(f"Executing command in {cwd}: {command}")
    try:
        process = subprocess.run(
            command if shell else command.split(), # For simple commands not needing shell features
            cwd=cwd,
            capture_output=True,
            text=True,
            shell=shell, # shell=True if command is a string to be interpreted by the shell
            check=False
        )
        if process.returncode != 0:
            logger.warning(f"Command '{command}' failed with exit code {process.returncode} in {cwd}.")
            if process.stdout:
                logger.warning(f"Stdout from failed command:\n{process.stdout.strip()}")
            if process.stderr:
                logger.warning(f"Stderr from failed command:\n{process.stderr.strip()}")
        else:
            if process.stdout: # Log stdout even for successful commands at DEBUG level
                logger.debug(f"Stdout: {process.stdout.strip()}")
            if process.stderr: # Log stderr even for successful commands at DEBUG level (some tools use stderr for info)
                logger.debug(f"Stderr: {process.stderr.strip()}")
            logger.info(f"Command finished successfully with exit code {process.returncode}")
        return process.returncode, process.stdout, process.stderr
    except Exception as e:
        logger.error(f"Failed to execute command '{command}': {e}", exc_info=True)
        return -1, "", str(e)


def run_and_monitor_build(
    project_config: Dict[str, Any],
    parallelism_level: int,
    monitoring_interval: int,
    log_dir: Path
):
    global current_pidstat_proc, current_build_proc
    project_name = project_config["NAME"]
    project_dir = Path(project_config["DIR"])
    build_command_template = project_config["BUILD_COMMAND_TEMPLATE"]
    process_pattern = project_config["PROCESS_PATTERN"]
    clean_command_template = project_config.get("CLEAN_COMMAND_TEMPLATE", "")
    setup_command_template = project_config.get("SETUP_COMMAND_TEMPLATE", "")

    current_timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Change to .csv extension
    output_log_filename = f"{project_name}_j{parallelism_level}_mem_{current_timestamp}.csv"
    output_log_file = log_dir / output_log_filename
    # pidstat_stderr_filename can remain .log
    pidstat_stderr_filename = f"{project_name}_j{parallelism_level}_pidstat_stderr_{current_timestamp}.log"
    pidstat_stderr_file = log_dir / pidstat_stderr_filename

    actual_build_command = build_command_template.replace("<N>", str(parallelism_level))

    logger.info("=" * 80)
    logger.info(f"Project: {project_name}")
    logger.info(f"Parallelism: -j{parallelism_level}")
    logger.info(f"Source Directory: {project_dir}")
    logger.info(f"Build Command: {actual_build_command}")
    logger.info(f"Process Pattern (pidstat -C): {process_pattern}")
    logger.info(f"Memory Log (CSV): {output_log_file}")
    logger.info("-" * 80)

    if not project_dir.is_dir():
        logger.error(f"Project directory '{project_dir}' does not exist. Skipping.")
        logger.info("=" * 80)
        return

    # Write log header to CSV
    header_info = [
        f"# Project: {project_name}",
        f"# Parallelism: -j{parallelism_level}",
        f"# Source Directory: {project_dir}",
        f"# Build Command: {actual_build_command}",
        f"# Process Pattern (for pidstat -C): {process_pattern}",
        f"# Monitoring Interval: {monitoring_interval} seconds",
        f"# Log Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    # CSV Header row
    csv_header = "Timestamp_epoch,Category,RSS_KB,VSZ_KB,PID,Command_Name,Full_Command"
    
    with open(output_log_file, "w") as f:
        for line in header_info:
            f.write(line + "\n")
        f.write(csv_header + "\n") # Write the actual CSV header

    # Execute setup command if defined
    if setup_command_template:
        logger.info(f"Executing setup command: {setup_command_template}")
        # If 'source' is present, or if it's a complex command, it likely needs a shell.
        # For simplicity and consistency with build command, we can default to using bash -c
        # if the command is non-trivial or explicitly needs shell features.
        # However, run_command itself takes a 'shell' boolean.
        # If setup_command_template is simple like "mkdir foo", shell=False is fine.
        # If it's "source env.sh && cmd", shell=True and passing the whole string is needed.

        # Let's refine how shell is determined for setup_cmd_to_run
        # If "source " is in the command, it definitely needs a shell that understands 'source'.
        # We'll pass the full string to `bash -c` in this case.
        # For other commands, we can let `run_command` decide based on its `shell` param.
        # The `run_command` function itself will handle `command.split()` if `shell=False`.

        # For setup commands, if they contain "source", we must use bash -c.
        # Otherwise, for commands like "make clean", shell=True with the string is robust.
        _use_shell_for_setup = True # Default to True for setup/clean for robustness with make etc.
        _setup_cmd_to_run = setup_command_template
        if "source " in setup_command_template:
            _setup_cmd_to_run = f"bash -c '{setup_command_template}'"
            # shell=True is already implied by passing the string to run_command with shell=True

        exit_code, _, _ = run_command(_setup_cmd_to_run, cwd=project_dir, shell=_use_shell_for_setup)
        if exit_code != 0:
            logger.warning(f"Setup command '{setup_command_template}' failed with exit code {exit_code}. Continuing...")
        else:
            logger.info("Setup command completed successfully.")


    # Execute clean command if defined
    if clean_command_template:
        logger.info(f"Executing clean command: {clean_command_template}")
        # Most clean commands (e.g., "make clean") work well with shell=True
        exit_code, _, _ = run_command(clean_command_template, cwd=project_dir, shell=True)
        if exit_code != 0:
            logger.warning(f"Clean command '{clean_command_template}' failed with exit code {exit_code}. Continuing...")
        else:
            logger.info("Clean command completed successfully.")

    pidstat_cmd = ["pidstat", "-r", "-l", "-C", process_pattern, str(monitoring_interval)]
    pidstat_env = os.environ.copy()
    pidstat_env["LC_ALL"] = "C"

    category_stats: Dict[str, Dict[str, Any]] = {}

    try:
        with open(pidstat_stderr_file, "w") as ps_err_f:
            logger.info(f"Starting pidstat: {' '.join(pidstat_cmd)}")
            current_pidstat_proc = subprocess.Popen(
                pidstat_cmd,
                stdout=subprocess.PIPE,
                stderr=ps_err_f,
                text=True,
                bufsize=1, 
                env=pidstat_env
            )

            logger.info(f"Starting build command: {actual_build_command}")
            start_time_seconds = time.time()
            
            # Always use bash -c for build commands to handle complex commands,
            # environment variables, and shell features consistently.
            # The command string itself is passed to bash -c.
            build_cmd_to_run = f"bash -c '{actual_build_command}'"

            current_build_proc = subprocess.Popen(
                build_cmd_to_run,
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )

            if current_pidstat_proc.stdout:
                for line in iter(current_pidstat_proc.stdout.readline, ""):
                    line = line.strip()
                    if not line or "Linux" in line or line.startswith("Average:"):
                        continue
                    
                    if re.match(r"^[0-9]{2}:[0-9]{2}:[0-9]{2}", line):
                        parts = line.split()
                        if len(parts) < 9 or parts[2] == "PID": 
                            continue

                        try:
                            current_epoch = int(time.time()) 
                            pid = parts[2]
                            vsz_kb = parts[5]
                            rss_kb = int(parts[6]) 
                            
                            command_name_short = parts[8]
                            command_full_str = " ".join(parts[8:])

                            category = get_process_category(command_name_short, command_full_str)

                            # Update stats
                            if category not in category_stats:
                                category_stats[category] = {"total_rss": 0, "peak_rss": 0, "count": 0}
                            category_stats[category]["total_rss"] += rss_kb
                            category_stats[category]["count"] += 1
                            if rss_kb > category_stats[category]["peak_rss"]:
                                category_stats[category]["peak_rss"] = rss_kb
                            
                            # Write data row as CSV
                            # Ensure fields with potential commas or special chars are handled if necessary
                            # For now, assuming simple string/numeric fields.
                            # If Command_Name or Full_Command can contain commas, they should be quoted.
                            # For simplicity, we'll join with comma. Pandas can usually handle this.
                            csv_data_row = f"{current_epoch},{category},{rss_kb},{vsz_kb},{pid},{command_name_short},\"{command_full_str.replace('\"', '\"\"')}\""

                            with open(output_log_file, "a") as f_out:
                                f_out.write(csv_data_row + "\n")

                        except (IndexError, ValueError) as e:
                            logger.warning(f"Error parsing pidstat line: '{line}'. Error: {e}")
                            continue
                    
                    # Check if build process has finished
                    if current_build_proc and current_build_proc.poll() is not None:
                        logger.info("Build process finished, stopping pidstat line reading.")
                        break
            
            # Ensure build process is waited for
            build_exit_code = -1
            if current_build_proc:
                build_stdout, build_stderr = current_build_proc.communicate() # Wait for completion
                build_exit_code = current_build_proc.returncode
                logger.info(f"Build stdout:\n{build_stdout}")
                if build_stderr:
                     logger.error(f"Build stderr:\n{build_stderr}")


        end_time_seconds = time.time()
        duration_seconds = int(end_time_seconds - start_time_seconds)
        h = duration_seconds // 3600
        m = (duration_seconds % 3600) // 60
        s = duration_seconds % 60
        formatted_duration = f"{h:02d}:{m:02d}:{s:02d}"


        logger.info(f"Build command finished with exit code: {build_exit_code}")
        logger.info(f"Build total time: {formatted_duration} ({duration_seconds} seconds)")

        # Write summary to CSV file as comments
        with open(output_log_file, "a") as f_out:
            f_out.write(f"# Build Duration: {formatted_duration} ({duration_seconds} seconds)\n")
            f_out.write("\n# === Memory Usage Summary by Category ===\n")
            # Summary header as comment
            f_out.write(f"# {'Category':<30} , {'Total RSS (KB)':>15} , {'Peak RSS (KB)':>15} , {'Count':>15}\n")
            f_out.write(f"# {'-'*30} , {'-'*15} , {'-'*15} , {'-'*15}\n")
            for cat, stats in sorted(category_stats.items()):
                # Summary data as comment
                f_out.write(f"# {cat:<30} , {stats['total_rss']:>15} , {stats['peak_rss']:>15} , {stats['count']:>15}\n")
            f_out.write("# === End of Summary ===\n")

        if build_exit_code == 0:
            logger.info(f"Project '{project_name}' (j{parallelism_level}) built successfully.")
        else:
            logger.warning(f"Project '{project_name}' (j{parallelism_level}) build failed with exit code {build_exit_code}.")

    except Exception as e:
        logger.error(f"An error occurred during monitoring for {project_name} j{parallelism_level}: {e}", exc_info=True)
    finally:
        if current_pidstat_proc and current_pidstat_proc.poll() is None:
            logger.info(f"Terminating pidstat process (PID: {current_pidstat_proc.pid})...")
            current_pidstat_proc.terminate()
            try:
                current_pidstat_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"pidstat process (PID: {current_pidstat_proc.pid}) did not terminate, killing...")
                current_pidstat_proc.kill()
            current_pidstat_proc = None
        
        if current_build_proc and current_build_proc.poll() is None: # Should be finished by now
            logger.info(f"Terminating build process (PID: {current_build_proc.pid}) if still running...")
            current_build_proc.terminate()
            try:
                current_build_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                current_build_proc.kill()
            current_build_proc = None
        logger.info(f"Monitoring for {project_name} j{parallelism_level} finished.")
        logger.info("=" * 80)

def check_pidstat_installed():
    try:
        subprocess.run(["pidstat", "-V"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def cleanup_processes():
    """Ensures monitored processes are cleaned up on exit."""
    logger.info("Cleaning up any running subprocesses...")
    global current_pidstat_proc, current_build_proc
    if current_pidstat_proc and current_pidstat_proc.poll() is None:
        logger.info(f"Terminating pidstat (PID {current_pidstat_proc.pid}) from cleanup handler.")
        current_pidstat_proc.terminate()
        current_pidstat_proc.kill() # Force kill if terminate doesn't work quickly
    if current_build_proc and current_build_proc.poll() is None:
        logger.info(f"Terminating build process (PID {current_build_proc.pid}) from cleanup handler.")
        current_build_proc.terminate()
        current_build_proc.kill()
    logger.info("Cleanup finished.")