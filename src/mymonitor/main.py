"""
Main command-line entry point for the MyMonitor application.

This script serves as the primary interface for users to run, monitor, and
analyze build processes. It orchestrates the entire workflow, from parsing
command-line arguments and loading configurations to executing builds,
collecting memory data, and generating reports and plots.

Key functionalities include:
- Parsing command-line arguments to provide flexible, single-run configurations.
- Loading default settings from TOML configuration files.
- Overriding file-based configurations with command-line arguments.
- Handling different execution modes, such as full monitoring runs or clean-only actions.
- Iterating through specified projects and parallelism levels.
- Invoking the core monitoring logic from `monitor_utils`.
- Triggering post-run analysis and plot generation from `plotter`.
- Ensuring graceful shutdown on system signals (e.g., Ctrl+C).
"""

import argparse
import logging
import psutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Local application imports
from . import config
from .data_models import RunContext
from .monitor_utils import (
    _execute_clean_step,
    _generate_run_paths,
    cleanup_processes,
    run_and_monitor_build,
)
from .process_utils import check_pidstat_installed

# --- Logging Setup ---
# Configure a basic logger to output informational messages to the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def signal_handler(sig: int, frame: Any) -> None:
    """Handles termination signals (SIGINT, SIGTERM) for graceful shutdown.

    This function is registered as a signal handler. When a signal like
    Ctrl+C is received, it logs a warning, attempts to clean up any running
    subprocesses via `cleanup_processes`, and then exits the application.

    Args:
        sig: The signal number received.
        frame: The current stack frame at the time of the signal.
    """
    logger.warning(f"Signal {signal.Signals(sig).name} received, initiating cleanup...")
    cleanup_processes()
    sys.exit(1)


def main_cli() -> None:
    """
    The main command-line interface (CLI) function for MyMonitor.

    This function orchestrates the entire application lifecycle:
    1.  Sets up signal handlers for graceful termination.
    2.  Defines and parses command-line arguments using `argparse`. These
        arguments allow for flexible, on-the-fly configuration.
    3.  Loads the base configuration from TOML files (`config.toml`,
        `projects.toml`, `rules.toml`).
    4.  Overrides the loaded configuration with any values provided via
        command-line arguments (CLI arguments have the highest priority).
    5.  Executes the requested action, which can be:
        - A "clean-only" run to prepare the build environment.
        - A full monitoring run, iterating through selected projects and
          parallelism levels.
    6.  After the monitoring run, it triggers the plot generation process
        unless disabled.
    """
    # Register signal handlers to ensure cleanup on Ctrl+C or other term signals.
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # --- Argument Parsing ---
    # Configure the argument parser with a description and help text formatting.
    # `argparse` automatically handles the -h/--help argument.
    parser = argparse.ArgumentParser(
        description="Monitor and analyze build process memory usage.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Group: Target Selectors - Arguments that specify *what* to run.
    parser.add_argument(
        "-p",
        "--project",
        nargs="+",
        metavar="NAME",
        help="Specify one or more project names to run (e.g., qemu chromium).\nThis overrides the default behavior of running all projects.",
    )

    # Group: Behavior Modifiers - Arguments that change *how* the run is executed.
    parser.add_argument(
        "-j",
        "--jobs",
        nargs="+",
        type=int,
        metavar="N",
        help="Specify parallelism levels (e.g., 4 16).\nOverrides 'default_jobs' in config.toml.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        metavar="SEC",
        help="Override the monitoring interval in seconds (e.g., 0.5).",
    )
    parser.add_argument(
        "--metric-type",
        choices=["pss_psutil", "rss_pidstat"],
        help="Override the memory metric collector type.",
    )
    parser.add_argument(
        "--no-pre-clean", action="store_true", help="Skip the pre-build clean step."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the plot generation step after the run.",
    )

    # Group: Action Flags - Arguments that define the main action to perform.
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only run the clean command for the selected projects and exit.",
    )

    # Group: Meta-Configuration - Arguments that control the tool's own configuration.
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a custom main config.toml file.",
    )
    args = parser.parse_args()

    # --- Configuration Loading ---
    # Load the application configuration. If a custom config path is provided via
    # the CLI, it will be used instead of the default.
    try:
        if args.config:
            if not args.config.is_file():
                logger.error(f"Specified config file not found: {args.config}")
                sys.exit(1)
            # This is the mechanism to point the singleton config loader to a different file.
            config._CONFIG_FILE_PATH = args.config
            config._CONFIG = None  # Force reload
        app_config = config.get_config()
    except Exception as e:
        logger.error(
            f"Failed to initialize application due to configuration error: {e}"
        )
        sys.exit(1)

    # --- Configuration Overriding ---
    # Apply command-line arguments to override the settings loaded from files.
    monitor_config = app_config.monitor
    projects_to_run = app_config.projects

    if args.project:
        project_map = {p.name: p for p in projects_to_run}
        filtered_projects = []
        for proj_name in args.project:
            if proj_name in project_map:
                filtered_projects.append(project_map[proj_name])
            else:
                logger.error(
                    f"Project '{proj_name}' not found. Available: {list(project_map.keys())}"
                )
                sys.exit(1)
        projects_to_run = filtered_projects

    if args.jobs:
        monitor_config.default_jobs = args.jobs
    if args.interval:
        monitor_config.interval_seconds = args.interval
    if args.metric_type:
        monitor_config.metric_type = args.metric_type
    if args.no_plots:
        monitor_config.skip_plots = True

    # --- Action Execution ---
    if not projects_to_run:
        logger.info("No projects selected to run. Exiting.")
        sys.exit(0)

    # Handle the --clean-only action flag.
    if args.clean_only:
        logger.info("--- Running in Clean-Only Mode ---")
        for project_config in projects_to_run:
            logger.info(f"Cleaning project: {project_config.name}")
            # Create a minimal context required for the clean step.
            dummy_paths = _generate_run_paths(Path("/tmp"), "clean", 0, "na", "na")
            run_context = RunContext(
                project_name=project_config.name,
                project_dir=project_config.dir,
                process_pattern="",
                actual_build_command="",
                parallelism_level=0,
                monitoring_interval=0,
                collector_type="",
                current_timestamp_str="",
                taskset_available=False,
                build_cores_target_str="",
                monitor_script_pinned_to_core_info="",
                monitor_core_id=None,
                paths=dummy_paths,
            )
            _execute_clean_step(run_context, project_config, "Clean-Only")
        logger.info("--- Clean-Only Mode Finished ---")
        sys.exit(0)

    # --- Main Monitoring Run ---
    # Verify system dependencies for the selected collector.
    if monitor_config.metric_type == "rss_pidstat" and not check_pidstat_installed():
        logger.error(
            "'pidstat' not found. Please install the 'sysstat' package to use the 'rss_pidstat' collector."
        )
        sys.exit(1)

    # Create a unique, timestamped directory for this run's output.
    current_run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_specific_log_dir_name = f"run_{current_run_timestamp}"
    current_run_output_dir = monitor_config.log_root_dir / run_specific_log_dir_name
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Logs and plots for this run will be saved in: {current_run_output_dir.resolve()}"
    )

    # --- CPU Pinning for the Monitor Script ---
    # This logic pins the main script itself to the configured core.
    actual_monitor_core_id: int | None = None
    monitor_script_pinned_to_core_info: str = "Not Pinned"

    if monitor_config.monitor_core >= 0:
        try:
            p = psutil.Process()
            available_cores = list(range(psutil.cpu_count()))
            if monitor_config.monitor_core in available_cores:
                p.cpu_affinity([monitor_config.monitor_core])
                actual_monitor_core_id = monitor_config.monitor_core
                monitor_script_pinned_to_core_info = f"Core {actual_monitor_core_id}"
                logger.info(
                    f"Successfully pinned main monitor script to CPU core {actual_monitor_core_id}."
                )
            else:
                monitor_script_pinned_to_core_info = (
                    f"Invalid Core ID ({monitor_config.monitor_core})"
                )
                logger.warning(
                    f"Configured monitor_core={monitor_config.monitor_core} is not available. Script will not be pinned."
                )
        except (AttributeError, NotImplementedError):
            monitor_script_pinned_to_core_info = "Not Supported on this OS"
            logger.warning(
                "CPU affinity is not supported on this OS. Script will not be pinned."
            )
        except Exception as e:
            monitor_script_pinned_to_core_info = f"Error ({e})"
            logger.error(
                f"An error occurred while trying to pin the monitor script: {e}",
                exc_info=True,
            )

    # Main loop: iterate through each selected project, and for each project,
    # iterate through each specified parallelism level.
    for project_config in projects_to_run:
        logger.info(f">>> Starting processing for project: {project_config.name}")
        for level in monitor_config.default_jobs:
            # Delegate the entire build-and-monitor task to the utility function.
            run_and_monitor_build(
                project_config=project_config,
                parallelism_level=level,
                monitoring_interval=monitor_config.interval_seconds,
                log_dir=current_run_output_dir,
                collector_type=monitor_config.metric_type,
                skip_pre_clean=args.no_pre_clean,
                monitor_core_id_for_collector_and_build_avoidance=actual_monitor_core_id,
                build_cpu_cores_policy=monitor_config.build_cores_policy,
                specific_build_cores_str=monitor_config.specific_build_cores,
                monitor_script_pinned_to_core_info=monitor_script_pinned_to_core_info,
            )
        logger.info(f"<<< Finished processing for project: {project_config.name}")

    logger.info("All specified build and monitoring tasks completed.")

    # --- Finalization ---
    # Generate plots from the collected data unless skipped.
    if not monitor_config.skip_plots:
        logger.info("--- Starting plot generation via external tool ---")
        # Execute the plotter tool as a separate process.
        # This decouples the main application from the plotting logic.
        plotter_script_path = Path(__file__).parent.parent.parent / "tools" / "plotter.py"
        try:
            # Use sys.executable to ensure the same Python interpreter is used.
            command = [
                sys.executable,
                str(plotter_script_path),
                "--log-dir",
                str(current_run_output_dir),
            ]
            logger.info(f"Executing plotter command: {' '.join(command)}")
            result = subprocess.run(
                command,
                check=True,  # Raise an exception if the plotter fails
                capture_output=True,
                text=True,
            )
            logger.info("Plotter tool output:\n" + result.stdout)
            if result.stderr:
                logger.warning("Plotter tool stderr:\n" + result.stderr)
            logger.info("--- Plot generation finished ---")
        except FileNotFoundError:
            logger.error(f"Plotter script not found at: {plotter_script_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Plot generation tool failed with exit code {e.returncode}:")
            logger.error("STDOUT:\n" + e.stdout)
            logger.error("STDERR:\n" + e.stderr)
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during plot generation: {e}",
                exc_info=True,
            )
    else:
        logger.info("Skipping plot generation as per config or command-line flag.")


# Standard Python entry point guard.
if __name__ == "__main__":
    main_cli()
