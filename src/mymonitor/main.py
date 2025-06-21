"""
Main script for the Build Memory Profiler.

This script orchestrates the process of building specified projects with varying
parallelism levels, monitoring their resource consumption (CPU, memory),
and generating plots from the collected data.

It supports configuration for multiple projects, allowing users to specify
build commands, cleanup commands, and patterns for processes to monitor.
The script uses pidstat or psutil for resource monitoring.
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

from . import config
from .monitor_utils import (
    run_and_monitor_build,
    cleanup_processes,
    _execute_clean_step,
    _generate_run_paths,
)
from .process_utils import check_pidstat_installed
from .plotter import generate_plots_for_logs
from .data_models import RunContext

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def signal_handler(sig: int, frame: Any) -> None:
    """
    Handles termination signals (SIGINT, SIGTERM) for graceful shutdown.

    Args:
        sig: The signal number.
        frame: The current stack frame.
    """
    logger.warning(f"Signal {signal.Signals(sig).name} received, initiating cleanup...")
    cleanup_processes()
    sys.exit(1)


def main_cli() -> None:
    """
    Main command-line interface function for the Build Memory Profiler.
    Parses command-line arguments to override configurations and runs the monitoring tasks.
    """
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # --- Define Command-Line Arguments ---
    parser = argparse.ArgumentParser(
        description="Monitor and analyze build process memory usage.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Target Selectors
    parser.add_argument(
        "-p",
        "--project",
        nargs="+",
        metavar="NAME",
        help="Specify one or more project names to run (e.g., qemu chromium).\nThis overrides the default behavior of running all projects.",
    )
    # Behavior Modifiers
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
    # Action Flags
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only run the clean command for the selected projects and exit.",
    )
    # Meta-Configuration
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a custom main config.toml file.",
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    try:
        if args.config:
            if not args.config.is_file():
                logger.error(f"Specified config file not found: {args.config}")
                sys.exit(1)
            config._CONFIG_FILE_PATH = args.config
            config._CONFIG = None  # Force reload
        app_config = config.get_config()
    except Exception as e:
        logger.error(
            f"Failed to initialize application due to configuration error: {e}"
        )
        sys.exit(1)

    # --- Override Configuration with CLI Arguments ---
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

    # --- Execute Actions ---
    if not projects_to_run:
        logger.info("No projects selected to run. Exiting.")
        sys.exit(0)

    # Handle --clean-only action
    if args.clean_only:
        logger.info("--- Running in Clean-Only Mode ---")
        for project_config in projects_to_run:
            logger.info(f"Cleaning project: {project_config.name}")
            # Create a minimal context for the clean step
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
            # We can't write to a real summary log, so we just execute the command.
            _execute_clean_step(run_context, project_config, "Clean-Only")
        logger.info("--- Clean-Only Mode Finished ---")
        sys.exit(0)

    # --- Main Monitoring Run ---
    # (The rest of the logic is similar to your previous version, but adapted for the new flags)
    if monitor_config.metric_type == "rss_pidstat" and not check_pidstat_installed():
        logger.error("'pidstat' not found. Please install 'sysstat' package.")
        sys.exit(1)

    current_run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_specific_log_dir_name = f"run_{current_run_timestamp}"
    current_run_output_dir = monitor_config.log_root_dir / run_specific_log_dir_name
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Logs for this run will be saved in: {current_run_output_dir.resolve()}"
    )

    # Pin monitor script to a core (logic can be kept as is)
    # ... (CPU pinning logic from your previous version would go here) ...
    actual_monitor_core_id = None  # Placeholder for your pinning logic

    for project_config in projects_to_run:
        logger.info(f">>> Starting processing for project: {project_config.name}")
        for level in monitor_config.default_jobs:
            # Pass the --no-pre-clean flag down to the monitoring utility
            run_and_monitor_build(
                project_config=project_config,
                parallelism_level=level,
                monitoring_interval=monitor_config.interval_seconds,
                log_dir=current_run_output_dir,
                collector_type=monitor_config.metric_type,
                skip_pre_clean=args.no_pre_clean,  # NEW
                monitor_core_id_for_collector_and_build_avoidance=actual_monitor_core_id,
                build_cpu_cores_policy=monitor_config.build_cores_policy,
                specific_build_cores_str=monitor_config.specific_build_cores,
                monitor_script_pinned_to_core_info="Not Pinned",  # Placeholder
            )
        logger.info(f"<<< Finished processing for project: {project_config.name}")

    logger.info("All specified build and monitoring tasks completed.")

    if not monitor_config.skip_plots:
        logger.info("--- Starting plot generation ---")
        try:
            generate_plots_for_logs(current_run_output_dir)
            logger.info("--- Plot generation finished ---")
        except Exception as e:
            logger.error(
                f"An error occurred during plot generation: {e}", exc_info=True
            )
    else:
        logger.info("Skipping plot generation as per config or command-line flag.")


if __name__ == "__main__":
    main_cli()
