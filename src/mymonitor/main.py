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
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from .monitor_utils import (
    run_and_monitor_build,
    check_pidstat_installed,
    cleanup_processes,
)
from .plotter import generate_plots_for_logs

# --- Global Configuration ---
MONITORING_INTERVAL_SECONDS: int = 1
"""Interval in seconds for resource monitoring polling."""

DEFAULT_PARALLELISM_LEVELS: List[str] = ["4", "8", "16"]
"""Default parallelism levels to use if not specified by the user."""


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

try:
    # Resolve project root directory based on this file's location.
    PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Fallback for environments where __file__ might not be defined (e.g. some REPLs)
    PROJECT_ROOT_DIR = Path(".").resolve()

LOG_ROOT_DIR = PROJECT_ROOT_DIR / "logs"
CONFIG_FILE_PATH = PROJECT_ROOT_DIR / "projects_config.json"  # Define config file path


def load_projects_config(config_path: Path) -> List[Dict[str, Any]]:
    """Loads project configurations from a JSON file."""
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)


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

    Parses command-line arguments, sets up monitoring, iterates through
    specified projects and parallelism levels to run builds, and optionally
    generates plots from the collected data.
    """
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    projects_config = load_projects_config(CONFIG_FILE_PATH)

    parser = argparse.ArgumentParser(description="Build Memory Profiler")
    parser.add_argument(
        "-p",
        "--projects",
        type=str,
        help="Comma-separated list of project names to build (e.g., qemu,aosp). Default: all configured.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=str,
        help=(
            "Comma-separated list of parallelism levels (e.g., 4,8,16). "
            f"Default: {','.join(DEFAULT_PARALLELISM_LEVELS)}"
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots after monitoring.",
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        default="pss_psutil",
        choices=["rss_pidstat", "pss_psutil"],
        help="Metric type of memory to be collected (e.g., rss_pidstat, pss_psutil).",
    )
    args = parser.parse_args()

    # Check dependencies for the selected collector
    if args.metric_type == "rss_pidstat":
        if not check_pidstat_installed():
            logger.error(
                "'pidstat' command not found, 'rss_pidstat' collector cannot be used. "
                "Please install the 'sysstat' package."
            )
            sys.exit(1)
    elif args.metric_type == "pss_psutil":
        try:
            import psutil  # Check if psutil can be imported

            logger.info(
                f"Using psutil version {psutil.__version__} for PSS collection."
            )
        except ImportError:
            # Ideally, this should be caught by dependency management (e.g., pyproject.toml)
            # However, a runtime check serves as a fallback.
            logger.error(
                "'psutil' library not found, 'pss_psutil' collector cannot be used. "
                "Please install it (e.g., pip install psutil)."
            )
            sys.exit(1)
        except AttributeError:  # For older psutil versions that might lack __version__
            logger.info(
                "Using psutil for PSS collection (version attribute not found)."
            )

    # Create a unique directory for this run's logs and plots
    current_run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_specific_log_dir_name = f"run_{current_run_timestamp}"
    current_run_output_dir = LOG_ROOT_DIR / run_specific_log_dir_name
    current_run_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Log files and plots for this run will be saved in: {current_run_output_dir.resolve()}"
    )

    selected_project_names: List[str] = []
    if args.projects:
        selected_project_names = [name.strip() for name in args.projects.split(",")]

    parallelism_levels_str: List[str]
    if args.jobs:
        parallelism_levels_str = [level.strip() for level in args.jobs.split(",")]
    else:
        parallelism_levels_str = DEFAULT_PARALLELISM_LEVELS

    parallelism_levels_int: List[int] = []
    for level_str in parallelism_levels_str:
        try:
            level_int = int(level_str)
            if level_int <= 0:
                logger.warning(
                    f"Invalid parallelism level '{level_str}', must be positive. Skipping."
                )
                continue
            parallelism_levels_int.append(level_int)
        except ValueError:
            logger.warning(
                f"Invalid parallelism level '{level_str}', not a number. Skipping."
            )

    if not parallelism_levels_int:
        logger.error("No valid parallelism levels specified. Exiting.")
        sys.exit(1)

    projects_to_run: List[Dict[str, Any]] = []
    if not selected_project_names:  # Run all if none specified
        projects_to_run = projects_config  # Use loaded config
        logger.info(f"Processing all {len(projects_config)} configured projects.")
    else:
        for proj_name in selected_project_names:
            found_project = next(
                (p for p in projects_config if p["NAME"] == proj_name), None # Use loaded config
            )
            if found_project:
                projects_to_run.append(found_project)
            else:
                logger.warning(
                    f"Project '{proj_name}' not found in configurations. Skipping."
                )
        logger.info(
            f"Processing selected projects: {[p['NAME'] for p in projects_to_run]}"
        )

    if not projects_to_run:
        logger.info(
            "No projects to process based on selection and configuration. Exiting."
        )
        sys.exit(0)

    # Main loop for processing each project with each parallelism level
    for project_config in projects_to_run:
        logger.info(f">>> Starting processing for project: {project_config['NAME']}")
        for level in parallelism_levels_int:
            run_and_monitor_build(
                project_config=project_config,
                parallelism_level=level,
                monitoring_interval=MONITORING_INTERVAL_SECONDS,
                log_dir=current_run_output_dir,  # Pass the run-specific directory
                collector_type=args.metric_type,  # Pass the selected collector type
            )
        logger.info(f"<<< Finished processing for project: {project_config['NAME']}")

    logger.info("All specified build and monitoring tasks completed.")

    # --- Generate plots ---
    if not args.skip_plots:
        logger.info("--- Starting plot generation ---")
        try:
            # Pass the same run-specific directory for plots
            generate_plots_for_logs(current_run_output_dir)
            logger.info("--- Plot generation finished ---")
        except Exception as e:
            logger.error(
                f"An error occurred during plot generation: {e}", exc_info=True
            )
    else:
        logger.info("Skipping plot generation as per --skip-plots flag.")


if __name__ == "__main__":
    main_cli()
