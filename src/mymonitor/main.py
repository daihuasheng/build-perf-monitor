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


# --- Project Configurations ---
# (Similar to PROJECTS_CONFIG in Bash)
PROJECTS_CONFIG: List[Dict[str, Any]] = [
    {
        "NAME": "qemu",
        "DIR": "/host/qemu/build",  # <<< MODIFY AS NEEDED
        "SETUP_COMMAND_TEMPLATE": "",
        "BUILD_COMMAND_TEMPLATE": "make -j<N>",
        "PROCESS_PATTERN": r"make|qemu.*|gcc|cc|g\+\+|c\+\+|clang|as|ld|cc1|collect2|configure|python[0-9._-]*",
        "CLEAN_COMMAND_TEMPLATE": "make clean",
    },
    {
        "NAME": "aosp",
        "DIR": "/host/aosp",  # <<< MODIFY AS NEEDED
        "SETUP_COMMAND_TEMPLATE": "source build/envsetup.sh && lunch aosp_arm64-userdebug",  # <<< MODIFY TARGET
        "BUILD_COMMAND_TEMPLATE": "m -j<N>",
        "PROCESS_PATTERN": r"make|soong_ui|soong_build|ninja|kati|javac|aapt[2]?|d8|r8|metalava|clang[^\s-]*|ld\.lld|lld|gcc|cc|g\+\+|c\+\+|python[0-9._-]*|bpfmt|aidl|hidl-gen|dex2oat|zip|rsync",
        "CLEAN_COMMAND_TEMPLATE": "m clean",
    },
    {
        "NAME": "chromium",
        "DIR": "/host/chromium/src",  # <<< MODIFY AS NEEDED
        "SETUP_COMMAND_TEMPLATE": "",  # "gn gen out/Default" - usually done once
        "BUILD_COMMAND_TEMPLATE": "autoninja -C out/Default chrome -j<N>",  # <<< MODIFY out/Default
        "PROCESS_PATTERN": r"ninja|gn|clang[^\s-]*|gomacc|siso|ld\.lld|lld|python[0-9._-]*|mojo[a-z_]*|lcc|ar|ranlib|strip",
        "CLEAN_COMMAND_TEMPLATE": "gn clean out/Default",  # <<< MODIFY out/Default
    },
]
"""
List of project configurations. Each configuration is a dictionary defining
how to build, clean, and monitor a specific project.
<N> in command templates is replaced with the parallelism level.
'PROCESS_PATTERN' is a regex used to identify relevant processes for monitoring.
"""

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
    # Fallback if __file__ is not defined (e.g., in some interactive environments).
    PROJECT_ROOT_DIR = Path.cwd()
    logger.warning(
        "__file__ not defined, using current working directory as project root for logs."
    )

LOG_ROOT_DIR = PROJECT_ROOT_DIR / "logs"
"""Root directory for storing all log files and generated plots."""


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
        projects_to_run = PROJECTS_CONFIG
        logger.info(f"Processing all {len(PROJECTS_CONFIG)} configured projects.")
    else:
        for proj_name in selected_project_names:
            found_project = next(
                (p for p in PROJECTS_CONFIG if p["NAME"] == proj_name), None
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
