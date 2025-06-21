"""
Main script for the Build Memory Profiler.

This script orchestrates the process of building specified projects with varying
parallelism levels, monitoring their resource consumption (CPU, memory),
and generating plots from the collected data.

It supports configuration for multiple projects, allowing users to specify
build commands, cleanup commands, and patterns for processes to monitor.
The script uses pidstat or psutil for resource monitoring.
"""

import logging
import psutil
import signal
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from . import config
from .monitor_utils import (
    run_and_monitor_build,
    cleanup_processes,
)
from .process_utils import check_pidstat_installed
from .plotter import generate_plots_for_logs

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
    """
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # --- Load all configuration from the central config module ---
    try:
        app_config = config.get_config()
        monitor_config = app_config.monitor
        projects_to_run = app_config.projects
    except Exception as e:
        logger.error(f"Failed to initialize application due to configuration error: {e}")
        sys.exit(1)

    # --- Pin main monitor script to a specific core based on config ---
    monitor_script_pinned_to_core_str = "Not Pinned"
    actual_monitor_core_id: Optional[int] = None

    if monitor_config.monitor_core >= 0:
        actual_monitor_core_id = monitor_config.monitor_core
        try:
            current_process = psutil.Process()
            available_cores_count = psutil.cpu_count()
            if not (
                available_cores_count
                and 0 <= actual_monitor_core_id < available_cores_count
            ):
                logger.warning(
                    f"Monitor core {actual_monitor_core_id} is invalid. "
                    f"Available cores: 0-{available_cores_count - 1 if available_cores_count else 'N/A'}. "
                    "Skipping pinning for monitor script."
                )
                actual_monitor_core_id = None  # Disable pinning if core is invalid
            else:
                current_affinity = current_process.cpu_affinity()
                if (
                    len(current_affinity) == 1
                    and actual_monitor_core_id in current_affinity
                ):
                    logger.info(
                        f"Monitor script is already pinned to CPU core {actual_monitor_core_id}."
                    )
                else:
                    current_process.cpu_affinity([actual_monitor_core_id])
                    logger.info(
                        f"Successfully pinned monitor script to CPU core {actual_monitor_core_id}."
                    )
                monitor_script_pinned_to_core_str = str(actual_monitor_core_id)
        except Exception as e:
            logger.error(
                f"Failed to pin monitor script to CPU core {actual_monitor_core_id}: {e}",
                exc_info=True,
            )
            actual_monitor_core_id = None  # Disable pinning on error
    else:
        logger.info(
            "Monitor script CPU pinning is disabled by user (monitor_core < 0)."
        )
        actual_monitor_core_id = None

    # Check dependencies for the selected collector
    if monitor_config.metric_type == "rss_pidstat":
        if not check_pidstat_installed():
            logger.error(
                "'pidstat' command not found, 'rss_pidstat' collector cannot be used. "
                "Please install the 'sysstat' package."
            )
            sys.exit(1)
    elif monitor_config.metric_type == "pss_psutil":
        try:
            logger.info(
                f"Using psutil version {psutil.__version__} for PSS collection."
            )
        except ImportError: # This case should ideally not be hit if psutil is a direct dependency
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
    current_run_output_dir = monitor_config.log_root_dir / run_specific_log_dir_name
    current_run_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Log files and plots for this run will be saved in: {current_run_output_dir.resolve()}"
    )

    parallelism_levels_int = monitor_config.default_jobs

    if not projects_to_run:
        logger.info("No projects configured in config.toml. Exiting.")
        sys.exit(0)

    # Main loop for processing each project with each parallelism level
    for project_config in projects_to_run:
        logger.info(f">>> Starting processing for project: {project_config.name}")
        for level in parallelism_levels_int:
            run_and_monitor_build(
                project_config=project_config,
                parallelism_level=level,
                monitoring_interval=monitor_config.interval_seconds,
                log_dir=current_run_output_dir,
                collector_type=monitor_config.metric_type,
                monitor_core_id_for_collector_and_build_avoidance=actual_monitor_core_id,
                build_cpu_cores_policy=monitor_config.build_cores_policy,
                specific_build_cores_str=monitor_config.specific_build_cores,
                monitor_script_pinned_to_core_info=monitor_script_pinned_to_core_str,
            )
        logger.info(f"<<< Finished processing for project: {project_config.name}")

    logger.info("All specified build and monitoring tasks completed.")

    # --- Generate plots ---
    if not monitor_config.skip_plots:
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
        logger.info("Skipping plot generation as per config setting.")


if __name__ == "__main__":
    main_cli()
