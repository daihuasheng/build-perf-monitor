"""
Command-line interface for the MyMonitor build performance monitoring application.

This module provides the main CLI entry point for the MyMonitor application,
handling command-line arguments, configuration validation, and orchestrating
monitoring runs across multiple projects and parallelism levels.
"""

import argparse
import logging
import asyncio
import signal
import subprocess
import sys
import time
from pathlib import Path

from ..config import get_config
from ..system.commands import check_pidstat_installed
from ..validation import (
    handle_cli_error,
    ValidationError,
    validate_jobs_list,
    validate_project_name,
)
from .orchestrator import BuildRunner

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s] %(name)s:%(filename)s:%(lineno)d\t %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main_cli() -> None:
    """
    Main command-line interface for the MyMonitor application.

    This function provides the primary entry point for the CLI, handling:
    - Multiprocessing configuration for cross-platform compatibility
    - Configuration loading and validation
    - Command-line argument parsing
    - Project and job validation
    - Monitoring execution coordination
    - Plot generation orchestration

    The function sets up multiprocessing to use the 'spawn' method for stability
    across different platforms and processes each specified project with each
    specified parallelism level.

    Raises:
        SystemExit: On configuration errors, validation failures, or execution problems.
    """
    # --- Global state for graceful shutdown ---
    shutdown_requested = False
    active_runner = None

    def global_signal_handler(signum, frame):
        """Handle signals globally to ensure a clean shutdown."""
        nonlocal shutdown_requested, active_runner
        if shutdown_requested:
            logger.warning("Shutdown already in progress. Please be patient.")
            return

        logger.info(
            f"Signal {signal.strsignal(signum)} received. Initiating graceful shutdown..."
        )
        shutdown_requested = True

        # If a runner is active, request it to shut down its components.
        if active_runner:
            logger.info("Requesting active build runner to shut down...")
            active_runner.request_shutdown()

    # Register the global signal handler for SIGINT and SIGTERM.
    signal.signal(signal.SIGINT, global_signal_handler)
    signal.signal(signal.SIGTERM, global_signal_handler)

    # Since we're using AsyncIO instead of multiprocessing, we don't need
    # to set the start method anymore
    logger.info("Starting MyMonitor with AsyncIO architecture")

    # Load application configuration
    try:
        app_config = get_config()
    except (FileNotFoundError, KeyError) as e:
        handle_cli_error(
            error=e,
            context="configuration loading",
            exit_code=1,
            include_traceback=True,
            logger=logger,
        )

    monitor_config = app_config.monitor

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Monitor build processes for performance analysis."
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        help=f"Specify a single project to run from the config. Available: {[p.name for p in app_config.projects]}",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=str,
        help=f"Specify parallelism levels (e.g., '8,16'). Defaults to {monitor_config.default_jobs} from config.",
    )
    parser.add_argument(
        "--no-pre-clean",
        action="store_true",
        help="Skip the pre-build clean step defined in the project config.",
    )
    parser.add_argument(
        "--no-post-clean",
        action="store_true",
        help="Skip the post-build clean step after all builds are completed.",
    )
    args = parser.parse_args()

    # Validate and filter projects
    if args.project:
        # Validate project name format first
        try:
            validated_project_name = validate_project_name(
                args.project, field_name="--project argument"
            )
        except ValidationError as e:
            handle_cli_error(
                error=e,
                context="project name validation",
                exit_code=1,
                include_traceback=False,
                logger=logger,
            )

        # Check if project exists in configuration
        projects_to_run = [
            p for p in app_config.projects if p.name == validated_project_name
        ]
        if not projects_to_run:
            available_projects = [p.name for p in app_config.projects]
            logger.error(
                f"Project '{validated_project_name}' not found in configuration."
            )
            logger.info(f"Available projects: {', '.join(available_projects)}")
            sys.exit(1)
    else:
        projects_to_run = app_config.projects

    # Validate and parse job levels
    if args.jobs:
        try:
            jobs_to_run = validate_jobs_list(
                args.jobs,
                min_jobs=1,
                max_jobs=1024,  # Reasonable upper limit
                field_name="--jobs argument",
            )
        except ValidationError as e:
            handle_cli_error(
                error=e,
                context="jobs argument validation",
                exit_code=1,
                include_traceback=False,
                logger=logger,
            )
    else:
        jobs_to_run = monitor_config.default_jobs

    # Create output directory for this run
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_run_output_dir = monitor_config.log_root_dir / f"run_{run_timestamp}"
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Log and plot outputs will be saved in: {current_run_output_dir}")

    # Execute monitoring for each project and job level combination
    last_runner_per_project = (
        {}
    )  # Keep track of last runner for each project for final cleanup

    for project in projects_to_run:
        if shutdown_requested:
            logger.warning("Shutdown requested, skipping further projects.")
            break

        logger.info(f">>> Starting processing for project: {project.name}")

        # Check system dependencies
        if (
            monitor_config.metric_type == "rss_pidstat"
            and not check_pidstat_installed()
        ):
            logger.error(
                "Collector type is 'rss_pidstat' but pidstat is not installed. "
                "Please install it (e.g., 'sudo apt-get install sysstat') or "
                "choose another collector type in config.toml. Skipping project."
            )
            continue

        # Run monitoring for each parallelism level
        for j_level in jobs_to_run:
            if shutdown_requested:
                logger.warning(
                    f"Shutdown requested, skipping further parallelism levels for project '{project.name}'."
                )
                break

            logger.info(f"--- Running with -j{j_level} ---")
            try:
                runner = BuildRunner(
                    project_config=project,
                    parallelism_level=j_level,
                    monitoring_interval=monitor_config.interval_seconds,
                    log_dir=current_run_output_dir,
                    collector_type=monitor_config.metric_type,
                    skip_pre_clean=args.no_pre_clean,
                )
                active_runner = runner
                last_runner_per_project[project.name] = (
                    runner  # Keep reference for final cleanup
                )
                runner.run()
            except Exception as e:
                logger.error(
                    f"Unexpected error during monitoring for project '{project.name}' with -j{j_level}: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                continue
            finally:
                active_runner = None  # Clear the active runner reference

        logger.info(f"<<< Finished processing for project: {project.name}")

    if shutdown_requested:
        logger.info(
            "Build monitoring was terminated prematurely due to a shutdown request."
        )
    else:
        logger.info("All specified build and monitoring tasks completed.")

    # Execute final cleanup for all projects (unless disabled)
    if not args.no_post_clean and not shutdown_requested:
        logger.info("--- Starting post-build cleanup ---")
        for project_name, runner in last_runner_per_project.items():
            try:
                logger.info(f"Executing final clean for project: {project_name}")
                runner.execute_final_clean()
            except Exception as e:
                logger.error(
                    f"Error during final cleanup for project '{project_name}': {e}",
                    exc_info=True,
                )
        logger.info("--- Post-build cleanup completed ---")
    elif args.no_post_clean:
        logger.info("Post-build cleanup skipped (--no-post-clean flag)")
    else:
        logger.info("Post-build cleanup skipped due to shutdown request")

    # Generate plots if not disabled and if not shut down prematurely
    if not monitor_config.skip_plots and not shutdown_requested:
        logger.info("--- Starting plot generation via external tool ---")
        try:
            # Find the plotter tool relative to the package
            plotter_path = (
                Path(__file__).parent.parent.parent.parent / "tools" / "plotter.py"
            )

            # Generate detailed plots
            plotter_cmd_detailed = [
                sys.executable,
                str(plotter_path),
                "--log-dir",
                str(current_run_output_dir),
            ]
            logger.info(
                f"Executing plotter for detailed plots: {' '.join(plotter_cmd_detailed)}"
            )
            result_detailed = subprocess.run(
                plotter_cmd_detailed, capture_output=True, text=True, check=False
            )
            logger.info("Detailed plotter tool output:\n" + result_detailed.stdout)
            if result_detailed.stderr:
                logger.warning(
                    "Detailed plotter tool stderr:\n" + result_detailed.stderr
                )

            # Generate summary plots
            plotter_cmd_summary = plotter_cmd_detailed + ["--summary-plot"]
            logger.info(
                f"Executing plotter for summary plot: {' '.join(plotter_cmd_summary)}"
            )
            result_summary = subprocess.run(
                plotter_cmd_summary, capture_output=True, text=True, check=False
            )
            logger.info("Summary plotter tool output:\n" + result_summary.stdout)
            if result_summary.stderr:
                logger.warning("Summary plotter tool stderr:\n" + result_summary.stderr)

        except Exception as e:
            logger.error(
                f"Failed to execute plotter tool: {type(e).__name__}: {e}",
                exc_info=True,
            )

        logger.info("--- Plot generation finished ---")


if __name__ == "__main__":
    main_cli()
