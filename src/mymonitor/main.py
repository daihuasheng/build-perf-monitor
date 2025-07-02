"""Main command-line interface for the MyMonitor build performance monitoring application.

This module serves as the primary entry point for the MyMonitor application, providing
a command-line interface for monitoring build processes and collecting performance data.
It coordinates configuration loading, project selection, build execution monitoring,
and result visualization.

The main workflow includes:
1. Loading configuration from TOML files
2. Parsing command-line arguments for project and job selection
3. Executing monitoring runs for each project/job combination
4. Generating performance plots and reports

Usage:
    python -m mymonitor.main [--project PROJECT] [--jobs JOBS] [--no-pre-clean]

Example:
    python -m mymonitor.main --project chromium --jobs 8,16 --no-pre-clean
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
import multiprocessing

# Local application imports
from .config import get_config
from .monitor_utils import BuildRunner
from .process_utils import (
    check_pidstat_installed,
)
from .data_models import handle_cli_error, ValidationError, validate_jobs_list, validate_project_name

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s] %(name)s:%(filename)s:%(lineno)d\t %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main_cli() -> None:
    """Main command-line interface for the MyMonitor application.
    
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
    # Set the start method for multiprocessing to 'spawn' to ensure stability
    # across different platforms and avoid potential deadlocks with fork.
    # This should be done early, ideally within the `if __name__ == '__main__':` block.
    # We place it here as it's the main entry point for the CLI.
    try:
        multiprocessing.set_start_method("spawn", force=True)
        logger.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        # The context can only be set once. If it's already set, we can ignore the error.
        logger.debug("Multiprocessing context already set.")
        pass

    try:
        app_config = get_config()
    except (FileNotFoundError, KeyError) as e:
        handle_cli_error(
            error=e,
            context="configuration loading",
            exit_code=1,
            include_traceback=True,
            logger=logger
        )

    monitor_config = app_config.monitor

    parser = argparse.ArgumentParser(
        description="Monitor build processes for performance analysis."
    )
    parser.add_argument(
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
    args = parser.parse_args()

    if args.project:
        # Validate project name format first
        try:
            validated_project_name = validate_project_name(
                args.project,
                field_name="--project argument"
            )
        except ValidationError as e:
            handle_cli_error(
                error=e,
                context="project name validation",
                exit_code=1,
                include_traceback=False,
                logger=logger
            )
        
        # Check if project exists in configuration
        projects_to_run = [p for p in app_config.projects if p.name == validated_project_name]
        if not projects_to_run:
            available_projects = [p.name for p in app_config.projects]
            logger.error(f"Project '{validated_project_name}' not found in configuration.")
            logger.info(f"Available projects: {', '.join(available_projects)}")
            sys.exit(1)
    else:
        projects_to_run = app_config.projects

    if args.jobs:
        try:
            jobs_to_run = validate_jobs_list(
                args.jobs,
                min_jobs=1,
                max_jobs=1024,  # Reasonable upper limit
                field_name="--jobs argument"
            )
        except ValidationError as e:
            handle_cli_error(
                error=e,
                context="jobs argument validation",
                exit_code=1,
                include_traceback=False,
                logger=logger
            )
    else:
        jobs_to_run = monitor_config.default_jobs

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_run_output_dir = monitor_config.log_root_dir / f"run_{run_timestamp}"
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Log and plot outputs will be saved in: {current_run_output_dir}")

    for project in projects_to_run:
        logger.info(f">>> Starting processing for project: {project.name}")

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

        for j_level in jobs_to_run:
            logger.info(f"--- Running with -j{j_level} ---")
            try:
                runner = BuildRunner(
                    project_config=project,
                    parallelism_level=j_level,
                monitoring_interval=monitor_config.interval_seconds,
                log_dir=current_run_output_dir,
                collector_type=monitor_config.metric_type,
                skip_pre_clean=args.no_pre_clean,
                    scheduling_policy=monitor_config.scheduling_policy,
                    manual_build_cores=monitor_config.manual_build_cores,
                    manual_monitoring_cores=monitor_config.manual_monitoring_cores,
                    monitor_core_id=monitor_config.monitor_core,
                )
                runner.run()
            except Exception as e:
                logger.error(
                    f"Unexpected error during monitoring for project '{project.name}' with -j{j_level}: {type(e).__name__}: {e}",
                    exc_info=True,
            )
                continue
        logger.info(f"<<< Finished processing for project: {project.name}")

    logger.info("All specified build and monitoring tasks completed.")

    if not monitor_config.skip_plots:
        logger.info("--- Starting plot generation via external tool ---")
        try:
            plotter_cmd_detailed = [
                sys.executable,
                str(Path(__file__).parent.parent / "tools" / "plotter.py"),
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
                exc_info=True
            )
        logger.info("--- Plot generation finished ---")


if __name__ == "__main__":
    main_cli()
