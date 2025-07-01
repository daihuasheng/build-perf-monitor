import argparse
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path
import multiprocessing

# Local application imports
from .config import get_config
from .monitor_utils import (
    BuildRunner,
    cleanup_processes,
)
from .process_utils import (
    check_pidstat_installed,
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s] %(name)s:%(filename)s:%(lineno)d\t %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main_cli():
    """
    Main command-line interface for the MyMonitor application.
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

    # Directly link the signal to the simple, safe cleanup function
    signal.signal(signal.SIGINT, cleanup_processes)
    signal.signal(signal.SIGTERM, cleanup_processes)

    try:
        app_config = get_config()
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        sys.exit(1)

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
        projects_to_run = [p for p in app_config.projects if p.name == args.project]
        if not projects_to_run:
            logger.error(f"Project '{args.project}' not found in configuration.")
            sys.exit(1)
    else:
        projects_to_run = app_config.projects

    if args.jobs:
        try:
            jobs_to_run = [int(j.strip()) for j in args.jobs.split(",")]
        except ValueError:
            logger.error(
                f"Invalid format for --jobs: '{args.jobs}'. Use comma-separated integers."
        )
        sys.exit(1)
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
                    f"An unexpected error occurred during monitoring for -j{j_level}: {e}",
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
            logger.error(f"Failed to execute plotter tool: {e}", exc_info=True)
        logger.info("--- Plot generation finished ---")


if __name__ == "__main__":
    main_cli()
