import argparse
import logging
import sys
import signal
from pathlib import Path
from typing import List, Dict, Any
import time # Add this import

from .monitor_utils import run_and_monitor_build, check_pidstat_installed, cleanup_processes
from .plotter import generate_plots_for_logs

# --- Global Configuration ---
MONITORING_INTERVAL_SECONDS = 1
DEFAULT_PARALLELISM_LEVELS = ["4", "8", "16"]


# --- Project Configurations ---
# (Similar to PROJECTS_CONFIG in Bash)
PROJECTS_CONFIG: List[Dict[str, Any]] = [
    {
        "NAME": "qemu",
        "DIR": "/host/qemu/build", # <<< MODIFY AS NEEDED
        "SETUP_COMMAND_TEMPLATE": "",
        "BUILD_COMMAND_TEMPLATE": "make -j<N>",
        "PROCESS_PATTERN": r"make|qemu.*|gcc|cc|g\+\+|c\+\+|clang|as|ld|cc1|collect2|configure|python[0-9._-]*",
        "CLEAN_COMMAND_TEMPLATE": "make clean",
    },
    {
        "NAME": "aosp",
        "DIR": "/host/aosp", # <<< MODIFY AS NEEDED
        "SETUP_COMMAND_TEMPLATE": "source build/envsetup.sh && lunch aosp_arm64-userdebug", # <<< MODIFY TARGET
        "BUILD_COMMAND_TEMPLATE": "m -j<N>",
        "PROCESS_PATTERN": r"make|soong_ui|soong_build|ninja|kati|javac|aapt[2]?|d8|r8|metalava|clang[^\s-]*|ld\.lld|lld|gcc|cc|g\+\+|c\+\+|python[0-9._-]*|bpfmt|aidl|hidl-gen|dex2oat|zip|rsync",
        "CLEAN_COMMAND_TEMPLATE": "m clean",
    },
    {
        "NAME": "chromium",
        "DIR": "/host/chromium/src", # <<< MODIFY AS NEEDED
        "SETUP_COMMAND_TEMPLATE": "", # "gn gen out/Default" - usually done once
        "BUILD_COMMAND_TEMPLATE": "autoninja -C out/Default chrome -j<N>", # <<< MODIFY out/Default
        "PROCESS_PATTERN": r"ninja|gn|clang[^\s-]*|gomacc|siso|ld\.lld|lld|python[0-9._-]*|mojo[a-z_]*|lcc|ar|ranlib|strip",
        "CLEAN_COMMAND_TEMPLATE": "gn clean out/Default", # <<< MODIFY out/Default
    }
]

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
try:
    PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    PROJECT_ROOT_DIR = Path.cwd()
    logger.warning("__file__ not defined, using current working directory as project root for logs.")

LOG_ROOT_DIR = PROJECT_ROOT_DIR / "logs"

def signal_handler(sig, frame):
    logger.warning(f"Signal {sig} received, initiating cleanup...")
    cleanup_processes()
    sys.exit(1)


def main_cli():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Build Memory Profiler")
    parser.add_argument(
        "-p", "--projects",
        type=str,
        help="Comma-separated list of project names to build (e.g., qemu,aosp). Default: all configured."
    )
    parser.add_argument(
        "-j", "--jobs",
        type=str,
        help=f"Comma-separated list of parallelism levels (e.g., 4,8,16). Default: {','.join(DEFAULT_PARALLELISM_LEVELS)}"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots after monitoring."
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        default="pss_psutil",
        choices=["rss_pidstat", "pss_psutil"],
        help="Metric type of memory to be collected (E.g. rss_pidstat, pss_psutil)。"
    )
    args = parser.parse_args()

    # 检查所选收集器的依赖项
    if args.metric_type == "rss_pidstat":
        if not check_pidstat_installed():
            logger.error("'pidstat' 命令未找到，无法使用 'rss_pidstat' 收集器。请安装 'sysstat' 包。")
            sys.exit(1)
    elif args.metric_type == "pss_psutil":
        try:
            import psutil # 检查 psutil 是否可以导入
            logger.info(f"将使用 psutil 版本 {psutil.__version__} 进行 PSS 收集。")
        except ImportError:
            # 理想情况下，这种情况应该由依赖管理 (pyproject.toml) 捕获
            # 但运行时检查是一个后备方案。
            logger.error("'psutil' 库未找到，无法使用 'pss_psutil' 收集器。请安装它 (例如: pip install psutil)。")
            sys.exit(1)
        except AttributeError: # 适用于可能没有 __version__ 属性的旧版 psutil
             logger.info(f"将使用 psutil 进行 PSS 收集 (未找到版本属性)。")


    # Create a unique directory for this run's logs and plots
    current_run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_specific_log_dir_name = f"run_{current_run_timestamp}"
    current_run_output_dir = LOG_ROOT_DIR / run_specific_log_dir_name
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Log files and plots for this run will be saved in: {current_run_output_dir.resolve()}")


    selected_project_names: List[str] = []
    if args.projects:
        selected_project_names = [name.strip() for name in args.projects.split(',')]

    parallelism_levels_str: List[str] = []
    if args.jobs:
        parallelism_levels_str = [level.strip() for level in args.jobs.split(',')]
    else:
        parallelism_levels_str = DEFAULT_PARALLELISM_LEVELS
    
    parallelism_levels_int: List[int] = []
    for level_str in parallelism_levels_str:
        try:
            level_int = int(level_str)
            if level_int <= 0:
                logger.warning(f"Invalid parallelism level '{level_str}', must be positive. Skipping.")
                continue
            parallelism_levels_int.append(level_int)
        except ValueError:
            logger.warning(f"Invalid parallelism level '{level_str}', not a number. Skipping.")
    
    if not parallelism_levels_int:
        logger.error("No valid parallelism levels specified. Exiting.")
        sys.exit(1)


    projects_to_run: List[Dict[str, Any]] = []
    if not selected_project_names: # Run all if none specified
        projects_to_run = PROJECTS_CONFIG
        logger.info(f"Processing all {len(PROJECTS_CONFIG)} configured projects.")
    else:
        for proj_name in selected_project_names:
            found_project = next((p for p in PROJECTS_CONFIG if p["NAME"] == proj_name), None)
            if found_project:
                projects_to_run.append(found_project)
            else:
                logger.warning(f"Project '{proj_name}' not found in configurations. Skipping.")
        logger.info(f"Processing selected projects: {[p['NAME'] for p in projects_to_run]}")

    if not projects_to_run:
        logger.info("No projects to process based on selection and configuration. Exiting.")
        sys.exit(0)

    for project_config in projects_to_run:
        logger.info(f">>> Starting processing for project: {project_config['NAME']}")
        for level in parallelism_levels_int:
            run_and_monitor_build(
                project_config=project_config,
                parallelism_level=level,
                monitoring_interval=MONITORING_INTERVAL_SECONDS,
                log_dir=current_run_output_dir, # Pass the new run-specific directory
                collector_type=args.metric_type # Pass the selected collector type
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
            logger.error(f"An error occurred during plot generation: {e}", exc_info=True)
    else:
        logger.info("Skipping plot generation as per --skip-plots flag.")


if __name__ == "__main__":
    main_cli()