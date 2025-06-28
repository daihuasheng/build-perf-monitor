"""
Process Tree Verification Tool for MyMonitor.

This script helps determine if the 'descendants_only' optimization mode is safe
to use for a specific project. It runs a build and simultaneously tracks two sets
of processes:
1. All processes on the system that match the project's `process_pattern`.
2. All processes that are descendants of the main build process.

After the build, it compares these two sets. If they are identical, it means
the 'descendants_only' mode is safe. If not, it reports the "orphan" processes
that would be missed by the optimization.
"""

import argparse
import logging
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import psutil

# Add the project root to the Python path to allow importing 'mymonitor' modules.
sys.path.insert(0, str(Path(__file__).parent.parent))
from mymonitor import config

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ProcessTreeVerifier")


def verify_project(project_name: str, parallelism: int) -> bool:
    """
    Runs the verification process for a given project.

    Returns:
        bool: True if verification is successful (safe to use 'descendants_only'), False otherwise.
    """
    logger.info(f"--- Starting Verification for Project: {project_name} ---")

    # --- 1. Load Configuration ---
    try:
        app_config = config.get_config()
        project_map = {p.name: p for p in app_config.projects}
        if project_name not in project_map:
            logger.error(f"Project '{project_name}' not found in configuration.")
            return False
        project_config = project_map[project_name]
        process_pattern_re = re.compile(project_config.process_pattern)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return False

    # --- 2. Prepare and Start Build Command ---
    base_build_cmd = project_config.build_command_template.replace(
        "<N>", str(parallelism)
    )
    actual_build_command = (
        f"{project_config.setup_command_template} && {base_build_cmd}"
        if project_config.setup_command_template
        else base_build_cmd
    )
    final_command = ["/bin/bash", "-c", actual_build_command]

    logger.info(f"Executing build command in '{project_config.dir}':")
    logger.info(f"  $ {shlex.join(final_command)}")

    build_proc = subprocess.Popen(
        final_command,
        cwd=project_config.dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    logger.info(f"Build process started with PID: {build_proc.pid}")

    # --- 3. Monitor Processes During Build ---
    all_matching_pids = set()
    descendant_pids = set()
    try:
        main_build_process = psutil.Process(build_proc.pid)
        descendant_pids.add(main_build_process.pid)
    except psutil.NoSuchProcess:
        logger.error(f"Build process with PID {build_proc.pid} disappeared immediately.")
        return False

    while build_proc.poll() is None:
        logger.info("Sampling process tree...")
        try:
            descendants = main_build_process.children(recursive=True)
            current_descendant_pids = {p.pid for p in descendants}
            descendant_pids.update(current_descendant_pids)
        except psutil.NoSuchProcess:
            logger.warning("Main build process disappeared during scan.")
            break
        time.sleep(0.1)

    # --- FIX: Grace period and final scan ---
    # Reduce the grace period to be shorter than the lifetime of the test's
    # detached process (0.3s). This ensures the final scan happens while the
    # orphan process is still running and can be detected.
    grace_period = 0.2
    logger.info(f"Build process finished. Waiting for grace period ({grace_period}s) before final scan...")
    time.sleep(grace_period)
    logger.info("Performing final process scan...")
    for proc in psutil.process_iter(["cmdline"]):
        cmdline = " ".join(proc.info["cmdline"] or [])
        if process_pattern_re.search(cmdline):
            all_matching_pids.add(proc.pid)

    # --- 4. Analyze and Report Results ---
    orphan_pids = all_matching_pids - descendant_pids

    print("\n" + "=" * 50)
    print("  PROCESS TREE VERIFICATION REPORT")
    print("=" * 50)
    print(f"Project: {project_name}")
    print(f"Total unique processes matching pattern: {len(all_matching_pids)}")
    print(f"Total unique descendant processes found: {len(descendant_pids)}")
    print("-" * 50)

    if not orphan_pids:
        print("\n[SUCCESS] Verification passed!")
        print("All relevant processes are descendants of the main build process.")
        print("It is SAFE to enable 'descendants_only' mode for this project.")
        print("\n" + "=" * 50)
        return True
    else:
        print(
            f"\n[FAILURE] Verification failed. Found {len(orphan_pids)} orphan process(es)."
        )
        print("These processes matched the project pattern but were NOT descendants.")
        print("Using 'descendants_only' mode will cause these to be MISSED.")
        print("\nOrphan Process Details:")
        for pid in sorted(list(orphan_pids)):
            try:
                p = psutil.Process(pid)
                print(
                    f"  - PID: {p.pid}, Name: {p.name()}, Cmd: {' '.join(p.cmdline())}"
                )
            except psutil.NoSuchProcess:
                print(
                    f"  - PID: {pid} (process terminated before details could be fetched)"
                )
        print("\n" + "=" * 50)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify if a project's build processes are all descendants of the main process."
    )
    parser.add_argument(
        "-p",
        "--project",
        required=True,
        help="The name of the project to verify (must be defined in projects.toml).",
    )
    parser.add_argument(
        "-j",
        type=int,
        default=8,
        help="The parallelism level (-j N) to use for the test build. Default: 8.",
    )
    args = parser.parse_args()
    verify_project(args.project, args.j)
