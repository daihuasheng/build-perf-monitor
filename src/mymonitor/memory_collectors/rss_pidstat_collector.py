import subprocess
import time
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

from .base import AbstractMemoryCollector, ProcessMemorySample

logger = logging.getLogger(__name__)

class RssPidstatCollector(AbstractMemoryCollector):
    """Collects RSS and VSZ memory metrics using the pidstat command."""

    PIDSTAT_METRIC_FIELDS = ["RSS_KB", "VSZ_KB"]

    def __init__(self, process_pattern: str, monitoring_interval: int, **kwargs):
        super().__init__(process_pattern, monitoring_interval, **kwargs)
        self.pidstat_proc: Optional[subprocess.Popen] = None
        self.pidstat_stderr_file: Optional[Path] = kwargs.get("pidstat_stderr_file")
        self._pidstat_stderr_handle: Optional[Any] = None

    def get_metric_fields(self) -> List[str]:
        return self.PIDSTAT_METRIC_FIELDS

    def start(self):
        pidstat_cmd = [
            "pidstat",
            "-r",  # Report page faults and memory utilization
            "-l",  # Display command name and all its arguments
            "-C", self.process_pattern,
            str(self.monitoring_interval), # Interval for pidstat to report
        ]
        pidstat_env = os.environ.copy()
        pidstat_env["LC_ALL"] = "C" # Ensures consistent output format from pidstat

        if self.pidstat_stderr_file:
            self._pidstat_stderr_handle = open(self.pidstat_stderr_file, "w")
            stderr_dest = self._pidstat_stderr_handle
        else:
            stderr_dest = subprocess.DEVNULL

        logger.info(f"Starting pidstat collector with command: {' '.join(pidstat_cmd)}")
        try:
            self.pidstat_proc = subprocess.Popen(
                pidstat_cmd,
                stdout=subprocess.PIPE,
                stderr=stderr_dest,
                text=True,
                bufsize=1, # Line-buffered
                env=pidstat_env,
            )
            logger.info(f"pidstat process started (PID: {self.pidstat_proc.pid}).")
        except Exception as e:
            logger.error(f"Failed to start pidstat: {e}", exc_info=True)
            if self._pidstat_stderr_handle:
                self._pidstat_stderr_handle.close()
            raise

    def stop(self):
        if self.pidstat_proc and self.pidstat_proc.poll() is None:
            logger.info(f"Stopping pidstat process (PID: {self.pidstat_proc.pid})...")
            self.pidstat_proc.terminate()
            try:
                self.pidstat_proc.wait(timeout=5)
                logger.info(f"pidstat process (PID: {self.pidstat_proc.pid}) terminated.")
            except subprocess.TimeoutExpired:
                logger.warning(f"pidstat process (PID: {self.pidstat_proc.pid}) did not terminate gracefully, killing...")
                self.pidstat_proc.kill()
            except Exception as e:
                logger.error(f"Error during pidstat stop: {e}", exc_info=True)
            self.pidstat_proc = None
        
        if self._pidstat_stderr_handle:
            self._pidstat_stderr_handle.close()
            self._pidstat_stderr_handle = None
        logger.info("RssPidstatCollector stopped.")


    def read_samples(self) -> Iterable[List[ProcessMemorySample]]:
        if not self.pidstat_proc or not self.pidstat_proc.stdout:
            logger.warning("pidstat process not running or stdout not available.")
            return []

        current_interval_samples: List[ProcessMemorySample] = []
        
        # pidstat outputs a header, then data for each interval, then potentially an average line.
        # We expect one set of process lines per interval.
        # This simplified version assumes each readline() that parses is part of the "current" interval.
        # A more robust parser might explicitly look for timestamp changes if pidstat provided them per line.
        # Since pidstat itself is interval-based, we yield after processing a block of lines assumed to be from one interval.

        for line in iter(self.pidstat_proc.stdout.readline, ""):
            line = line.strip()
            if not line or "Linux" in line or line.startswith("Average:") or "UID" in line : # Skip headers/footers
                if current_interval_samples: # If we had samples and hit a non-data line, yield them
                    yield current_interval_samples
                    current_interval_samples = []
                continue

            # Example pidstat -r -l line (after header):
            # 10:23:01      1000    12345  ...  VSZ   RSS  %MEM Command
            # Time          UID     PID           col5 col6      col8 onwards
            # Note: pidstat output format can vary slightly. This parsing is based on common sysstat versions.
            # With -l, the command can be long.
            # Without UID (if not run as root or -u not specified, but -r implies it for older versions)
            # Time PID %usr %system %guest %CPU CPU Command
            # If -r is used, it's typically:
            # Time UID PID minflt/s majflt/s VSZ RSS %MEM Command
            
            parts = line.split(None, 8) # Split max 8 times, last part is the command
                                        # For "Time UID PID X Y VSZ RSS %MEM Command"
            if len(parts) < 9 or parts[2] == "PID": # Basic check for data line and skip inner headers
                if current_interval_samples: # If we had samples and hit a non-data line, yield them
                    yield current_interval_samples
                    current_interval_samples = []
                continue

            try:
                # Assuming standard output of `pidstat -r -l` where:
                # parts[0] = Time (HH:MM:SS)
                # parts[1] = UID
                # parts[2] = PID
                # parts[3] = minflt/s
                # parts[4] = majflt/s
                # parts[5] = VSZ (KB)
                # parts[6] = RSS (KB)
                # parts[7] = %MEM
                # parts[8] = Command name and arguments
                
                pid_str = parts[2]
                vsz_kb_str = parts[5]
                rss_kb_str = parts[6]
                
                # The command name is the first word of parts[8], full command is parts[8]
                cmd_full_str = parts[8]
                cmd_parts = cmd_full_str.split(None, 1)
                cmd_name_short = os.path.basename(cmd_parts[0]) if cmd_parts else "unknown"
                # Base name for cmd_name_short if it's a path
                cmd_name_short = os.path.basename(cmd_name_short)


                metrics = {
                    "RSS_KB": int(rss_kb_str),
                    "VSZ_KB": int(vsz_kb_str)
                }
                
                sample = ProcessMemorySample(
                    pid=pid_str,
                    command_name=cmd_name_short,
                    full_command=cmd_full_str,
                    metrics=metrics
                )
                current_interval_samples.append(sample)

            except (IndexError, ValueError) as e:
                logger.warning(f"Error parsing pidstat line: '{line}'. Error: {e}. Parts: {parts}")
                # If parsing fails, we might have pending samples from the same interval
                if current_interval_samples:
                    yield current_interval_samples
                    current_interval_samples = []
                continue
        
        # Yield any remaining samples after loop finishes (e.g., EOF)
        if current_interval_samples:
            yield current_interval_samples