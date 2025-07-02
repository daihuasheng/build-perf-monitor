"""
Shared data structures for the orchestration module.

This module defines the configuration objects, runtime state,
and constants used across different orchestration components.
"""

import multiprocessing
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..data_models import (
    MonitoringResults, ProjectConfig, RunContext
)
from ..memory_collectors.base import AbstractMemoryCollector


@dataclass
class BuildRunnerConfig:
    """
    Configuration parameters for BuildRunner.
    
    This replaces the long parameter list in the original BuildRunner.__init__
    """
    project_config: ProjectConfig
    parallelism_level: int
    monitoring_interval: float
    log_dir: Path
    collector_type: str
    skip_pre_clean: bool
    scheduling_policy: str
    manual_build_cores: str
    manual_monitoring_cores: str
    monitor_core_id: int


@dataclass  
class RuntimeState:
    """
    Runtime state shared across orchestration components.
    
    This centralizes all the state variables that were scattered
    across the original BuildRunner class.
    """
    # Core execution state
    run_context: Optional[RunContext] = None
    build_process: Optional[subprocess.Popen] = None
    collector: Optional[AbstractMemoryCollector] = None
    shutdown_requested: threading.Event = field(default_factory=threading.Event)
    results: Optional[MonitoringResults] = None
    
    # Build command preparation
    final_build_command: Optional[str] = None
    executable_shell: Optional[str] = None
    build_command_prefix: str = ""
    
    # Monitoring infrastructure
    monitoring_worker_processes: List[multiprocessing.Process] = field(default_factory=list)
    producer_thread: Optional[threading.Thread] = None
    input_queue: Optional[Any] = None
    output_queue: Optional[Any] = None
    monitoring_cores: List[int] = field(default_factory=list)
    num_workers: int = 0
    
    # Synchronization events for coordination
    producer_finished: threading.Event = field(default_factory=threading.Event)
    workers_finished: threading.Event = field(default_factory=threading.Event)
    all_data_queued: threading.Event = field(default_factory=threading.Event)
    
    # Timestamp and system info
    current_timestamp_str: str = ""
    taskset_available: bool = False


class TimeoutConstants:
    """
    Centralized timeout configuration.
    
    This replaces the scattered magic numbers throughout the original code.
    """
    # Queue operation timeouts
    QUEUE_GET_TIMEOUT = 1.0
    QUEUE_PUT_TIMEOUT = 2.0
    SENTINEL_PUT_TIMEOUT = 0.5
    
    # Process and thread timeouts
    COLLECTOR_STOP_TIMEOUT = 8.0
    WORKER_JOIN_TIMEOUT = 15.0
    PRODUCER_JOIN_TIMEOUT = 10.0
    BUILD_WAIT_TIMEOUT = 1.0
    
    # Result processing timeouts
    QUEUE_PROCESS_TIMEOUT = 0.5
    MAX_CONSECUTIVE_TIMEOUTS = 3
    WORKERS_WRITE_DELAY = 0.5
    
    # Process termination timeouts  
    TERMINATION_GRACEFUL_TIMEOUT = 3
    TERMINATION_INTERRUPT_TIMEOUT = 2
    TERMINATION_FORCE_TIMEOUT = 2