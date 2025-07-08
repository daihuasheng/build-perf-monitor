"""
Configuration data models.

This module contains all configuration-related data structures for projects,
monitoring settings, categorization rules, and application configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class MonitorConfig:
    """
    Configuration for the monitor's global behavior, loaded from `config.toml`.
    """

    # [monitor.general]
    default_jobs: List[int]
    skip_plots: bool
    log_root_dir: Path
    categorization_cache_size: int

    # [monitor.collection]
    interval_seconds: float
    metric_type: str
    pss_collector_mode: str
    process_check_interval: float
    monitoring_timeout: float
    graceful_shutdown_timeout: float

    # [monitor.scheduling] - Unified Policy
    scheduling_policy: str
    monitor_core: int
    manual_build_cores: str
    manual_monitoring_cores: str
    enable_cpu_affinity: bool
    max_concurrent_monitors: int
    thread_name_prefix: str
    
    # [monitor.async_settings] - AsyncIO-specific Configuration
    enable_thread_pool_optimization: bool = True


@dataclass
class ProjectConfig:
    """
    Configuration for a single project to be monitored, loaded from `projects.toml`.
    """

    # A unique, descriptive name for the project (e.g., "qemu", "chromium").
    name: str
    # The root directory of the project where commands will be executed.
    dir: str
    # The command used to build the project. '<N>' is a placeholder for the parallelism level.
    build_command_template: str
    # A regex pattern used by the memory collector to identify relevant processes for this project.
    process_pattern: str
    # The command used to clean build artifacts. Can be an empty string.
    clean_command_template: str
    # An optional command to run before the build (e.g., 'source env.sh'). Can be an empty string.
    setup_command_template: str


@dataclass
class RuleConfig:
    """
    Configuration for a categorization rule, loaded from `rules.toml`.
    """

    # Priority level for the rule (higher numbers processed first).
    priority: int
    # The major category this rule assigns.
    major_category: str
    # The minor category this rule assigns.
    category: str
    # The field to match against ('current_cmd_name' or 'current_full_cmd').
    match_field: str
    # The type of match to perform ('exact', 'in_list', 'regex', 'contains').
    match_type: str
    # For backward compatibility, support both pattern (string) and patterns (Union)
    # patterns can be either a string (for exact/regex/contains) or list (for in_list).
    patterns: Union[str, List[str]] = ""
    # Legacy field for single pattern (maintained for compatibility)
    pattern: Optional[str] = None
    # Optional comment describing the rule.
    comment: str = ""
    
    def __post_init__(self):
        """Post-initialization processing to handle pattern/patterns compatibility."""
        # If both pattern and patterns are provided, prefer patterns
        if self.pattern is not None and self.patterns == "":
            self.patterns = self.pattern
        # If patterns is empty and pattern is None, this will be caught by validation
        # Ensure pattern field stays in sync for backward compatibility
        if isinstance(self.patterns, str) and self.pattern is None:
            self.pattern = self.patterns


@dataclass
class AppConfig:
    """
    The root configuration object that aggregates all loaded settings.
    """

    # The global monitor configuration.
    monitor: MonitorConfig
    # A list of all configured projects.
    projects: List[ProjectConfig]
    # A list of all categorization rules, sorted by priority.
    rules: List[RuleConfig]
