"""
Configuration data models.

This module contains all configuration-related data structures for projects,
monitoring settings, categorization rules, and application configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from ..config.storage_config import StorageConfig


@dataclass
class MonitorConfig:
    """
    Configuration for the monitor's global behavior, loaded from `config.toml`.
    """

    # [monitor.general] - 必需字段
    default_jobs: List[int]
    skip_plots: bool
    log_root_dir: Path
    categorization_cache_size: int

    # [monitor.collection] - 必需字段
    interval_seconds: float
    metric_type: str  # "pss_psutil" 或 "rss_pidstat" - 指标类型
    pss_collector_mode: str
    process_check_interval: float
    monitoring_timeout: float
    graceful_shutdown_timeout: float

    # [monitor.scheduling] - 必需字段
    scheduling_policy: str
    monitor_core: int
    manual_build_cores: str
    manual_monitoring_cores: str
    enable_cpu_affinity: bool
    max_concurrent_monitors: int
    thread_name_prefix: str

    # [monitor.hybrid] - 混合架构配置（有默认值）
    hybrid_discovery_interval: float = 0.01  # 发现Worker扫描间隔（秒）
    hybrid_sampling_workers: int = 4  # 采样Worker数量
    hybrid_task_queue_size: int = 1000  # 任务队列大小
    hybrid_result_queue_size: int = 2000  # 结果队列大小
    hybrid_enable_prioritization: bool = True  # 启用任务优先级
    hybrid_max_retry_count: int = 3  # 最大重试次数
    hybrid_queue_timeout: float = 0.1  # 队列操作超时时间（秒）
    hybrid_worker_timeout: float = 5.0  # Worker操作超时时间（秒）
    hybrid_enable_queue_monitoring: bool = True  # 启用队列监控
    hybrid_batch_result_size: int = 50  # 结果批处理大小

    # [monitor.storage] - 存储配置（有默认值）
    storage: StorageConfig = None  # 存储配置，将在验证时设置默认值


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
