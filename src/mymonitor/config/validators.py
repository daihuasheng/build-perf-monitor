"""
Configuration validation utilities.

This module provides specialized validation functions for different types
of configuration data including monitor settings, projects, and rules.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from ..models.config import MonitorConfig, ProjectConfig, RuleConfig
from ..validation import (
    ValidationError,
    validate_positive_integer,
    validate_positive_float,
    validate_enum_choice,
    validate_regex_pattern,
    validate_command_template,
    validate_simple_command,
    validate_project_name,
)

logger = logging.getLogger(__name__)


def validate_monitor_config(monitor_data: Dict[str, Any]) -> MonitorConfig:
    """
    Validate and create a MonitorConfig from raw configuration data.

    Args:
        monitor_data: Raw monitor configuration from TOML

    Returns:
        Validated MonitorConfig instance

    Raises:
        ValidationError: If validation fails
    """
    general_settings = monitor_data.get("general", {})
    collection_settings = monitor_data.get("collection", {})
    scheduling_settings = monitor_data.get("scheduling", {})
    hybrid_settings = monitor_data.get("hybrid", {})

    try:
        # Validate collection settings
        interval_seconds = validate_positive_float(
            collection_settings.get("interval_seconds", 1),
            min_value=0.001,  # 1ms minimum
            max_value=60.0,  # 60s maximum
            field_name="monitor.collection.interval_seconds",
        )

        process_check_interval = validate_positive_float(
            collection_settings.get("process_check_interval", 0.1),
            min_value=0.001,  # 1ms minimum
            max_value=10.0,  # 10s maximum
            field_name="monitor.collection.process_check_interval",
        )

        monitoring_timeout = validate_positive_float(
            collection_settings.get("monitoring_timeout", 30.0),
            min_value=1.0,  # 1s minimum
            max_value=300.0,  # 5m maximum
            field_name="monitor.collection.monitoring_timeout",
        )

        graceful_shutdown_timeout = validate_positive_float(
            collection_settings.get("graceful_shutdown_timeout", 5.0),
            min_value=0.1,  # 100ms minimum
            max_value=60.0,  # 1m maximum
            field_name="monitor.collection.graceful_shutdown_timeout",
        )

        metric_type = validate_enum_choice(
            collection_settings.get("metric_type", "pss_psutil"),
            valid_choices=["pss_psutil", "rss_pidstat"],
            field_name="monitor.collection.metric_type",
        )

        pss_collector_mode = validate_enum_choice(
            collection_settings.get("pss_collector_mode", "full_scan"),
            valid_choices=["full_scan", "descendants_only"],
            field_name="monitor.collection.pss_collector_mode",
        )

        # Validate scheduling settings
        monitor_core = validate_positive_integer(
            scheduling_settings.get("monitor_core", 0),
            min_value=0,
            max_value=1023,  # Reasonable upper bound
            field_name="monitor.scheduling.monitor_core",
        )

        scheduling_policy = validate_enum_choice(
            scheduling_settings.get("scheduling_policy", "adaptive"),
            valid_choices=["adaptive", "manual"],
            field_name="monitor.scheduling.scheduling_policy",
        )

        enable_cpu_affinity = scheduling_settings.get("enable_cpu_affinity", True)
        if not isinstance(enable_cpu_affinity, bool):
            raise ValidationError(
                "monitor.scheduling.enable_cpu_affinity must be a boolean"
            )

        max_concurrent_monitors = validate_positive_integer(
            scheduling_settings.get("max_concurrent_monitors", 8),
            min_value=1,
            max_value=128,
            field_name="monitor.scheduling.max_concurrent_monitors",
        )

        thread_name_prefix = scheduling_settings.get(
            "thread_name_prefix", "MonitorWorker"
        )
        if not isinstance(thread_name_prefix, str) or not thread_name_prefix.strip():
            raise ValidationError(
                "monitor.scheduling.thread_name_prefix must be a non-empty string"
            )

        # Validate adaptive strategy specific settings
        if scheduling_policy == "adaptive":
            _validate_adaptive_strategy_settings(scheduling_settings)
        elif scheduling_policy == "manual":
            _validate_manual_strategy_settings(scheduling_settings)

        # Validate general settings
        default_jobs = general_settings.get("default_jobs", [4, 8, 16])
        if not isinstance(default_jobs, list) or not default_jobs:
            raise ValidationError(
                "monitor.general.default_jobs must be a non-empty list"
            )

        validated_default_jobs = []
        for i, job in enumerate(default_jobs):
            validated_job = validate_positive_integer(
                job,
                min_value=1,
                max_value=1024,
                field_name=f"monitor.general.default_jobs[{i}]",
            )
            validated_default_jobs.append(validated_job)

        categorization_cache_size = validate_positive_integer(
            general_settings.get("categorization_cache_size", 4096),
            min_value=16,
            max_value=1048576,  # 1M entries max
            field_name="monitor.general.categorization_cache_size",
        )

        # Validate log root directory (create if it doesn't exist)
        log_root_dir_str = general_settings.get("log_root_dir", "logs")
        log_root_dir = Path(log_root_dir_str)
        try:
            log_root_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValidationError(f"Cannot create log directory '{log_root_dir}': {e}")

        # Validate hybrid settings (混合架构是唯一架构)
        hybrid_discovery_interval = validate_positive_float(
            hybrid_settings.get("hybrid_discovery_interval", 0.01),
            min_value=0.001,  # 1ms minimum
            max_value=1.0,  # 1s maximum
            field_name="monitor.hybrid.hybrid_discovery_interval",
        )

        hybrid_sampling_workers = validate_positive_integer(
            hybrid_settings.get("hybrid_sampling_workers", 4),
            min_value=1,
            max_value=64,  # Reasonable upper bound
            field_name="monitor.hybrid.hybrid_sampling_workers",
        )

        hybrid_task_queue_size = validate_positive_integer(
            hybrid_settings.get("hybrid_task_queue_size", 1000),
            min_value=10,
            max_value=100000,
            field_name="monitor.hybrid.hybrid_task_queue_size",
        )

        hybrid_result_queue_size = validate_positive_integer(
            hybrid_settings.get("hybrid_result_queue_size", 2000),
            min_value=10,
            max_value=100000,
            field_name="monitor.hybrid.hybrid_result_queue_size",
        )

        hybrid_enable_prioritization = hybrid_settings.get(
            "hybrid_enable_prioritization", True
        )
        if not isinstance(hybrid_enable_prioritization, bool):
            raise ValidationError(
                "monitor.hybrid.hybrid_enable_prioritization must be a boolean"
            )

        # Validate additional hybrid settings
        hybrid_max_retry_count = validate_positive_integer(
            hybrid_settings.get("hybrid_max_retry_count", 3),
            min_value=1,
            max_value=10,
            field_name="monitor.hybrid.hybrid_max_retry_count",
        )

        hybrid_queue_timeout = validate_positive_float(
            hybrid_settings.get("hybrid_queue_timeout", 0.1),
            min_value=0.01,  # 10ms minimum
            max_value=10.0,  # 10s maximum
            field_name="monitor.hybrid.hybrid_queue_timeout",
        )

        hybrid_worker_timeout = validate_positive_float(
            hybrid_settings.get("hybrid_worker_timeout", 5.0),
            min_value=1.0,  # 1s minimum
            max_value=300.0,  # 5min maximum
            field_name="monitor.hybrid.hybrid_worker_timeout",
        )

        hybrid_enable_queue_monitoring = hybrid_settings.get(
            "hybrid_enable_queue_monitoring", True
        )
        if not isinstance(hybrid_enable_queue_monitoring, bool):
            raise ValidationError(
                "monitor.hybrid.hybrid_enable_queue_monitoring must be a boolean"
            )

        hybrid_batch_result_size = validate_positive_integer(
            hybrid_settings.get("hybrid_batch_result_size", 50),
            min_value=1,
            max_value=1000,
            field_name="monitor.hybrid.hybrid_batch_result_size",
        )

        return MonitorConfig(
            # from [monitor.collection]
            interval_seconds=interval_seconds,
            metric_type=metric_type,
            pss_collector_mode=pss_collector_mode,
            process_check_interval=process_check_interval,
            monitoring_timeout=monitoring_timeout,
            graceful_shutdown_timeout=graceful_shutdown_timeout,
            # from [monitor.scheduling]
            scheduling_policy=scheduling_policy,
            monitor_core=monitor_core,
            manual_build_cores=scheduling_settings.get("manual_build_cores", ""),
            manual_monitoring_cores=scheduling_settings.get(
                "manual_monitoring_cores", ""
            ),
            enable_cpu_affinity=enable_cpu_affinity,
            max_concurrent_monitors=max_concurrent_monitors,
            thread_name_prefix=thread_name_prefix,
            # from [monitor.general]
            default_jobs=validated_default_jobs,
            skip_plots=general_settings.get("skip_plots", False),
            log_root_dir=log_root_dir,
            categorization_cache_size=categorization_cache_size,
            # from [monitor.hybrid] (混合架构是唯一架构)
            hybrid_discovery_interval=hybrid_discovery_interval,
            hybrid_sampling_workers=hybrid_sampling_workers,
            hybrid_task_queue_size=hybrid_task_queue_size,
            hybrid_result_queue_size=hybrid_result_queue_size,
            hybrid_enable_prioritization=hybrid_enable_prioritization,
            hybrid_max_retry_count=hybrid_max_retry_count,
            hybrid_queue_timeout=hybrid_queue_timeout,
            hybrid_worker_timeout=hybrid_worker_timeout,
            hybrid_enable_queue_monitoring=hybrid_enable_queue_monitoring,
            hybrid_batch_result_size=hybrid_batch_result_size,
        )

    except ValidationError as e:
        logger.error(f"Monitor configuration validation failed: {e}")
        raise


def validate_projects_config(
    projects_data: List[Dict[str, Any]],
) -> List[ProjectConfig]:
    """
    Validate and create ProjectConfig instances from raw configuration data.

    Args:
        projects_data: List of raw project configurations from TOML

    Returns:
        List of validated ProjectConfig instances

    Raises:
        ValidationError: If validation fails
    """
    projects_config = []
    existing_project_names = []

    for i, project_data in enumerate(projects_data):
        try:
            # Validate project name
            project_name = validate_project_name(
                project_data.get("name", ""),
                existing_names=existing_project_names,
                field_name=f"projects[{i}].name",
            )
            existing_project_names.append(project_name)

            # Validate project directory - store as string without checking existence
            project_dir_str = project_data.get("dir", "").strip()
            if not project_dir_str:
                raise ValidationError(f"projects[{i}].dir cannot be empty")

            # Validate build command template
            build_command = validate_command_template(
                project_data.get("build_command_template", ""),
                field_name=f"projects[{i}].build_command_template",
            )

            # Validate process pattern (regex)
            process_pattern = validate_regex_pattern(
                project_data.get("process_pattern", ""),
                field_name=f"projects[{i}].process_pattern",
            )

            # Validate optional command templates
            clean_command = project_data.get("clean_command_template", "")
            if clean_command:
                clean_command = validate_simple_command(
                    clean_command, field_name=f"projects[{i}].clean_command_template"
                )

            setup_command = project_data.get("setup_command_template", "")
            if setup_command:
                setup_command = validate_simple_command(
                    setup_command, field_name=f"projects[{i}].setup_command_template"
                )

            project_config = ProjectConfig(
                name=project_name,
                dir=project_dir_str,
                build_command_template=build_command,
                process_pattern=process_pattern,
                clean_command_template=clean_command,
                setup_command_template=setup_command,
            )
            projects_config.append(project_config)

        except ValidationError as e:
            logger.error(f"Project configuration validation failed: {e}")
            raise

    if not projects_config:
        raise ValidationError("No valid projects found in configuration")

    return projects_config


def validate_rules_config(rules_data: List[Dict[str, Any]]) -> List[RuleConfig]:
    """
    Validate and create RuleConfig instances from raw configuration data.

    Args:
        rules_data: List of raw rule configurations from TOML

    Returns:
        List of validated RuleConfig instances, sorted by priority

    Raises:
        ValidationError: If validation fails
    """
    rules_config = []

    for i, rule_data in enumerate(rules_data):
        try:
            # Validate priority
            priority = validate_positive_integer(
                rule_data.get("priority", 0),
                min_value=1,
                max_value=10000,
                field_name=f"rules[{i}].priority",
            )

            # Validate major_category
            major_category = rule_data.get("major_category", "").strip()
            if not major_category:
                raise ValidationError(f"rules[{i}].major_category cannot be empty")

            # Validate category
            category = rule_data.get("category", "").strip()
            if not category:
                raise ValidationError(f"rules[{i}].category cannot be empty")

            # Validate match_field
            match_field = validate_enum_choice(
                rule_data.get("match_field", ""),
                valid_choices=["current_cmd_name", "current_cmd_full", "orig_cmd_full"],
                field_name=f"rules[{i}].match_field",
            )

            # Validate match_type
            match_type = validate_enum_choice(
                rule_data.get("match_type", ""),
                valid_choices=["exact", "in_list", "regex", "contains"],
                field_name=f"rules[{i}].match_type",
            )

            # Validate patterns - handle both 'pattern' (singular) and 'patterns' (plural)
            patterns_data = rule_data.get("patterns")
            pattern_data = rule_data.get("pattern")

            # Determine which field to use based on what's provided
            if patterns_data is not None and pattern_data is not None:
                # Both provided - use patterns and warn about duplication
                logger.warning(
                    f"rules[{i}]: both 'pattern' and 'patterns' provided, using 'patterns'"
                )
                final_patterns = patterns_data
            elif patterns_data is not None:
                # Only patterns provided
                final_patterns = patterns_data
            elif pattern_data is not None:
                # Only pattern provided
                final_patterns = pattern_data
            else:
                # Neither provided
                raise ValidationError(
                    f"rules[{i}]: must have either 'pattern' or 'patterns' field"
                )

            # Validate patterns based on match_type
            if match_type == "in_list":
                # For in_list, we need a list
                if not isinstance(final_patterns, list):
                    # Try to convert single pattern to list for convenience
                    if isinstance(final_patterns, str):
                        final_patterns = [final_patterns]
                    else:
                        raise ValidationError(
                            f"rules[{i}]: match_type 'in_list' requires patterns to be a list"
                        )

                if not final_patterns:
                    raise ValidationError(
                        f"rules[{i}]: match_type 'in_list' requires non-empty patterns list"
                    )

                validated_patterns = []
                for j, pattern in enumerate(final_patterns):
                    if not isinstance(pattern, str) or not pattern.strip():
                        raise ValidationError(
                            f"rules[{i}].patterns[{j}] must be a non-empty string"
                        )
                    validated_patterns.append(pattern.strip())

                final_patterns = validated_patterns

            else:
                # For other match types ('exact', 'regex', 'contains'), expect string
                if isinstance(final_patterns, list):
                    if len(final_patterns) == 1:
                        # Convert single-item list to string for convenience
                        final_patterns = final_patterns[0]
                    else:
                        raise ValidationError(
                            f"rules[{i}]: match_type '{match_type}' requires a single pattern, not a list"
                        )

                if not isinstance(final_patterns, str) or not final_patterns.strip():
                    raise ValidationError(
                        f"rules[{i}]: match_type '{match_type}' requires a non-empty string pattern"
                    )

                # For regex patterns, validate the regex
                if match_type == "regex":
                    validate_regex_pattern(
                        final_patterns, field_name=f"rules[{i}].pattern"
                    )

                final_patterns = final_patterns.strip()

            # Create validated rule with both pattern and patterns for compatibility
            patterns_value = final_patterns
            pattern_value = final_patterns if isinstance(final_patterns, str) else None

            rule_config = RuleConfig(
                priority=priority,
                major_category=major_category,
                category=category,
                match_field=match_field,
                match_type=match_type,
                patterns=patterns_value,
                pattern=pattern_value,
                comment=rule_data.get("comment", ""),
            )
            rules_config.append(rule_config)

        except ValidationError as e:
            logger.error(f"Rule configuration validation failed: {e}")
            raise

    # Sort rules by priority in descending order. This is crucial for the
    # classification logic to work correctly, as it ensures more specific
    # rules are checked before more general ones.
    rules_config.sort(key=lambda r: r.priority, reverse=True)
    logger.info(f"Loaded and validated {len(rules_config)} categorization rules.")

    return rules_config


def _validate_adaptive_strategy_settings(scheduling_settings: Dict[str, Any]) -> None:
    """
    Validate adaptive CPU allocation strategy specific settings.

    Args:
        scheduling_settings: Scheduling configuration section

    Raises:
        ValidationError: If adaptive strategy settings are invalid
    """
    # For adaptive strategy, manual core settings should not be specified
    manual_build_cores = scheduling_settings.get("manual_build_cores", "")
    manual_monitoring_cores = scheduling_settings.get("manual_monitoring_cores", "")

    if manual_build_cores and manual_build_cores.strip():
        logger.warning(
            "manual_build_cores is specified but will be ignored in adaptive mode"
        )

    if manual_monitoring_cores and manual_monitoring_cores.strip():
        logger.warning(
            "manual_monitoring_cores is specified but will be ignored in adaptive mode"
        )

    # Validate that CPU affinity is enabled for adaptive strategy
    enable_cpu_affinity = scheduling_settings.get("enable_cpu_affinity", True)
    if not enable_cpu_affinity:
        logger.warning(
            "CPU affinity is disabled in adaptive mode, which may reduce performance"
        )


def _validate_manual_strategy_settings(scheduling_settings: Dict[str, Any]) -> None:
    """
    Validate manual CPU allocation strategy specific settings.

    Args:
        scheduling_settings: Scheduling configuration section

    Raises:
        ValidationError: If manual strategy settings are invalid
    """
    manual_build_cores = scheduling_settings.get("manual_build_cores", "")
    manual_monitoring_cores = scheduling_settings.get("manual_monitoring_cores", "")

    # For manual strategy, core specifications are required
    if not manual_build_cores or not manual_build_cores.strip():
        raise ValidationError(
            "manual_build_cores must be specified when using manual scheduling policy"
        )

    if not manual_monitoring_cores or not manual_monitoring_cores.strip():
        raise ValidationError(
            "manual_monitoring_cores must be specified when using manual scheduling policy"
        )

    # Validate core range format (basic validation)
    try:
        _validate_core_range_format(manual_build_cores, "manual_build_cores")
        _validate_core_range_format(manual_monitoring_cores, "manual_monitoring_cores")
    except Exception as e:
        raise ValidationError(f"Invalid core range format: {e}")


def _validate_core_range_format(core_range: str, field_name: str) -> None:
    """
    Validate CPU core range format.

    Args:
        core_range: Core range string (e.g., "0-3", "0,2,4", "0-1,4-7")
        field_name: Field name for error reporting

    Raises:
        ValidationError: If core range format is invalid
    """
    if not core_range or not core_range.strip():
        raise ValidationError(f"{field_name} cannot be empty")

    # Basic format validation - allow digits, commas, hyphens
    import re

    if not re.match(r"^[\d,\-\s]+$", core_range.strip()):
        raise ValidationError(
            f"{field_name} contains invalid characters. "
            f"Use format like '0-3' or '0,2,4' or '0-1,4-7'"
        )

    # First check for negative numbers specifically (not ranges like "0-7")
    if re.search(r"(^|,)-\d+", core_range.strip()):
        # Extract the negative number for error message
        negative_nums = re.findall(r"(?:^|,)-(\d+)", core_range)
        if negative_nums:
            raise ValidationError(
                f"{field_name} contains invalid core number -{negative_nums[0]}. "
                f"Core numbers must be between 0 and 1023"
            )

    # Check for reasonable core numbers (0-1023)
    parts = re.findall(r"\d+", core_range)
    if not parts:  # No digits found
        raise ValidationError(f"{field_name} must contain at least one core number")

    for part in parts:
        core_num = int(part)
        if core_num < 0 or core_num > 1023:
            raise ValidationError(
                f"{field_name} contains invalid core number {core_num}. "
                f"Core numbers must be between 0 and 1023"
            )

    # Check for invalid patterns like "0-", "0--3", ",0", "0,"
    if re.search(r"(-$)|(--)|^,|,$", core_range.strip()):
        raise ValidationError(
            f"{field_name} has invalid format. "
            f"Use format like '0-3' or '0,2,4' or '0-1,4-7'"
        )
