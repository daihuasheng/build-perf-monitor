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
    
    try:
        # Validate collection settings
        interval_seconds = validate_positive_float(
            collection_settings.get("interval_seconds", 1),
            min_value=0.001,  # 1ms minimum
            max_value=60.0,   # 60s maximum
            field_name="monitor.collection.interval_seconds"
        )
        
        metric_type = validate_enum_choice(
            collection_settings.get("metric_type", "pss_psutil"),
            valid_choices=["pss_psutil", "rss_pidstat"],
            field_name="monitor.collection.metric_type"
        )
        
        pss_collector_mode = validate_enum_choice(
            collection_settings.get("pss_collector_mode", "full_scan"),
            valid_choices=["full_scan", "descendants_only"],
            field_name="monitor.collection.pss_collector_mode"
        )
        
        # Validate scheduling settings
        monitor_core = validate_positive_integer(
            scheduling_settings.get("monitor_core", 0),
            min_value=0,
            max_value=1023,  # Reasonable upper bound
            field_name="monitor.scheduling.monitor_core"
        )
        
        scheduling_policy = validate_enum_choice(
            scheduling_settings.get("scheduling_policy", "adaptive"),
            valid_choices=["adaptive", "manual"],
            field_name="monitor.scheduling.scheduling_policy"
        )
        
        # Validate general settings
        default_jobs = general_settings.get("default_jobs", [4, 8, 16])
        if not isinstance(default_jobs, list) or not default_jobs:
            raise ValidationError("monitor.general.default_jobs must be a non-empty list")
        
        validated_default_jobs = []
        for i, job in enumerate(default_jobs):
            validated_job = validate_positive_integer(
                job,
                min_value=1,
                max_value=1024,
                field_name=f"monitor.general.default_jobs[{i}]"
            )
            validated_default_jobs.append(validated_job)
        
        categorization_cache_size = validate_positive_integer(
            general_settings.get("categorization_cache_size", 4096),
            min_value=16,
            max_value=1048576,  # 1M entries max
            field_name="monitor.general.categorization_cache_size"
        )
        
        # Validate log root directory (create if it doesn't exist)
        log_root_dir_str = general_settings.get("log_root_dir", "logs")
        log_root_dir = Path(log_root_dir_str)
        try:
            log_root_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValidationError(f"Cannot create log directory '{log_root_dir}': {e}")
        
        return MonitorConfig(
            # from [monitor.collection]
            interval_seconds=interval_seconds,
            metric_type=metric_type,
            pss_collector_mode=pss_collector_mode,
            # from [monitor.scheduling]
            scheduling_policy=scheduling_policy,
            monitor_core=monitor_core,
            manual_build_cores=scheduling_settings.get("manual_build_cores", ""),
            manual_monitoring_cores=scheduling_settings.get("manual_monitoring_cores", ""),
            # from [monitor.general]
            default_jobs=validated_default_jobs,
            skip_plots=general_settings.get("skip_plots", False),
            log_root_dir=log_root_dir,
            categorization_cache_size=categorization_cache_size,
        )
        
    except ValidationError as e:
        logger.error(f"Monitor configuration validation failed: {e}")
        raise


def validate_projects_config(projects_data: List[Dict[str, Any]]) -> List[ProjectConfig]:
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
                field_name=f"projects[{i}].name"
            )
            existing_project_names.append(project_name)
            
            # Validate project directory - store as string without checking existence
            project_dir_str = project_data.get("dir", "").strip()
            if not project_dir_str:
                raise ValidationError(f"projects[{i}].dir cannot be empty")
            
            # Validate build command template
            build_command = validate_command_template(
                project_data.get("build_command_template", ""),
                required_placeholders=["<N>"],  # Accept <N> format for backward compatibility
                field_name=f"projects[{i}].build_command_template"
            )
            
            # Validate process pattern (regex)
            process_pattern = validate_regex_pattern(
                project_data.get("process_pattern", ""),
                field_name=f"projects[{i}].process_pattern"
            )
            
            # Validate optional command templates
            clean_command = project_data.get("clean_command_template", "")
            if clean_command:
                clean_command = validate_command_template(
                    clean_command,
                    field_name=f"projects[{i}].clean_command_template"
                )
            
            setup_command = project_data.get("setup_command_template", "")
            if setup_command:
                setup_command = validate_command_template(
                    setup_command,
                    field_name=f"projects[{i}].setup_command_template"
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
                field_name=f"rules[{i}].priority"
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
                field_name=f"rules[{i}].match_field"
            )
            
            # Validate match_type
            match_type = validate_enum_choice(
                rule_data.get("match_type", ""),
                valid_choices=["exact", "in_list", "regex", "contains"],
                field_name=f"rules[{i}].match_type"
            )
            
            # Validate patterns - handle both 'pattern' (singular) and 'patterns' (plural)
            patterns_data = rule_data.get("patterns")
            pattern_data = rule_data.get("pattern")
            
            # Determine which field to use based on what's provided
            if patterns_data is not None and pattern_data is not None:
                # Both provided - use patterns and warn about duplication
                logger.warning(f"rules[{i}]: both 'pattern' and 'patterns' provided, using 'patterns'")
                final_patterns = patterns_data
            elif patterns_data is not None:
                # Only patterns provided
                final_patterns = patterns_data
            elif pattern_data is not None:
                # Only pattern provided
                final_patterns = pattern_data
            else:
                # Neither provided
                raise ValidationError(f"rules[{i}]: must have either 'pattern' or 'patterns' field")
            
            # Validate patterns based on match_type
            if match_type == "in_list":
                # For in_list, we need a list
                if not isinstance(final_patterns, list):
                    # Try to convert single pattern to list for convenience
                    if isinstance(final_patterns, str):
                        final_patterns = [final_patterns]
                    else:
                        raise ValidationError(f"rules[{i}]: match_type 'in_list' requires patterns to be a list")
                
                if not final_patterns:
                    raise ValidationError(f"rules[{i}]: match_type 'in_list' requires non-empty patterns list")
                
                validated_patterns = []
                for j, pattern in enumerate(final_patterns):
                    if not isinstance(pattern, str) or not pattern.strip():
                        raise ValidationError(f"rules[{i}].patterns[{j}] must be a non-empty string")
                    validated_patterns.append(pattern.strip())
                
                final_patterns = validated_patterns
            
            else:
                # For other match types ('exact', 'regex', 'contains'), expect string
                if isinstance(final_patterns, list):
                    if len(final_patterns) == 1:
                        # Convert single-item list to string for convenience
                        final_patterns = final_patterns[0]
                    else:
                        raise ValidationError(f"rules[{i}]: match_type '{match_type}' requires a single pattern, not a list")
                
                if not isinstance(final_patterns, str) or not final_patterns.strip():
                    raise ValidationError(f"rules[{i}]: match_type '{match_type}' requires a non-empty string pattern")
                
                # For regex patterns, validate the regex
                if match_type == "regex":
                    validate_regex_pattern(
                        final_patterns,
                        field_name=f"rules[{i}].pattern"
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
                comment=rule_data.get("comment", "")
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
