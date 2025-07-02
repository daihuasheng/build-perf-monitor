"""
Configuration loading and management for the MyMonitor application.

This module is responsible for loading settings from a hierarchy of TOML files,
validating them against data models, and providing a single, typed configuration
object for use throughout the application.

It employs a singleton pattern, ensuring that configuration files are read and
parsed only once per application run. The main entry point for other modules
is the `get_config()` function.

The configuration structure is hierarchical:
1. A main `config.toml` file defines global settings and points to other files.
2. A `projects.toml` file defines the specific projects to be monitored.
3. A `rules.toml` file defines the rules for process categorization.
"""

import logging
import tomllib
from pathlib import Path
from typing import NoReturn, Optional, Union

# Import the data models from the central location to ensure type safety.
from .data_models import (
    AppConfig, MonitorConfig, ProjectConfig, RuleConfig, ErrorSeverity, handle_config_error,
    ValidationError, validate_positive_integer, validate_positive_float, validate_path_exists,
    validate_enum_choice, validate_regex_pattern, validate_command_template, validate_project_name
)

logger = logging.getLogger(__name__)

# --- Global Singleton for Configuration ---

# This global variable will hold the single instance of the loaded AppConfig.
_CONFIG: Optional[AppConfig] = None

# Defines the default path to the main configuration file, relative to this script's location.
# This can be programmatically overridden (e.g., in tests or by the CLI main.py)
# to load a different configuration.
_CONFIG_FILE_PATH = Path(__file__).parent.parent.parent / "conf" / "config.toml"

# Configuration loading and management now uses centralized error handling from data_models


def _load_config(config_path: Path) -> AppConfig:
    """
    Loads the main TOML configuration and any referenced sub-configuration files.

    This is the internal workhorse function that performs the file I/O and parsing.
    It reads the main config file, then uses the paths specified within it to
    load the project and rule configurations. It populates the dataclasses defined
    in `data_models.py` and returns a fully constructed `AppConfig` object.

    Args:
        config_path: The absolute path to the main `config.toml` file.

    Returns:
        A fully populated AppConfig object.

    Raises:
        FileNotFoundError: If the main config file or any referenced sub-config
                           file does not exist.
        KeyError: If a required key (like 'projects_config' or 'rules_config')
                  is missing from the configuration.
        tomllib.TOMLDecodeError: If any of the TOML files are malformed.
        TypeError: If the data in the TOML files does not match the expected types.
    """
    logger.info(f"Loading main configuration from: {config_path}")
    if not config_path.exists():
        logger.error(f"Main configuration file not found: {config_path}")
        raise FileNotFoundError(f"Main configuration file not found: {config_path}")

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # --- Load Monitor Config (from the new nested structure) ---
        monitor_data = data.get("monitor", {})
        general_settings = monitor_data.get("general", {})
        collection_settings = monitor_data.get("collection", {})
        scheduling_settings = monitor_data.get("scheduling", {})

        # Validate monitor configuration values
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
            
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

        monitor_config = MonitorConfig(
            # from [monitor.collection]
            interval_seconds=interval_seconds,
            metric_type=metric_type,
            pss_collector_mode=pss_collector_mode,
            # from [monitor.scheduling]
            monitor_core=monitor_core,
            scheduling_policy=scheduling_policy,
            manual_build_cores=scheduling_settings.get("manual_build_cores", ""),
            manual_monitoring_cores=scheduling_settings.get("manual_monitoring_cores", ""),
            # from [monitor.general]
            default_jobs=validated_default_jobs,
            skip_plots=general_settings.get("skip_plots", False),
            log_root_dir=log_root_dir,
            categorization_cache_size=categorization_cache_size,
        )

        # --- Load Projects and Rules from referenced files ---
        paths_data = data.get("paths", {})
        # The directory of the main config file is used as the base for relative paths.
        config_dir = config_path.parent

        # Load Projects
        projects_file = paths_data.get("projects_config")
        if not projects_file:
            raise KeyError(
                "Missing 'projects_config' path in [paths] section of config.toml"
            )
        projects_path = config_dir / projects_file
        logger.info(f"Loading projects from: {projects_path}")
        with open(projects_path, "rb") as f_proj:
            projects_data = tomllib.load(f_proj)
        
        # Validate and create projects configuration
        projects_config = []
        existing_project_names = []
        
        for i, project_data in enumerate(projects_data.get("projects", [])):
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
                    required_placeholders=["N"],  # Accept <N> format for backward compatibility
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

        # Load Rules
        rules_file = paths_data.get("rules_config")
        if not rules_file:
            raise KeyError(
                "Missing 'rules_config' path in [paths] section of config.toml"
            )
        rules_path = config_dir / rules_file
        logger.info(f"Loading rules from: {rules_path}")
        with open(rules_path, "rb") as f_rules:
            rules_data = tomllib.load(f_rules)
        
        # Validate and create rules configuration
        rules_config = []
        for i, rule_data in enumerate(rules_data.get("rules", [])):
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

        # Assemble the final, comprehensive configuration object.
        return AppConfig(
            monitor=monitor_config,
            projects=projects_config,
            rules=rules_config,
        )

    except FileNotFoundError as e:
        handle_config_error(
            error=e,
            context="loading configuration file",
            severity=ErrorSeverity.CRITICAL,
            reraise=True,
            logger=logger
        )
        raise  # Explicit re-raise for linter clarity
    except (tomllib.TOMLDecodeError, TypeError, KeyError) as e:
        handle_config_error(
            error=e,
            context="parsing configuration files",
            severity=ErrorSeverity.CRITICAL,
            reraise=True,
            logger=logger
        )
        raise  # Explicit re-raise for linter clarity


def get_config() -> AppConfig:
    """
    Returns the global application configuration, loading it if necessary.

    This function implements the singleton pattern for configuration. The first
    time it is called, it invokes `_load_config` to read and parse the files.
    On all subsequent calls, it returns the already-loaded configuration object,
    avoiding redundant file I/O and ensuring a consistent state.

    This is the designated public function for accessing configuration from
    any other module in the application.

    Returns:
        The singleton AppConfig instance.
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config(_CONFIG_FILE_PATH)
    return _CONFIG
