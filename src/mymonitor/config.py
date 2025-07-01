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
from typing import Optional

# Import the data models from the central location to ensure type safety.
from .data_models import AppConfig, MonitorConfig, ProjectConfig, RuleConfig

logger = logging.getLogger(__name__)

# --- Global Singleton for Configuration ---

# This global variable will hold the single instance of the loaded AppConfig.
_CONFIG: Optional[AppConfig] = None

# Defines the default path to the main configuration file, relative to this script's location.
# This can be programmatically overridden (e.g., in tests or by the CLI main.py)
# to load a different configuration.
_CONFIG_FILE_PATH = Path(__file__).parent.parent.parent / "conf" / "config.toml"


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

        monitor_config = MonitorConfig(
            # from [monitor.collection]
            interval_seconds=collection_settings.get("interval_seconds", 1),
            metric_type=collection_settings.get("metric_type", "pss_psutil"),
            pss_collector_mode=collection_settings.get(
                "pss_collector_mode", "full_scan"
            ),
            # from [monitor.scheduling]
            monitor_core=scheduling_settings.get("monitor_core", 0),
            scheduling_policy=scheduling_settings.get(
                "scheduling_policy", "adaptive"
            ),
            manual_build_cores=scheduling_settings.get("manual_build_cores", ""),
            manual_monitoring_cores=scheduling_settings.get(
                "manual_monitoring_cores", ""
            ),
            # from [monitor.general]
            default_jobs=general_settings.get("default_jobs", [4, 8, 16]),
            skip_plots=general_settings.get("skip_plots", False),
            log_root_dir=Path(general_settings.get("log_root_dir", "logs")),
            categorization_cache_size=general_settings.get(
                "categorization_cache_size", 4096
            ),
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
        projects_config = [
            ProjectConfig(
                name=p.get("name"),
                dir=Path(p.get("dir")),
                build_command_template=p.get("build_command_template"),
                process_pattern=p.get("process_pattern"),
                clean_command_template=p.get("clean_command_template", ""),
                setup_command_template=p.get("setup_command_template", ""),
            )
            for p in projects_data.get("projects", [])
        ]

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
        # Unpack each rule dictionary directly into the RuleConfig dataclass.
        rules_config = [RuleConfig(**r) for r in rules_data.get("rules", [])]
        # Sort rules by priority in descending order. This is crucial for the
        # classification logic to work correctly, as it ensures more specific
        # rules are checked before more general ones.
        rules_config.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Loaded and sorted {len(rules_config)} categorization rules.")

        # Assemble the final, comprehensive configuration object.
        return AppConfig(
            monitor=monitor_config,
            projects=projects_config,
            rules=rules_config,
        )

    except FileNotFoundError as e:
        logger.error(f"A configuration file was not found: {e}", exc_info=True)
        raise
    except (tomllib.TOMLDecodeError, TypeError, KeyError) as e:
        logger.error(f"Error parsing configuration files: {e}", exc_info=True)
        raise


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
