"""
Configuration loading and management for the MyMonitor application.

This module loads settings from a central 'config.toml' file, validates them,
and provides a typed configuration object for use throughout the application.
"""

import logging
import tomllib
from pathlib import Path
from typing import List, Optional

# Import the data models from the central location
from .data_models import AppConfig, MonitorConfig, ProjectConfig, RuleConfig

logger = logging.getLogger(__name__)

# --- Global Singleton for Configuration ---

_CONFIG: Optional[AppConfig] = None
_CONFIG_FILE_PATH = Path(__file__).parent.parent.parent / "conf" / "config.toml"


def _load_config(config_path: Path) -> AppConfig:
    """
    Loads the main TOML configuration and any referenced sub-configuration files.
    """
    logger.info(f"Loading main configuration from: {config_path}")
    if not config_path.exists():
        logger.error(f"Main configuration file not found: {config_path}")
        raise FileNotFoundError(f"Main configuration file not found: {config_path}")

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # --- Load Monitor Config (from main file) ---
        monitor_data = data.get("monitor", {})
        monitor_config = MonitorConfig(
            interval_seconds=monitor_data.get("interval_seconds", 1),
            default_jobs=monitor_data.get("default_jobs", [4, 8, 16]),
            metric_type=monitor_data.get("metric_type", "pss_psutil"),
            monitor_core=monitor_data.get("monitor_core", 0),
            build_cores_policy=monitor_data.get("build_cores_policy", "all_others"),
            specific_build_cores=monitor_data.get("specific_build_cores", ""),
            skip_plots=monitor_data.get("skip_plots", False),
            log_root_dir=Path(monitor_data.get("log_root_dir", "logs")),
        )

        # --- Load Projects and Rules from referenced files ---
        paths_data = data.get("paths", {})
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
        rules_config = [RuleConfig(**r) for r in rules_data.get("rules", [])]
        rules_config.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Loaded and sorted {len(rules_config)} categorization rules.")

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
    This ensures the configuration is loaded only once.
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config(_CONFIG_FILE_PATH)
    return _CONFIG
