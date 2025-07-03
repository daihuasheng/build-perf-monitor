"""
Configuration management for the mymonitor package.

This module provides a clean interface for loading, validating, and accessing
configuration data from TOML files with singleton pattern management.
"""

# Main configuration interface
from .manager import (
    clear_config_cache,
    get_config,
    get_config_info,
    is_config_loaded,
    set_config_path,
)

# For advanced usage - direct access to loaders and validators
from .loader import (
    get_config_paths,
    load_main_config,
    load_projects_config,
    load_rules_config,
    load_toml_file,
)
from .validators import (
    validate_monitor_config,
    validate_projects_config,
    validate_rules_config,
)

__all__ = [
    # Main interface
    "get_config",
    "set_config_path",
    "clear_config_cache",
    "is_config_loaded",
    "get_config_info",
    # Advanced interface
    "load_toml_file",
    "load_main_config", 
    "load_projects_config",
    "load_rules_config",
    "get_config_paths",
    "validate_monitor_config",
    "validate_projects_config",
    "validate_rules_config",
]
