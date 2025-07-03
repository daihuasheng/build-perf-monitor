"""
Configuration management and singleton pattern.

This module provides the main configuration loading and management interface,
implementing a singleton pattern to ensure configuration is loaded only once.
"""

import logging
from pathlib import Path
from typing import Optional

from ..models.config import AppConfig
from ..validation import handle_config_error, ErrorSeverity
from .loader import load_main_config, load_projects_config, load_rules_config, get_config_paths
from .validators import validate_monitor_config, validate_projects_config, validate_rules_config

logger = logging.getLogger(__name__)

# --- Global Singleton for Configuration ---

# This global variable will hold the single instance of the loaded AppConfig.
_CONFIG: Optional[AppConfig] = None

# Defines the default path to the main configuration file, relative to this script's location.
# This can be programmatically overridden (e.g., in tests or by the CLI main.py)
# to load a different configuration.
_CONFIG_FILE_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "config.toml"


def set_config_path(config_path: Path) -> None:
    """
    Set a custom configuration file path.
    
    This function allows overriding the default configuration path,
    useful for testing or when running from different locations.
    
    Args:
        config_path: Path to the main config.toml file
        
    Note:
        This must be called before the first call to get_config()
        to have any effect.
    """
    global _CONFIG_FILE_PATH, _CONFIG
    _CONFIG_FILE_PATH = config_path
    # Clear cached config to force reload with new path
    _CONFIG = None
    logger.info(f"Configuration path set to: {config_path}")


def clear_config_cache() -> None:
    """
    Clear the cached configuration, forcing a reload on next access.
    
    Useful for testing or when configuration files have been modified
    and you want to reload them.
    """
    global _CONFIG
    _CONFIG = None
    logger.debug("Configuration cache cleared")


def _load_config(config_path: Path) -> AppConfig:
    """
    Load the complete application configuration from TOML files.
    
    This is the internal function that performs the actual loading and validation
    of all configuration files. It coordinates between the loader and validator
    modules to create a fully validated AppConfig instance.
    
    Args:
        config_path: Path to the main config.toml file
        
    Returns:
        Fully validated AppConfig instance
        
    Raises:
        FileNotFoundError: If configuration files are missing
        ValidationError: If configuration validation fails
        tomllib.TOMLDecodeError: If TOML files are malformed
        KeyError: If required configuration keys are missing
    """
    try:
        # Load main configuration file
        main_config_data = load_main_config(config_path)
        config_dir = config_path.parent
        
        # Get paths to sub-configuration files
        config_paths = get_config_paths(main_config_data, config_dir)
        
        # Load and validate monitor configuration
        monitor_data = main_config_data.get("monitor", {})
        monitor_config = validate_monitor_config(monitor_data)
        
        # Load and validate projects configuration
        projects_data = load_projects_config(config_paths["projects"])
        projects_config = validate_projects_config(projects_data)
        
        # Load and validate rules configuration
        rules_data = load_rules_config(config_paths["rules"])
        rules_config = validate_rules_config(rules_data)
        
        # Assemble the final configuration object
        app_config = AppConfig(
            monitor=monitor_config,
            projects=projects_config,
            rules=rules_config,
        )
        
        logger.info(f"Successfully loaded configuration with {len(projects_config)} projects and {len(rules_config)} rules")
        return app_config
        
    except FileNotFoundError as e:
        handle_config_error(
            error=e,
            context="loading configuration file",
            severity=ErrorSeverity.CRITICAL,
            reraise=True,
            logger=logger
        )
        raise
    except Exception as e:
        handle_config_error(
            error=e,
            context="processing configuration",
            severity=ErrorSeverity.CRITICAL,
            reraise=True,
            logger=logger
        )
        raise


def get_config() -> AppConfig:
    """
    Get the global application configuration, loading it if necessary.
    
    This function implements the singleton pattern for configuration access.
    The first time it's called, it loads and validates the configuration files.
    Subsequent calls return the cached configuration instance.
    
    This is the main public interface for accessing configuration throughout
    the application.
    
    Returns:
        The singleton AppConfig instance
        
    Raises:
        FileNotFoundError: If configuration files are missing
        ValidationError: If configuration validation fails
        tomllib.TOMLDecodeError: If TOML files are malformed
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config(_CONFIG_FILE_PATH)
    return _CONFIG


def is_config_loaded() -> bool:
    """
    Check if configuration has been loaded and cached.
    
    Returns:
        True if configuration is cached, False otherwise
    """
    return _CONFIG is not None


def get_config_info() -> dict:
    """
    Get information about the current configuration state.
    
    Returns:
        Dictionary with configuration metadata
    """
    return {
        "config_loaded": is_config_loaded(),
        "config_path": str(_CONFIG_FILE_PATH),
        "projects_count": len(_CONFIG.projects) if _CONFIG else 0,
        "rules_count": len(_CONFIG.rules) if _CONFIG else 0,
    }
