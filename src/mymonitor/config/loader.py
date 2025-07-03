"""
Configuration file loading utilities.

This module handles the low-level loading and parsing of TOML configuration files,
including the main config.toml, projects.toml, and rules.toml files.
"""

import logging
import tomllib
from pathlib import Path
from typing import Any, Dict, List

from ..validation import handle_config_error, ErrorSeverity

logger = logging.getLogger(__name__)


def load_toml_file(file_path: Path, description: str = "configuration file") -> Dict[str, Any]:
    """
    Load and parse a TOML file with error handling.
    
    Args:
        file_path: Path to the TOML file to load
        description: Human-readable description for error messages
        
    Returns:
        Parsed TOML data as a dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        tomllib.TOMLDecodeError: If the file is malformed
    """
    logger.info(f"Loading {description} from: {file_path}")
    
    if not file_path.exists():
        logger.error(f"{description} not found: {file_path}")
        raise FileNotFoundError(f"{description} not found: {file_path}")
    
    try:
        with open(file_path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        handle_config_error(
            error=e,
            context=f"parsing {description}",
            severity=ErrorSeverity.CRITICAL,
            reraise=True,
            logger=logger
        )
        raise


def load_main_config(config_path: Path) -> Dict[str, Any]:
    """
    Load the main configuration file (config.toml).
    
    Args:
        config_path: Path to the main config.toml file
        
    Returns:
        Parsed configuration data
    """
    return load_toml_file(config_path, "main configuration file")


def load_projects_config(projects_path: Path) -> List[Dict[str, Any]]:
    """
    Load the projects configuration file (projects.toml).
    
    Args:
        projects_path: Path to the projects.toml file
        
    Returns:
        List of project configuration dictionaries
    """
    projects_data = load_toml_file(projects_path, "projects configuration file")
    return projects_data.get("projects", [])


def load_rules_config(rules_path: Path) -> List[Dict[str, Any]]:
    """
    Load the rules configuration file (rules.toml).
    
    Args:
        rules_path: Path to the rules.toml file
        
    Returns:
        List of rule configuration dictionaries
    """
    rules_data = load_toml_file(rules_path, "rules configuration file")
    return rules_data.get("rules", [])


def get_config_paths(main_config_data: Dict[str, Any], config_dir: Path) -> Dict[str, Path]:
    """
    Extract and resolve configuration file paths from main config.
    
    Args:
        main_config_data: Parsed main configuration data
        config_dir: Directory containing the main config file (for relative paths)
        
    Returns:
        Dictionary mapping config types to resolved paths
        
    Raises:
        KeyError: If required path keys are missing
    """
    paths_data = main_config_data.get("paths", {})
    
    # Check for required path keys
    projects_file = paths_data.get("projects_config")
    if not projects_file:
        raise KeyError("Missing 'projects_config' path in [paths] section of config.toml")
        
    rules_file = paths_data.get("rules_config")
    if not rules_file:
        raise KeyError("Missing 'rules_config' path in [paths] section of config.toml")
    
    return {
        "projects": config_dir / projects_file,
        "rules": config_dir / rules_file,
    }
