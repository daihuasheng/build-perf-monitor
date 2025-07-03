"""
Process classification and categorization utilities.

This module provides functionality to categorize processes into meaningful groups
based on configurable rules, with intelligent caching and shell wrapper parsing.
"""

import logging
import re
from typing import Dict, Tuple

from .. import config
from ..system.processes import parse_shell_wrapper_command

logger = logging.getLogger(__name__)

# Cache for process categorization to avoid repeated rule evaluation
_categorization_cache: Dict[Tuple[str, str], Tuple[str, str]] = {}


def get_process_category(cmd_name: str, cmd_full: str) -> Tuple[str, str]:
    """Categorize a process based on configured classification rules.

    This function applies a set of prioritized rules to classify processes into
    major and minor categories (e.g., 'Compiler', 'gcc'). It uses caching to
    avoid repeated rule evaluation for identical process command lines, which
    significantly improves performance on builds with many similar processes.

    The function also handles shell wrapper commands by attempting to extract
    the actual executed command from shell wrappers like 'sh -c "gcc ..."'.

    Args:
        cmd_name: Base name of the command executable (e.g., 'gcc').
        cmd_full: Complete command line with all arguments.

    Returns:
        Tuple of (major_category, minor_category) strings.
        Returns ('Other', 'Other_<cmd_name>') if no rules match.

    Note:
        Rules are loaded from the application configuration and applied in
        priority order. Results are cached up to the configured cache size limit.

    Examples:
        >>> get_process_category('gcc', 'gcc -O2 -c file.c')
        ('Compiler', 'gcc')
        >>> get_process_category('unknown_tool', 'unknown_tool --help')
        ('Other', 'Other_unknown_tool')
    """
    # Check cache first
    cache_key = (cmd_name, cmd_full)
    if cache_key in _categorization_cache:
        return _categorization_cache[cache_key]
    
    app_config = config.get_config()
    
    # Try to parse shell wrapper commands
    current_cmd_name, current_cmd_full = parse_shell_wrapper_command(cmd_name, cmd_full)

    for rule in app_config.rules:
        target_field_value = (
            current_cmd_name
            if rule.match_field == "current_cmd_name"
            else current_cmd_full
        )

        match = False
        if rule.match_type == "exact":
            # For exact match, patterns should be a string
            pattern_to_match = rule.patterns if isinstance(rule.patterns, str) else rule.patterns[0] if rule.patterns else ""
            match = target_field_value == pattern_to_match
        elif rule.match_type == "contains":
            # For contains match, patterns should be a string
            pattern_to_match = rule.patterns if isinstance(rule.patterns, str) else rule.patterns[0] if rule.patterns else ""
            match = pattern_to_match and pattern_to_match in target_field_value
        elif rule.match_type == "regex":
            # For regex match, patterns should be a string
            pattern_to_match = rule.patterns if isinstance(rule.patterns, str) else rule.patterns[0] if rule.patterns else ""
            if pattern_to_match:
                match = bool(re.search(pattern_to_match, target_field_value))
        elif rule.match_type == "in_list":
            # For in_list match, patterns should be a list
            patterns_to_check = rule.patterns if isinstance(rule.patterns, list) else [rule.patterns] if rule.patterns else []
            match = target_field_value in patterns_to_check

        if match:
            result = (rule.major_category, rule.category)
            # Cache the result, but respect the cache size limit
            if len(_categorization_cache) < app_config.monitor.categorization_cache_size:
                _categorization_cache[cache_key] = result
            return result

    # If no rules match, generate a classification based on the original command name
    result = ("Other", f"Other_{current_cmd_name}")
    # Cache the result, but respect the cache size limit
    if len(_categorization_cache) < app_config.monitor.categorization_cache_size:
        _categorization_cache[cache_key] = result
    return result


def clear_categorization_cache() -> None:
    """Clear the process categorization cache.
    
    Useful for testing or when rules have been updated and you want
    to ensure fresh categorization results.
    """
    global _categorization_cache
    _categorization_cache.clear()
    logger.debug("Process categorization cache cleared")


def get_cache_stats() -> Dict[str, int]:
    """Get statistics about the categorization cache.
    
    Returns:
        Dictionary with cache statistics including size and hit rate info.
    """
    return {
        "cache_size": len(_categorization_cache),
        "cache_entries": len(_categorization_cache),
    }
