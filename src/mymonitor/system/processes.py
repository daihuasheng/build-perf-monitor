"""
Process analysis and parsing utilities.

This module provides functions for analyzing and parsing process information,
including extraction of actual commands from shell wrappers.
"""

import logging
import shlex
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def parse_shell_wrapper_command(cmd_name: str, cmd_full: str) -> Tuple[str, str]:
    """Parse shell wrapper commands to extract the actual executed command.
    
    This function identifies common shell wrapper patterns (like sh -c, bash -c, etc.)
    and extracts the actual command being executed, while excluding cases where
    the shell is used to launch scripts.
    
    Args:
        cmd_name: The process command name (e.g., 'sh', 'bash').
        cmd_full: The complete command line string.
        
    Returns:
        A tuple of (parsed_command_name, parsed_full_command).
        If not a shell wrapper or parsing fails, returns the original (cmd_name, cmd_full).
        
    Examples:
        >>> parse_shell_wrapper_command('sh', 'sh -c "gcc -o test test.c"')
        ('gcc', 'gcc -o test test.c')
        >>> parse_shell_wrapper_command('bash', 'bash script.sh')
        ('bash', 'bash script.sh')
    """
    
    # Only handle common shell commands
    if cmd_name not in ['sh', 'bash', 'zsh', 'dash']:
        return cmd_name, cmd_full
    
    # Check if this is a shell -c pattern
    if ' -c ' not in cmd_full:
        return cmd_name, cmd_full
    
    try:
        # Use shlex to safely parse the command line
        parts = shlex.split(cmd_full)
        
        # Find the position of the -c argument
        c_index = -1
        for i, part in enumerate(parts):
            if part == '-c' and i + 1 < len(parts):
                c_index = i
                break
        
        if c_index == -1:
            return cmd_name, cmd_full
            
        # Get the command string after -c
        wrapped_command = parts[c_index + 1]
        
        # Parse the wrapped command again
        try:
            wrapped_parts = shlex.split(wrapped_command)
        except ValueError as e:
            # If parsing fails, it might be complex shell syntax, keep original
            logger.debug(f"Failed to parse shell command '{wrapped_command}': {e}")
            return cmd_name, cmd_full
            
        if not wrapped_parts:
            return cmd_name, cmd_full
            
        # Get the base name of the wrapped command
        wrapped_cmd_name = Path(wrapped_parts[0]).name
        
        # Check if this is a script file - if so, keep the original shell classification
        if (wrapped_cmd_name.endswith('.sh') or 
            wrapped_cmd_name.endswith('.py') or 
            wrapped_cmd_name.endswith('.pl') or
            wrapped_cmd_name.endswith('.rb') or
            wrapped_cmd_name.endswith('.js') or
            wrapped_parts[0].endswith('.sh') or
            wrapped_parts[0].endswith('.py') or
            wrapped_parts[0].endswith('.pl') or
            wrapped_parts[0].endswith('.rb') or
            wrapped_parts[0].endswith('.js')):
            return cmd_name, cmd_full
            
        # Check if this contains obvious script content (shell syntax)
        if any(syntax in wrapped_command for syntax in ['&&', '||', '|', ';', '$(', '`']):
            # If it contains shell syntax, keep original shell classification as this is script logic
            return cmd_name, cmd_full
                
        # If the wrapped command looks like a simple program call, return the parsed result
        logger.debug(f"Shell wrapper detected: '{cmd_full}' -> unwrapped: '{wrapped_cmd_name}', '{wrapped_command}'")
        return wrapped_cmd_name, wrapped_command
        
    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to parse shell wrapper command '{cmd_full}': {e}")
        return cmd_name, cmd_full
