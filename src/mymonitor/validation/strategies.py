"""
Simplified error handling strategies.

This module provides essential error handling patterns without over-engineering,
focusing on the functionality that is actually used in the application.
"""

import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def simple_retry(
    func: Callable[[], T],
    max_attempts: int = 3,
    delay: float = 1.0,
    context: str = "operation"
) -> T:
    """
    Simple retry mechanism for basic operations.
    
    This replaces the complex retry strategies with a simple,
    effective retry mechanism for the few cases that need it.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        context: Context description for error messages
        
    Returns:
        Result from func if successful
        
    Raises:
        Exception: Last exception if all attempts fail
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            result = func()
            if attempt > 0:
                logger.info(f"Operation '{context}' succeeded on attempt {attempt + 1}")
            return result
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                logger.debug(f"Attempt {attempt + 1} failed for {context}: {e}")
                time.sleep(delay)
            else:
                logger.error(f"All {max_attempts} attempts failed for {context}: {e}")
    
    # This should never be reached, but for type safety
    raise last_exception or Exception(f"All attempts failed for {context}")


def with_simple_retry(max_attempts: int = 3, delay: float = 1.0, context: str = ""):
    """
    Decorator for simple retry functionality.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        context: Context description for error messages
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            return simple_retry(
                lambda: func(*args, **kwargs),
                max_attempts=max_attempts,
                delay=delay,
                context=context or func.__name__
            )
        return wrapper
    return decorator


# Legacy compatibility functions for existing code
def get_error_recovery_strategy(name: str) -> 'SimpleRetryStrategy':
    """
    Legacy compatibility function.
    Returns a simple retry strategy regardless of name.
    """
    return SimpleRetryStrategy()


class SimpleRetryStrategy:
    """
    Simple retry strategy that replaces complex error recovery.
    """
    
    def __init__(self, max_attempts: int = 3, delay: float = 1.0):
        self.max_attempts = max_attempts
        self.delay = delay
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with simple retry.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to func
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Result from func
        """
        # Extract context and logger from kwargs if present
        context = kwargs.pop('error_context', func.__name__)
        logger_instance = kwargs.pop('logger_instance', None)  # Remove unused logger parameter
        
        return simple_retry(
            lambda: func(*args, **kwargs),
            max_attempts=self.max_attempts,
            delay=self.delay,
            context=context
        )


def with_error_recovery(strategy_name: str = "default", error_context: str = ""):
    """
    Legacy compatibility decorator.
    
    Args:
        strategy_name: Ignored for compatibility
        error_context: Context description for error messages
        
    Returns:
        Decorated function with simple retry
    """
    return with_simple_retry(context=error_context)


# Legacy compatibility aliases
create_file_operation_strategy = lambda: SimpleRetryStrategy()
create_network_operation_strategy = lambda: SimpleRetryStrategy()
create_process_operation_strategy = lambda: SimpleRetryStrategy()
create_monitoring_operation_strategy = lambda: SimpleRetryStrategy()

def register_error_recovery_strategy(name: str, strategy: Any) -> None:
    """Legacy compatibility function - does nothing."""
    pass
