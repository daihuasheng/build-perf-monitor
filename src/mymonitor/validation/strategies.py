"""
Advanced error handling strategies and patterns.

This module provides sophisticated error handling strategies including retry mechanisms,
circuit breaker patterns, timeout handling, and error recovery strategies.
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from .exceptions import ErrorSeverity, handle_error

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryStrategy(Enum):
    """Defines different retry strategies."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


class CircuitBreakerState(Enum):
    """States for circuit breaker pattern."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, rejecting calls
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[Type[BaseException]] = field(default_factory=lambda: [
        ConnectionError, TimeoutError, OSError, IOError
    ])
    non_retryable_exceptions: List[Type[BaseException]] = field(default_factory=lambda: [
        KeyboardInterrupt, SystemExit, ValueError, TypeError
    ])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # For half-open state
    timeout: float = 30.0


@dataclass
class TimeoutConfig:
    """Configuration for timeout handling."""
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    total_timeout: float = 60.0


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Prevents cascading failures by monitoring error rates and temporarily
    cutting off failing services.
    """
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
        
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time < self.config.recovery_timeout:
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
            else:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN state")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _on_success(self):
        """Handle successful function execution."""
        self.last_success_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' moved to CLOSED state")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self, error: Exception):
        """Handle failed function execution."""
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' moved to OPEN state after {self.failure_count} failures")
        
        logger.debug(f"Circuit breaker '{self.name}' failure count: {self.failure_count}")


class RetryHandler:
    """
    Advanced retry handler with multiple backoff strategies.
    
    Provides intelligent retry logic with configurable backoff strategies,
    jitter, and exception filtering.
    """
    
    def __init__(self, config: RetryConfig):
        self.config = config
        
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable(e):
                    logger.debug(f"Non-retryable exception {type(e).__name__}: {e}")
                    raise e
                
                # Don't sleep after last attempt
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.debug(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
                else:
                    logger.debug(f"Final attempt {attempt + 1} failed: {e}")
        
        # All attempts failed
        if last_exception:
            raise last_exception
        else:
            raise Exception("All retry attempts failed")
    
    def _is_retryable(self, exception: Exception) -> bool:
        """Check if an exception is retryable."""
        # Check non-retryable exceptions first
        for exc_type in self.config.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # Check retryable exceptions
        for exc_type in self.config.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        # Default to retryable for unknown exceptions
        return True
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the next retry attempt."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.initial_delay
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.initial_delay * (self.config.backoff_multiplier ** attempt)
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.initial_delay * (1 + attempt)
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.config.initial_delay * self._fibonacci(attempt + 1)
        else:
            delay = self.config.initial_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number (for fibonacci backoff)."""
        if n <= 1:
            return n
        return self._fibonacci(n - 1) + self._fibonacci(n - 2)


class ErrorRecoveryStrategy:
    """
    High-level error recovery strategy coordinator.
    
    Combines retry logic, circuit breaker, and timeout handling for
    comprehensive error recovery.
    """
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        timeout_config: Optional[TimeoutConfig] = None,
        name: str = "default"
    ):
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config
        self.timeout_config = timeout_config or TimeoutConfig()
        self.name = name
        
        # Initialize components
        self.retry_handler = RetryHandler(self.retry_config)
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config, name) if circuit_breaker_config else None
        
    def execute(
        self,
        func: Callable[..., T],
        *args,
        error_context: str = "",
        logger_instance: Optional[logging.Logger] = None,
        **kwargs
    ) -> T:
        """
        Execute a function with comprehensive error recovery.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            error_context: Context description for error logging
            logger_instance: Logger instance for error reporting
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If all recovery attempts fail
        """
        def wrapped_func(*args, **kwargs):
            if self.circuit_breaker:
                return self.circuit_breaker.call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        try:
            return self.retry_handler.execute(wrapped_func, *args, **kwargs)
        except Exception as e:
            # Log the final failure
            handle_error(
                error=e,
                context=error_context or f"executing {func.__name__}",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger_instance or logger
            )
            raise


# Pre-configured error recovery strategies for common scenarios

def create_file_operation_strategy() -> ErrorRecoveryStrategy:
    """Create error recovery strategy for file operations."""
    retry_config = RetryConfig(
        max_attempts=3,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        initial_delay=0.5,
        max_delay=5.0,
        retryable_exceptions=[OSError, IOError, PermissionError]
    )
    
    return ErrorRecoveryStrategy(
        retry_config=retry_config,
        name="file_operations"
    )


def create_network_operation_strategy() -> ErrorRecoveryStrategy:
    """Create error recovery strategy for network operations."""
    retry_config = RetryConfig(
        max_attempts=5,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        initial_delay=1.0,
        max_delay=30.0,
        retryable_exceptions=[ConnectionError, TimeoutError, OSError]
    )
    
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        success_threshold=2
    )
    
    return ErrorRecoveryStrategy(
        retry_config=retry_config,
        circuit_breaker_config=circuit_breaker_config,
        name="network_operations"
    )


def create_process_operation_strategy() -> ErrorRecoveryStrategy:
    """Create error recovery strategy for process operations."""
    retry_config = RetryConfig(
        max_attempts=3,
        strategy=RetryStrategy.FIXED_DELAY,
        initial_delay=2.0,
        retryable_exceptions=[OSError, subprocess.SubprocessError],
        non_retryable_exceptions=[KeyboardInterrupt, SystemExit, FileNotFoundError]
    )
    
    return ErrorRecoveryStrategy(
        retry_config=retry_config,
        name="process_operations"
    )


def create_monitoring_operation_strategy() -> ErrorRecoveryStrategy:
    """Create error recovery strategy for monitoring operations."""
    retry_config = RetryConfig(
        max_attempts=2,
        strategy=RetryStrategy.FIXED_DELAY,
        initial_delay=1.0,
        retryable_exceptions=[OSError, IOError, RuntimeError],
        non_retryable_exceptions=[KeyboardInterrupt, SystemExit, ValueError]
    )
    
    return ErrorRecoveryStrategy(
        retry_config=retry_config,
        name="monitoring_operations"
    )


# Global registry for error recovery strategies
_strategy_registry: Dict[str, ErrorRecoveryStrategy] = {
    "file_operations": create_file_operation_strategy(),
    "network_operations": create_network_operation_strategy(),
    "process_operations": create_process_operation_strategy(),
    "monitoring_operations": create_monitoring_operation_strategy(),
}


def get_error_recovery_strategy(name: str) -> ErrorRecoveryStrategy:
    """
    Get a pre-configured error recovery strategy by name.
    
    Args:
        name: Name of the strategy
        
    Returns:
        ErrorRecoveryStrategy instance
        
    Raises:
        KeyError: If strategy name is not found
    """
    if name not in _strategy_registry:
        raise KeyError(f"Unknown error recovery strategy: {name}. "
                      f"Available strategies: {list(_strategy_registry.keys())}")
    
    return _strategy_registry[name]


def register_error_recovery_strategy(name: str, strategy: ErrorRecoveryStrategy) -> None:
    """
    Register a custom error recovery strategy.
    
    Args:
        name: Name for the strategy
        strategy: ErrorRecoveryStrategy instance
    """
    _strategy_registry[name] = strategy
    logger.info(f"Registered error recovery strategy: {name}")


# Decorator for easy error recovery
def with_error_recovery(strategy_name: str = "default", error_context: str = ""):
    """
    Decorator to apply error recovery to a function.
    
    Args:
        strategy_name: Name of the error recovery strategy to use
        error_context: Context description for error logging
        
    Returns:
        Decorated function with error recovery
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            try:
                strategy = get_error_recovery_strategy(strategy_name)
            except KeyError:
                # Fall back to default strategy
                strategy = ErrorRecoveryStrategy(name=strategy_name)
            
            return strategy.execute(
                func,
                *args,
                error_context=error_context or f"executing {func.__name__}",
                **kwargs
            )
        return wrapper
    return decorator 