"""
Asynchronous error handling utilities for async monitoring system.

This module provides structured error handling, logging, and recovery
mechanisms specifically designed for async operations.
"""

import asyncio
import logging
import sys
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Callable, List, Union, AsyncGenerator
from datetime import datetime

from ..validation import ErrorSeverity


class AsyncErrorType(Enum):
    """Types of async errors."""
    MONITORING_ERROR = "monitoring_error"
    BUILD_ERROR = "build_error"
    THREAD_POOL_ERROR = "thread_pool_error"
    TIMEOUT_ERROR = "timeout_error"
    CANCELLATION_ERROR = "cancellation_error"
    RESOURCE_ERROR = "resource_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class AsyncErrorContext:
    """Context information for async errors."""
    error_type: AsyncErrorType
    severity: ErrorSeverity
    component: str
    operation: str
    timestamp: datetime
    task_name: Optional[str] = None
    correlation_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class AsyncErrorHandler:
    """
    Centralized async error handling with structured logging and recovery.
    
    This class provides:
    - Structured error logging with context
    - Automatic error classification
    - Recovery mechanism registration
    - Error aggregation and reporting
    - Async-safe exception handling
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the async error handler.
        
        Args:
            logger: Optional logger instance, defaults to module logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[AsyncErrorContext] = []
        self.recovery_handlers: Dict[AsyncErrorType, List[Callable]] = {}
        self.error_counts: Dict[AsyncErrorType, int] = {}
        self.max_history_size = 1000
        self._lock = asyncio.Lock()
        
    async def handle_error(
        self,
        error: Exception,
        context: AsyncErrorContext,
        reraise: bool = True,
        attempt_recovery: bool = True
    ) -> bool:
        """
        Handle an async error with structured logging and optional recovery.
        
        Args:
            error: The exception that occurred
            context: Error context information
            reraise: Whether to reraise the exception
            attempt_recovery: Whether to attempt recovery
            
        Returns:
            True if error was handled/recovered, False otherwise
            
        Raises:
            The original exception if reraise=True and recovery fails
        """
        async with self._lock:
            # Update error counts
            self.error_counts[context.error_type] = self.error_counts.get(context.error_type, 0) + 1
            
            # Add to history
            self.error_history.append(context)
            if len(self.error_history) > self.max_history_size:
                self.error_history.pop(0)
        
        # Log the error
        await self._log_error(error, context)
        
        # Attempt recovery if requested
        recovery_successful = False
        if attempt_recovery:
            recovery_successful = await self._attempt_recovery(error, context)
        
        # Reraise if requested and recovery failed
        if reraise and not recovery_successful:
            raise error
            
        return recovery_successful
    
    async def handle_task_error(
        self,
        task: asyncio.Task,
        component: str,
        operation: str,
        error_type: AsyncErrorType = AsyncErrorType.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR
    ) -> Optional[Exception]:
        """
        Handle error from a completed asyncio task.
        
        Args:
            task: The completed task
            component: Component name where error occurred
            operation: Operation being performed
            error_type: Type of error
            severity: Error severity
            
        Returns:
            The exception if one occurred, None otherwise
        """
        if not task.done():
            return None
            
        exception = task.exception()
        if exception is None:
            return None
            
        context = AsyncErrorContext(
            error_type=error_type,
            severity=severity,
            component=component,
            operation=operation,
            timestamp=datetime.now(),
            task_name=task.get_name(),
            correlation_id=str(id(task))
        )
        
        await self.handle_error(exception, context, reraise=False)
        return exception
    
    @asynccontextmanager
    async def error_context(
        self,
        component: str,
        operation: str,
        error_type: AsyncErrorType = AsyncErrorType.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        reraise: bool = True,
        attempt_recovery: bool = True
    ) -> AsyncGenerator[None, None]:
        """
        Async context manager for error handling.
        
        Args:
            component: Component name
            operation: Operation being performed
            error_type: Type of error
            severity: Error severity
            reraise: Whether to reraise exceptions
            attempt_recovery: Whether to attempt recovery
            
        Usage:
            async with error_handler.error_context("monitor", "collect_data"):
                # Your async code here
                pass
        """
        try:
            yield
        except Exception as e:
            context = AsyncErrorContext(
                error_type=error_type,
                severity=severity,
                component=component,
                operation=operation,
                timestamp=datetime.now()
            )
            
            await self.handle_error(e, context, reraise=reraise, attempt_recovery=attempt_recovery)
    
    def register_recovery_handler(
        self,
        error_type: AsyncErrorType,
        handler: Callable[[Exception, AsyncErrorContext], Any]
    ) -> None:
        """
        Register a recovery handler for specific error types.
        
        Args:
            error_type: Type of error to handle
            handler: Recovery handler function (can be async)
        """
        if error_type not in self.recovery_handlers:
            self.recovery_handlers[error_type] = []
        self.recovery_handlers[error_type].append(handler)
    
    async def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all errors handled.
        
        Returns:
            Dictionary containing error statistics and recent errors
        """
        async with self._lock:
            recent_errors = self.error_history[-10:] if self.error_history else []
            
            return {
                'total_errors': len(self.error_history),
                'error_counts': self.error_counts.copy(),
                'recent_errors': [
                    {
                        'error_type': ctx.error_type.value,
                        'severity': ctx.severity.value,
                        'component': ctx.component,
                        'operation': ctx.operation,
                        'timestamp': ctx.timestamp.isoformat(),
                        'task_name': ctx.task_name,
                        'correlation_id': ctx.correlation_id
                    }
                    for ctx in recent_errors
                ]
            }
    
    async def clear_error_history(self) -> None:
        """Clear the error history."""
        async with self._lock:
            self.error_history.clear()
            self.error_counts.clear()
    
    async def _log_error(self, error: Exception, context: AsyncErrorContext) -> None:
        """
        Log an error with structured context.
        
        Args:
            error: The exception
            context: Error context
        """
        log_level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(context.severity, logging.ERROR)
        
        # Create structured log message
        log_data = {
            'error_type': context.error_type.value,
            'severity': context.severity.value,
            'component': context.component,
            'operation': context.operation,
            'timestamp': context.timestamp.isoformat(),
            'task_name': context.task_name,
            'correlation_id': context.correlation_id,
            'exception_type': type(error).__name__,
            'exception_message': str(error)
        }
        
        if context.additional_data:
            log_data.update(context.additional_data)
        
        # Log with traceback for errors and critical issues
        if context.severity in (ErrorSeverity.ERROR, ErrorSeverity.CRITICAL):
            self.logger.log(
                log_level,
                f"Async error in {context.component}.{context.operation}: {error}",
                extra=log_data,
                exc_info=True
            )
        else:
            self.logger.log(
                log_level,
                f"Async {context.severity.value} in {context.component}.{context.operation}: {error}",
                extra=log_data
            )
    
    async def _attempt_recovery(self, error: Exception, context: AsyncErrorContext) -> bool:
        """
        Attempt to recover from an error using registered handlers.
        
        Args:
            error: The exception
            context: Error context
            
        Returns:
            True if recovery was successful, False otherwise
        """
        handlers = self.recovery_handlers.get(context.error_type, [])
        if not handlers:
            return False
            
        for handler in handlers:
            try:
                # Call handler (may be async)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(error, context)
                else:
                    result = handler(error, context)
                
                # If handler returns True, recovery was successful
                if result is True:
                    self.logger.info(
                        f"Recovery successful for {context.error_type.value} in {context.component}.{context.operation}"
                    )
                    return True
                    
            except Exception as recovery_error:
                self.logger.warning(
                    f"Recovery handler failed for {context.error_type.value}: {recovery_error}"
                )
                
        return False


class AsyncErrorDecorator:
    """
    Decorator for async functions to handle errors automatically.
    
    This decorator provides a convenient way to add error handling
    to async functions without manually wrapping them.
    """
    
    def __init__(
        self,
        error_handler: AsyncErrorHandler,
        component: str,
        error_type: AsyncErrorType = AsyncErrorType.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        reraise: bool = True,
        attempt_recovery: bool = True
    ):
        """
        Initialize the decorator.
        
        Args:
            error_handler: Error handler instance
            component: Component name
            error_type: Type of error
            severity: Error severity
            reraise: Whether to reraise exceptions
            attempt_recovery: Whether to attempt recovery
        """
        self.error_handler = error_handler
        self.component = component
        self.error_type = error_type
        self.severity = severity
        self.reraise = reraise
        self.attempt_recovery = attempt_recovery
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorate an async function with error handling.
        
        Args:
            func: The async function to decorate
            
        Returns:
            Decorated function
        """
        if not asyncio.iscoroutinefunction(func):
            raise ValueError("AsyncErrorDecorator can only be used with async functions")
        
        async def wrapper(*args, **kwargs):
            operation = func.__name__
            
            async with self.error_handler.error_context(
                component=self.component,
                operation=operation,
                error_type=self.error_type,
                severity=self.severity,
                reraise=self.reraise,
                attempt_recovery=self.attempt_recovery
            ):
                return await func(*args, **kwargs)
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__annotations__ = func.__annotations__
        
        return wrapper


# Global error handler instance
_global_error_handler: Optional[AsyncErrorHandler] = None


def get_async_error_handler() -> AsyncErrorHandler:
    """
    Get the global async error handler instance.
    
    Returns:
        Global AsyncErrorHandler instance
    """
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = AsyncErrorHandler()
    return _global_error_handler


def async_error_handled(
    component: str,
    error_type: AsyncErrorType = AsyncErrorType.UNKNOWN_ERROR,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    reraise: bool = True,
    attempt_recovery: bool = True
) -> AsyncErrorDecorator:
    """
    Decorator factory for async error handling.
    
    Args:
        component: Component name
        error_type: Type of error
        severity: Error severity
        reraise: Whether to reraise exceptions
        attempt_recovery: Whether to attempt recovery
        
    Returns:
        AsyncErrorDecorator instance
        
    Usage:
        @async_error_handled("monitor", AsyncErrorType.MONITORING_ERROR)
        async def my_async_function():
            # Your code here
            pass
    """
    error_handler = get_async_error_handler()
    return AsyncErrorDecorator(
        error_handler=error_handler,
        component=component,
        error_type=error_type,
        severity=severity,
        reraise=reraise,
        attempt_recovery=attempt_recovery
    )
