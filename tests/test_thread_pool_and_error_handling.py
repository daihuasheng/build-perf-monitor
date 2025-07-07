"""
Tests for thread pool manager and async error handler.
"""

import asyncio
import pytest
import time
import threading
from concurrent.futures import Future
from unittest.mock import Mock, patch

from mymonitor.executor.thread_pool import (
    ThreadPoolConfig, ManagedThreadPoolExecutor, ThreadPoolManager,
    get_thread_pool_manager, initialize_global_thread_pools,
    shutdown_global_thread_pools
)
from mymonitor.validation.error_handler import (
    AsyncErrorHandler, AsyncErrorType, AsyncErrorContext, AsyncErrorDecorator,
    get_async_error_handler, async_error_handled
)
from mymonitor.validation import ErrorSeverity
from datetime import datetime


class TestThreadPoolManager:
    """Test cases for thread pool management."""
    
    def test_thread_pool_config_defaults(self):
        """Test thread pool configuration defaults."""
        config = ThreadPoolConfig()
        assert config.max_workers == 4
        assert config.thread_name_prefix == "AsyncMonitor"
        assert config.enable_cpu_affinity is True
        assert config.dedicated_cores is None
        assert config.monitor_resource_usage is True
        assert config.shutdown_timeout == 10.0
    
    def test_managed_thread_pool_executor_lifecycle(self):
        """Test managed thread pool executor lifecycle."""
        config = ThreadPoolConfig(max_workers=2, enable_cpu_affinity=False)
        executor = ManagedThreadPoolExecutor(config)
        
        # Test initial state
        assert executor.executor is None
        assert not executor.is_shutdown
        
        # Start executor
        executor.start()
        assert executor.executor is not None
        assert not executor.is_shutdown
        
        # Submit task
        def simple_task():
            return "success"
        
        future = executor.submit(simple_task)
        assert isinstance(future, Future)
        
        # Wait for completion
        result = future.result(timeout=5.0)
        assert result == "success"
        
        # Check stats
        stats = executor.get_stats()
        assert stats['tasks_submitted'] >= 1
        assert stats['tasks_completed'] >= 1
        assert stats['success_rate'] > 0
        
        # Shutdown
        executor.shutdown()
        assert executor.is_shutdown
        assert executor.executor is None
    
    def test_managed_thread_pool_executor_error_handling(self):
        """Test error handling in managed thread pool executor."""
        config = ThreadPoolConfig(max_workers=2, enable_cpu_affinity=False)
        executor = ManagedThreadPoolExecutor(config)
        executor.start()
        
        def failing_task():
            raise ValueError("Test error")
        
        future = executor.submit(failing_task)
        
        # Should capture the exception
        with pytest.raises(ValueError):
            future.result(timeout=5.0)
        
        # Check stats reflect the failure
        stats = executor.get_stats()
        assert stats['tasks_failed'] >= 1
        
        executor.shutdown()
    
    def test_thread_pool_manager_lifecycle(self):
        """Test thread pool manager lifecycle."""
        manager = ThreadPoolManager()
        
        # Test initialization
        assert not manager.is_initialized
        assert len(manager.pools) == 0
        
        # Initialize with default configs
        manager.initialize()
        assert manager.is_initialized
        assert len(manager.pools) >= 3  # monitoring, build, io
        
        # Test pool retrieval
        monitoring_pool = manager.get_pool('monitoring')
        assert monitoring_pool is not None
        assert isinstance(monitoring_pool, ManagedThreadPoolExecutor)
        
        # Test task submission
        def test_task():
            return "pool_task_success"
        
        future = manager.submit_to_pool('monitoring', test_task)
        result = future.result(timeout=5.0)
        assert result == "pool_task_success"
        
        # Test stats
        stats = manager.get_all_stats()
        assert 'monitoring' in stats
        assert 'build' in stats
        assert 'io' in stats
        
        # Shutdown
        manager.shutdown_all()
        assert not manager.is_initialized
        assert len(manager.pools) == 0
    
    def test_thread_pool_manager_custom_config(self):
        """Test thread pool manager with custom configurations."""
        manager = ThreadPoolManager()
        
        custom_configs = {
            'test_pool': ThreadPoolConfig(
                max_workers=1,
                thread_name_prefix="Test",
                enable_cpu_affinity=False
            )
        }
        
        manager.initialize(custom_configs)
        
        test_pool = manager.get_pool('test_pool')
        assert test_pool is not None
        assert test_pool.config.max_workers == 1
        assert test_pool.config.thread_name_prefix == "Test"
        
        manager.shutdown_all()
    
    def test_thread_pool_manager_context_manager(self):
        """Test thread pool manager as context manager."""
        with ThreadPoolManager() as manager:
            assert manager.is_initialized
            
            def test_task():
                return "context_manager_success"
            
            future = manager.submit_to_pool('monitoring', test_task)
            result = future.result(timeout=5.0)
            assert result == "context_manager_success"
        
        # Should be shutdown after context exit
        assert not manager.is_initialized
    
    def test_global_thread_pool_manager(self):
        """Test global thread pool manager functions."""
        # Clean up any existing global manager
        shutdown_global_thread_pools()
        
        # Test initialization
        initialize_global_thread_pools()
        
        manager = get_thread_pool_manager()
        assert manager.is_initialized
        
        # Test usage
        def global_task():
            return "global_success"
        
        future = manager.submit_to_pool('monitoring', global_task)
        result = future.result(timeout=5.0)
        assert result == "global_success"
        
        # Cleanup
        shutdown_global_thread_pools()
        
        # Should create new manager if needed
        new_manager = get_thread_pool_manager()
        assert new_manager is not None
        assert not new_manager.is_initialized


class TestAsyncErrorHandler:
    """Test cases for async error handler."""
    
    @pytest.mark.asyncio
    async def test_async_error_handler_basic(self):
        """Test basic async error handler functionality."""
        handler = AsyncErrorHandler()
        
        # Test error handling
        test_error = ValueError("Test error")
        context = AsyncErrorContext(
            error_type=AsyncErrorType.MONITORING_ERROR,
            severity=ErrorSeverity.ERROR,
            component="test",
            operation="test_operation",
            timestamp=datetime.now()
        )
        
        # Handle error without reraising
        result = await handler.handle_error(test_error, context, reraise=False)
        assert result is False  # No recovery handler, so False
        
        # Check error was recorded
        summary = await handler.get_error_summary()
        assert summary['total_errors'] == 1
        assert summary['error_counts'][AsyncErrorType.MONITORING_ERROR] == 1
        
        # Test with reraise
        with pytest.raises(ValueError):
            await handler.handle_error(test_error, context, reraise=True)
    
    @pytest.mark.asyncio
    async def test_async_error_context_manager(self):
        """Test async error context manager."""
        handler = AsyncErrorHandler()
        
        # Test successful operation
        async with handler.error_context("test", "success_op"):
            pass  # Should not raise
        
        # Test error handling
        with pytest.raises(ValueError):
            async with handler.error_context("test", "error_op"):
                raise ValueError("Context error")
        
        # Check error was recorded
        summary = await handler.get_error_summary()
        assert summary['total_errors'] == 1
        
        # Test without reraise
        async with handler.error_context("test", "no_reraise", reraise=False):
            raise ValueError("No reraise error")
        
        # Should have 2 errors now
        summary = await handler.get_error_summary()
        assert summary['total_errors'] == 2
    
    @pytest.mark.asyncio
    async def test_async_error_recovery(self):
        """Test async error recovery mechanism."""
        handler = AsyncErrorHandler()
        
        # Register recovery handler
        recovery_called = False
        
        def recovery_handler(error, context):
            nonlocal recovery_called
            recovery_called = True
            return True  # Indicate successful recovery
        
        handler.register_recovery_handler(AsyncErrorType.MONITORING_ERROR, recovery_handler)
        
        # Test recovery
        test_error = ValueError("Recoverable error")
        context = AsyncErrorContext(
            error_type=AsyncErrorType.MONITORING_ERROR,
            severity=ErrorSeverity.ERROR,
            component="test",
            operation="recovery_test",
            timestamp=datetime.now()
        )
        
        result = await handler.handle_error(test_error, context, reraise=False)
        assert result is True  # Recovery successful
        assert recovery_called
    
    @pytest.mark.asyncio
    async def test_async_error_task_handling(self):
        """Test async task error handling."""
        handler = AsyncErrorHandler()
        
        async def failing_task():
            raise ValueError("Task error")
        
        task = asyncio.create_task(failing_task())
        
        # Wait for task to complete
        try:
            await task
        except ValueError:
            pass  # Expected
        
        # Handle task error
        exception = await handler.handle_task_error(
            task, "test", "task_operation", AsyncErrorType.MONITORING_ERROR
        )
        
        assert isinstance(exception, ValueError)
        assert str(exception) == "Task error"
        
        # Check error was recorded
        summary = await handler.get_error_summary()
        assert summary['total_errors'] == 1
    
    @pytest.mark.asyncio
    async def test_async_error_decorator(self):
        """Test async error decorator."""
        handler = AsyncErrorHandler()
        
        @AsyncErrorDecorator(
            handler, "test", AsyncErrorType.MONITORING_ERROR, reraise=False
        )
        async def decorated_function():
            raise ValueError("Decorated error")
        
        # Should not raise due to reraise=False
        result = await decorated_function()
        assert result is None
        
        # Check error was recorded
        summary = await handler.get_error_summary()
        assert summary['total_errors'] == 1
    
    @pytest.mark.asyncio
    async def test_async_error_handled_decorator(self):
        """Test async_error_handled decorator factory."""
        
        @async_error_handled("test", AsyncErrorType.MONITORING_ERROR, reraise=False)
        async def decorated_function():
            raise ValueError("Factory decorated error")
        
        # Should not raise due to reraise=False
        result = await decorated_function()
        assert result is None
        
        # Check error was recorded in global handler
        global_handler = get_async_error_handler()
        summary = await global_handler.get_error_summary()
        assert summary['total_errors'] >= 1
    
    @pytest.mark.asyncio
    async def test_async_error_history_management(self):
        """Test error history management."""
        handler = AsyncErrorHandler()
        handler.max_history_size = 3  # Small size for testing
        
        # Add multiple errors
        for i in range(5):
            context = AsyncErrorContext(
                error_type=AsyncErrorType.MONITORING_ERROR,
                severity=ErrorSeverity.ERROR,
                component="test",
                operation=f"operation_{i}",
                timestamp=datetime.now()
            )
            await handler.handle_error(ValueError(f"Error {i}"), context, reraise=False)
        
        # Should only keep last 3 errors
        summary = await handler.get_error_summary()
        assert summary['total_errors'] == 3
        assert len(summary['recent_errors']) == 3
        
        # Clear history
        await handler.clear_error_history()
        summary = await handler.get_error_summary()
        assert summary['total_errors'] == 0
        assert len(summary['recent_errors']) == 0


class TestAsyncIntegration:
    """Integration tests for async components."""
    
    @pytest.mark.asyncio
    async def test_thread_pool_with_error_handler(self):
        """Test thread pool integration with error handler."""
        # Setup
        manager = ThreadPoolManager()
        manager.initialize()
        
        handler = AsyncErrorHandler()
        
        # Define a task that might fail
        def risky_task(should_fail=False):
            if should_fail:
                raise ValueError("Intentional failure")
            return "success"
        
        # Test successful execution
        future = manager.submit_to_pool('monitoring', risky_task, False)
        result = future.result(timeout=5.0)
        assert result == "success"
        
        # Test error handling
        future = manager.submit_to_pool('monitoring', risky_task, True)
        
        try:
            future.result(timeout=5.0)
        except ValueError as e:
            # Handle the error
            context = AsyncErrorContext(
                error_type=AsyncErrorType.THREAD_POOL_ERROR,
                severity=ErrorSeverity.ERROR,
                component="thread_pool",
                operation="risky_task",
                timestamp=datetime.now()
            )
            await handler.handle_error(e, context, reraise=False)
        
        # Check error was recorded
        summary = await handler.get_error_summary()
        assert summary['total_errors'] == 1
        assert summary['error_counts'][AsyncErrorType.THREAD_POOL_ERROR] == 1
        
        # Cleanup
        manager.shutdown_all()
    
    @pytest.mark.asyncio
    async def test_async_operations_with_timeout(self):
        """Test async operations with timeout and error handling."""
        handler = AsyncErrorHandler()
        
        async def slow_operation():
            await asyncio.sleep(2.0)
            return "completed"
        
        # Test timeout handling
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=0.1)
        
        # Test with error context
        try:
            async with handler.error_context("test", "timeout_operation"):
                await asyncio.wait_for(slow_operation(), timeout=0.1)
        except asyncio.TimeoutError:
            pass  # Expected
        
        # Check error was recorded
        summary = await handler.get_error_summary()
        assert summary['total_errors'] == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_async_operations(self):
        """Test concurrent async operations with error handling."""
        handler = AsyncErrorHandler()
        
        async def worker_task(worker_id: int, should_fail: bool = False):
            await asyncio.sleep(0.1)  # Simulate work
            if should_fail:
                raise ValueError(f"Worker {worker_id} failed")
            return f"Worker {worker_id} success"
        
        # Create multiple concurrent tasks
        tasks = []
        for i in range(5):
            task = asyncio.create_task(worker_task(i, i % 2 == 0))  # Even workers fail
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle errors for failed tasks
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                context = AsyncErrorContext(
                    error_type=AsyncErrorType.MONITORING_ERROR,
                    severity=ErrorSeverity.ERROR,
                    component="worker",
                    operation=f"worker_task_{i}",
                    timestamp=datetime.now()
                )
                await handler.handle_error(result, context, reraise=False)
        
        # Check error handling
        summary = await handler.get_error_summary()
        # Should have 3 errors (workers 0, 2, 4)
        assert summary['total_errors'] == 3
        
        # Check successful results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 2  # Workers 1, 3
