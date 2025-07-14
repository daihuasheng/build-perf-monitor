"""
Thread pool manager specifically for monitoring tasks.

This module provides thread pool management for memory monitoring tasks only.
Build processes and I/O tasks use single-threaded execution, while monitoring
tasks use dedicated thread pools for optimal performance.
"""

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List, Dict, Any, Callable, Set
from dataclasses import dataclass

from ..system.cpu_manager import set_current_thread_affinity, get_available_cores
from ..validation import handle_error, ErrorSeverity

logger = logging.getLogger(__name__)


@dataclass
class ThreadPoolConfig:
    """Configuration for monitoring thread pool management."""

    max_workers: int = 4
    thread_name_prefix: str = "MonitorWorker"
    enable_cpu_affinity: bool = True
    dedicated_cores: Optional[List[int]] = None
    monitor_resource_usage: bool = True
    shutdown_timeout: float = 10.0


class ManagedThreadPoolExecutor:
    """
    Enhanced ThreadPoolExecutor for monitoring tasks with CPU affinity.

    This class provides monitoring-specific thread pool management with:
    - CPU affinity management for monitoring workers
    - Resource usage monitoring for memory collection tasks
    - Graceful shutdown handling for long-running monitoring
    - Thread-local initialization for monitoring collectors
    """

    def __init__(self, config: ThreadPoolConfig):
        """
        Initialize the managed thread pool executor.

        Args:
            config: Thread pool configuration
        """
        self.config = config
        self.executor: Optional[ThreadPoolExecutor] = None
        self.thread_affinities: Dict[int, int] = {}  # thread_id -> core_id
        self.active_futures: Set[Future] = set()
        self.is_shutdown = False
        self._lock = threading.Lock()

        # Resource monitoring
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "threads_created": 0,
            "current_active_threads": 0,
        }

    def start(self) -> None:
        """
        Start the thread pool executor.

        Raises:
            RuntimeError: If already started or configuration is invalid
        """
        if self.executor is not None:
            raise RuntimeError("Thread pool already started")

        try:
            # Validate configuration
            available_cores = get_available_cores()
            if self.config.dedicated_cores:
                invalid_cores = set(self.config.dedicated_cores) - set(available_cores)
                if invalid_cores:
                    logger.warning(
                        f"Invalid cores specified: {invalid_cores}, using available cores"
                    )
                    self.config.dedicated_cores = [
                        c for c in self.config.dedicated_cores if c in available_cores
                    ]

            # Determine optimal worker count
            if self.config.dedicated_cores:
                optimal_workers = min(
                    len(self.config.dedicated_cores), self.config.max_workers
                )
            else:
                optimal_workers = min(len(available_cores), self.config.max_workers)

            # Create executor with thread initializer
            self.executor = ThreadPoolExecutor(
                max_workers=optimal_workers,
                thread_name_prefix=self.config.thread_name_prefix,
                initializer=(
                    self._thread_initializer
                    if self.config.enable_cpu_affinity
                    else None
                ),
            )

            self.is_shutdown = False
            logger.info(
                f"Started thread pool with {optimal_workers} workers, CPU affinity: {self.config.enable_cpu_affinity}"
            )

        except Exception as e:
            handle_error(
                error=e,
                context="starting managed thread pool",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger,
            )

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """
        Submit a task to the thread pool.

        Args:
            fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future representing the task

        Raises:
            RuntimeError: If executor is not started or is shutdown
        """
        if self.executor is None:
            raise RuntimeError("Thread pool not started")
        if self.is_shutdown:
            raise RuntimeError("Thread pool is shutdown")

        try:
            with self._lock:
                self.stats["tasks_submitted"] += 1

            future = self.executor.submit(fn, *args, **kwargs)

            # Track active futures
            self.active_futures.add(future)

            # Add completion callback
            future.add_done_callback(self._task_completed)

            return future

        except Exception as e:
            with self._lock:
                self.stats["tasks_failed"] += 1
            handle_error(
                error=e,
                context="submitting task to thread pool",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger,
            )

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """
        Shutdown the thread pool executor.

        Args:
            wait: Whether to wait for completion
            cancel_futures: Whether to cancel pending futures
        """
        if self.executor is None or self.is_shutdown:
            return

        try:
            self.is_shutdown = True

            if cancel_futures:
                # Cancel all active futures
                with self._lock:
                    for future in self.active_futures:
                        future.cancel()

            # Shutdown executor
            self.executor.shutdown(wait=wait)

            if wait:
                logger.info("Thread pool shutdown completed")
            else:
                logger.info("Thread pool shutdown initiated")

        except Exception as e:
            handle_error(
                error=e,
                context="shutting down thread pool",
                severity=ErrorSeverity.WARNING,
                reraise=False,
                logger=logger,
            )
        finally:
            self.executor = None
            self.active_futures.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current thread pool statistics.

        Returns:
            Dictionary containing usage statistics
        """
        with self._lock:
            stats = self.stats.copy()

        # Add current status
        stats["is_shutdown"] = self.is_shutdown
        stats["active_futures"] = len(self.active_futures)
        stats["success_rate"] = (
            stats["tasks_completed"] / max(1, stats["tasks_submitted"]) * 100
        )

        return stats

    def _thread_initializer(self) -> None:
        """
        Initialize thread with CPU affinity settings.

        This runs once per thread when the thread is created.
        """
        try:
            with self._lock:
                self.stats["threads_created"] += 1
                self.stats["current_active_threads"] += 1

            # Set CPU affinity if configured
            if self.config.enable_cpu_affinity and self.config.dedicated_cores:
                # Round-robin assignment of cores to threads
                thread_id = threading.get_ident()
                core_index = (self.stats["threads_created"] - 1) % len(
                    self.config.dedicated_cores
                )
                assigned_core = self.config.dedicated_cores[core_index]

                try:
                    set_current_thread_affinity([assigned_core])
                    self.thread_affinities[thread_id] = assigned_core
                    logger.debug(f"Thread {thread_id} bound to core {assigned_core}")
                except Exception as e:
                    logger.warning(
                        f"Failed to set thread affinity for thread {thread_id}: {e}"
                    )

        except Exception as e:
            logger.warning(f"Thread initialization failed: {e}")

    def _task_completed(self, future: Future) -> None:
        """
        Callback executed when a task completes.

        Args:
            future: The completed future
        """
        try:
            with self._lock:
                self.active_futures.discard(future)

                if future.cancelled():
                    # Task was cancelled, don't count as completed or failed
                    pass
                elif future.exception() is not None:
                    self.stats["tasks_failed"] += 1
                else:
                    self.stats["tasks_completed"] += 1

        except Exception as e:
            logger.warning(f"Error in task completion callback: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True)


class ThreadPoolManager:
    """
    Thread pool manager for task scheduling.

    This class manages thread pools for various task scheduling needs.
    Focuses on thread pool lifecycle management and task coordination.
    """

    def __init__(self):
        """Initialize the thread pool manager."""
        self.pools: Dict[str, ManagedThreadPoolExecutor] = {}
        self.default_config = ThreadPoolConfig()
        self.is_initialized = False

    def initialize(self, configs: Optional[Dict[str, ThreadPoolConfig]] = None) -> None:
        """
        Initialize thread pools based on configuration.

        Args:
            configs: Optional configuration for different thread pools
        """
        if self.is_initialized:
            logger.warning("Thread pool manager already initialized")
            return

        try:
            # Create thread pools based on provided configurations
            if configs:
                for pool_name, config in configs.items():
                    pool = ManagedThreadPoolExecutor(config)
                    pool.start()
                    self.pools[pool_name] = pool
                    logger.info(
                        f"Initialized thread pool '{pool_name}' with {config.max_workers} workers"
                    )
            else:
                # Create default pool if no configuration provided
                default_pool = ManagedThreadPoolExecutor(self.default_config)
                default_pool.start()
                self.pools["default"] = default_pool
                logger.info(
                    f"Initialized default thread pool with {self.default_config.max_workers} workers"
                )

            self.is_initialized = True
            logger.info("Thread pool manager initialized successfully")

        except Exception as e:
            handle_error(
                error=e,
                context="initializing thread pool manager",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger,
            )

    def get_pool(self, pool_name: str) -> Optional[ManagedThreadPoolExecutor]:
        """
        Get a thread pool by name.

        Args:
            pool_name: Name of the pool to retrieve

        Returns:
            Thread pool executor or None if not found
        """
        return self.pools.get(pool_name)

    def submit_monitoring_task(self, fn: Callable, *args, **kwargs) -> Future:
        """
        Submit a monitoring task to the monitoring thread pool.

        Args:
            fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future representing the task

        Raises:
            ValueError: If monitoring pool doesn't exist
        """
        pool = self.get_pool("monitoring")
        if pool is None:
            raise ValueError("Monitoring thread pool not initialized")

        return pool.submit(fn, *args, **kwargs)

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the monitoring thread pool.

        Returns:
            Dictionary of statistics
        """
        pool = self.get_pool("monitoring")
        return pool.get_stats() if pool else {}

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all thread pools.

        Returns:
            Dictionary of pool name -> statistics
        """
        return {name: pool.get_stats() for name, pool in self.pools.items()}

    def shutdown_all(self, wait: bool = True) -> None:
        """
        Shutdown the monitoring thread pool.

        Args:
            wait: Whether to wait for completion
        """
        if not self.is_initialized:
            return

        try:
            monitoring_pool = self.pools.get("monitoring")
            if monitoring_pool:
                try:
                    monitoring_pool.shutdown(wait=wait)
                except Exception as e:
                    logger.warning(f"Error shutting down monitoring pool: {e}")

            self.pools.clear()
            self.is_initialized = False
            logger.info("Monitoring thread pool shutdown")

        except Exception as e:
            handle_error(
                error=e,
                context="shutting down monitoring thread pool",
                severity=ErrorSeverity.WARNING,
                reraise=False,
                logger=logger,
            )

    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown_all(wait=True)


# Global thread pool manager instance
_global_thread_pool_manager: Optional[ThreadPoolManager] = None


def get_thread_pool_manager() -> ThreadPoolManager:
    """
    Get the global thread pool manager instance.

    Returns:
        Global ThreadPoolManager instance
    """
    global _global_thread_pool_manager
    if _global_thread_pool_manager is None:
        _global_thread_pool_manager = ThreadPoolManager()
    return _global_thread_pool_manager


def initialize_global_thread_pools(
    monitoring_config: Optional[ThreadPoolConfig] = None,
) -> None:
    """
    Initialize global thread pools for monitoring tasks.

    Args:
        monitoring_config: Configuration for monitoring thread pool
    """
    manager = get_thread_pool_manager()

    if monitoring_config:
        configs = {"monitoring": monitoring_config}
    else:
        # Default monitoring configuration
        default_monitoring_config = ThreadPoolConfig(
            max_workers=4,
            thread_name_prefix="MonitorWorker",
            enable_cpu_affinity=True,
            shutdown_timeout=10.0,
        )
        configs = {"monitoring": default_monitoring_config}

    manager.initialize(configs)


def shutdown_global_thread_pools(wait: bool = True) -> None:
    """
    Shutdown all global thread pools.

    Args:
        wait: Whether to wait for completion
    """
    global _global_thread_pool_manager
    if _global_thread_pool_manager is not None:
        _global_thread_pool_manager.shutdown_all(wait=wait)
        _global_thread_pool_manager = None
