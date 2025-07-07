"""
Thread pool manager for optimal resource utilization in async monitoring.

This module provides intelligent thread pool management with CPU affinity
support and resource optimization for the async monitoring system.
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
    """Configuration for thread pool management."""
    max_workers: int = 4
    thread_name_prefix: str = "AsyncMonitor"
    enable_cpu_affinity: bool = True
    dedicated_cores: Optional[List[int]] = None
    monitor_resource_usage: bool = True
    shutdown_timeout: float = 10.0


class ManagedThreadPoolExecutor:
    """
    Enhanced ThreadPoolExecutor with CPU affinity and resource monitoring.
    
    This class extends ThreadPoolExecutor functionality with:
    - CPU affinity management per thread
    - Resource usage monitoring
    - Graceful shutdown handling
    - Thread-local initialization
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
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'threads_created': 0,
            'current_active_threads': 0
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
                    logger.warning(f"Invalid cores specified: {invalid_cores}, using available cores")
                    self.config.dedicated_cores = [c for c in self.config.dedicated_cores if c in available_cores]
            
            # Determine optimal worker count
            if self.config.dedicated_cores:
                optimal_workers = min(len(self.config.dedicated_cores), self.config.max_workers)
            else:
                optimal_workers = min(len(available_cores), self.config.max_workers)
            
            # Create executor with thread initializer
            self.executor = ThreadPoolExecutor(
                max_workers=optimal_workers,
                thread_name_prefix=self.config.thread_name_prefix,
                initializer=self._thread_initializer if self.config.enable_cpu_affinity else None
            )
            
            self.is_shutdown = False
            logger.info(f"Started thread pool with {optimal_workers} workers, CPU affinity: {self.config.enable_cpu_affinity}")
            
        except Exception as e:
            handle_error(
                error=e,
                context="starting managed thread pool",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger
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
                self.stats['tasks_submitted'] += 1
                
            future = self.executor.submit(fn, *args, **kwargs)
            
            # Track active futures
            self.active_futures.add(future)
            
            # Add completion callback
            future.add_done_callback(self._task_completed)
            
            return future
            
        except Exception as e:
            with self._lock:
                self.stats['tasks_failed'] += 1
            handle_error(
                error=e,
                context="submitting task to thread pool",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger
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
                logger=logger
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
        stats['is_shutdown'] = self.is_shutdown
        stats['active_futures'] = len(self.active_futures)
        stats['success_rate'] = (
            stats['tasks_completed'] / max(1, stats['tasks_submitted']) * 100
        )
        
        return stats
    
    def _thread_initializer(self) -> None:
        """
        Initialize thread with CPU affinity settings.
        
        This runs once per thread when the thread is created.
        """
        try:
            with self._lock:
                self.stats['threads_created'] += 1
                self.stats['current_active_threads'] += 1
            
            # Set CPU affinity if configured
            if self.config.enable_cpu_affinity and self.config.dedicated_cores:
                # Round-robin assignment of cores to threads
                thread_id = threading.get_ident()
                core_index = (self.stats['threads_created'] - 1) % len(self.config.dedicated_cores)
                assigned_core = self.config.dedicated_cores[core_index]
                
                try:
                    set_current_thread_affinity([assigned_core])
                    self.thread_affinities[thread_id] = assigned_core
                    logger.debug(f"Thread {thread_id} bound to core {assigned_core}")
                except Exception as e:
                    logger.warning(f"Failed to set thread affinity for thread {thread_id}: {e}")
                    
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
                    self.stats['tasks_failed'] += 1
                else:
                    self.stats['tasks_completed'] += 1
                    
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
    Global thread pool manager for async monitoring system.
    
    This class manages multiple thread pools for different purposes:
    - Monitoring collectors
    - Build execution
    - Data processing
    - I/O operations
    """
    
    def __init__(self):
        """Initialize the thread pool manager."""
        self.pools: Dict[str, ManagedThreadPoolExecutor] = {}
        self.default_config = ThreadPoolConfig()
        self.is_initialized = False
        
    def initialize(self, configs: Optional[Dict[str, ThreadPoolConfig]] = None) -> None:
        """
        Initialize thread pools with configurations.
        
        Args:
            configs: Dictionary of pool name -> configuration
        """
        if self.is_initialized:
            logger.warning("Thread pool manager already initialized")
            return
            
        try:
            configs = configs or {}
            
            # Create default pools
            default_pools = {
                'monitoring': ThreadPoolConfig(
                    max_workers=4,
                    thread_name_prefix="Monitor",
                    enable_cpu_affinity=True
                ),
                'build': ThreadPoolConfig(
                    max_workers=2,
                    thread_name_prefix="Build",
                    enable_cpu_affinity=True
                ),
                'io': ThreadPoolConfig(
                    max_workers=2,
                    thread_name_prefix="IO",
                    enable_cpu_affinity=False
                )
            }
            
            # Merge with provided configs
            final_configs = {**default_pools, **configs}
            
            # Create and start pools
            for pool_name, config in final_configs.items():
                pool = ManagedThreadPoolExecutor(config)
                pool.start()
                self.pools[pool_name] = pool
                
            self.is_initialized = True
            logger.info(f"Thread pool manager initialized with {len(self.pools)} pools")
            
        except Exception as e:
            handle_error(
                error=e,
                context="initializing thread pool manager",
                severity=ErrorSeverity.ERROR,
                reraise=True,
                logger=logger
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
    
    def submit_to_pool(self, pool_name: str, fn: Callable, *args, **kwargs) -> Future:
        """
        Submit a task to a specific thread pool.
        
        Args:
            pool_name: Name of the pool
            fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Future representing the task
            
        Raises:
            ValueError: If pool doesn't exist
        """
        pool = self.get_pool(pool_name)
        if pool is None:
            raise ValueError(f"Thread pool '{pool_name}' not found")
            
        return pool.submit(fn, *args, **kwargs)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all thread pools.
        
        Returns:
            Dictionary of pool name -> statistics
        """
        return {name: pool.get_stats() for name, pool in self.pools.items()}
    
    def shutdown_all(self, wait: bool = True) -> None:
        """
        Shutdown all thread pools.
        
        Args:
            wait: Whether to wait for completion
        """
        if not self.is_initialized:
            return
            
        try:
            for pool_name, pool in self.pools.items():
                try:
                    pool.shutdown(wait=wait)
                except Exception as e:
                    logger.warning(f"Error shutting down pool '{pool_name}': {e}")
                    
            self.pools.clear()
            self.is_initialized = False
            logger.info("All thread pools shutdown")
            
        except Exception as e:
            handle_error(
                error=e,
                context="shutting down all thread pools",
                severity=ErrorSeverity.WARNING,
                reraise=False,
                logger=logger
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


def initialize_global_thread_pools(configs: Optional[Dict[str, ThreadPoolConfig]] = None) -> None:
    """
    Initialize global thread pools with configurations.
    
    Args:
        configs: Dictionary of pool name -> configuration
    """
    manager = get_thread_pool_manager()
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
