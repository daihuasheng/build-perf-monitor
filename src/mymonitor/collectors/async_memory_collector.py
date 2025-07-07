"""
Asynchronous memory collector that wraps existing collectors for thread-safe operation.

This module provides an AsyncMemoryCollector that wraps existing memory collectors
to make them thread-safe and compatible with AsyncIO operations, while supporting
CPU affinity for optimal performance.
"""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Iterable, Optional, Type, Union

from .base import AbstractMemoryCollector, ProcessMemorySample
from ..system.cpu_manager import set_current_thread_affinity

logger = logging.getLogger(__name__)


class AsyncMemoryCollector:
    """
    Asynchronous wrapper for memory collectors that provides thread-safe operations.
    
    This class wraps existing memory collector implementations to make them
    compatible with AsyncIO operations while maintaining CPU affinity for
    optimal performance.
    """
    
    def __init__(
        self,
        collector_class: Type[AbstractMemoryCollector],
        process_pattern: str,
        monitoring_interval: float,
        cpu_core: Optional[int] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        **collector_kwargs
    ):
        """
        Initialize the async memory collector.
        
        Args:
            collector_class: The memory collector class to wrap
            process_pattern: Process pattern to monitor
            monitoring_interval: Monitoring interval in seconds
            cpu_core: CPU core to bind to (optional)
            executor: ThreadPoolExecutor to use (optional)
            **collector_kwargs: Additional arguments for the collector
        """
        self.collector_class = collector_class
        self.process_pattern = process_pattern
        self.monitoring_interval = monitoring_interval
        self.cpu_core = cpu_core
        self.executor = executor
        self.collector_kwargs = collector_kwargs
        
        # Thread-local storage for collector instances
        self._thread_local = threading.local()
        
        # State management
        self._is_running = False
        self._lock = threading.Lock()
        self._build_process_pid: Optional[int] = None
        
    @property
    def build_process_pid(self) -> Optional[int]:
        """Get the build process PID."""
        with self._lock:
            return self._build_process_pid
    
    @build_process_pid.setter
    def build_process_pid(self, value: Optional[int]) -> None:
        """Set the build process PID."""
        with self._lock:
            self._build_process_pid = value
    
    def _get_thread_collector(self) -> AbstractMemoryCollector:
        """
        Get or create a collector instance for the current thread.
        
        Returns:
            Memory collector instance for the current thread
        """
        if not hasattr(self._thread_local, 'collector'):
            # Initialize CPU affinity for this thread if specified
            if self.cpu_core is not None:
                success = set_current_thread_affinity([self.cpu_core])
                if success:
                    logger.debug(f"Thread {threading.current_thread().name} bound to core {self.cpu_core}")
            
            # Create collector instance for this thread
            self._thread_local.collector = self.collector_class(
                process_pattern=self.process_pattern,
                monitoring_interval=self.monitoring_interval,
                collector_cpu_core=self.cpu_core,
                **self.collector_kwargs
            )
            
            # Set build process PID if available
            if self._build_process_pid is not None:
                self._thread_local.collector.build_process_pid = self._build_process_pid
                
        return self._thread_local.collector
    
    async def start_async(self) -> None:
        """
        Start the memory collection asynchronously.
        """
        if self._is_running:
            logger.warning("Collector is already running")
            return
            
        try:
            loop = asyncio.get_event_loop()
            
            # Run collector start in thread pool
            if self.executor:
                await loop.run_in_executor(self.executor, self._start_sync)
            else:
                await loop.run_in_executor(None, self._start_sync)
                
            with self._lock:
                self._is_running = True
                
            logger.debug(f"AsyncMemoryCollector started for pattern: {self.process_pattern}")
            
        except Exception as e:
            logger.error(f"Failed to start async memory collector: {e}")
            raise
    
    async def stop_async(self, timeout: float = 10.0) -> bool:
        """
        Stop the memory collection asynchronously.
        
        Args:
            timeout: Maximum time to wait for stop
            
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self._is_running:
            logger.warning("Collector is not running")
            return True
            
        try:
            loop = asyncio.get_event_loop()
            
            # Run collector stop in thread pool
            if self.executor:
                success = await loop.run_in_executor(
                    self.executor, 
                    lambda: self._stop_sync(timeout)
                )
            else:
                success = await loop.run_in_executor(
                    None, 
                    lambda: self._stop_sync(timeout)
                )
                
            with self._lock:
                self._is_running = False
                
            logger.debug(f"AsyncMemoryCollector stopped for pattern: {self.process_pattern}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to stop async memory collector: {e}")
            return False
    
    async def read_samples_async(self) -> List[ProcessMemorySample]:
        """
        Read memory samples asynchronously.
        
        Returns:
            List of ProcessMemorySample objects
        """
        if not self._is_running:
            logger.warning("Collector is not running")
            return []
            
        try:
            loop = asyncio.get_event_loop()
            
            # Run sample reading in thread pool
            if self.executor:
                samples = await loop.run_in_executor(self.executor, self._read_samples_sync)
            else:
                samples = await loop.run_in_executor(None, self._read_samples_sync)
                
            return samples
            
        except Exception as e:
            logger.error(f"Failed to read samples asynchronously: {e}")
            return []
    
    def _start_sync(self) -> None:
        """Start the collector synchronously (thread-safe)."""
        collector = self._get_thread_collector()
        collector.start()
    
    def _stop_sync(self, timeout: float) -> bool:
        """Stop the collector synchronously (thread-safe)."""
        if hasattr(self._thread_local, 'collector'):
            return self._thread_local.collector.stop(timeout)
        return True
    
    def _read_samples_sync(self) -> List[ProcessMemorySample]:
        """Read samples synchronously (thread-safe)."""
        collector = self._get_thread_collector()
        
        # Get one batch of samples
        for sample_batch in collector.read_samples():
            if sample_batch:
                return sample_batch
            break
            
        return []
    
    def get_metric_fields(self) -> List[str]:
        """
        Get the metric fields provided by this collector.
        
        Returns:
            List of metric field names
        """
        # Create a temporary collector to get metric fields
        temp_collector = self.collector_class(
            process_pattern=self.process_pattern,
            monitoring_interval=self.monitoring_interval,
            **self.collector_kwargs
        )
        return temp_collector.get_metric_fields()
    
    def get_primary_metric_field(self) -> str:
        """
        Get the primary metric field name.
        
        Returns:
            Primary metric field name
        """
        # Create a temporary collector to get primary metric field
        temp_collector = self.collector_class(
            process_pattern=self.process_pattern,
            monitoring_interval=self.monitoring_interval,
            **self.collector_kwargs
        )
        return temp_collector.get_primary_metric_field()
    
    def is_running(self) -> bool:
        """
        Check if the collector is running.
        
        Returns:
            True if running, False otherwise
        """
        with self._lock:
            return self._is_running
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_async()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_async()


class AsyncMemoryCollectorFactory:
    """
    Factory for creating async memory collectors.
    """
    
    @staticmethod
    def create_pss_psutil_collector(
        process_pattern: str,
        monitoring_interval: float,
        cpu_core: Optional[int] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        mode: str = 'full_scan',
        **kwargs
    ) -> AsyncMemoryCollector:
        """
        Create an async PSS psutil collector.
        
        Args:
            process_pattern: Process pattern to monitor
            monitoring_interval: Monitoring interval in seconds
            cpu_core: CPU core to bind to
            executor: ThreadPoolExecutor to use
            mode: Collection mode ('full_scan' or 'descendants_only')
            **kwargs: Additional collector arguments
            
        Returns:
            AsyncMemoryCollector instance
        """
        from .pss_psutil import PssPsutilCollector
        
        collector_kwargs = {
            'mode': mode,
            'taskset_available': True,  # Will be validated in the collector
            **kwargs
        }
        
        return AsyncMemoryCollector(
            collector_class=PssPsutilCollector,
            process_pattern=process_pattern,
            monitoring_interval=monitoring_interval,
            cpu_core=cpu_core,
            executor=executor,
            **collector_kwargs
        )
    
    @staticmethod
    def create_rss_pidstat_collector(
        process_pattern: str,
        monitoring_interval: float,
        cpu_core: Optional[int] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        pidstat_stderr_file: Optional[str] = None,
        **kwargs
    ) -> AsyncMemoryCollector:
        """
        Create an async RSS pidstat collector.
        
        Args:
            process_pattern: Process pattern to monitor
            monitoring_interval: Monitoring interval in seconds
            cpu_core: CPU core to bind to
            executor: ThreadPoolExecutor to use
            pidstat_stderr_file: File for pidstat stderr output
            **kwargs: Additional collector arguments
            
        Returns:
            AsyncMemoryCollector instance
        """
        from .rss_pidstat import RssPidstatCollector
        
        collector_kwargs = {
            'pidstat_stderr_file': pidstat_stderr_file,
            'taskset_available': True,  # Will be validated in the collector
            **kwargs
        }
        
        return AsyncMemoryCollector(
            collector_class=RssPidstatCollector,
            process_pattern=process_pattern,
            monitoring_interval=monitoring_interval,
            cpu_core=cpu_core,
            executor=executor,
            **collector_kwargs
        )
    
    @staticmethod
    def create_collector(
        collector_type: str,
        process_pattern: str,
        monitoring_interval: float,
        cpu_core: Optional[int] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        **kwargs
    ) -> AsyncMemoryCollector:
        """
        Create an async memory collector based on type.
        
        Args:
            collector_type: Type of collector ('pss_psutil' or 'rss_pidstat')
            process_pattern: Process pattern to monitor
            monitoring_interval: Monitoring interval in seconds
            cpu_core: CPU core to bind to
            executor: ThreadPoolExecutor to use
            **kwargs: Additional collector arguments
            
        Returns:
            AsyncMemoryCollector instance
            
        Raises:
            ValueError: If collector type is unknown
        """
        if collector_type == 'pss_psutil':
            return AsyncMemoryCollectorFactory.create_pss_psutil_collector(
                process_pattern=process_pattern,
                monitoring_interval=monitoring_interval,
                cpu_core=cpu_core,
                executor=executor,
                **kwargs
            )
        elif collector_type == 'rss_pidstat':
            return AsyncMemoryCollectorFactory.create_rss_pidstat_collector(
                process_pattern=process_pattern,
                monitoring_interval=monitoring_interval,
                cpu_core=cpu_core,
                executor=executor,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown collector type: {collector_type}")


# Convenience functions for backward compatibility
async def create_async_collector(
    collector_type: str,
    process_pattern: str,
    monitoring_interval: float,
    cpu_core: Optional[int] = None,
    executor: Optional[ThreadPoolExecutor] = None,
    **kwargs
) -> AsyncMemoryCollector:
    """
    Create and start an async memory collector.
    
    Args:
        collector_type: Type of collector ('pss_psutil' or 'rss_pidstat')
        process_pattern: Process pattern to monitor
        monitoring_interval: Monitoring interval in seconds
        cpu_core: CPU core to bind to
        executor: ThreadPoolExecutor to use
        **kwargs: Additional collector arguments
        
    Returns:
        Started AsyncMemoryCollector instance
    """
    collector = AsyncMemoryCollectorFactory.create_collector(
        collector_type=collector_type,
        process_pattern=process_pattern,
        monitoring_interval=monitoring_interval,
        cpu_core=cpu_core,
        executor=executor,
        **kwargs
    )
    
    await collector.start_async()
    return collector
