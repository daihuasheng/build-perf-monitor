"""
Unit tests for thread pool management functionality.

Tests the thread pool configuration, lifecycle management,
CPU affinity, and task execution.
"""

import pytest
import threading
import time
from unittest.mock import patch, Mock, MagicMock
from concurrent.futures import Future

from mymonitor.executor.thread_pool import (
    ThreadPoolConfig,
    ManagedThreadPoolExecutor,
    ThreadPoolManager,
    get_thread_pool_manager,
    initialize_global_thread_pools,
)


@pytest.mark.unit
class TestThreadPoolConfig:
    """Test cases for ThreadPoolConfig."""

    def test_thread_pool_config_defaults(self):
        """Test ThreadPoolConfig default values."""
        config = ThreadPoolConfig()

        assert config.max_workers == 4
        assert config.thread_name_prefix == "MonitorWorker"
        assert config.enable_cpu_affinity is True
        assert config.dedicated_cores is None
        assert config.monitor_resource_usage is True
        assert config.shutdown_timeout == 10.0

    def test_thread_pool_config_custom_values(self):
        """Test ThreadPoolConfig with custom values."""
        config = ThreadPoolConfig(
            max_workers=8,
            thread_name_prefix="CustomWorker",
            enable_cpu_affinity=False,
            dedicated_cores=[0, 1, 2, 3],
            monitor_resource_usage=False,
            shutdown_timeout=5.0,
        )

        assert config.max_workers == 8
        assert config.thread_name_prefix == "CustomWorker"
        assert config.enable_cpu_affinity is False
        assert config.dedicated_cores == [0, 1, 2, 3]
        assert config.monitor_resource_usage is False
        assert config.shutdown_timeout == 5.0


@pytest.mark.unit
class TestManagedThreadPoolExecutor:
    """Test cases for ManagedThreadPoolExecutor."""

    def test_managed_thread_pool_initialization(self):
        """Test ManagedThreadPoolExecutor initialization."""
        config = ThreadPoolConfig(max_workers=2)
        executor = ManagedThreadPoolExecutor(config)

        assert executor.config == config
        assert executor.executor is None
        assert executor.is_shutdown is False
        assert executor.stats["tasks_submitted"] == 0

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_managed_thread_pool_start(self, mock_get_cores):
        """Test starting the managed thread pool."""
        mock_get_cores.return_value = [0, 1, 2, 3]

        config = ThreadPoolConfig(max_workers=2, enable_cpu_affinity=False)
        executor = ManagedThreadPoolExecutor(config)

        executor.start()

        assert executor.executor is not None
        assert executor.is_shutdown is False

        # Clean up
        executor.shutdown(wait=True)

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_managed_thread_pool_start_with_dedicated_cores(self, mock_get_cores):
        """Test starting with dedicated cores."""
        mock_get_cores.return_value = [0, 1, 2, 3]

        config = ThreadPoolConfig(
            max_workers=4,
            dedicated_cores=[0, 1],
            enable_cpu_affinity=False,  # Disable to avoid system calls
        )
        executor = ManagedThreadPoolExecutor(config)

        executor.start()

        assert executor.executor is not None

        # Clean up
        executor.shutdown(wait=True)

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_managed_thread_pool_start_invalid_cores(self, mock_get_cores, caplog):
        """Test starting with invalid dedicated cores."""
        mock_get_cores.return_value = [0, 1, 2, 3]

        config = ThreadPoolConfig(
            max_workers=2,
            dedicated_cores=[0, 1, 8, 9],  # 8, 9 are invalid
            enable_cpu_affinity=False,
        )
        executor = ManagedThreadPoolExecutor(config)

        executor.start()

        # Should log warning about invalid cores
        assert "Invalid cores specified" in caplog.text
        assert executor.config.dedicated_cores == [0, 1]  # Invalid cores removed

        # Clean up
        executor.shutdown(wait=True)

    def test_managed_thread_pool_start_already_started(self):
        """Test starting an already started thread pool."""
        config = ThreadPoolConfig(max_workers=1, enable_cpu_affinity=False)
        executor = ManagedThreadPoolExecutor(config)

        with patch(
            "mymonitor.executor.thread_pool.get_available_cores", return_value=[0, 1]
        ):
            executor.start()

            # Should raise RuntimeError on second start
            with pytest.raises(RuntimeError, match="already started"):
                executor.start()

            # Clean up
            executor.shutdown(wait=True)

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_managed_thread_pool_submit_task(self, mock_get_cores):
        """Test submitting tasks to the thread pool."""
        mock_get_cores.return_value = [0, 1, 2, 3]

        config = ThreadPoolConfig(max_workers=2, enable_cpu_affinity=False)
        executor = ManagedThreadPoolExecutor(config)
        executor.start()

        def test_function(x):
            return x * 2

        # Submit task
        future = executor.submit(test_function, 5)
        result = future.result(timeout=1.0)

        assert result == 10
        assert executor.stats["tasks_submitted"] == 1
        assert executor.stats["tasks_completed"] == 1

        # Clean up
        executor.shutdown(wait=True)

    def test_managed_thread_pool_submit_not_started(self):
        """Test submitting task to non-started thread pool."""
        config = ThreadPoolConfig(max_workers=1)
        executor = ManagedThreadPoolExecutor(config)

        with pytest.raises(RuntimeError, match="not started"):
            executor.submit(lambda: None)

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_managed_thread_pool_submit_after_shutdown(self, mock_get_cores):
        """Test submitting task after shutdown."""
        mock_get_cores.return_value = [0, 1]

        config = ThreadPoolConfig(max_workers=1, enable_cpu_affinity=False)
        executor = ManagedThreadPoolExecutor(config)
        executor.start()
        executor.shutdown(wait=True)

        with pytest.raises(RuntimeError, match="not started"):
            executor.submit(lambda: None)

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_managed_thread_pool_shutdown(self, mock_get_cores):
        """Test thread pool shutdown."""
        mock_get_cores.return_value = [0, 1]

        config = ThreadPoolConfig(max_workers=1, enable_cpu_affinity=False)
        executor = ManagedThreadPoolExecutor(config)
        executor.start()

        # Submit a task
        future = executor.submit(lambda: time.sleep(0.1))

        # Shutdown
        executor.shutdown(wait=True)

        assert executor.is_shutdown is True
        assert future.done()

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_managed_thread_pool_context_manager(self, mock_get_cores):
        """Test using ManagedThreadPoolExecutor as context manager."""
        mock_get_cores.return_value = [0, 1]

        config = ThreadPoolConfig(max_workers=1, enable_cpu_affinity=False)
        executor = ManagedThreadPoolExecutor(config)

        with executor:
            future = executor.submit(lambda: 42)
            result = future.result()
            assert result == 42

        # Should be shutdown after context exit
        assert executor.is_shutdown is True


@pytest.mark.unit
class TestThreadPoolManager:
    """Test cases for ThreadPoolManager."""

    def test_thread_pool_manager_singleton(self):
        """Test that ThreadPoolManager follows singleton pattern."""
        manager1 = get_thread_pool_manager()
        manager2 = get_thread_pool_manager()
        assert manager1 is manager2

    def test_thread_pool_manager_initialization(self):
        """Test ThreadPoolManager initialization."""
        manager = ThreadPoolManager()

        assert len(manager.pools) == 0
        assert manager.is_initialized is False
        assert isinstance(manager.default_config, ThreadPoolConfig)

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_thread_pool_manager_initialize_with_configs(self, mock_get_cores):
        """Test initializing ThreadPoolManager with custom configurations."""
        mock_get_cores.return_value = [0, 1, 2, 3]

        manager = ThreadPoolManager()

        configs = {
            "monitoring": ThreadPoolConfig(
                max_workers=2,
                thread_name_prefix="MonitorWorker",
                enable_cpu_affinity=False,
            ),
            "processing": ThreadPoolConfig(
                max_workers=4,
                thread_name_prefix="ProcessWorker",
                enable_cpu_affinity=False,
            ),
        }

        manager.initialize(configs)

        assert manager.is_initialized is True
        assert len(manager.pools) == 2
        assert "monitoring" in manager.pools
        assert "processing" in manager.pools

        # Clean up
        manager.shutdown_all()

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_thread_pool_manager_initialize_default(self, mock_get_cores):
        """Test initializing ThreadPoolManager with default configuration."""
        mock_get_cores.return_value = [0, 1, 2, 3]

        manager = ThreadPoolManager()
        manager.initialize()

        assert manager.is_initialized is True
        assert len(manager.pools) == 1
        assert "default" in manager.pools

        # Clean up
        manager.shutdown_all()

    def test_thread_pool_manager_initialize_already_initialized(self, caplog):
        """Test initializing already initialized ThreadPoolManager."""
        manager = ThreadPoolManager()
        manager.is_initialized = True

        manager.initialize()

        assert "already initialized" in caplog.text

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_thread_pool_manager_get_pool(self, mock_get_cores):
        """Test getting thread pool from manager."""
        mock_get_cores.return_value = [0, 1, 2, 3]

        manager = ThreadPoolManager()
        config = ThreadPoolConfig(max_workers=2, enable_cpu_affinity=False)
        manager.initialize({"test": config})

        pool = manager.get_pool("test")
        assert pool is not None
        assert isinstance(pool, ManagedThreadPoolExecutor)

        # Test non-existent pool
        assert manager.get_pool("nonexistent") is None

        # Clean up
        manager.shutdown_all()

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_thread_pool_manager_submit_monitoring_task(self, mock_get_cores):
        """Test submitting monitoring task through manager."""
        mock_get_cores.return_value = [0, 1, 2, 3]

        manager = ThreadPoolManager()
        config = ThreadPoolConfig(max_workers=1, enable_cpu_affinity=False)
        manager.initialize({"monitoring": config})

        def test_function(x):
            return x * 3

        future = manager.submit_monitoring_task(test_function, 7)
        result = future.result(timeout=1.0)

        assert result == 21

        # Clean up
        manager.shutdown_all()

    def test_thread_pool_manager_submit_task_invalid_pool(self):
        """Test submitting task to invalid pool."""
        manager = ThreadPoolManager()

        with pytest.raises(ValueError, match="Monitoring thread pool not initialized"):
            manager.submit_monitoring_task(lambda: None)


@pytest.mark.unit
class TestGlobalThreadPoolFunctions:
    """Test cases for global thread pool functions."""

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_initialize_global_thread_pools_with_config(self, mock_get_cores):
        """Test initializing global thread pools with custom config."""
        mock_get_cores.return_value = [0, 1, 2, 3]

        monitoring_config = ThreadPoolConfig(
            max_workers=3, thread_name_prefix="TestWorker", enable_cpu_affinity=False
        )

        initialize_global_thread_pools(monitoring_config=monitoring_config)

        manager = get_thread_pool_manager()
        assert manager.is_initialized is True

        pool = manager.get_pool("monitoring")
        assert pool is not None
        assert pool.config.max_workers == 3
        assert pool.config.thread_name_prefix == "TestWorker"

        # Clean up
        manager.shutdown_all()
        manager.is_initialized = False
        manager.pools.clear()

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_initialize_global_thread_pools_default(self, mock_get_cores):
        """Test initializing global thread pools with default config."""
        mock_get_cores.return_value = [0, 1, 2, 3]

        initialize_global_thread_pools()

        manager = get_thread_pool_manager()
        assert manager.is_initialized is True

        pool = manager.get_pool("monitoring")
        assert pool is not None
        assert pool.config.max_workers == 4
        assert pool.config.thread_name_prefix == "MonitorWorker"

        # Clean up
        manager.shutdown_all()
        manager.is_initialized = False
        manager.pools.clear()
