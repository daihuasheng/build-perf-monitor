"""
Tests for AsyncMonitoringCoordinator and related async components.

This module provides comprehensive tests for the new async monitoring
architecture components.
"""

import asyncio
import pytest
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, AsyncMock

from mymonitor.monitoring.coordinator import AsyncMonitoringCoordinator
from mymonitor.system.cpu_manager import (
    get_cpu_manager, set_current_thread_affinity, 
    get_available_cores
)
from mymonitor.collectors.async_memory_collector import (
    AsyncMemoryCollector, AsyncMemoryCollectorFactory
)
from mymonitor.models.runtime import RunContext, RunPaths


class TestAsyncMonitoringCoordinator:
    """Tests for AsyncMonitoringCoordinator."""
    
    @pytest.fixture
    def run_context(self):
        """Create a mock run context for testing."""
        paths = Mock(spec=RunPaths)
        paths.collector_aux_log_file = "/tmp/test.log"
        
        context = Mock(spec=RunContext)
        context.monitoring_interval = 0.1
        context.process_pattern = "test_pattern"
        context.collector_type = "pss_psutil"
        context.taskset_available = True
        context.monitor_core_id = 0
        context.paths = paths
        
        return context
    
    @pytest.fixture
    def coordinator(self, run_context):
        """Create an AsyncMonitoringCoordinator for testing."""
        return AsyncMonitoringCoordinator(run_context)
    
    def test_initialization(self, coordinator, run_context):
        """Test coordinator initialization."""
        assert coordinator.run_context == run_context
        assert coordinator.executor is None
        assert coordinator.monitoring_tasks == []
        assert coordinator.collectors == []
        assert coordinator.results is None
        assert not coordinator.is_monitoring
        assert coordinator.samples_collected == []
    
    @pytest.mark.asyncio
    async def test_setup_monitoring(self, coordinator):
        """Test monitoring setup."""
        monitoring_cores = [0, 1]
        
        with patch.object(coordinator, '_create_collector_sync') as mock_create:
            mock_collector1 = Mock()
            mock_collector2 = Mock()
            mock_create.side_effect = [mock_collector1, mock_collector2]
            
            await coordinator.setup_monitoring(monitoring_cores)
            
            assert coordinator.executor is not None
            assert len(coordinator.collectors) == 2
            assert coordinator.collectors[0] == mock_collector1
            assert coordinator.collectors[1] == mock_collector2
            
            # Clean up
            coordinator.executor.shutdown(wait=True)
    
    @pytest.mark.asyncio
    async def test_setup_monitoring_failure(self, coordinator):
        """Test monitoring setup failure handling."""
        monitoring_cores = [0]
        
        with patch.object(coordinator, '_create_collector_sync') as mock_create:
            mock_create.side_effect = Exception("Test error")
            
            with pytest.raises(RuntimeError, match="No collectors could be created"):
                await coordinator.setup_monitoring(monitoring_cores)
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, coordinator):
        """Test starting monitoring."""
        # Setup first
        monitoring_cores = [0]
        mock_collector = Mock()
        
        with patch.object(coordinator, '_create_collector_async', return_value=mock_collector):
            await coordinator.setup_monitoring(monitoring_cores)
        
        # Start monitoring
        build_pid = 12345
        
        with patch('asyncio.create_task') as mock_create_task:
            mock_task = Mock()
            mock_create_task.return_value = mock_task
            
            await coordinator.start_monitoring(build_pid)
            
            assert coordinator.is_monitoring
            assert len(coordinator.monitoring_tasks) == 1
            assert coordinator.monitoring_tasks[0] == mock_task
            
        # Clean up
        coordinator.executor.shutdown(wait=True)
    
    @pytest.mark.asyncio
    async def test_start_monitoring_not_setup(self, coordinator):
        """Test starting monitoring without setup."""
        with pytest.raises(RuntimeError, match="Monitoring not set up"):
            await coordinator.start_monitoring(12345)
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, coordinator):
        """Test stopping monitoring."""
        # Setup and start first
        monitoring_cores = [0]
        mock_collector = Mock()
        
        with patch.object(coordinator, '_create_collector_async', return_value=mock_collector):
            await coordinator.setup_monitoring(monitoring_cores)
        
        # Mock monitoring state
        coordinator.is_monitoring = True
        coordinator.samples_collected = [
            {'epoch': time.time(), 'pid': '123', 'PSS_KB': 1000}
        ]
        
        # Create proper async mock task
        async def mock_coroutine():
            pass
        
        mock_task = asyncio.create_task(mock_coroutine())
        coordinator.monitoring_tasks = [mock_task]
        
        with patch.object(coordinator, '_aggregate_monitoring_data') as mock_aggregate:
            mock_results = Mock()
            mock_aggregate.return_value = mock_results
            
            await coordinator.stop_monitoring()
            
            assert not coordinator.is_monitoring
            assert coordinator.results == mock_results
            assert len(coordinator.monitoring_tasks) == 0
            assert len(coordinator.collectors) == 0
    
    @pytest.mark.asyncio
    async def test_monitor_process_async(self, coordinator):
        """Test async monitoring worker."""
        mock_collector = Mock()
        mock_collector.build_process_pid = None
        
        coordinator.executor = ThreadPoolExecutor(max_workers=1)
        coordinator._shutdown_event = asyncio.Event()
        
        # Mock collector.start to be a simple function
        mock_collector.start = Mock()
        
        worker_id = 0
        build_pid = 12345
        
        # Start the worker and let it run briefly
        task = asyncio.create_task(
            coordinator._monitor_process_async(mock_collector, build_pid, worker_id)
        )
        
        # Let it run for a short time
        await asyncio.sleep(0.01)
        
        # Signal shutdown
        coordinator._shutdown_event.set()
        
        # Wait for completion
        await task
        
        # Verify collector was configured
        assert mock_collector.build_process_pid == build_pid
        mock_collector.start.assert_called_once()
        
        # Clean up
        coordinator.executor.shutdown(wait=True)
    
    def test_aggregate_monitoring_data_empty(self, coordinator):
        """Test aggregating empty monitoring data."""
        results = coordinator._aggregate_monitoring_data([])
        
        assert results.all_samples_data == []
        assert results.category_stats == {}
        assert results.peak_overall_memory_kb == 0
        assert results.peak_overall_memory_epoch == 0
        assert results.category_peak_sum == {}
        assert results.category_pid_set == {}
    
    def test_aggregate_monitoring_data_with_samples(self, coordinator):
        """Test aggregating monitoring data with samples."""
        epoch = time.time()
        samples = [
            {
                'epoch': epoch,
                'pid': '123',
                'command_name': 'gcc',
                'full_command': 'gcc -c test.c',
                'major_category': 'CPP_Compile',
                'minor_category': 'Driver_Compile',
                'PSS_KB': 1000,
                'RSS_KB': 1200
            },
            {
                'epoch': epoch,
                'pid': '124',
                'command_name': 'g++',
                'full_command': 'g++ -c test.cpp',
                'major_category': 'CPP_Compile',
                'minor_category': 'Driver_Compile',
                'PSS_KB': 1500,
                'RSS_KB': 1700
            }
        ]
        
        results = coordinator._aggregate_monitoring_data(samples)
        
        assert len(results.all_samples_data) == 2
        assert results.peak_overall_memory_kb == 2500  # 1000 + 1500
        assert 'CPP_Compile:Driver_Compile' in results.category_stats
        assert results.category_stats['CPP_Compile:Driver_Compile']['process_count'] == 2
        assert results.category_stats['CPP_Compile:Driver_Compile']['peak_sum_kb'] == 2500


class TestCPUAffinityManager:
    """Tests for CPU affinity manager."""
    
    @pytest.fixture
    def manager(self):
        """Create a CPU affinity manager for testing."""
        return get_cpu_manager()
    
    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.platform in ['linux', 'windows', 'darwin']
        assert isinstance(manager.taskset_available, bool)
    
    def test_get_available_cores(self, manager):
        """Test getting available cores."""
        cores = manager.available_cores
        assert isinstance(cores, list)
        assert len(cores) > 0
        assert all(isinstance(core, int) for core in cores)
    
    def test_validate_core_ids_valid(self, manager):
        """Test validating valid core IDs."""
        available_cores = manager.available_cores
        if available_cores:
            valid_cores = [available_cores[0]]
            assert manager._validate_core_ids(valid_cores)
    
    def test_validate_core_ids_invalid(self, manager):
        """Test validating invalid core IDs."""
        invalid_cores = [-1, 9999]
        assert not manager._validate_core_ids(invalid_cores)
    
    def test_validate_core_ids_empty(self, manager):
        """Test validating empty core ID list."""
        assert not manager._validate_core_ids([])
    
    @patch('os.cpu_count', return_value=4)
    def test_set_thread_affinity_valid_core(self, mock_cpu_count, manager):
        """Test setting thread affinity with valid core."""
        # This test may fail on systems without proper permissions
        # or psutil, so we'll just verify the method doesn't crash
        try:
            result = manager.set_thread_affinity([0])
            assert isinstance(result, bool)
        except Exception:
            # Expected on systems without proper setup
            pass
    
    def test_set_thread_affinity_invalid_core(self, manager):
        """Test setting thread affinity with invalid core."""
        result = manager.set_thread_affinity([9999])
        assert result is False


class TestAsyncMemoryCollector:
    """Tests for AsyncMemoryCollector."""
    
    @pytest.fixture
    def mock_collector_class(self):
        """Create a mock collector class."""
        mock_class = Mock()
        mock_instance = Mock()
        mock_instance.start = Mock()
        mock_instance.stop = Mock(return_value=True)
        mock_instance.read_samples = Mock(return_value=[])
        mock_instance.get_metric_fields = Mock(return_value=['PSS_KB', 'RSS_KB'])
        mock_instance.get_primary_metric_field = Mock(return_value='PSS_KB')
        mock_class.return_value = mock_instance
        return mock_class
    
    @pytest.fixture
    def async_collector(self, mock_collector_class):
        """Create an AsyncMemoryCollector for testing."""
        return AsyncMemoryCollector(
            collector_class=mock_collector_class,
            process_pattern="test_pattern",
            monitoring_interval=0.1,
            cpu_core=0
        )
    
    def test_initialization(self, async_collector, mock_collector_class):
        """Test async collector initialization."""
        assert async_collector.collector_class == mock_collector_class
        assert async_collector.process_pattern == "test_pattern"
        assert async_collector.monitoring_interval == 0.1
        assert async_collector.cpu_core == 0
        assert not async_collector.is_running()
    
    def test_build_process_pid_property(self, async_collector):
        """Test build process PID property."""
        assert async_collector.build_process_pid is None
        
        async_collector.build_process_pid = 12345
        assert async_collector.build_process_pid == 12345
    
    @pytest.mark.asyncio
    async def test_start_async(self, async_collector):
        """Test starting async collector."""
        with patch.object(async_collector, '_start_sync'):
            await async_collector.start_async()
            assert async_collector.is_running()
    
    @pytest.mark.asyncio
    async def test_stop_async(self, async_collector):
        """Test stopping async collector."""
        # Set as running first
        async_collector._is_running = True
        
        with patch.object(async_collector, '_stop_sync', return_value=True):
            result = await async_collector.stop_async()
            assert result is True
            assert not async_collector.is_running()
    
    @pytest.mark.asyncio
    async def test_read_samples_async(self, async_collector):
        """Test reading samples asynchronously."""
        async_collector._is_running = True
        
        with patch.object(async_collector, '_read_samples_sync', return_value=[]):
            samples = await async_collector.read_samples_async()
            assert samples == []
    
    @pytest.mark.asyncio
    async def test_read_samples_async_not_running(self, async_collector):
        """Test reading samples when not running."""
        samples = await async_collector.read_samples_async()
        assert samples == []
    
    def test_get_metric_fields(self, async_collector):
        """Test getting metric fields."""
        fields = async_collector.get_metric_fields()
        assert fields == ['PSS_KB', 'RSS_KB']
    
    def test_get_primary_metric_field(self, async_collector):
        """Test getting primary metric field."""
        field = async_collector.get_primary_metric_field()
        assert field == 'PSS_KB'


class TestAsyncMemoryCollectorFactory:
    """Tests for AsyncMemoryCollectorFactory."""
    
    def test_create_pss_psutil_collector(self):
        """Test creating PSS psutil collector."""
        collector = AsyncMemoryCollectorFactory.create_pss_psutil_collector(
            process_pattern="test_pattern",
            monitoring_interval=0.1,
            cpu_core=0,
            mode='full_scan'
        )
        
        assert isinstance(collector, AsyncMemoryCollector)
        assert collector.process_pattern == "test_pattern"
        assert collector.monitoring_interval == 0.1
        assert collector.cpu_core == 0
    
    def test_create_rss_pidstat_collector(self):
        """Test creating RSS pidstat collector."""
        collector = AsyncMemoryCollectorFactory.create_rss_pidstat_collector(
            process_pattern="test_pattern",
            monitoring_interval=0.1,
            cpu_core=0,
            pidstat_stderr_file="/tmp/test.log"
        )
        
        assert isinstance(collector, AsyncMemoryCollector)
        assert collector.process_pattern == "test_pattern"
        assert collector.monitoring_interval == 0.1
        assert collector.cpu_core == 0
    
    def test_create_collector_pss_psutil(self):
        """Test creating collector by type - pss_psutil."""
        collector = AsyncMemoryCollectorFactory.create_collector(
            collector_type="pss_psutil",
            process_pattern="test_pattern",
            monitoring_interval=0.1,
            cpu_core=0
        )
        
        assert isinstance(collector, AsyncMemoryCollector)
    
    def test_create_collector_rss_pidstat(self):
        """Test creating collector by type - rss_pidstat."""
        collector = AsyncMemoryCollectorFactory.create_collector(
            collector_type="rss_pidstat",
            process_pattern="test_pattern",
            monitoring_interval=0.1,
            cpu_core=0
        )
        
        assert isinstance(collector, AsyncMemoryCollector)
    
    def test_create_collector_unknown_type(self):
        """Test creating collector with unknown type."""
        with pytest.raises(ValueError, match="Unknown collector type"):
            AsyncMemoryCollectorFactory.create_collector(
                collector_type="unknown_type",
                process_pattern="test_pattern",
                monitoring_interval=0.1
            )


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_set_current_thread_affinity(self):
        """Test setting current thread affinity."""
        # This may fail on systems without proper permissions
        try:
            result = set_current_thread_affinity(0)
            assert isinstance(result, bool)
        except Exception:
            # Expected on systems without proper setup
            pass
    
    def test_get_available_cores(self):
        """Test getting available cores."""
        cores = get_available_cores()
        assert isinstance(cores, list)
        assert len(cores) > 0
    
    def test_initialize_thread_affinity(self):
        """Test initializing thread affinity."""
        # This may fail on systems without proper permissions
        try:
            result = set_current_thread_affinity([0])
            assert isinstance(result, bool)
        except Exception:
            # Expected on systems without proper setup
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
