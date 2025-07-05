"""
Tests for the monitoring coordinator and memory aggregation logic.

This module tests the core monitoring functionality, particularly the
memory aggregation logic that was recently fixed to properly handle
both total peak memory and individual process peak memory.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock
from mymonitor.monitoring.coordinator import MonitoringCoordinator
from mymonitor.models.runtime import RunContext


def create_mock_run_context():
    """Create a mock RunContext for testing."""
    from mymonitor.models.runtime import RunPaths
    
    mock_paths = RunPaths(
        output_parquet_file=Path("/tmp/test.parquet"),
        output_summary_log_file=Path("/tmp/test_summary.log"),
        collector_aux_log_file=Path("/tmp/test_aux.log")
    )
    
    return RunContext(
        project_name="TestProject",
        project_dir=Path("/tmp"),
        process_pattern="gcc|clang",
        actual_build_command="make -j4",
        parallelism_level=4,
        monitoring_interval=0.1,
        collector_type="pss_psutil",
        current_timestamp_str="20250705_120000",
        taskset_available=True,
        build_cores_target_str="1-3",
        monitor_script_pinned_to_core_info="core 0",
        monitor_core_id=0,
        paths=mock_paths
    )


class TestMonitoringCoordinator:
    """Test cases for the MonitoringCoordinator class."""
    
    def test_aggregate_monitoring_data_basic(self):
        """Test basic aggregation of monitoring data."""
        coordinator = MonitoringCoordinator(create_mock_run_context())
        
        # Create sample monitoring data
        sample_data = [
            {'epoch': 1000, 'pid': 'proc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 100},
            {'epoch': 1000, 'pid': 'proc2', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 200},
            {'epoch': 2000, 'pid': 'proc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 150},
            {'epoch': 2000, 'pid': 'proc2', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 250},
            {'epoch': 3000, 'pid': 'proc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 80},
            {'epoch': 3000, 'pid': 'proc2', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 180}
        ]
        
        result = coordinator._aggregate_monitoring_data(sample_data)
        
        # Should have category stats
        assert result.category_stats is not None
        category_key = 'CPP_Compile:GCC'
        assert category_key in result.category_stats
        
        category_stats = result.category_stats[category_key]
        
        # Check total peak memory (should be the peak sum across all processes)
        # At epoch 2000: proc1=150 + proc2=250 = 400
        assert category_stats['peak_memory_kb'] == 400
        
        # Check individual peak memory (should be the max of any single process)
        # proc2 has the highest peak at 250
        assert category_stats['individual_peak_memory_kb'] == 250
    
    def test_aggregate_monitoring_data_multiple_categories(self):
        """Test aggregation with multiple categories."""
        coordinator = MonitoringCoordinator(create_mock_run_context())
        
        sample_data = [
            {'epoch': 1000, 'pid': 'gcc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 100},
            {'epoch': 1000, 'pid': 'ninja1', 'major_category': 'BuildSystem', 'minor_category': 'Ninja', 'PSS_KB': 50},
            {'epoch': 2000, 'pid': 'gcc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 150},
            {'epoch': 2000, 'pid': 'ninja1', 'major_category': 'BuildSystem', 'minor_category': 'Ninja', 'PSS_KB': 80}
        ]
        
        result = coordinator._aggregate_monitoring_data(sample_data)
        
        # Should have two categories
        assert len(result.category_stats) == 2
        assert 'CPP_Compile:GCC' in result.category_stats
        assert 'BuildSystem:Ninja' in result.category_stats
        
        # Check GCC category
        gcc_stats = result.category_stats['CPP_Compile:GCC']
        assert gcc_stats['peak_memory_kb'] == 150  # gcc1's peak
        assert gcc_stats['individual_peak_memory_kb'] == 150
        
        # Check Ninja category
        ninja_stats = result.category_stats['BuildSystem:Ninja']
        assert ninja_stats['peak_memory_kb'] == 80  # ninja1's peak
        assert ninja_stats['individual_peak_memory_kb'] == 80
    
    def test_aggregate_monitoring_data_same_category_different_processes(self):
        """Test aggregation with same category but different processes."""
        coordinator = MonitoringCoordinator(create_mock_run_context())
        
        sample_data = [
            {'epoch': 1000, 'pid': 'gcc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 100},
            {'epoch': 1000, 'pid': 'gcc2', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 80},
            {'epoch': 2000, 'pid': 'gcc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 200},
            {'epoch': 2000, 'pid': 'gcc2', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 160},
            {'epoch': 3000, 'pid': 'gcc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 50},
            {'epoch': 3000, 'pid': 'gcc2', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 40}
        ]
        
        result = coordinator._aggregate_monitoring_data(sample_data)
        
        category_stats = result.category_stats['CPP_Compile:GCC']
        
        # Total peak should be the maximum sum at any epoch
        # At epoch 2000: gcc1=200 + gcc2=160 = 360
        assert category_stats['peak_memory_kb'] == 360
        
        # Individual peak should be the max of any single process (gcc1's 200)
        assert category_stats['individual_peak_memory_kb'] == 200
    
    def test_aggregate_monitoring_data_empty_list(self):
        """Test aggregation with empty sample list."""
        coordinator = MonitoringCoordinator(create_mock_run_context())
        
        result = coordinator._aggregate_monitoring_data([])
        
        # Should return empty results
        assert len(result.all_samples_data) == 0
        assert len(result.category_stats) == 0
        assert result.peak_overall_memory_kb == 0
    
    def test_aggregate_monitoring_data_single_process(self):
        """Test aggregation with single process."""
        coordinator = MonitoringCoordinator(create_mock_run_context())
        
        sample_data = [
            {'epoch': 1000, 'pid': 'proc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 100},
            {'epoch': 2000, 'pid': 'proc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 200},
            {'epoch': 3000, 'pid': 'proc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 150}
        ]
        
        result = coordinator._aggregate_monitoring_data(sample_data)
        
        category_stats = result.category_stats['CPP_Compile:GCC']
        
        # For single process, total peak and individual peak should be the same
        assert category_stats['peak_memory_kb'] == 200
        assert category_stats['individual_peak_memory_kb'] == 200
    
    def test_aggregate_monitoring_data_rss_metric(self):
        """Test aggregation with RSS metric instead of PSS."""
        coordinator = MonitoringCoordinator(create_mock_run_context())
        
        sample_data = [
            {'epoch': 1000, 'pid': 'proc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'RSS_KB': 100},
            {'epoch': 2000, 'pid': 'proc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'RSS_KB': 200}
        ]
        
        result = coordinator._aggregate_monitoring_data(sample_data)
        
        category_stats = result.category_stats['CPP_Compile:GCC']
        assert category_stats['peak_memory_kb'] == 200
        assert category_stats['individual_peak_memory_kb'] == 200


class TestMonitoringCoordinatorIntegration:
    """Integration tests for monitoring coordinator with real-world scenarios."""
    
    def test_real_world_build_scenario(self):
        """Test with a realistic build scenario."""
        coordinator = MonitoringCoordinator(create_mock_run_context())
        
        # Simulate a build with multiple GCC processes and other tools
        sample_data = [
            {'epoch': 1000, 'pid': 'gcc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 50000},
            {'epoch': 1000, 'pid': 'gcc2', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 30000},
            {'epoch': 1000, 'pid': 'ninja1', 'major_category': 'BuildSystem', 'minor_category': 'Ninja', 'PSS_KB': 10000},
            {'epoch': 2000, 'pid': 'gcc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 80000},
            {'epoch': 2000, 'pid': 'gcc2', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 60000},
            {'epoch': 2000, 'pid': 'ninja1', 'major_category': 'BuildSystem', 'minor_category': 'Ninja', 'PSS_KB': 15000},
            {'epoch': 3000, 'pid': 'gcc1', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 20000},
            {'epoch': 3000, 'pid': 'gcc2', 'major_category': 'CPP_Compile', 'minor_category': 'GCC', 'PSS_KB': 10000},
            {'epoch': 3000, 'pid': 'ninja1', 'major_category': 'BuildSystem', 'minor_category': 'Ninja', 'PSS_KB': 8000}
        ]
        
        result = coordinator._aggregate_monitoring_data(sample_data)
        
        # Should have both categories
        assert 'CPP_Compile:GCC' in result.category_stats
        assert 'BuildSystem:Ninja' in result.category_stats
        
        gcc_stats = result.category_stats['CPP_Compile:GCC']
        ninja_stats = result.category_stats['BuildSystem:Ninja']
        
        # GCC: peak total at epoch 2000: 80000 + 60000 = 140000
        # GCC: individual peak is gcc1's 80000
        assert gcc_stats['peak_memory_kb'] == 140000
        assert gcc_stats['individual_peak_memory_kb'] == 80000
        
        # Ninja: peak at epoch 2000: 15000
        assert ninja_stats['peak_memory_kb'] == 15000
        assert ninja_stats['individual_peak_memory_kb'] == 15000
