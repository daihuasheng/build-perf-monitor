"""
Unit tests for DataStorageManager.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import polars as pl

from src.mymonitor.storage.data_manager import DataStorageManager
from src.mymonitor.models.results import MonitoringResults
from src.mymonitor.config.storage_config import StorageConfig


class MockRunContext:
    """Mock run context for testing."""
    
    def __init__(self):
        self.project_name = "test_project"
        self.project_dir = "/tmp/test"
        self.process_pattern = "gcc|ld"
        self.actual_build_command = "make -j4"
        self.parallelism_level = 4
        self.monitoring_interval = 0.05
        self.collector_type = "pss_psutil"
        self.current_timestamp_str = "20230416_123456"
        self.taskset_available = True
        self.build_cores_target_str = "0-3"
        self.monitor_script_pinned_to_core_info = "core 4"
        self.monitor_core_id = 4


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    with patch("src.mymonitor.storage.data_manager.get_config") as mock_get_config:
        mock_config = MagicMock()
        mock_config.monitor.storage = StorageConfig(
            format="parquet",
            compression="snappy",
            generate_legacy_formats=False
        )
        mock_get_config.return_value = mock_config
        yield mock_get_config


@pytest.fixture
def test_results():
    """Create test monitoring results."""
    test_samples = [
        {
            "timestamp": 1650123456.789,
            "pid": 12345,
            "process_name": "gcc",
            "rss_kb": 102400,
            "vms_kb": 204800,
            "pss_kb": 98304,
            "category": "compiler"
        },
        {
            "timestamp": 1650123457.789,
            "pid": 12346,
            "process_name": "ld",
            "rss_kb": 51200,
            "vms_kb": 102400,
            "pss_kb": 49152,
            "category": "linker"
        }
    ]
    
    return MonitoringResults(
        all_samples_data=test_samples,
        category_stats={
            "compiler": {"peak_sum_kb": 98304, "process_count": 1, "average_peak_kb": 98304},
            "linker": {"peak_sum_kb": 49152, "process_count": 1, "average_peak_kb": 49152}
        },
        peak_overall_memory_kb=147456,
        peak_overall_memory_epoch=1650123457,
        category_peak_sum={"compiler": 98304, "linker": 49152},
        category_pid_set={"compiler": {"12345"}, "linker": {"12346"}}
    )


class TestDataStorageManager:
    """Test cases for DataStorageManager class."""
    
    def test_initialization(self, mock_config):
        """Test DataStorageManager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataStorageManager(Path(tmpdir))
            
            assert manager.storage_format == "parquet"
            assert manager.compression == "snappy"
            assert manager.generate_legacy is False
            assert manager.output_dir == Path(tmpdir)
    
    def test_save_monitoring_results(self, mock_config, test_results):
        """Test saving monitoring results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataStorageManager(Path(tmpdir))
            run_context = MockRunContext()
            
            # Save results
            manager.save_monitoring_results(test_results, run_context)
            
            # Check files were created
            assert (Path(tmpdir) / "memory_samples.parquet").exists()
            assert (Path(tmpdir) / "metadata.json").exists()
            assert (Path(tmpdir) / "analysis_results.json").exists()
            assert (Path(tmpdir) / "summary.log").exists()
    
    def test_save_memory_samples(self, mock_config, test_results):
        """Test saving memory samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataStorageManager(Path(tmpdir))
            
            # Save memory samples
            manager._save_memory_samples(test_results.all_samples_data)
            
            # Check file was created
            assert (Path(tmpdir) / "memory_samples.parquet").exists()
            
            # Load and verify data
            df = pl.read_parquet(Path(tmpdir) / "memory_samples.parquet")
            assert len(df) == 2
            assert df["pid"].to_list() == [12345, 12346]
            assert df["process_name"].to_list() == ["gcc", "ld"]
    
    def test_save_metadata(self, mock_config):
        """Test saving metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataStorageManager(Path(tmpdir))
            run_context = MockRunContext()
            
            # Save metadata
            manager._save_metadata(run_context)
            
            # Check file was created
            assert (Path(tmpdir) / "metadata.json").exists()
            
            # Load and verify data
            with open(Path(tmpdir) / "metadata.json", "r") as f:
                import json
                data = json.load(f)
                
                assert data["project_name"] == "test_project"
                assert data["parallelism_level"] == 4
                assert data["collector_type"] == "pss_psutil"
    
    def test_save_analysis_results(self, mock_config, test_results):
        """Test saving analysis results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataStorageManager(Path(tmpdir))
            
            # Save analysis results
            manager._save_analysis_results(test_results)
            
            # Check file was created
            assert (Path(tmpdir) / "analysis_results.json").exists()
            
            # Load and verify data
            with open(Path(tmpdir) / "analysis_results.json", "r") as f:
                import json
                data = json.load(f)
                
                assert data["peak_overall_memory_kb"] == 147456
                assert data["category_peak_sum"]["compiler"] == 98304
                assert data["category_peak_sum"]["linker"] == 49152
    
    def test_save_summary_log(self, mock_config, test_results):
        """Test saving summary log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataStorageManager(Path(tmpdir))
            run_context = MockRunContext()
            
            # Save summary log
            manager._save_summary_log(test_results, run_context)
            
            # Check file was created
            assert (Path(tmpdir) / "summary.log").exists()
            
            # Load and verify content
            with open(Path(tmpdir) / "summary.log", "r") as f:
                content = f.read()
                
                assert "Build Monitoring Summary" in content
                assert "Project: test_project" in content
                assert "Parallelism: -j4" in content
                assert "Peak Overall Memory" in content
                assert "compiler" in content
                assert "linker" in content
    
    def test_load_memory_samples(self, mock_config, test_results):
        """Test loading memory samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataStorageManager(Path(tmpdir))
            
            # Save memory samples
            manager._save_memory_samples(test_results.all_samples_data)
            
            # Load memory samples
            df = manager.load_memory_samples()
            
            # Verify data
            assert len(df) == 2
            assert df["pid"].to_list() == [12345, 12346]
            assert df["process_name"].to_list() == ["gcc", "ld"]
    
    def test_load_memory_samples_with_columns(self, mock_config, test_results):
        """Test loading memory samples with specific columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataStorageManager(Path(tmpdir))
            
            # Save memory samples
            manager._save_memory_samples(test_results.all_samples_data)
            
            # Load specific columns
            df = manager.load_memory_samples(columns=["pid", "process_name"])
            
            # Verify data
            assert len(df) == 2
            assert df.columns == ["pid", "process_name"]
            assert "rss_kb" not in df.columns
            assert df["pid"].to_list() == [12345, 12346]
            assert df["process_name"].to_list() == ["gcc", "ld"]
    
    def test_load_memory_samples_file_not_found(self, mock_config):
        """Test loading memory samples when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataStorageManager(Path(tmpdir))
            
            # Attempt to load non-existent file
            with pytest.raises(FileNotFoundError):
                manager.load_memory_samples()
    
    def test_get_storage_info(self, mock_config, test_results):
        """Test getting storage information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataStorageManager(Path(tmpdir))
            
            # Save memory samples
            manager._save_memory_samples(test_results.all_samples_data)
            
            # Get storage info
            info = manager.get_storage_info()
            
            # Verify info
            assert info["storage_format"] == "parquet"
            assert info["compression"] == "snappy"
            assert info["output_dir"] == str(Path(tmpdir))
            assert "memory_samples.parquet" in info["files"]
            assert info["files"]["memory_samples.parquet"]["exists"] is True
            assert info["files"]["memory_samples.parquet"]["size_bytes"] > 0
    
    def test_legacy_format_generation(self, mock_config, test_results):
        """Test generation of legacy formats."""
        # Override mock config to enable legacy formats
        mock_config.return_value.monitor.storage = StorageConfig(
            format="parquet",
            compression="snappy",
            generate_legacy_formats=True
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataStorageManager(Path(tmpdir))
            
            # Verify legacy format flag is set
            assert manager.generate_legacy is True
            
            # Save memory samples
            manager._save_memory_samples(test_results.all_samples_data)
            
            # Check both formats were created
            assert (Path(tmpdir) / "memory_samples.parquet").exists()
            assert (Path(tmpdir) / "memory_samples.csv").exists()
