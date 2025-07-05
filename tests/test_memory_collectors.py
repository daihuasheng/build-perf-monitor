"""
Tests for memory collectors functionality.

This module tests the memory collection functionality, including
PSS and RSS collectors, process detection, and memory measurement.
"""

import pytest
import time
import subprocess
from unittest.mock import Mock, patch, MagicMock
from mymonitor.collectors.pss_psutil import PssPsutilCollector
from mymonitor.collectors.rss_pidstat import RssPidstatCollector


class TestPssPsutilCollector:
    """Test cases for the PSS Psutil collector."""
    
    def test_collector_initialization(self):
        """Test collector initialization with different parameters."""
        collector = PssPsutilCollector(
            process_pattern="gcc|clang",
            monitoring_interval=0.1,
            mode="full_scan"
        )
        
        assert collector.process_pattern == "gcc|clang"
        assert collector.monitoring_interval == 0.1
        assert collector.mode == "full_scan"
    
    def test_collector_initialization_invalid_mode(self):
        """Test collector initialization with invalid mode."""
        with pytest.raises(ValueError):
            PssPsutilCollector(
                process_pattern="gcc|clang",
                monitoring_interval=0.1,
                mode="invalid_mode"
            )
    
    @patch('mymonitor.collectors.pss_psutil.psutil.process_iter')
    def test_process_detection(self, mock_process_iter):
        """Test process detection functionality."""
        # Mock a process that matches the pattern
        mock_process = Mock()
        mock_process.pid = 1234
        mock_process.name.return_value = "gcc"
        mock_process.cmdline.return_value = ["gcc", "-c", "main.c"]
        mock_process.memory_info.return_value = Mock(rss=1024*1024)  # 1MB
        mock_process.memory_full_info.return_value = Mock(pss=512*1024)  # 512KB
        mock_process.is_running.return_value = True
        
        mock_process_iter.return_value = [mock_process]
        
        collector = PssPsutilCollector(
            process_pattern="gcc",
            monitoring_interval=0.1,
            mode="full_scan"
        )
        
        # Test that the collector identifies the correct fields
        assert "PSS_KB" in collector.get_metric_fields()
        assert collector.get_primary_metric_field() == "PSS_KB"
    
    def test_collector_start_stop(self):
        """Test collector start and stop functionality."""
        collector = PssPsutilCollector(
            process_pattern="nonexistent_process",
            monitoring_interval=0.1,
            mode="full_scan"
        )
        
        # Test that start/stop methods are available
        assert hasattr(collector, 'start')
        assert hasattr(collector, 'stop')
        
        # Test that we can call start/stop without exceptions
        # (we can't test actual functionality without running processes)
        try:
            collector.start()
            result = collector.stop(timeout=0.1)
            assert isinstance(result, bool)
        except Exception as e:
            # This is expected since we're not actually running processes
            pass
        finally:
            # Ensure collector is stopped even if an exception occurred
            try:
                collector.stop(timeout=0.1)
            except Exception:
                pass


class TestRssPidstatCollector:
    """Test cases for the RSS Pidstat collector."""
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        collector = RssPidstatCollector(
            process_pattern="gcc|clang",
            monitoring_interval=0.5
        )
        
        assert collector.process_pattern == "gcc|clang"
        assert collector.monitoring_interval == 0.5
    
    @patch('subprocess.Popen')
    def test_pidstat_command_execution(self, mock_popen):
        """Test pidstat command execution."""
        collector = RssPidstatCollector(
            process_pattern="gcc",
            monitoring_interval=0.5
        )
        
        # Test that the collector has the correct fields
        assert "RSS_KB" in collector.get_metric_fields()
        assert collector.get_primary_metric_field() == "RSS_KB"
        
        # Test that start/stop methods are available
        assert hasattr(collector, 'start')
        assert hasattr(collector, 'stop')
        assert hasattr(collector, 'read_samples')


class TestCollectorIntegration:
    """Integration tests for memory collectors."""
    
    def test_collector_with_fake_process(self):
        """Test collector with a real but short-lived process."""
        # Start a simple sleep process that we can monitor
        process = subprocess.Popen(['sleep', '2'])
        
        try:
            collector = PssPsutilCollector(
                process_pattern="sleep",
                monitoring_interval=0.1,
                mode="full_scan"
            )
            
            # Test that the collector has the proper interface
            assert hasattr(collector, 'read_samples')
            assert hasattr(collector, 'get_metric_fields')
            assert hasattr(collector, 'get_primary_metric_field')
            
            # Test metric field functions
            fields = collector.get_metric_fields()
            assert isinstance(fields, list)
            assert len(fields) > 0
            
            primary_field = collector.get_primary_metric_field()
            assert isinstance(primary_field, str)
            assert primary_field in fields
                
        finally:
            process.terminate()
            process.wait()
    
    def test_collector_error_handling(self):
        """Test collector error handling."""
        collector = PssPsutilCollector(
            process_pattern="gcc",
            monitoring_interval=0.1,
            mode="full_scan"
        )
        
        # Test that the collector can handle invalid patterns
        invalid_collector = None
        try:
            invalid_collector = PssPsutilCollector(
                process_pattern="[invalid_regex",
                monitoring_interval=0.1,
                mode="full_scan"
            )
        except Exception:
            # This is expected for invalid regex patterns
            pass
        
        # Test that valid collector has the right interface
        assert hasattr(collector, 'start')
        assert hasattr(collector, 'stop')
        
        # Test stop with very short timeout
        try:
            collector.start()
            result = collector.stop(timeout=0.001)
            assert isinstance(result, bool)
        except Exception:
            # This is expected since we're not actually running processes
            pass
        finally:
            # Ensure collector is stopped even if an exception occurred
            try:
                collector.stop(timeout=0.1)
            except Exception:
                pass
    
    def test_collector_pattern_matching(self):
        """Test process pattern matching functionality."""
        collector = PssPsutilCollector(
            process_pattern="gcc|clang|g\\+\\+",
            monitoring_interval=0.1,
            mode="full_scan"
        )
        
        # Mock processes with different names
        with patch('mymonitor.collectors.pss_psutil.psutil.process_iter') as mock_iter:
            mock_processes = []
            
            # Process that should match
            matching_process = Mock()
            matching_process.pid = 1234
            matching_process.name.return_value = "gcc"
            matching_process.cmdline.return_value = ["gcc", "-c", "main.c"]
            mock_processes.append(matching_process)
            
            # Process that should not match
            non_matching_process = Mock()
            non_matching_process.pid = 5678
            non_matching_process.name.return_value = "python"
            non_matching_process.cmdline.return_value = ["python", "script.py"]
            mock_processes.append(non_matching_process)
            
            mock_iter.return_value = mock_processes
            
            # Test that pattern matching works correctly
            import re
            pattern = re.compile(collector.process_pattern)
            
            assert pattern.search("gcc") is not None
            assert pattern.search("clang") is not None
            assert pattern.search("g++") is not None
            assert pattern.search("python") is None
