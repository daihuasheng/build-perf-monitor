"""
Performance tests for CPU allocation functionality.

Tests the performance characteristics of CPU allocation algorithms,
including execution time, memory usage, and scalability.
"""

import pytest
import time
import psutil
import threading
from unittest.mock import patch

from mymonitor.system.cpu_manager import CPUManager
from mymonitor.classification.classifier import get_process_category, clear_categorization_cache


@pytest.mark.performance
class TestCPUAllocationPerformance:
    """Performance tests for CPU allocation algorithms."""
    
    def setup_method(self):
        """Set up test environment."""
        self.cpu_manager = CPUManager()
    
    def test_adaptive_allocation_performance(self, performance_config):
        """Test performance of adaptive allocation algorithm."""
        # Test with various system sizes
        system_sizes = [4, 8, 16, 32, 64, 128]
        parallelism_levels = [1, 2, 4, 8, 16, 32]
        
        execution_times = []
        
        for cores in system_sizes:
            for parallelism in parallelism_levels:
                if parallelism > cores:
                    continue
                
                self.cpu_manager.available_cores = list(range(cores))
                
                start_time = time.perf_counter()
                
                # Run allocation multiple times to get average
                for _ in range(100):
                    plan = self.cpu_manager._plan_adaptive_allocation_v2(
                        parallelism_level=parallelism,
                        monitoring_workers=min(4, parallelism)
                    )
                
                end_time = time.perf_counter()
                avg_time = (end_time - start_time) / 100
                execution_times.append(avg_time)
                
                # Verify performance constraint
                assert avg_time < 0.001, f"Allocation too slow: {avg_time:.6f}s for {cores} cores, {parallelism} parallelism"
        
        # Overall performance should be reasonable
        max_time = max(execution_times)
        assert max_time < performance_config["max_execution_time"] / 1000  # Convert to ms
    
    def test_manual_allocation_performance(self, performance_config):
        """Test performance of manual allocation algorithm."""
        self.cpu_manager.available_cores = list(range(64))
        
        core_specifications = [
            "0-7",
            "0,2,4,6,8,10,12,14",
            "0-15,32-47",
            "0-3,8-11,16-19,24-27",
            "0-31"
        ]
        
        for core_spec in core_specifications:
            start_time = time.perf_counter()
            
            # Run allocation multiple times
            for _ in range(100):
                plan = self.cpu_manager._plan_manual_allocation(
                    cores_string=core_spec,
                    parallelism_level=8,
                    monitoring_workers=4
                )
            
            end_time = time.perf_counter()
            avg_time = (end_time - start_time) / 100
            
            # Should be fast
            assert avg_time < 0.001, f"Manual allocation too slow: {avg_time:.6f}s for {core_spec}"
    
    def test_core_assignment_performance(self, performance_config):
        """Test performance of core assignment logic."""
        self.cpu_manager.available_cores = list(range(128))
        
        test_cases = [
            (8, 4, False),   # Independent assignment
            (16, 8, False),
            (32, 16, False),
            (64, 32, False),
            (8, 4, True),    # Shared assignment
            (16, 8, True),
            (32, 16, True),
            (64, 32, True),
        ]
        
        for build_count, monitor_count, shared in test_cases:
            start_time = time.perf_counter()
            
            for _ in range(1000):
                build_cores, monitor_cores = self.cpu_manager._assign_cores(
                    build_core_count=build_count,
                    monitoring_core_count=monitor_count,
                    total_cores=128,
                    shared=shared
                )
            
            end_time = time.perf_counter()
            avg_time = (end_time - start_time) / 1000
            
            # Core assignment should be very fast
            assert avg_time < 0.0001, f"Core assignment too slow: {avg_time:.6f}s"
    
    def test_allocation_memory_usage(self, performance_config):
        """Test memory usage of allocation algorithms."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Simulate large system
        self.cpu_manager.available_cores = list(range(256))
        
        # Run many allocations
        plans = []
        for i in range(1000):
            plan = self.cpu_manager._plan_adaptive_allocation_v2(
                parallelism_level=16,
                monitoring_workers=8
            )
            plans.append(plan)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
        
        # Memory increase should be reasonable
        assert memory_increase < performance_config["max_memory_mb"], \
            f"Memory usage too high: {memory_increase:.2f}MB"
        
        # Clean up
        del plans
    
    def test_concurrent_allocation_performance(self, performance_config):
        """Test performance under concurrent allocation requests."""
        self.cpu_manager.available_cores = list(range(32))
        
        results = []
        errors = []
        
        def allocation_worker(worker_id):
            try:
                start_time = time.perf_counter()
                
                for i in range(100):
                    plan = self.cpu_manager._plan_adaptive_allocation_v2(
                        parallelism_level=4 + (worker_id % 4),
                        monitoring_workers=2 + (worker_id % 3)
                    )
                
                end_time = time.perf_counter()
                results.append(end_time - start_time)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent allocations
        threads = []
        for i in range(8):
            thread = threading.Thread(target=allocation_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0, f"Concurrent allocation errors: {errors}"
        
        # Verify performance
        max_time = max(results)
        assert max_time < performance_config["max_execution_time"], \
            f"Concurrent allocation too slow: {max_time:.3f}s"
    
    def test_scalability_with_system_size(self, performance_config):
        """Test scalability of allocation algorithms with system size."""
        system_sizes = [4, 8, 16, 32, 64, 128, 256]
        execution_times = []
        
        for size in system_sizes:
            self.cpu_manager.available_cores = list(range(size))
            
            start_time = time.perf_counter()
            
            # Run allocation
            plan = self.cpu_manager._plan_adaptive_allocation_v2(
                parallelism_level=min(16, size // 2),
                monitoring_workers=min(8, size // 4)
            )
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Should scale reasonably
            assert execution_time < 0.01, f"Allocation too slow for {size} cores: {execution_time:.6f}s"
        
        # Check that execution time doesn't grow exponentially
        # Allow for some variance but should be roughly linear or better
        if len(execution_times) >= 3:
            # Compare first and last execution times
            ratio = execution_times[-1] / execution_times[0]
            size_ratio = system_sizes[-1] / system_sizes[0]
            
            # Execution time should not grow faster than system size
            assert ratio <= size_ratio * 2, f"Poor scalability: time ratio {ratio:.2f}, size ratio {size_ratio:.2f}"


@pytest.mark.performance
class TestClassificationPerformance:
    """Performance tests for process classification."""
    
    def setup_method(self):
        """Set up test environment."""
        clear_categorization_cache()
    
    def teardown_method(self):
        """Clean up after test."""
        clear_categorization_cache()
    
    @patch('mymonitor.config.get_config')
    def test_classification_performance(self, mock_get_config, performance_config):
        """Test performance of process classification."""
        # Mock configuration with realistic rules
        mock_config = Mock()
        mock_config.rules = [
            Mock(priority=100, major_category="Compiler", category="gcc",
                 match_type="regex", match_field="cmd_name", patterns=["^gcc.*", "^g\\+\\+.*"]),
            Mock(priority=90, major_category="Linker", category="ld",
                 match_type="regex", match_field="cmd_name", patterns=["^ld$", "^ld\\..*"]),
            Mock(priority=80, major_category="Build_Tool", category="make",
                 match_type="in_list", match_field="cmd_name", patterns=["make", "gmake", "ninja"]),
            Mock(priority=70, major_category="Archiver", category="ar",
                 match_type="regex", match_field="cmd_name", patterns=["^ar$"]),
            Mock(priority=60, major_category="Preprocessor", category="cpp",
                 match_type="regex", match_field="cmd_name", patterns=["^cpp$"]),
        ]
        mock_config.monitor.categorization_cache_size = 10000
        mock_get_config.return_value = mock_config
        
        # Test data
        test_processes = [
            ("gcc", "gcc -O2 -c file.c"),
            ("g++", "g++ -std=c++17 file.cpp"),
            ("ld", "ld -o output file.o"),
            ("make", "make -j4"),
            ("ar", "ar rcs libtest.a file.o"),
            ("unknown", "unknown_tool --help"),
        ] * 100  # Repeat for performance testing
        
        start_time = time.perf_counter()
        
        # Classify all processes
        for cmd_name, full_cmd in test_processes:
            major, minor = get_process_category(cmd_name, full_cmd)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time = total_time / len(test_processes)
        
        # Should be fast
        assert avg_time < 0.001, f"Classification too slow: {avg_time:.6f}s per process"
        assert total_time < performance_config["max_execution_time"], \
            f"Total classification time too slow: {total_time:.3f}s"
    
    @patch('mymonitor.config.get_config')
    def test_classification_cache_performance(self, mock_get_config, performance_config):
        """Test performance of classification caching."""
        mock_config = Mock()
        mock_config.rules = [
            Mock(priority=100, major_category="Compiler", category="gcc",
                 match_type="regex", match_field="cmd_name", patterns=["^gcc.*"]),
        ]
        mock_config.monitor.categorization_cache_size = 1000
        mock_get_config.return_value = mock_config
        
        # First run - populate cache
        start_time = time.perf_counter()
        for i in range(100):
            get_process_category("gcc", f"gcc -O2 file{i}.c")
        first_run_time = time.perf_counter() - start_time
        
        # Second run - use cache
        start_time = time.perf_counter()
        for i in range(100):
            get_process_category("gcc", f"gcc -O2 file{i}.c")
        second_run_time = time.perf_counter() - start_time
        
        # Cache should provide significant speedup
        speedup = first_run_time / second_run_time
        assert speedup > 2.0, f"Cache speedup insufficient: {speedup:.2f}x"
        
        # Cached operations should be very fast
        avg_cached_time = second_run_time / 100
        assert avg_cached_time < 0.0001, f"Cached classification too slow: {avg_cached_time:.6f}s"
    
    @patch('mymonitor.config.get_config')
    def test_classification_memory_usage(self, mock_get_config, performance_config):
        """Test memory usage of classification system."""
        mock_config = Mock()
        mock_config.rules = [
            Mock(priority=100, major_category="Compiler", category="gcc",
                 match_type="regex", match_field="cmd_name", patterns=["^gcc.*"]),
        ]
        mock_config.monitor.categorization_cache_size = 10000
        mock_get_config.return_value = mock_config
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Classify many unique processes to fill cache
        for i in range(5000):
            get_process_category(f"tool{i}", f"tool{i} --arg{i}")
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
        
        # Memory usage should be reasonable
        assert memory_increase < performance_config["max_memory_mb"], \
            f"Classification memory usage too high: {memory_increase:.2f}MB"
