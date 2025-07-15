"""
Integration tests for CPU allocation functionality.

Tests the complete CPU allocation workflow including configuration validation,
allocation planning, and thread pool integration.
"""

import pytest
from unittest.mock import patch, Mock

from mymonitor.system.cpu_manager import CPUManager
from mymonitor.config.validators import validate_monitor_config
from mymonitor.executor.thread_pool import (
    ThreadPoolConfig,
    initialize_global_thread_pools,
)
from mymonitor.models.runtime import RunContext


@pytest.mark.integration
class TestCPUAllocationIntegration:
    """Integration tests for CPU allocation workflow."""

    def setup_method(self):
        """Set up test environment."""
        self.cpu_manager = CPUManager()

    def test_adaptive_allocation_full_workflow(self, sample_config_data):
        """Test complete adaptive allocation workflow."""
        # 1. Validate configuration
        sample_config_data["scheduling"]["scheduling_policy"] = "adaptive"
        config = validate_monitor_config(sample_config_data)

        assert config.scheduling_policy == "adaptive"

        # 2. Plan CPU allocation
        self.cpu_manager.available_cores = list(range(16))
        cpu_plan = self.cpu_manager.plan_cpu_allocation(
            cores_policy=config.scheduling_policy,
            cores_string=config.manual_build_cores,
            parallelism_level=4,
            monitoring_workers=4,
        )

        # 3. Verify allocation plan
        assert cpu_plan.allocation_strategy == "adaptive_v2"
        assert len(cpu_plan.build_cores) == 8  # max(1.25*4, 4+4) = 8
        assert len(cpu_plan.monitoring_cores) == 8  # remaining cores
        assert cpu_plan.shared_cores is False
        assert cpu_plan.has_resource_warning is False

        # 4. Create thread pool configuration
        thread_config = ThreadPoolConfig(
            max_workers=len(cpu_plan.monitoring_cores),
            dedicated_cores=cpu_plan.monitoring_cores,
            enable_cpu_affinity=config.enable_cpu_affinity,
        )

        assert thread_config.dedicated_cores == cpu_plan.monitoring_cores
        assert thread_config.max_workers == 8

    def test_manual_allocation_full_workflow(self, sample_config_data):
        """Test complete manual allocation workflow."""
        # 1. Configure for manual allocation
        sample_config_data["scheduling"]["scheduling_policy"] = "manual"
        sample_config_data["scheduling"]["manual_build_cores"] = "0-7"
        sample_config_data["scheduling"]["manual_monitoring_cores"] = "8-11"

        config = validate_monitor_config(sample_config_data)

        assert config.scheduling_policy == "manual"
        assert config.manual_build_cores == "0-7"
        assert config.manual_monitoring_cores == "8-11"

        # 2. Plan CPU allocation
        self.cpu_manager.available_cores = list(range(16))
        cpu_plan = self.cpu_manager.plan_cpu_allocation(
            cores_policy=config.scheduling_policy,
            cores_string=config.manual_build_cores,
            parallelism_level=4,
            monitoring_workers=4,
        )

        # 3. Verify allocation plan
        assert cpu_plan.allocation_strategy == "manual"
        assert len(cpu_plan.build_cores) >= 1
        assert len(cpu_plan.monitoring_cores) >= 1

        # 4. Verify taskset prefixes
        assert cpu_plan.taskset_prefix != ""
        assert cpu_plan.monitoring_taskset_prefix != ""

    def test_resource_warning_integration(self, sample_config_data):
        """Test resource warning integration across components."""
        # 1. Configure adaptive allocation
        sample_config_data["scheduling"]["scheduling_policy"] = "adaptive"
        config = validate_monitor_config(sample_config_data)

        # 2. Simulate resource-constrained environment
        self.cpu_manager.available_cores = list(range(4))  # Only 4 cores
        cpu_plan = self.cpu_manager.plan_cpu_allocation(
            cores_policy=config.scheduling_policy,
            cores_string=config.manual_build_cores,
            parallelism_level=8,  # High parallelism
            monitoring_workers=4,
        )

        # 3. Verify warning is triggered
        assert cpu_plan.has_resource_warning is True
        assert cpu_plan.shared_cores is True
        assert len(cpu_plan.build_cores) < 8  # Less than requested parallelism

        # 4. Verify thread pool adapts to constraints
        thread_config = ThreadPoolConfig(
            max_workers=min(len(cpu_plan.monitoring_cores), 4),
            dedicated_cores=cpu_plan.monitoring_cores,
            enable_cpu_affinity=config.enable_cpu_affinity,
        )

        assert thread_config.max_workers <= 4

    @patch("mymonitor.executor.thread_pool.get_available_cores")
    def test_thread_pool_integration_with_allocation(
        self, mock_get_cores, sample_config_data
    ):
        """Test thread pool integration with CPU allocation."""
        mock_get_cores.return_value = list(range(8))

        # 1. Plan CPU allocation
        self.cpu_manager.available_cores = list(range(8))
        cpu_plan = self.cpu_manager.plan_cpu_allocation(
            cores_policy="adaptive",
            cores_string="",
            parallelism_level=2,
            monitoring_workers=2,
        )

        # 2. Create thread pool configuration with allocated cores
        monitoring_config = ThreadPoolConfig(
            max_workers=len(cpu_plan.monitoring_cores),
            dedicated_cores=cpu_plan.monitoring_cores,
            enable_cpu_affinity=False,  # Disable for testing
        )

        # 3. Initialize global thread pools
        initialize_global_thread_pools(monitoring_config=monitoring_config)

        # 4. Verify integration
        from mymonitor.executor.thread_pool import get_thread_pool_manager

        manager = get_thread_pool_manager()

        assert manager.is_initialized is True
        pool = manager.get_pool("monitoring")
        assert pool is not None
        assert pool.config.dedicated_cores == cpu_plan.monitoring_cores

        # Clean up
        manager.shutdown_all()
        manager.is_initialized = False
        manager.pools.clear()

    def test_run_context_integration(self, sample_config_data, temp_dir):
        """Test RunContext integration with CPU allocation."""
        # 1. Create run paths
        from mymonitor.models.runtime import RunPaths

        run_paths = RunPaths(
            output_parquet_file=temp_dir / "output.parquet",
            output_summary_log_file=temp_dir / "summary.log",
            collector_aux_log_file=temp_dir / "collector_aux.log",
        )

        # 2. Create run context
        run_context = RunContext(
            project_name="test_project",
            project_dir=temp_dir,
            process_pattern=".*",
            actual_build_command="make -j4",
            parallelism_level=4,
            monitoring_interval=1.0,
            collector_type="pss_psutil",
            current_timestamp_str="20240101_120000",
            taskset_available=True,
            build_cores_target_str="0-3",
            monitor_script_pinned_to_core_info="core 4",
            monitor_core_id=4,
            paths=run_paths,
        )

        # 2. Plan CPU allocation
        self.cpu_manager.available_cores = list(range(12))
        cpu_plan = self.cpu_manager.plan_cpu_allocation(
            cores_policy="adaptive",
            cores_string="",
            parallelism_level=run_context.parallelism_level,
            monitoring_workers=4,
        )

        # 3. Update run context with CPU plan
        run_context.build_cores_target_str = cpu_plan.build_cores_desc
        run_context.monitor_script_pinned_to_core_info = cpu_plan.monitoring_cores_desc
        run_context.monitor_core_id = (
            cpu_plan.monitoring_cores[0] if cpu_plan.monitoring_cores else None
        )
        run_context.taskset_available = cpu_plan.taskset_available

        # 4. Verify integration
        assert run_context.build_cores_target_str == cpu_plan.build_cores_desc
        assert (
            run_context.monitor_script_pinned_to_core_info
            == cpu_plan.monitoring_cores_desc
        )
        assert run_context.taskset_available == cpu_plan.taskset_available

        # 5. Create monitoring thread pool config
        monitoring_config = run_context.create_monitoring_thread_pool_config()

        # Update with CPU allocation
        monitoring_config.dedicated_cores = cpu_plan.monitoring_cores
        monitoring_config.max_workers = min(
            len(cpu_plan.monitoring_cores), monitoring_config.max_workers
        )

        assert monitoring_config.dedicated_cores == cpu_plan.monitoring_cores

    def test_configuration_validation_integration(self, sample_config_data):
        """Test configuration validation integration with CPU allocation."""
        # Test adaptive strategy validation
        sample_config_data["scheduling"]["scheduling_policy"] = "adaptive"
        sample_config_data["scheduling"][
            "manual_build_cores"
        ] = "0-3"  # Should be ignored

        config = validate_monitor_config(sample_config_data)

        # Should succeed but manual cores should be ignored
        assert config.scheduling_policy == "adaptive"
        assert config.manual_build_cores == "0-3"  # Present but ignored

        # Test manual strategy validation
        sample_config_data["scheduling"]["scheduling_policy"] = "manual"
        sample_config_data["scheduling"]["manual_build_cores"] = "0-7"
        sample_config_data["scheduling"]["manual_monitoring_cores"] = "8-11"

        config = validate_monitor_config(sample_config_data)

        # Should succeed with manual configuration
        assert config.scheduling_policy == "manual"
        assert config.manual_build_cores == "0-7"
        assert config.manual_monitoring_cores == "8-11"

        # Plan allocation with manual configuration
        self.cpu_manager.available_cores = list(range(16))
        cpu_plan = self.cpu_manager.plan_cpu_allocation(
            cores_policy=config.scheduling_policy,
            cores_string=config.manual_build_cores,
            parallelism_level=4,
            monitoring_workers=4,
        )

        assert cpu_plan.allocation_strategy == "manual"

    def test_edge_case_integration(self, sample_config_data):
        """Test edge cases in CPU allocation integration."""
        # Test single core system
        self.cpu_manager.available_cores = [0]

        cpu_plan = self.cpu_manager.plan_cpu_allocation(
            cores_policy="adaptive",
            cores_string="",
            parallelism_level=4,
            monitoring_workers=2,
        )

        # Should handle gracefully
        assert len(cpu_plan.build_cores) == 1
        assert len(cpu_plan.monitoring_cores) == 1
        assert cpu_plan.shared_cores is True
        assert cpu_plan.has_resource_warning is True

        # Thread pool should adapt
        thread_config = ThreadPoolConfig(
            max_workers=1,  # Limited by available cores
            dedicated_cores=cpu_plan.monitoring_cores,
            enable_cpu_affinity=False,
        )

        assert thread_config.max_workers == 1
        assert thread_config.dedicated_cores == [0]

    def test_large_system_integration(self, sample_config_data):
        """Test integration on large systems with many cores."""
        # Simulate 64-core system
        self.cpu_manager.available_cores = list(range(64))

        cpu_plan = self.cpu_manager.plan_cpu_allocation(
            cores_policy="adaptive",
            cores_string="",
            parallelism_level=8,
            monitoring_workers=20,
        )

        # Build cores: max(1.25*8, 8+4) = max(10, 12) = 12
        # Remaining: 64-12 = 52, monitoring: min(52, 16) = 16
        # Excess: 52-16 = 36, back to build: 12+36 = 48
        assert len(cpu_plan.build_cores) == 48
        assert len(cpu_plan.monitoring_cores) == 16  # Limited to 16
        assert cpu_plan.shared_cores is False
        assert cpu_plan.has_resource_warning is False

        # Thread pool should use all monitoring cores
        thread_config = ThreadPoolConfig(
            max_workers=16,
            dedicated_cores=cpu_plan.monitoring_cores,
            enable_cpu_affinity=True,
        )

        assert len(thread_config.dedicated_cores) == 16
