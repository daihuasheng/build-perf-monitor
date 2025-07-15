"""
Unit tests for CPU manager functionality.

Tests the CPU allocation strategies, core assignment logic,
and CPU affinity management.
"""

import pytest
from unittest.mock import patch, Mock

from mymonitor.system.cpu_manager import CPUManager, get_cpu_manager
from mymonitor.models.runtime import CpuAllocationPlan


@pytest.mark.unit
class TestCPUManager:
    """Test cases for CPUManager class."""

    def test_cpu_manager_singleton(self):
        """Test that CPUManager follows singleton pattern."""
        manager1 = get_cpu_manager()
        manager2 = get_cpu_manager()
        assert manager1 is manager2

    def test_cpu_manager_initialization(self, mock_psutil):
        """Test CPUManager initialization."""
        manager = CPUManager()
        assert manager.available_cores == list(range(8))
        assert manager.taskset_available is not None

    @patch("mymonitor.system.cpu_manager.subprocess.run")
    def test_taskset_availability_detection(self, mock_run):
        """Test taskset availability detection."""
        # Test when taskset is available
        mock_run.return_value = Mock(returncode=0)
        manager = CPUManager()
        assert manager.taskset_available is True

        # Test when taskset is not available
        mock_run.return_value = Mock(returncode=1)
        manager = CPUManager()
        assert manager.taskset_available is False


@pytest.mark.unit
class TestAdaptiveAllocation:
    """Test cases for adaptive CPU allocation strategy."""

    def setup_method(self):
        """Set up test environment."""
        self.manager = CPUManager()

    def test_adaptive_allocation_sufficient_cores(self):
        """Test adaptive allocation with sufficient CPU cores."""
        # 16 cores, 4 parallelism
        self.manager.available_cores = list(range(16))
        plan = self.manager._plan_adaptive_allocation_v2(
            parallelism_level=4, monitoring_workers=4
        )

        # Build cores: max(1.25*4, 4+4) = max(5, 8) = 8
        assert len(plan.build_cores) == 8
        assert plan.build_cores == [0, 1, 2, 3, 4, 5, 6, 7]

        # Monitoring cores: remaining 8 cores
        assert len(plan.monitoring_cores) == 8
        assert plan.monitoring_cores == [8, 9, 10, 11, 12, 13, 14, 15]

        # Should not be shared or have warnings
        assert plan.shared_cores is False
        assert plan.has_resource_warning is False
        assert plan.allocation_strategy == "adaptive_v2"

    def test_adaptive_allocation_insufficient_cores(self):
        """Test adaptive allocation with insufficient CPU cores."""
        # 4 cores, 8 parallelism
        self.manager.available_cores = list(range(4))
        plan = self.manager._plan_adaptive_allocation_v2(
            parallelism_level=8, monitoring_workers=4
        )

        # Build cores: limited to available cores
        assert len(plan.build_cores) == 4
        assert plan.build_cores == [0, 1, 2, 3]

        # Monitoring cores: shared with build cores
        assert len(plan.monitoring_cores) == 4
        assert plan.monitoring_cores == [0, 1, 2, 3]

        # Should be shared and have warnings
        assert plan.shared_cores is True
        assert plan.has_resource_warning is True

    def test_adaptive_allocation_monitoring_limit(self):
        """Test adaptive allocation with monitoring core limit."""
        # 32 cores, 4 parallelism
        self.manager.available_cores = list(range(32))
        plan = self.manager._plan_adaptive_allocation_v2(
            parallelism_level=4, monitoring_workers=20
        )

        # Build cores: 8 + excess from monitoring limit
        # Initial: max(5, 8) = 8, remaining: 24, monitoring: 16, excess: 8
        # Final build cores: 8 + 8 = 16
        assert len(plan.build_cores) == 16

        # Monitoring cores: limited to 16
        assert len(plan.monitoring_cores) == 16
        assert plan.shared_cores is False

    def test_adaptive_allocation_edge_cases(self):
        """Test adaptive allocation edge cases."""
        # Single core system
        self.manager.available_cores = [0]
        plan = self.manager._plan_adaptive_allocation_v2(
            parallelism_level=4, monitoring_workers=2
        )

        assert len(plan.build_cores) == 1
        assert len(plan.monitoring_cores) == 1
        assert plan.shared_cores is True
        assert plan.has_resource_warning is True


@pytest.mark.unit
class TestManualAllocation:
    """Test cases for manual CPU allocation strategy."""

    def setup_method(self):
        """Set up test environment."""
        self.manager = CPUManager()
        self.manager.available_cores = list(range(8))

    def test_manual_allocation_normal(self):
        """Test normal manual allocation."""
        plan = self.manager._plan_manual_allocation(
            cores_string="0-5", parallelism_level=4, monitoring_workers=2
        )

        # Should allocate cores as specified
        assert len(plan.build_cores) == 5  # 0-4
        assert len(plan.monitoring_cores) == 1  # 5
        assert plan.allocation_strategy == "manual"
        assert plan.has_resource_warning is False

    def test_manual_allocation_insufficient_cores(self):
        """Test manual allocation with insufficient cores."""
        plan = self.manager._plan_manual_allocation(
            cores_string="0-1", parallelism_level=4, monitoring_workers=2
        )

        assert plan.has_resource_warning is True

    def test_manual_allocation_shared_cores(self):
        """Test manual allocation with shared cores."""
        # This would happen if manual config specifies overlapping cores
        plan = self.manager._plan_manual_allocation(
            cores_string="0-3", parallelism_level=2, monitoring_workers=2
        )

        # Check if shared cores are detected
        shared = set(plan.build_cores) & set(plan.monitoring_cores)
        assert plan.shared_cores == bool(shared)

    def test_manual_allocation_invalid_cores(self):
        """Test manual allocation with invalid core specification."""
        # Should fall back to adaptive allocation
        plan = self.manager._plan_manual_allocation(
            cores_string="invalid", parallelism_level=4, monitoring_workers=2
        )

        # Should fall back to adaptive allocation
        assert plan.allocation_strategy == "adaptive_v2"


@pytest.mark.unit
class TestCoreAssignment:
    """Test cases for core assignment logic."""

    def setup_method(self):
        """Set up test environment."""
        self.manager = CPUManager()
        self.manager.available_cores = list(range(12))

    def test_independent_core_assignment(self):
        """Test independent core assignment."""
        build_cores, monitoring_cores = self.manager._assign_cores(
            build_core_count=6, monitoring_core_count=4, total_cores=12, shared=False
        )

        assert build_cores == [0, 1, 2, 3, 4, 5]
        assert monitoring_cores == [6, 7, 8, 9]

        # No overlap
        assert not (set(build_cores) & set(monitoring_cores))

    def test_shared_core_assignment(self):
        """Test shared core assignment."""
        build_cores, monitoring_cores = self.manager._assign_cores(
            build_core_count=6, monitoring_core_count=4, total_cores=12, shared=True
        )

        assert build_cores == [0, 1, 2, 3, 4, 5]
        assert monitoring_cores == [0, 1, 2, 3]

        # Should have overlap
        assert set(monitoring_cores).issubset(set(build_cores))


@pytest.mark.unit
class TestCPUAffinityManagement:
    """Test cases for CPU affinity management."""

    def setup_method(self):
        """Set up test environment."""
        self.manager = CPUManager()

    @patch("psutil.Process")
    def test_set_process_affinity_success(self, mock_process_class):
        """Test successful process affinity setting."""
        mock_process = Mock()
        mock_process.cpu_affinity.return_value = None
        mock_process_class.return_value = mock_process

        result = self.manager.set_process_affinity(12345, [0, 1, 2])

        assert result is True
        mock_process.cpu_affinity.assert_called_once_with([0, 1, 2])

    @patch("psutil.Process")
    def test_set_process_affinity_failure(self, mock_process_class):
        """Test process affinity setting failure."""
        mock_process_class.side_effect = Exception("Process not found")

        result = self.manager.set_process_affinity(12345, [0, 1, 2])

        assert result is False

    def test_validate_core_ids_valid(self):
        """Test validation of valid core IDs."""
        self.manager.available_cores = [0, 1, 2, 3]

        assert self.manager._validate_core_ids([0, 1, 2]) is True
        assert self.manager._validate_core_ids([0]) is True

    def test_validate_core_ids_invalid(self):
        """Test validation of invalid core IDs."""
        self.manager.available_cores = [0, 1, 2, 3]

        assert self.manager._validate_core_ids([0, 1, 2, 4]) is False
        assert self.manager._validate_core_ids([5]) is False


@pytest.mark.unit
class TestCPUAllocationPlan:
    """Test cases for CpuAllocationPlan model."""

    def test_cpu_allocation_plan_creation(self):
        """Test CpuAllocationPlan creation with all fields."""
        plan = CpuAllocationPlan(
            build_cores=[0, 1, 2, 3],
            monitoring_cores=[4, 5],
            build_cores_desc="0-3",
            monitoring_cores_desc="4-5",
            taskset_prefix="taskset -c 0-3",
            taskset_available=True,
            has_resource_warning=True,
            shared_cores=False,
            allocation_strategy="adaptive_v2",
            monitoring_taskset_prefix="taskset -c 4-5",
        )

        assert plan.build_cores == [0, 1, 2, 3]
        assert plan.monitoring_cores == [4, 5]
        assert plan.has_resource_warning is True
        assert plan.shared_cores is False
        assert plan.allocation_strategy == "adaptive_v2"
        assert plan.monitoring_taskset_prefix == "taskset -c 4-5"

    def test_cpu_allocation_plan_backward_compatibility(self):
        """Test backward compatibility properties."""
        plan = CpuAllocationPlan(
            build_cores=[0, 1, 2, 3],
            monitoring_cores=[4, 5],
            build_cores_desc="0-3",
            monitoring_cores_desc="4-5",
            taskset_prefix="taskset -c 0-3",
            taskset_available=True,
        )

        # Test backward compatibility properties
        assert plan.build_command_prefix == "taskset -c 0-3"
        assert plan.build_taskset_prefix == "taskset -c 0-3"

    def test_cpu_allocation_plan_defaults(self):
        """Test CpuAllocationPlan default values."""
        plan = CpuAllocationPlan(
            build_cores=[0, 1],
            monitoring_cores=[2, 3],
            build_cores_desc="0-1",
            monitoring_cores_desc="2-3",
            taskset_prefix="taskset -c 0-1",
            taskset_available=True,
        )

        # Test default values
        assert plan.has_resource_warning is False
        assert plan.shared_cores is False
        assert plan.allocation_strategy == "adaptive"
        assert plan.monitoring_taskset_prefix == ""
