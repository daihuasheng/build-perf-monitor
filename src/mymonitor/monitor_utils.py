"""
Core utilities for orchestrating the build monitoring process.

This module provides the main BuildRunner class interface for backward compatibility.
The implementation has been refactored to use specialized orchestration components
for better maintainability and separation of concerns.

The main `BuildRunner` class now delegates to:
- BuildConfiguration: Configuration validation and preparation  
- ProcessManager: Process lifecycle management
- LogManager: Log file management
- MonitoringOrchestrator: Data collection coordination
- SignalHandler: Signal handling management
"""

import logging
from pathlib import Path
from typing import Optional

from .data_models import MonitoringResults, ProjectConfig
from .orchestration import BuildRunner as OrchestrationBuildRunner
from .orchestration.shared_state import BuildRunnerConfig

logger = logging.getLogger(__name__)


class BuildRunner:
    """
    Compatibility wrapper around the refactored orchestration BuildRunner.
    
    This class maintains the original BuildRunner interface for backward compatibility
    while delegating all implementation to the new orchestration components.
    """

    def __init__(
        self,
        project_config: ProjectConfig,
        parallelism_level: int,
        monitoring_interval: float,
        log_dir: Path,
        collector_type: str,
        skip_pre_clean: bool,
        # --- Scheduling Configuration ---
        scheduling_policy: str,
        manual_build_cores: str,
        manual_monitoring_cores: str,
        monitor_core_id: int,
    ):
        """
        Initialize the BuildRunner with the same interface as before.
        
        All parameters are passed through to the new orchestration BuildRunner
        via the BuildRunnerConfig object.
        """
        # Create configuration object for the new implementation
        config = BuildRunnerConfig(
            project_config=project_config,
            parallelism_level=parallelism_level,
            monitoring_interval=monitoring_interval,
            log_dir=log_dir,
            collector_type=collector_type,
            skip_pre_clean=skip_pre_clean,
            scheduling_policy=scheduling_policy,
            manual_build_cores=manual_build_cores,
            manual_monitoring_cores=manual_monitoring_cores,
            monitor_core_id=monitor_core_id,
        )
        
        # Delegate to the new orchestration implementation
        self._orchestration_runner = OrchestrationBuildRunner(config)
        
        # Expose shutdown_requested for backward compatibility
        # (in case external code accesses it directly)
        self.shutdown_requested = self._orchestration_runner.state.shutdown_requested

    def run(self) -> None:
        """
        Execute the entire build and monitor lifecycle.
        
        This method maintains the exact same interface as the original BuildRunner.
        """
        self._orchestration_runner.run()

    def get_results(self) -> Optional[MonitoringResults]:
        """
        Get the monitoring results after the run completes.
        
        Returns:
            The monitoring results or None if no data was collected
        """
        return self._orchestration_runner.get_results()

    # --- Deprecated methods for backward compatibility ---
    # These methods are preserved in case any external code calls them directly,
    # but they delegate to the orchestration implementation.

    def setup(self) -> None:
        """Deprecated: Use run() instead. Maintained for compatibility."""
        logger.warning("setup() method is deprecated. Use run() instead.")
        self._orchestration_runner.setup()

    def teardown(self) -> None:
        """Deprecated: Use run() instead. Maintained for compatibility."""
        logger.warning("teardown() method is deprecated. Use run() instead.")
        self._orchestration_runner.teardown()


# --- Deprecated module-level functions ---
# These functions are no longer used by the new implementation but are kept
# for any external code that might import them directly.

def _global_signal_handler(signum: int, frame) -> None:
    """
    Deprecated: Signal handling is now managed by SignalHandler class.
    
    This function is kept for backward compatibility but should not be used.
    """
    logger.warning("_global_signal_handler is deprecated. Signal handling is now managed by SignalHandler class.")
    # Delegate to the new signal handler if needed
    from .orchestration.signal_handler import SignalHandler
    SignalHandler._global_signal_handler(signum, frame)


def _monitoring_worker_entry(core_id, input_queue, output_queue, primary_metric_field):
    """
    Deprecated: Worker function is now managed by MonitoringOrchestrator class.
    
    This function is kept for backward compatibility but should not be used.
    """
    logger.warning("_monitoring_worker_entry is deprecated. Worker logic is now managed by MonitoringOrchestrator class.")
    # Delegate to the new implementation
    from .orchestration.monitoring_orchestrator import MonitoringOrchestrator
    return MonitoringOrchestrator._worker_entry_point(core_id, input_queue, output_queue, primary_metric_field)
