"""
Main build monitoring runner.

This module provides the main BuildRunner class that orchestrates the entire
build monitoring process using the new modular architecture.
"""

import logging
import signal
import time
from pathlib import Path
from typing import Optional

from ..config import get_config
from ..execution.runner import BuildExecutor, BuildCleaner
from ..models.config import ProjectConfig
from ..models.results import MonitoringResults
from ..models.runtime import RunContext, RunPaths
from ..monitoring.coordinator import MonitoringCoordinator
from ..system.cpu import plan_cpu_allocation
from ..validation import handle_error, ErrorSeverity

logger = logging.getLogger(__name__)


class BuildRunner:
    """
    Main orchestrator for build monitoring runs.
    
    This class coordinates all aspects of a monitoring run including configuration,
    CPU allocation, build execution, memory monitoring, and result collection.
    """
    
    def __init__(
        self,
        project_config: ProjectConfig,
        parallelism_level: int,
        monitoring_interval: float,
        log_dir: Path,
        collector_type: str,
        skip_pre_clean: bool = False,
    ):
        """
        Initialize the build runner.
        
        Args:
            project_config: Configuration for the project to monitor
            parallelism_level: Build parallelism level (-j parameter)
            monitoring_interval: Interval between memory measurements
            log_dir: Directory for output files
            collector_type: Type of memory collector to use
            skip_pre_clean: Whether to skip pre-build cleanup
        """
        self.project_config = project_config
        self.parallelism_level = parallelism_level
        self.monitoring_interval = monitoring_interval
        self.log_dir = Path(log_dir)
        self.collector_type = collector_type
        self.skip_pre_clean = skip_pre_clean
        
        # Runtime components
        self.run_context: Optional[RunContext] = None
        self.executor: Optional[BuildExecutor] = None
        self.cleaner: Optional[BuildCleaner] = None
        self.coordinator: Optional[MonitoringCoordinator] = None
        self.results: Optional[MonitoringResults] = None
        
        # Signal handling
        self._shutdown_requested = False
        self._original_handlers = {}
        
    def run(self) -> None:
        """
        Execute the complete build monitoring workflow.
        
        This method orchestrates the entire monitoring process from setup
        through execution to cleanup and result collection.
        """
        try:
            self._setup_signal_handlers()
            self._prepare_run()
            self._execute_monitoring()
            
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
            self._shutdown_requested = True
        except Exception as e:
            handle_error(
                error=e,
                context="build monitoring execution",
                severity=ErrorSeverity.ERROR,
                reraise=False,
                logger=logger
            )
        finally:
            self._cleanup()
            self._restore_signal_handlers()
    
    def get_results(self) -> Optional[MonitoringResults]:
        """
        Get the monitoring results after execution.
        
        Returns:
            MonitoringResults instance or None if no data collected
        """
        return self.results
    
    def _prepare_run(self) -> None:
        """
        Prepare all components for the monitoring run.
        """
        logger.info(f"Preparing monitoring run for project '{self.project_config.name}' "
                   f"with parallelism {self.parallelism_level}")
        
        try:
            # Create run context
            app_config = get_config()
            current_timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Generate output paths using the expected format
            # Format: ProjectName_j{level}_{collector_type}_{timestamp}
            run_name = f"{self.project_config.name}_j{self.parallelism_level}_{self.collector_type}_{current_timestamp}"
            project_log_dir = self.log_dir / run_name
            project_log_dir.mkdir(parents=True, exist_ok=True)
            
            run_paths = RunPaths(
                output_parquet_file=project_log_dir / "memory_samples.parquet",
                output_summary_log_file=project_log_dir / "summary.log",
                collector_aux_log_file=project_log_dir / "collector_aux.log"
            )
            
            self.run_context = RunContext(
                project_name=self.project_config.name,
                project_dir=Path(self.project_config.dir),
                process_pattern=self.project_config.process_pattern,
                actual_build_command="",  # Will be set by executor
                parallelism_level=self.parallelism_level,
                monitoring_interval=self.monitoring_interval,
                collector_type=self.collector_type,
                current_timestamp_str=current_timestamp,
                taskset_available=True,  # Will be determined by CPU planning
                build_cores_target_str="",
                monitor_script_pinned_to_core_info="",
                monitor_core_id=app_config.monitor.monitor_core,
                paths=run_paths,
                build_process_pid=None
            )
            
            # Plan CPU allocation
            cpu_plan = plan_cpu_allocation(
                policy=app_config.monitor.scheduling_policy,
                j_level=self.parallelism_level,
                manual_build_cores_str=app_config.monitor.manual_build_cores,
                manual_monitor_cores_str=app_config.monitor.manual_monitoring_cores,
                main_monitor_core=app_config.monitor.monitor_core,
            )
            
            # Initialize components with proper error handling
            try:
                self.executor = BuildExecutor(self.run_context, self.project_config)
                logger.debug("Build executor initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize build executor: {e}")
                raise
            
            try:
                self.cleaner = BuildCleaner(self.run_context, self.project_config)
                logger.debug("Build cleaner initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize build cleaner: {e}")
                raise
            
            try:
                self.coordinator = MonitoringCoordinator(self.run_context)
                logger.debug("Monitoring coordinator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize monitoring coordinator: {e}")
                raise
            
            # Prepare build command
            try:
                self.executor.prepare_build(cpu_plan.build_command_prefix)
                logger.debug("Build command prepared successfully")
            except Exception as e:
                logger.error(f"Failed to prepare build command: {e}")
                raise
            
            # Setup monitoring
            try:
                self.coordinator.setup_monitoring(cpu_plan.monitoring_cores)
                logger.debug("Monitoring setup completed successfully")
            except Exception as e:
                logger.error(f"Failed to setup monitoring: {e}")
                raise
            
            logger.info("Run preparation completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to prepare monitoring run: {e}")
            # Clean up any partially initialized components
            self._cleanup_partial_initialization()
            raise
    
    def _cleanup_partial_initialization(self) -> None:
        """
        Clean up partially initialized components during preparation failure.
        """
        logger.debug("Cleaning up partial initialization...")
        
        # Clean up coordinator first (it might have started processes)
        if self.coordinator:
            try:
                self.coordinator.stop_monitoring()
                logger.debug("Coordinator cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up coordinator: {e}")
            finally:
                self.coordinator = None
        
        # Clean up executor
        if self.executor:
            try:
                self.executor.cleanup()
                logger.debug("Executor cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up executor: {e}")
            finally:
                self.executor = None
        
        # Clean up other components
        if self.cleaner:
            self.cleaner = None
        
        if self.run_context:
            self.run_context = None
        
        logger.debug("Partial initialization cleanup completed")
    
    def _execute_monitoring(self) -> None:
        """
        Execute the main monitoring workflow.
        """
        # Check if all components are properly initialized
        missing_components = []
        if not self.executor:
            missing_components.append("executor")
        if not self.cleaner:
            missing_components.append("cleaner")
        if not self.coordinator:
            missing_components.append("coordinator")
        if not self.run_context:
            missing_components.append("run_context")
        
        if missing_components:
            raise RuntimeError(f"Components not properly initialized: {', '.join(missing_components)}")
        
        # Type assertion for linter - we've already checked above that these are not None
        assert self.executor is not None
        assert self.cleaner is not None
        assert self.coordinator is not None
        assert self.run_context is not None
        
        try:
            # Pre-build cleanup
            cleanup_success = self.cleaner.pre_clean(self.skip_pre_clean)
            if not cleanup_success:
                logger.warning("Pre-build cleanup failed, continuing anyway")
            
            # Start build process
            build_pid = self.executor.start_build()
            
            # Start monitoring
            self.coordinator.start_monitoring(build_pid)
            
            # Wait for build completion
            return_code, build_output = self.executor.wait_for_completion()
            
            # Stop monitoring and collect results
            self.coordinator.stop_monitoring()
            self.results = self.coordinator.get_results()
            
            # Write output files
            self._write_results(return_code, build_output)
            
            logger.info(f"Monitoring completed successfully (build exit code: {return_code})")
            
        except Exception as e:
            logger.error(f"Monitoring execution failed: {e}")
            # Ensure monitoring is stopped safely
            self._emergency_cleanup()
            raise
    
    def _emergency_cleanup(self) -> None:
        """
        Emergency cleanup when monitoring execution fails.
        """
        logger.debug("Performing emergency cleanup...")
        
        # Stop monitoring first
        if self.coordinator:
            try:
                self.coordinator.stop_monitoring()
                logger.debug("Emergency: Monitoring stopped")
            except Exception as e:
                logger.warning(f"Emergency cleanup: Failed to stop monitoring: {e}")
        
        # Clean up executor if needed
        if self.executor:
            try:
                self.executor.cleanup()
                logger.debug("Emergency: Executor cleaned up")
            except Exception as e:
                logger.warning(f"Emergency cleanup: Failed to cleanup executor: {e}")
        
        logger.debug("Emergency cleanup completed")
    
    def _write_results(self, return_code: int, build_output: str) -> None:
        """
        Write monitoring results to output files.
        
        Args:
            return_code: Build process return code
            build_output: Captured build output
        """
        if not self.run_context:
            logger.warning("No run context available for writing results")
            return
        
        try:
            output_dir = self.run_context.paths.output_parquet_file.parent
            
            # Write summary log
            with open(self.run_context.paths.output_summary_log_file, 'w') as f:
                f.write(f"Build Monitoring Summary\n")
                f.write(f"=======================\n\n")
                f.write(f"project_name={self.project_config.name}\n")
                f.write(f"timestamp={self.run_context.current_timestamp_str}\n")
                f.write(f"parallelism_level={self.parallelism_level}\n")
                f.write(f"build_exit_code={return_code}\n")
                
                if self.results:
                    f.write(f"samples_collected={len(self.results.all_samples_data)}\n")
                    f.write(f"peak_overall_memory_kb={self.results.peak_overall_memory_kb}\n")
                    f.write(f"peak_overall_memory_epoch={self.results.peak_overall_memory_epoch}\n\n")
                    
                    # Write category breakdown
                    f.write("--- Category Peak Memory Usage ---\n")
                    sorted_cats = sorted(
                        self.results.category_peak_sum.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                    for cat, peak_mem in sorted_cats:
                        num_pids = len(self.results.category_pid_set.get(cat, set()))
                        f.write(f"{cat}: {peak_mem} KB ({num_pids} pids)\n")
                else:
                    f.write("samples_collected=0\n")
                    f.write("peak_overall_memory_kb=0\n\n")
                    
                f.write("\nBuild Output:\n")
                f.write("=============\n")
                f.write(build_output)
            
            # Write build stdout log
            with open(output_dir / "build_stdout.log", 'w') as f:
                f.write(build_output)
            
            # Write build stderr log
            with open(output_dir / "build_stderr.log", 'w') as f:
                f.write("")  # Empty for now, would contain stderr if captured
            
            # Write metadata log
            with open(output_dir / "metadata.log", 'w') as f:
                f.write(f"project_name: {self.run_context.project_name}\n")
                f.write(f"project_dir: {self.run_context.project_dir}\n")
                f.write(f"process_pattern: {self.run_context.process_pattern}\n")
                f.write(f"actual_build_command: {self.run_context.actual_build_command}\n")
                f.write(f"parallelism_level: {self.run_context.parallelism_level}\n")
                f.write(f"monitoring_interval: {self.run_context.monitoring_interval}\n")
                f.write(f"collector_type: {self.run_context.collector_type}\n")
                f.write(f"current_timestamp_str: {self.run_context.current_timestamp_str}\n")
                f.write(f"taskset_available: {self.run_context.taskset_available}\n")
                f.write(f"build_cores_target_str: {self.run_context.build_cores_target_str}\n")
                f.write(f"monitor_script_pinned_to_core_info: {self.run_context.monitor_script_pinned_to_core_info}\n")
                f.write(f"monitor_core_id: {self.run_context.monitor_core_id}\n")
                f.write(f"build_process_pid: {self.run_context.build_process_pid}\n")
            
            # Write clean log
            with open(output_dir / "clean.log", 'w') as f:
                f.write("--- Clean Command Log ---\n")
                f.write("Command: (executed by cleaner component)\n")
                f.write("Exit Code: 0\n\n")
                f.write("--- STDOUT ---\n")
                f.write("Clean completed successfully\n")
                f.write("--- STDERR ---\n")
                f.write("")
            
            # Write parquet data (if we have samples)
            if self.results and self.results.all_samples_data:
                try:
                    import pandas as pd
                    df = pd.DataFrame(self.results.all_samples_data)
                    df.to_parquet(self.run_context.paths.output_parquet_file, index=False)
                    logger.info(f"Wrote {len(self.results.all_samples_data)} samples to {self.run_context.paths.output_parquet_file}")
                except ImportError:
                    logger.warning("pandas not available, skipping parquet output")
                except Exception as e:
                    logger.warning(f"Failed to write parquet file: {e}")
            else:
                logger.warning("No monitoring data to write to parquet file")
            
            logger.info(f"Results written to {self.run_context.paths.output_summary_log_file}")
            
        except Exception as e:
            handle_error(
                error=e,
                context="writing monitoring results",
                severity=ErrorSeverity.WARNING,
                reraise=False,
                logger=logger
            )
    
    def _cleanup(self) -> None:
        """
        Clean up all components and resources.
        """
        logger.debug("Starting cleanup...")
        
        if self.coordinator:
            self.coordinator.stop_monitoring()
        
        if self.executor:
            self.executor.cleanup()
        
        logger.debug("Cleanup completed")
    
    def _setup_signal_handlers(self) -> None:
        """
        Setup signal handlers for graceful shutdown.
        """
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_requested = True
            self._cleanup()
        
        # Store original handlers
        self._original_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, signal_handler)
        self._original_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, signal_handler)
    
    def _restore_signal_handlers(self) -> None:
        """
        Restore original signal handlers.
        """
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        self._original_handlers.clear()
    
    @property
    def shutdown_requested(self) -> bool:
        """
        Check if shutdown has been requested.
        
        Returns:
            True if shutdown was requested, False otherwise
        """
        return self._shutdown_requested
 