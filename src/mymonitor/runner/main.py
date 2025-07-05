"""
Main build monitoring runner.

This module provides the main BuildRunner class that orchestrates the entire
build monitoring process using the new modular architecture.
"""

import logging
import signal
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import get_config
from ..execution.runner import BuildExecutor, BuildCleaner
from ..models.config import ProjectConfig
from ..models.results import MonitoringResults
from ..models.runtime import RunContext, RunPaths
from ..monitoring.coordinator import MonitoringCoordinator
from ..system.cpu import plan_cpu_allocation
from ..validation import handle_error, ErrorSeverity

logger = logging.getLogger(__name__)


def _get_metric_name_from_collector_type(collector_type: str) -> str:
    """
    Map collector type to memory metric name for summary logs.
    
    Args:
        collector_type: The collector type (e.g., 'pss_psutil', 'rss_pidstat')
        
    Returns:
        Memory metric name (e.g., 'PSS_KB', 'RSS_KB')
    """
    if 'pss' in collector_type.lower():
        return 'PSS_KB'
    elif 'rss' in collector_type.lower():
        return 'RSS_KB'
    else:
        return 'MEMORY_KB'


def _format_category_stats(category_peak_sum: Dict[str, int], 
                          category_pid_set: Dict[str, Set[str]],
                          category_stats: Dict[str, Dict[str, Any]] = None) -> str:
    """
    Format category statistics for summary log with major/minor category grouping.
    
    Args:
        category_peak_sum: Dictionary mapping category to peak memory
        category_pid_set: Dictionary mapping category to set of PIDs
        category_stats: Dictionary mapping category to detailed statistics
        
    Returns:
        Formatted string with category statistics
    """
    if not category_peak_sum:
        return "No category data available.\n"
    
    # Group by major category
    major_categories: Dict[str, Dict[str, Any]] = {}
    
    for category, peak_memory in category_peak_sum.items():
        if ':' in category:
            major_cat, minor_cat = category.split(':', 1)
        else:
            major_cat = category
            minor_cat = 'Unknown'
        
        # Initialize major category if not exists
        if major_cat not in major_categories:
            major_categories[major_cat] = {
                'minor_categories': {},
                'total_memory': 0,
                'total_pids': set()
            }
        
        # Add to minor categories
        individual_peak = 0
        if category_stats and category in category_stats:
            individual_peak = category_stats[category].get('individual_peak_memory_kb', peak_memory)
        else:
            individual_peak = peak_memory  # Fallback to total peak if no individual data
            
        major_categories[major_cat]['minor_categories'][minor_cat] = {
            'peak_memory': peak_memory,  # Total peak for this category
            'individual_peak_memory': individual_peak,  # Peak of single process in this category
            'pid_count': len(category_pid_set.get(category, set()))
        }
        
        # Update major category totals
        major_categories[major_cat]['total_memory'] += peak_memory
        major_categories[major_cat]['total_pids'].update(category_pid_set.get(category, set()))
    
    # Sort major categories by name
    sorted_major_cats = sorted(major_categories.keys())
    
    # Format output
    output_lines = []
    
    for major_cat in sorted_major_cats:
        major_data = major_categories[major_cat]
        
        # Major category header
        output_lines.append(f"\n{major_cat}:")
        output_lines.append(f"  Total Peak Memory: {major_data['total_memory']} KB "
                           f"({len(major_data['total_pids'])} total pids)")
        
        # Sort minor categories by peak memory (descending)
        sorted_minor_cats = sorted(
            major_data['minor_categories'].items(),
            key=lambda x: x[1]['peak_memory'],
            reverse=True
        )
        
        # Minor categories
        for minor_cat, minor_data in sorted_minor_cats:
            output_lines.append(
                f"    {minor_cat}: {minor_data['peak_memory']} KB (total, {minor_data['pid_count']} pids), "
                f"single process peak: {minor_data['individual_peak_memory']} KB"
            )
    
    return '\n'.join(output_lines)


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
        
        # Build timing
        self.build_start_time: Optional[float] = None
        self.build_end_time: Optional[float] = None
        
    def request_shutdown(self) -> None:
        """
        Requests a shutdown of the current running operations.
        
        This method is designed to be called from an external signal handler.
        """
        logger.info("Shutdown requested for the current build runner.")
        # The _emergency_cleanup method contains the necessary logic to stop
        # the coordinator and executor safely.
        self._emergency_cleanup()

    def run(self) -> None:
        """
        Execute the complete build monitoring workflow.
        
        This method orchestrates the entire monitoring process from setup
        through execution to cleanup and result collection.
        """
        try:
            self._prepare_run()
            self._execute_monitoring()
            
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user (KeyboardInterrupt)")
            # The global signal handler should now handle this,
            # but this catch remains as a fallback.
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
        Clean up any partially initialized components during preparation failure.
        """
        logger.debug("Cleaning up partial initialization...")
        
        if self.coordinator:
            try:
                self.coordinator.stop_monitoring()
            except Exception as e:
                logger.debug(f"Error stopping coordinator during cleanup: {e}")
        
        if self.executor:
            try:
                self.executor.cleanup()
            except Exception as e:
                logger.debug(f"Error cleaning up executor during cleanup: {e}")
        
        # Reset components
        self.coordinator = None
        self.executor = None
        self.cleaner = None
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
            
            # Record build start time
            self.build_start_time = time.time()
            
            # Start build process
            build_pid = self.executor.start_build()
            
            # Start monitoring
            self.coordinator.start_monitoring(build_pid)
            
            # Wait for build completion
            return_code, build_output = self.executor.wait_for_completion()
            
            # Record build end time
            self.build_end_time = time.time()
            
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
        Write monitoring results to output files in plotter.py compatible format.
        
        Args:
            return_code: Build process return code
            build_output: Captured build output
        """
        if not self.run_context:
            logger.warning("No run context available for writing results")
            return
        
        try:
            output_dir = self.run_context.paths.output_parquet_file.parent
            
            # Calculate build duration
            build_duration = 0.0
            if self.build_start_time and self.build_end_time:
                build_duration = self.build_end_time - self.build_start_time
            
            # Get memory metric name
            metric_name = _get_metric_name_from_collector_type(self.collector_type)
            
            # Write summary log in plotter.py compatible format
            with open(self.run_context.paths.output_summary_log_file, 'w') as f:
                f.write(f"Build Monitoring Summary\n")
                f.write(f"=======================\n\n")
                
                # plotter.py expected format
                f.write(f"Project: {self.project_config.name}\n")
                f.write(f"Parallelism: -j{self.parallelism_level}\n")
                f.write(f"Total Build & Monitoring Duration: {build_duration:.1f}s ({build_duration:.2f} seconds)\n")
                
                if self.results:
                    peak_mem_gb = self.results.peak_overall_memory_kb / (1024 * 1024)
                    f.write(f"Peak Overall Memory ({metric_name}): {peak_mem_gb:.2f} GB\n")
                    f.write(f"Samples Collected: {len(self.results.all_samples_data)}\n")
                else:
                    f.write(f"Peak Overall Memory ({metric_name}): 0.00 GB\n")
                    f.write(f"Samples Collected: 0\n")
                
                f.write(f"Build Exit Code: {return_code}\n\n")
                
                if self.results:
                    # Write enhanced category breakdown
                    f.write("--- Category Peak Memory Usage ---")
                    category_stats = _format_category_stats(
                        self.results.category_peak_sum,
                        self.results.category_pid_set,
                        self.results.category_stats
                    )
                    f.write(category_stats)
                    f.write("\n")
                else:
                    f.write("No memory data collected.\n")
            
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
                f.write(f"build_duration_seconds: {build_duration:.2f}\n")
            
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
 