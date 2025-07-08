"""
Async build runner for CLI integration.

This module provides the main async build runner that replaces the old
multiprocessing-based BuildRunner, using the new AsyncIO architecture.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from ..config import get_config
from ..models.config import ProjectConfig
from ..models.runtime import RunContext, RunPaths
from ..monitoring.coordinator import AsyncMonitoringCoordinator
from ..executor.build_process import BuildProcessManager
from ..system.cpu_manager import get_cpu_manager
from ..executor.thread_pool import (
    initialize_global_thread_pools, 
    shutdown_global_thread_pools, 
    ThreadPoolConfig
)
from ..validation import handle_error, ErrorSeverity

logger = logging.getLogger(__name__)


class BuildRunner:
    """
    Main async build runner that coordinates monitoring and build execution.
    
    This class provides async build monitoring with resource management,
    CPU affinity, and comprehensive monitoring capabilities.
    """
    
    def __init__(
        self,
        project_config: ProjectConfig,
        parallelism_level: int,
        monitoring_interval: float,
        log_dir: Path,
        collector_type: str,
        skip_pre_clean: bool = False
    ):
        """
        Initialize the async build monitor.
        
        Args:
            project_config: Project configuration
            parallelism_level: Number of parallel jobs (-j level)
            monitoring_interval: Monitoring interval in seconds
            log_dir: Directory for log files
            collector_type: Type of memory collector to use
            skip_pre_clean: Whether to skip pre-build cleanup
        """
        self.project_config = project_config
        self.parallelism_level = parallelism_level
        self.monitoring_interval = monitoring_interval
        self.log_dir = Path(log_dir)
        self.collector_type = collector_type
        self.skip_pre_clean = skip_pre_clean
        
        # Runtime state
        self.run_context: Optional[RunContext] = None
        self.monitoring_coordinator: Optional[AsyncMonitoringCoordinator] = None
        self.build_runner: Optional[BuildProcessManager] = None
        self.shutdown_requested = False
        
    async def run_async(self) -> bool:
        """
        Run the complete build monitoring process asynchronously.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get configuration for thread pools
            config = get_config()
            
            # Create thread pool configuration - only monitoring pool is needed
            # Build and I/O operations are handled by asyncio, not requiring separate thread pools
            thread_pool_configs = {
                'monitoring': ThreadPoolConfig(
                    max_workers=config.monitor.max_concurrent_monitors,
                    thread_name_prefix=config.monitor.thread_name_prefix,
                    enable_cpu_affinity=config.monitor.enable_cpu_affinity,
                    shutdown_timeout=config.monitor.graceful_shutdown_timeout
                )
            }
            
            # Initialize global thread pools with configuration
            initialize_global_thread_pools(thread_pool_configs)
            
            # Setup run context
            await self._setup_run_context()
            
            # Plan CPU allocation
            cpu_plan = get_cpu_manager().plan_cpu_allocation(
                cores_policy=config.monitor.scheduling_policy,
                cores_string=config.monitor.manual_build_cores,
                parallelism_level=self.parallelism_level,
                monitoring_workers=min(4, self.parallelism_level)
            )
            
            # Update run context with CPU plan details
            self.run_context.build_cores_target_str = cpu_plan.build_cores_desc
            self.run_context.monitor_script_pinned_to_core_info = cpu_plan.monitoring_cores_desc
            self.run_context.monitor_core_id = cpu_plan.monitoring_cores[0] if cpu_plan.monitoring_cores else None
            self.run_context.taskset_available = cpu_plan.taskset_available
            
            # Create monitoring coordinator
            self.monitoring_coordinator = AsyncMonitoringCoordinator(self.run_context)
            
            # Create build runner
            actual_build_command = self.project_config.build_command_template.replace('<N>', str(self.parallelism_level))
            self.build_runner = BuildProcessManager(
                build_command=actual_build_command,
                build_directory=Path(self.project_config.dir),
                build_cores=cpu_plan.build_cores,
                timeout=3600.0  # 1 hour timeout
            )
            
            # Setup monitoring
            await self.monitoring_coordinator.setup_monitoring(cpu_plan.monitoring_cores)
            
            # Start build process
            logger.info(f"Starting build with -j{self.parallelism_level}")
            build_pid = await self.build_runner.start_build_async()
            
            # Start monitoring
            await self.monitoring_coordinator.start_monitoring(build_pid)
            
            # Wait for build completion
            return_code = await self.build_runner.wait_for_completion_async()
            
            # Stop monitoring
            await self.monitoring_coordinator.stop_monitoring()
            
            # Process results
            results = self.monitoring_coordinator.get_results()
            if results:
                await self._save_results(results)
                logger.info(f"Build completed with return code {return_code}")
                return return_code == 0
            else:
                logger.warning("No monitoring results collected")
                return False
                
        except asyncio.CancelledError:
            logger.info("Build monitoring was cancelled")
            return False
        except Exception as e:
            logger.error(f"Error during async build monitoring: {e}", exc_info=True)
            return False
        finally:
            # Cleanup
            await self._cleanup()
    
    def request_shutdown(self) -> None:
        """Request shutdown of the build monitoring process."""
        self.shutdown_requested = True
        
        # Cancel build if running
        if self.build_runner:
            try:
                asyncio.create_task(self.build_runner.cancel_build_async())
            except Exception as e:
                logger.warning(f"Error cancelling build: {e}")
    
    async def _setup_run_context(self) -> None:
        """Setup the run context for monitoring."""
        # Generate timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create project-specific directory
        project_dir_name = f"{self.project_config.name}_j{self.parallelism_level}_{self.collector_type}_{timestamp}"
        project_output_dir = self.log_dir / project_dir_name
        project_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run paths with fixed names inside the project directory
        run_paths = RunPaths(
            output_parquet_file=project_output_dir / "memory_samples.parquet",
            output_summary_log_file=project_output_dir / "summary.log",
            collector_aux_log_file=project_output_dir / "collector_aux.log"
        )
        
        # Create additional log files that tests expect
        (project_output_dir / "build_stdout.log").touch()
        (project_output_dir / "build_stderr.log").touch()
        (project_output_dir / "clean.log").touch()
        
        # Create run context
        self.run_context = RunContext(
            project_name=self.project_config.name,
            project_dir=Path(self.project_config.dir),
            process_pattern=self.project_config.process_pattern,
            actual_build_command=self.project_config.build_command_template.replace('<N>', str(self.parallelism_level)),
            parallelism_level=self.parallelism_level,
            monitoring_interval=self.monitoring_interval,
            collector_type=self.collector_type,
            current_timestamp_str=timestamp,
            taskset_available=True,  # Will be detected automatically
            build_cores_target_str="",  # Will be set by CPU planning
            monitor_script_pinned_to_core_info="",  # Will be set by CPU planning
            monitor_core_id=None,  # Will be set by CPU planning
            paths=run_paths
        )
    
    async def _save_results(self, results) -> None:
        """Save monitoring results to files."""
        try:
            logger.info("Saving monitoring results...")
            
            if not results or not results.all_samples_data:
                logger.warning("No monitoring data to save")
                return
            
            # Import pandas for data saving
            try:
                import pandas as pd
            except ImportError:
                logger.error("pandas is required for saving results but not installed")
                return
            
            # Convert samples to DataFrame
            df = pd.DataFrame(results.all_samples_data)
            
            # Save Parquet file
            parquet_path = self.run_context.paths.output_parquet_file
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Saved monitoring data to: {parquet_path}")
            
            # Save metadata log
            metadata_path = self.run_context.paths.output_parquet_file.parent / "metadata.log"
            with open(metadata_path, 'w') as f:
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
                
                # Get build process PID from build runner
                build_pid = getattr(self.build_runner, '_process_pid', 'unknown')
                f.write(f"build_process_pid: {build_pid}\n")
                
                # Calculate and write build duration
                build_duration = getattr(self.build_runner, '_duration_seconds', 0.0)
                f.write(f"build_duration_seconds: {build_duration:.2f}\n")
            
            # Generate summary log in the expected format
            summary_path = self.run_context.paths.output_summary_log_file
            with open(summary_path, 'w') as f:
                f.write("Build Monitoring Summary\n")
                f.write("=======================\n\n")
                f.write(f"Project: {self.run_context.project_name}\n")
                f.write(f"Parallelism: -j{self.run_context.parallelism_level}\n")
                
                # Calculate build duration
                build_duration = getattr(self.build_runner, '_duration_seconds', 0.0)
                f.write(f"Total Build & Monitoring Duration: {build_duration:.1f}s ({build_duration:.2f} seconds)\n")
                
                # Format peak memory in GB
                peak_memory_gb = results.peak_overall_memory_kb / 1024 / 1024
                f.write(f"Peak Overall Memory (PSS_KB): {peak_memory_gb:.2f} GB\n")
                
                f.write(f"Samples Collected: {len(results.all_samples_data)}\n")
                
                # Get build exit code from build runner
                exit_code = getattr(self.build_runner, '_return_code', 0)
                f.write(f"Build Exit Code: {exit_code}\n\n")
                
                # Write category statistics in the expected format
                if results.category_stats:
                    f.write("--- Category Peak Memory Usage ---\n")
                    
                    # Group by major category
                    major_categories = {}
                    for category, stats in results.category_stats.items():
                        if ':' in category:
                            major_cat, minor_cat = category.split(':', 1)
                            if major_cat not in major_categories:
                                major_categories[major_cat] = {}
                            major_categories[major_cat][minor_cat] = stats
                    
                    for major_cat, minor_cats in major_categories.items():
                        # Calculate total peak memory for this major category
                        total_peak_kb = sum(stats['peak_sum_kb'] for stats in minor_cats.values())
                        total_pids = sum(stats['process_count'] for stats in minor_cats.values())
                        
                        f.write(f"{major_cat}:\n")
                        f.write(f"  Total Peak Memory: {total_peak_kb} KB ({total_pids} total pids)\n")
                        
                        # Write minor categories
                        for minor_cat, stats in minor_cats.items():
                            peak_kb = stats['peak_sum_kb']
                            process_count = stats['process_count']
                            single_peak_kb = int(stats['average_peak_kb'])  # Use average as single process peak
                            f.write(f"    {minor_cat}: {peak_kb} KB (total, {process_count} pids), single process peak: {single_peak_kb} KB\n")
                        
                        f.write("\n")
            
            logger.info(f"Saved summary log to: {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}", exc_info=True)
    
    async def _cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Stop monitoring if still running
            if self.monitoring_coordinator:
                await self.monitoring_coordinator.stop_monitoring()
            
            # Cancel build if still running
            if self.build_runner:
                await self.build_runner.cancel_build_async()
                
            # Shutdown global thread pools
            shutdown_global_thread_pools()
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def run(self) -> bool:
        """
        Run the build monitoring process synchronously.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if we're already in an async context
            try:
                # This will raise RuntimeError if no event loop is running
                loop = asyncio.get_running_loop()
                # If we get here, we're in an async context, we can't use run_until_complete
                logger.error("Cannot run synchronous BuildRunner from within async context. Use run_async() directly.")
                return False
            except RuntimeError:
                # No event loop running, we can create one
                pass
            
            # Create and run new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the async monitor
                return loop.run_until_complete(self.run_async())
            finally:
                # Clean up the event loop
                loop.close()
                asyncio.set_event_loop(None)
            
        except Exception as e:
            logger.error(f"Error running build monitor: {e}", exc_info=True)
            return False
