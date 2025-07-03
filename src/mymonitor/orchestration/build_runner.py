"""
Refactored BuildRunner for the orchestration module.

This module contains the main BuildRunner class that has been simplified
to focus on high-level flow coordination by delegating specific tasks
to specialized component classes.
"""

import logging
from typing import Optional

from ..system import prepare_command_with_setup, run_command
from ..models.results import MonitoringResults
from ..validation import ErrorSeverity, handle_error
from ..collectors.base import AbstractMemoryCollector
from .build_configuration import BuildConfiguration
from .log_manager import LogManager
from .monitoring_orchestrator import MonitoringOrchestrator
from .process_manager import ProcessManager
from .shared_state import BuildRunnerConfig, RuntimeState
from .signal_handler import SignalHandler

logger = logging.getLogger(__name__)


class BuildRunner:
    """
    Simplified BuildRunner that delegates tasks to specialized components.
    
    This class serves as the main orchestrator for the build monitoring process,
    coordinating between the configuration, process management, logging,
    monitoring, and signal handling components.
    """
    
    def __init__(self, config: BuildRunnerConfig):
        """
        Initialize the BuildRunner with specialized components.
        
        Args:
            config: Configuration object containing all necessary parameters
        """
        # Shared state across all components
        self.state = RuntimeState()
        self.config = config
        
        # Component initialization with dependency injection
        self.configuration = BuildConfiguration(config, self.state)
        self.process_manager = ProcessManager(self.state)
        self.log_manager = LogManager(self.state)
        self.monitoring = MonitoringOrchestrator(self.state, config)
        self.signal_handler = SignalHandler(self.state)
        
        # Validate and prepare configuration
        self.configuration.validate_and_prepare()

    def run(self) -> None:
        """
        Execute the entire build and monitor lifecycle.
        
        This is the main entry point that coordinates all phases of the
        build monitoring process.
        """
        # Register this instance for signal handling
        runner_id = id(self)
        self.signal_handler.register_runner(runner_id, self)
        
        # Set up signal handlers
        self.signal_handler.setup_signal_handlers()
        
        try:
            self.setup()
            if self.state.run_context:
                self.execute_build_step()
                self.wait_and_report()
        except Exception as e:
            logger.error(f"An error occurred during the run: {e}", exc_info=True)
        finally:
            self.teardown()
            # Clean up signal handlers
            self.signal_handler.cleanup_signal_handlers()
            # Unregister this instance
            self.signal_handler.unregister_runner(runner_id)

    def setup(self) -> None:
        """
        Prepare the environment for the build and monitoring.
        
        This coordinates the setup phase across all components.
        """
        logger.info("Setting up build and monitoring environment...")
        
        # Prepare CPU allocation and build command
        cpu_plan = self.configuration.prepare_cpu_allocation()
        self.configuration.prepare_build_command(cpu_plan)
        
        # Create run context
        run_context = self.configuration.create_run_context(cpu_plan)
        
        # Open log files
        self.log_manager.open_log_files(run_context)
        
        # Log run prologue
        self.log_manager.log_run_prologue(run_context)
        
        # Execute clean step if needed
        if not self.config.skip_pre_clean:
            self.execute_clean_step()
        
        # Initialize collector
        collector = self.initialize_collector()
        
        # Setup monitoring infrastructure
        self.monitoring.setup_monitoring(collector)
        
        logger.info("Setup complete.")

    def teardown(self) -> None:
        """
        Clean up all resources after the run.
        
        This coordinates the cleanup phase across all components.
        """
        logger.info("Tearing down monitoring resources...")
        
        # Shutdown monitoring system
        self.monitoring.shutdown_monitoring()
        
        # Terminate build process if still running
        if self.state.build_process:
            self.process_manager.terminate_process_tree(
                self.state.build_process.pid, "build process"
            )
        
        # Close log files
        self.log_manager.close_log_files()
        
        logger.info("Teardown complete.")

    def execute_clean_step(self) -> None:
        """Execute the pre-build clean command if defined."""
        if not self.state.run_context:
            raise ValueError("RunContext not initialized.")

        clean_command_template = self.config.project_config.clean_command_template
        setup_command_template = self.config.project_config.setup_command_template
        
        if not clean_command_template:
            logger.info("No clean command defined. Skipping pre-clean.")
            return

        logger.info("--- Executing Pre-build Clean Step ---")
        
        final_clean_command, executable = prepare_command_with_setup(
            clean_command_template, setup_command_template
        )

        return_code, stdout, stderr = run_command(
            final_clean_command,
            self.state.run_context.project_dir,
            shell=True,
            executable_shell=executable,
        )

        # Log the clean command results
        self.log_manager.write_clean_log(final_clean_command, return_code, stdout, stderr)

        if return_code != 0:
            logger.error("Clean command failed. See clean log for details.")
        else:
            logger.info("Clean command executed successfully.")
            
        logger.info("--- Finished Pre-build Clean Step ---")

    def execute_build_step(self) -> None:
        """Start the build process and begin monitoring."""
        if not self.state.run_context or not self.state.collector:
            raise ValueError("RunContext or Collector not initialized.")
        
        if not self.state.final_build_command:
            raise ValueError("Final build command not prepared.")

        logger.info("--- Starting build process and monitoring ---")

        # Start the build process
        self.process_manager.start_build_process(
            command=self.state.final_build_command,
            executable=self.state.executable_shell,
            project_dir=self.state.run_context.project_dir,
            log_files=self.log_manager.log_files
        )

        # Start monitoring
        self.monitoring.start_producer_thread()

    def wait_and_report(self) -> None:
        """
        Wait for the build to complete, collect results, and report.
        
        This coordinates the result collection and reporting phase.
        """
        # Monitor build completion
        build_exit_code = self.process_manager.monitor_build_completion()
        
        # Shutdown monitoring and collect results
        self.monitoring.shutdown_monitoring()
        results = self.monitoring.collect_and_aggregate_results()
        
        # Write summary log
        if results:
            self.log_manager.write_summary_log(build_exit_code, results)
            
            # Save results to parquet file
            if self.state.run_context:
                self._save_results_to_file(results)
        
        # Report build status
        if build_exit_code != 0:
            if self.state.run_context:
                logger.error(f"Build failed with exit code {build_exit_code}. "
                            f"Check logs in {self.state.run_context.paths.output_parquet_file.parent}.")
            else:
                logger.error(f"Build failed with exit code {build_exit_code}.")
        else:
            logger.info("Build completed successfully.")

    def initialize_collector(self) -> AbstractMemoryCollector:
        """Initialize and return the appropriate memory collector."""
        if not self.state.run_context:
            raise ValueError("RunContext not initialized.")

        # Simplified factory logic for collector initialization
        if self.config.collector_type == "pss_psutil":
            from ..collectors.pss_psutil import PssPsutilCollector
            collector_class = PssPsutilCollector
        elif self.config.collector_type == "rss_pidstat":
            from ..collectors.rss_pidstat import RssPidstatCollector
            collector_class = RssPidstatCollector
        else:
            raise ValueError(f"Unknown collector type: {self.config.collector_type}")

        collector = collector_class(
            process_pattern=self.state.run_context.process_pattern,
            monitoring_interval=self.state.run_context.monitoring_interval,
            # Pass collector-specific arguments
            pidstat_stderr_file=self.state.run_context.paths.collector_aux_log_file,
            collector_cpu_core=self.state.run_context.monitor_core_id,
            taskset_available=self.state.run_context.taskset_available,
        )
        
        return collector

    def get_results(self) -> Optional[MonitoringResults]:
        """
        Get the monitoring results after the run completes.
        
        Returns:
            The monitoring results or None if no data was collected
        """
        return self.state.results

    def _save_results_to_file(self, results: MonitoringResults) -> None:
        """Save monitoring results to parquet file."""
        if not self.state.run_context:
            return
            
        try:
            import pandas as pd
            
            final_df = pd.DataFrame(results.all_samples_data)
            if not final_df.empty:
                final_df.to_parquet(
                    self.state.run_context.paths.output_parquet_file, index=False
                )
                logger.info(f"Memory data saved to {self.state.run_context.paths.output_parquet_file}")
        except Exception as e:
            handle_error(
                error=e,
                context=f"Saving Parquet file to {self.state.run_context.paths.output_parquet_file}",
                severity=ErrorSeverity.ERROR,
                include_traceback=True,
                reraise=False,
                logger=logger
            )
 