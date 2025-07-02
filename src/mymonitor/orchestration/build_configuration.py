"""
Build configuration management for the orchestration module.

This module handles configuration validation, CPU allocation planning,
command preparation, and run context creation.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

from .. import process_utils
from ..data_models import (
    RunContext, RunPaths, ValidationError,
    validate_positive_integer, validate_positive_float,
    validate_enum_choice, validate_cpu_core_range, validate_path_exists
)
from .shared_state import BuildRunnerConfig, RuntimeState


class BuildConfiguration:
    """
    Handles build configuration validation, preparation and context creation.
    
    This class extracts the configuration-related logic from the original
    BuildRunner.__init__ and setup methods.
    """
    
    def __init__(self, config: BuildRunnerConfig, state: RuntimeState):
        self.config = config
        self.state = state
        self.validated = False
        
    def validate_and_prepare(self) -> None:
        """
        Validate all configuration parameters and prepare the build environment.
        
        This replaces the validation logic from BuildRunner.__init__
        """
        if self.validated:
            return
            
        try:
            # Validate basic numeric parameters
            self.config.parallelism_level = validate_positive_integer(
                self.config.parallelism_level,
                min_value=1,
                max_value=1024,
                field_name="parallelism_level"
            )
            
            self.config.monitoring_interval = validate_positive_float(
                self.config.monitoring_interval,
                min_value=0.001,  # 1ms minimum
                max_value=60.0,   # 60s maximum
                field_name="monitoring_interval"
            )
            
            self.config.monitor_core_id = validate_positive_integer(
                self.config.monitor_core_id,
                min_value=0,
                max_value=1023,
                field_name="monitor_core_id"
            )
            
            # Validate enum choices
            self.config.collector_type = validate_enum_choice(
                self.config.collector_type,
                valid_choices=["pss_psutil", "rss_pidstat"],
                field_name="collector_type"
            )
            
            self.config.scheduling_policy = validate_enum_choice(
                self.config.scheduling_policy,
                valid_choices=["adaptive", "manual"],
                field_name="scheduling_policy"
            )
            
            # Validate CPU core ranges
            max_cores = os.cpu_count() or 1
            
            self.config.manual_build_cores = validate_cpu_core_range(
                self.config.manual_build_cores,
                max_cores=max_cores,
                field_name="manual_build_cores"
            )
            
            self.config.manual_monitoring_cores = validate_cpu_core_range(
                self.config.manual_monitoring_cores,
                max_cores=max_cores,
                field_name="manual_monitoring_cores"
            )
            
            # Validate log directory
            if not isinstance(self.config.log_dir, Path):
                self.config.log_dir = Path(self.config.log_dir)
            if not self.config.log_dir.parent.exists():
                raise ValidationError(f"Parent directory of log_dir does not exist: {self.config.log_dir.parent}")
            
            # Initialize system state
            self.state.current_timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            self.state.taskset_available = shutil.which("taskset") is not None
            
            self.validated = True
            
        except ValidationError as e:
            raise ValidationError(f"BuildRunner parameter validation failed: {e}")

    def prepare_cpu_allocation(self) -> 'process_utils.CpuAllocationPlan':
        """
        Prepare CPU allocation plan based on configuration.
        
        Returns the CPU allocation plan for build and monitoring processes.
        """
        if not self.validated:
            raise RuntimeError("Configuration must be validated before preparing CPU allocation")
            
        cpu_plan = process_utils.plan_cpu_allocation(
            policy=self.config.scheduling_policy,
            j_level=self.config.parallelism_level,
            manual_build_cores_str=self.config.manual_build_cores,
            manual_monitor_cores_str=self.config.manual_monitoring_cores,
            main_monitor_core=self.config.monitor_core_id,
        )
        
        # Store results in state
        self.state.build_command_prefix = cpu_plan.build_command_prefix
        self.state.monitoring_cores = cpu_plan.monitoring_cores
        
        return cpu_plan

    def prepare_build_command(self, cpu_plan: 'process_utils.CpuAllocationPlan') -> Tuple[str, Optional[str]]:
        """
        Prepare the final build command with setup and CPU allocation.
        
        Returns:
            Tuple of (final_build_command, executable_shell)
            executable_shell can be None if no specific shell is required
        """
        if not self.validated:
            raise RuntimeError("Configuration must be validated before preparing build command")
            
        final_build_command, executable_shell = process_utils.prepare_full_build_command(
            main_command_template=self.config.project_config.build_command_template,
            j_level=self.config.parallelism_level,
            taskset_prefix=cpu_plan.build_command_prefix,
            setup_command=self.config.project_config.setup_command_template,
        )
        
        # Store results in state
        self.state.final_build_command = final_build_command
        self.state.executable_shell = executable_shell
        
        return final_build_command, executable_shell

    def create_run_context(self, cpu_plan: 'process_utils.CpuAllocationPlan') -> RunContext:
        """
        Create the RunContext for this monitoring task.
        
        This replaces BuildRunner._create_run_context method.
        """
        if not self.validated:
            raise RuntimeError("Configuration must be validated before creating run context")
            
        if not self.state.final_build_command:
            raise RuntimeError("Build command must be prepared before creating run context")

        # Validate project directory path at runtime
        project_dir_path = validate_path_exists(
            self.config.project_config.dir,
            must_be_dir=True,
            check_readable=True,
            field_name=f"project '{self.config.project_config.name}' directory"
        )

        # Generate run paths
        run_paths = self.generate_run_paths(
            self.config.log_dir,
            self.config.project_config.name,
            self.config.parallelism_level,
            self.config.collector_type,
            self.state.current_timestamp_str,
        )

        monitor_script_pinned_to_core_info = f"Core {self.config.monitor_core_id}"

        run_context = RunContext(
            project_name=self.config.project_config.name,
            project_dir=project_dir_path,
            process_pattern=self.config.project_config.process_pattern,
            actual_build_command=self.state.final_build_command,
            parallelism_level=self.config.parallelism_level,
            monitoring_interval=self.config.monitoring_interval,
            collector_type=self.config.collector_type,
            current_timestamp_str=self.state.current_timestamp_str,
            taskset_available=self.state.taskset_available,
            build_cores_target_str=cpu_plan.build_cores_desc,
            monitor_script_pinned_to_core_info=monitor_script_pinned_to_core_info,
            monitor_core_id=self.config.monitor_core_id,
            paths=run_paths,
        )
        
        # Store in state
        self.state.run_context = run_context
        
        return run_context

    @staticmethod
    def generate_run_paths(
        log_dir_base: Path,
        project_name: str,
        parallelism_level: int,
        collector_type: str,
        timestamp: str,
    ) -> RunPaths:
        """
        Generate all necessary output paths for a given run.
        
        This is moved from BuildRunner._generate_run_paths as a static method.
        """
        sanitized_project_name = "".join(
            c if c.isalnum() else "_" for c in project_name
        )
        run_name = f"{sanitized_project_name}_j{parallelism_level}_{collector_type}_{timestamp}"
        log_dir_for_run = log_dir_base / run_name
        log_dir_for_run.mkdir(parents=True, exist_ok=True)

        return RunPaths(
            output_parquet_file=log_dir_for_run / "memory_samples.parquet",
            output_summary_log_file=log_dir_for_run / "summary.log",
            collector_aux_log_file=log_dir_for_run / "collector_aux.log",
        )