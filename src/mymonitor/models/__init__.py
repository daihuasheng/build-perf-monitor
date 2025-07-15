"""
Data models and structures for the monitoring system.

This module provides comprehensive data models used throughout the application,
organized by their functional purpose and lifecycle:

Configuration Models:
- Application-wide configuration settings
- Project-specific build configurations
- Process classification rules and patterns
- Storage and monitoring parameters

Runtime Models:
- CPU allocation plans and resource management
- Execution context and environment state
- File paths and directory structures
- Process monitoring coordination

Result Models:
- Monitoring results and statistical analysis
- Memory usage samples and aggregations
- Performance metrics and categorized data

Hybrid Monitoring Models:
- Producer-consumer architecture components
- Task scheduling and execution coordination
- Sample collection and processing pipelines
- Performance statistics and monitoring metadata

All models use type hints and dataclasses for better code clarity,
runtime validation, and IDE support.
"""

# Configuration models
from .config import AppConfig, MonitorConfig, ProjectConfig, RuleConfig

# Runtime models
from .runtime import CpuAllocationPlan, RunContext, RunPaths

# Result models
from .results import MonitoringResults

# Hybrid monitoring models
from .hybrid_monitoring import (
    ProcessTask,
    SampleResult,
    HybridCollectorConfig,
    HybridCollectorStats,
)

__all__ = [
    # Configuration
    "AppConfig",
    "MonitorConfig",
    "ProjectConfig",
    "RuleConfig",
    # Runtime
    "CpuAllocationPlan",
    "RunContext",
    "RunPaths",
    # Results
    "MonitoringResults",
    # Hybrid monitoring
    "ProcessTask",
    "SampleResult",
    "HybridCollectorConfig",
    "HybridCollectorStats",
]
