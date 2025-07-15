"""
Unit tests for configuration validation functionality.

Tests the validation of monitor, project, and rule configurations,
including strategy-specific validation and error handling.
"""

import pytest
from unittest.mock import patch

from mymonitor.config.validators import (
    validate_monitor_config,
    validate_projects_config,
    validate_rules_config,
    _validate_adaptive_strategy_settings,
    _validate_manual_strategy_settings,
    _validate_core_range_format,
)
from mymonitor.validation import ValidationError


@pytest.mark.unit
class TestMonitorConfigValidation:
    """Test cases for monitor configuration validation."""

    def test_validate_monitor_config_adaptive_success(self, sample_config_data):
        """Test successful validation of adaptive monitor configuration."""
        config = validate_monitor_config(sample_config_data)

        assert config.scheduling_policy == "adaptive"
        assert config.enable_cpu_affinity is True
        assert config.thread_name_prefix == "MonitorWorker"
        assert config.max_concurrent_monitors == 4

    def test_validate_monitor_config_manual_success(self, sample_config_data):
        """Test successful validation of manual monitor configuration."""
        sample_config_data["scheduling"]["scheduling_policy"] = "manual"
        sample_config_data["scheduling"]["manual_build_cores"] = "0-7"
        sample_config_data["scheduling"]["manual_monitoring_cores"] = "8-11"

        config = validate_monitor_config(sample_config_data)

        assert config.scheduling_policy == "manual"
        assert config.manual_build_cores == "0-7"
        assert config.manual_monitoring_cores == "8-11"

    def test_validate_monitor_config_invalid_policy(self, sample_config_data):
        """Test validation failure with invalid scheduling policy."""
        sample_config_data["scheduling"]["scheduling_policy"] = "invalid"

        with pytest.raises(ValidationError) as exc_info:
            validate_monitor_config(sample_config_data)

        assert "scheduling_policy" in str(exc_info.value)

    def test_validate_monitor_config_missing_sections(self):
        """Test validation with missing configuration sections."""
        minimal_config = {
            "general": {"default_jobs": [4]},
            "collection": {"interval_seconds": 0.05},
            "scheduling": {"scheduling_policy": "adaptive"},
            "hybrid": {"hybrid_discovery_interval": 0.01},
        }

        # Should succeed with defaults
        config = validate_monitor_config(minimal_config)
        assert config.scheduling_policy == "adaptive"

    def test_validate_monitor_config_invalid_values(self, sample_config_data):
        """Test validation failure with invalid values."""
        # Invalid interval
        sample_config_data["collection"]["interval_seconds"] = -1.0

        with pytest.raises(ValidationError) as exc_info:
            validate_monitor_config(sample_config_data)

        assert "interval_seconds" in str(exc_info.value)


@pytest.mark.unit
class TestAdaptiveStrategyValidation:
    """Test cases for adaptive strategy validation."""

    def test_adaptive_strategy_valid_config(self):
        """Test validation of valid adaptive strategy configuration."""
        scheduling_settings = {
            "scheduling_policy": "adaptive",
            "enable_cpu_affinity": True,
        }

        # Should not raise any exceptions
        _validate_adaptive_strategy_settings(scheduling_settings)

    def test_adaptive_strategy_with_manual_cores_warning(self, caplog):
        """Test adaptive strategy with manual cores specified (should warn)."""
        scheduling_settings = {
            "scheduling_policy": "adaptive",
            "manual_build_cores": "0-3",
            "manual_monitoring_cores": "4-5",
            "enable_cpu_affinity": True,
        }

        _validate_adaptive_strategy_settings(scheduling_settings)

        # Should have warning messages
        assert "will be ignored in adaptive mode" in caplog.text

    def test_adaptive_strategy_disabled_affinity_warning(self, caplog):
        """Test adaptive strategy with disabled CPU affinity (should warn)."""
        scheduling_settings = {
            "scheduling_policy": "adaptive",
            "enable_cpu_affinity": False,
        }

        _validate_adaptive_strategy_settings(scheduling_settings)

        # Should have warning about performance
        assert "may reduce performance" in caplog.text


@pytest.mark.unit
class TestManualStrategyValidation:
    """Test cases for manual strategy validation."""

    def test_manual_strategy_valid_config(self):
        """Test validation of valid manual strategy configuration."""
        scheduling_settings = {
            "scheduling_policy": "manual",
            "manual_build_cores": "0-7",
            "manual_monitoring_cores": "8-11",
        }

        # Should not raise any exceptions
        _validate_manual_strategy_settings(scheduling_settings)

    def test_manual_strategy_missing_build_cores(self):
        """Test manual strategy validation failure with missing build cores."""
        scheduling_settings = {
            "scheduling_policy": "manual",
            "manual_monitoring_cores": "8-11",
        }

        with pytest.raises(ValidationError) as exc_info:
            _validate_manual_strategy_settings(scheduling_settings)

        assert "manual_build_cores must be specified" in str(exc_info.value)

    def test_manual_strategy_missing_monitoring_cores(self):
        """Test manual strategy validation failure with missing monitoring cores."""
        scheduling_settings = {
            "scheduling_policy": "manual",
            "manual_build_cores": "0-7",
        }

        with pytest.raises(ValidationError) as exc_info:
            _validate_manual_strategy_settings(scheduling_settings)

        assert "manual_monitoring_cores must be specified" in str(exc_info.value)

    def test_manual_strategy_empty_cores(self):
        """Test manual strategy validation failure with empty core specifications."""
        scheduling_settings = {
            "scheduling_policy": "manual",
            "manual_build_cores": "",
            "manual_monitoring_cores": "",
        }

        with pytest.raises(ValidationError) as exc_info:
            _validate_manual_strategy_settings(scheduling_settings)

        assert "manual_build_cores must be specified" in str(exc_info.value)


@pytest.mark.unit
class TestCoreRangeFormatValidation:
    """Test cases for core range format validation."""

    def test_valid_core_range_formats(self):
        """Test validation of valid core range formats."""
        valid_formats = ["0", "0-3", "0,2,4", "0-1,4-7", "0,2-5,8", "0-15"]

        for fmt in valid_formats:
            # Should not raise any exceptions
            _validate_core_range_format(fmt, "test_field")

    def test_invalid_core_range_formats(self):
        """Test validation failure with invalid core range formats."""
        invalid_formats = [
            "",  # Empty
            "a-b",  # Non-numeric
            "0-",  # Incomplete range
            "-3",  # Leading dash
            "0--3",  # Double dash
            "0,",  # Trailing comma
            ",0",  # Leading comma
        ]

        for fmt in invalid_formats:
            with pytest.raises(ValidationError):
                _validate_core_range_format(fmt, "test_field")

    def test_core_range_out_of_bounds(self):
        """Test validation failure with out-of-bounds core numbers."""
        invalid_ranges = [
            "1024",  # Too high
            "-1",  # Negative
            "0-1024",  # Range too high
        ]

        for fmt in invalid_ranges:
            with pytest.raises(ValidationError) as exc_info:
                _validate_core_range_format(fmt, "test_field")

            assert "invalid core number" in str(exc_info.value)

    def test_core_range_validation_error_messages(self):
        """Test that validation error messages are informative."""
        # Test empty string
        with pytest.raises(ValidationError) as exc_info:
            _validate_core_range_format("", "test_field")
        assert "test_field cannot be empty" in str(exc_info.value)

        # Test invalid characters
        with pytest.raises(ValidationError) as exc_info:
            _validate_core_range_format("a-b", "test_field")
        assert "invalid characters" in str(exc_info.value)

        # Test invalid format
        with pytest.raises(ValidationError) as exc_info:
            _validate_core_range_format("0-", "test_field")
        assert "invalid format" in str(exc_info.value)


@pytest.mark.unit
class TestProjectConfigValidation:
    """Test cases for project configuration validation."""

    def test_validate_projects_config_success(self, sample_project_config):
        """Test successful projects configuration validation."""
        projects_data = [sample_project_config]
        configs = validate_projects_config(projects_data)

        assert len(configs) == 1
        config = configs[0]
        assert config.name == "test_project"
        assert config.dir == "/tmp/test_project"
        assert config.build_command_template == "make -j<N>"
        assert config.clean_command_template == "make clean"

    def test_validate_projects_config_missing_required(self):
        """Test projects configuration validation with missing required fields."""
        incomplete_config = {
            "name": "test_project"
            # Missing dir, build_command
        }

        with pytest.raises(ValidationError):
            validate_projects_config([incomplete_config])

    def test_validate_projects_config_invalid_command(self):
        """Test projects configuration validation with invalid build command."""
        invalid_config = {
            "name": "test_project",
            "dir": "/tmp/test_project",
            "build_command_template": "",  # Empty command
            "clean_command_template": "make clean",
            "process_pattern": ".*",
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_projects_config([invalid_config])

        assert "build_command_template" in str(exc_info.value)


@pytest.mark.unit
class TestRulesConfigValidation:
    """Test cases for rules configuration validation."""

    def test_validate_rules_config_success(self, sample_rules_config):
        """Test successful rules configuration validation."""
        rules = validate_rules_config(sample_rules_config)

        assert len(rules) == 3

        # Check that rules are sorted by priority (descending)
        priorities = [rule.priority for rule in rules]
        assert priorities == sorted(priorities, reverse=True)

        # Check first rule (highest priority)
        assert rules[0].major_category == "Compiler"
        assert rules[0].category == "gcc"
        assert rules[0].priority == 100

    def test_validate_rules_config_invalid_rule(self):
        """Test rules configuration validation with invalid rule."""
        invalid_rules = [
            {
                "priority": "invalid",  # Should be integer
                "major_category": "Compiler",
                "category": "gcc",
                "match_type": "regex",
                "match_field": "cmd_name",
                "patterns": ["^gcc.*"],
            }
        ]

        with pytest.raises(ValidationError):
            validate_rules_config(invalid_rules)

    def test_validate_rules_config_missing_fields(self):
        """Test rules configuration validation with missing required fields."""
        incomplete_rules = [
            {
                "priority": 100,
                "major_category": "Compiler",
                # Missing category, match_type, etc.
            }
        ]

        with pytest.raises(ValidationError):
            validate_rules_config(incomplete_rules)

    def test_validate_rules_config_empty_list(self):
        """Test rules configuration validation with empty rules list."""
        rules = validate_rules_config([])
        assert len(rules) == 0
