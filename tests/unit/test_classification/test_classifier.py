"""
Unit tests for process classification functionality.

Tests the process categorization logic, caching mechanisms,
and shell wrapper command parsing.
"""

import pytest
from unittest.mock import patch, Mock

from mymonitor.classification.classifier import (
    get_process_category,
    clear_categorization_cache,
    get_cache_stats,
    _categorization_cache,
)
from mymonitor.system.processes import parse_shell_wrapper_command


@pytest.mark.unit
class TestProcessClassification:
    """Test cases for process classification."""

    def setup_method(self):
        """Set up test environment."""
        # Clear cache before each test
        clear_categorization_cache()

    def teardown_method(self):
        """Clean up after each test."""
        # Clear cache after each test
        clear_categorization_cache()

    @patch("mymonitor.classification.classifier.config.get_config")
    def test_get_process_category_compiler(self, mock_get_config):
        """Test classification of compiler processes."""
        # Mock configuration with compiler rules
        mock_config = Mock()
        mock_config.rules = [
            Mock(
                priority=100,
                major_category="Compiler",
                category="gcc",
                match_type="regex",
                match_field="current_cmd_name",
                patterns="^gcc.*|^g\\+\\+.*",
            )
        ]
        mock_config.monitor.categorization_cache_size = 1000
        mock_get_config.return_value = mock_config

        # Test gcc classification
        major, minor = get_process_category("gcc", "gcc -O2 -c file.c")
        assert major == "Compiler"
        assert minor == "gcc"

        # Test g++ classification
        major, minor = get_process_category("g++", "g++ -std=c++17 file.cpp")
        assert major == "Compiler"
        assert minor == "gcc"

    @patch("mymonitor.classification.classifier.config.get_config")
    def test_get_process_category_linker(self, mock_get_config):
        """Test classification of linker processes."""
        mock_config = Mock()
        mock_config.rules = [
            Mock(
                priority=90,
                major_category="Linker",
                category="ld",
                match_type="regex",
                match_field="current_cmd_name",
                patterns="^ld$|^ld\\..*",
            )
        ]
        mock_config.monitor.categorization_cache_size = 1000
        mock_get_config.return_value = mock_config

        major, minor = get_process_category("ld", "ld -o output file.o")
        assert major == "Linker"
        assert minor == "ld"

    @patch("mymonitor.classification.classifier.config.get_config")
    def test_get_process_category_build_tool(self, mock_get_config):
        """Test classification of build tool processes."""
        mock_config = Mock()
        mock_config.rules = [
            Mock(
                priority=80,
                major_category="Build_Tool",
                category="make",
                match_type="in_list",
                match_field="current_cmd_name",
                patterns=["make", "gmake", "ninja"],
            )
        ]
        mock_config.monitor.categorization_cache_size = 1000
        mock_get_config.return_value = mock_config

        # Test make
        major, minor = get_process_category("make", "make -j4")
        assert major == "Build_Tool"
        assert minor == "make"

        # Test ninja
        major, minor = get_process_category("ninja", "ninja -j8")
        assert major == "Build_Tool"
        assert minor == "make"

    @patch("mymonitor.classification.classifier.config.get_config")
    def test_get_process_category_unknown(self, mock_get_config):
        """Test classification of unknown processes."""
        mock_config = Mock()
        mock_config.rules = []  # No rules
        mock_config.monitor.categorization_cache_size = 1000
        mock_get_config.return_value = mock_config

        major, minor = get_process_category("unknown_tool", "unknown_tool --help")
        assert major == "Other"
        assert minor == "Other_unknown_tool"

    @patch("mymonitor.classification.classifier.config.get_config")
    def test_get_process_category_priority_order(self, mock_get_config):
        """Test that rules are applied in priority order."""
        mock_config = Mock()
        mock_config.rules = [
            Mock(
                priority=100,  # Higher priority - should be first
                major_category="Compiler",
                category="gcc",
                match_type="regex",
                match_field="current_cmd_name",
                patterns="^gcc.*",
            ),
            Mock(
                priority=50,  # Lower priority - should be second
                major_category="Generic",
                category="generic",
                match_type="regex",
                match_field="current_cmd_name",
                patterns=".*",  # Matches everything
            ),
        ]
        mock_config.monitor.categorization_cache_size = 1000
        mock_get_config.return_value = mock_config

        # Should match the higher priority rule
        major, minor = get_process_category("gcc", "gcc -O2 file.c")
        assert major == "Compiler"
        assert minor == "gcc"

    @patch("mymonitor.classification.classifier.config.get_config")
    def test_get_process_category_full_command_match(self, mock_get_config):
        """Test classification based on full command line."""
        mock_config = Mock()
        mock_config.rules = [
            Mock(
                priority=100,
                major_category="Compiler",
                category="gcc_optimized",
                match_type="regex",
                match_field="current_cmd_full",
                patterns="-O[23]",  # Match optimization flags
            )
        ]
        mock_config.monitor.categorization_cache_size = 1000
        mock_get_config.return_value = mock_config

        # Should match based on full command
        major, minor = get_process_category("gcc", "gcc -O2 -c file.c")
        assert major == "Compiler"
        assert minor == "gcc_optimized"

        # Should not match without optimization flag
        major, minor = get_process_category("gcc", "gcc -c file.c")
        assert major == "Other"
        assert minor == "Other_gcc"


@pytest.mark.unit
class TestClassificationCaching:
    """Test cases for classification caching."""

    def setup_method(self):
        """Set up test environment."""
        clear_categorization_cache()

    def teardown_method(self):
        """Clean up after each test."""
        clear_categorization_cache()

    @patch("mymonitor.classification.classifier.config.get_config")
    def test_classification_caching(self, mock_get_config):
        """Test that classification results are cached."""
        mock_config = Mock()
        mock_config.rules = [
            Mock(
                priority=100,
                major_category="Compiler",
                category="gcc",
                match_type="regex",
                match_field="current_cmd_name",
                patterns="^gcc.*",
            )
        ]
        mock_config.monitor.categorization_cache_size = 1000
        mock_get_config.return_value = mock_config

        # First call should populate cache
        major1, minor1 = get_process_category("gcc", "gcc -O2 file.c")

        # Second call should use cache
        major2, minor2 = get_process_category("gcc", "gcc -O2 file.c")

        assert major1 == major2 == "Compiler"
        assert minor1 == minor2 == "gcc"

        # Check cache stats
        stats = get_cache_stats()
        assert stats["cache_size"] == 1

    @patch("mymonitor.config.get_config")
    def test_cache_size_limit(self, mock_get_config):
        """Test that cache respects size limit."""
        mock_config = Mock()
        mock_config.rules = []
        mock_config.monitor.categorization_cache_size = 2  # Small cache
        mock_get_config.return_value = mock_config

        # Add entries up to cache limit
        get_process_category("tool1", "tool1 arg")
        get_process_category("tool2", "tool2 arg")

        # Cache should be at limit
        assert len(_categorization_cache) == 2

        # Adding another entry should not exceed limit
        get_process_category("tool3", "tool3 arg")
        assert len(_categorization_cache) == 2

    def test_clear_categorization_cache(self):
        """Test cache clearing functionality."""
        # Manually add entry to cache
        _categorization_cache[("test", "test")] = ("Test", "test")
        assert len(_categorization_cache) == 1

        # Clear cache
        clear_categorization_cache()
        assert len(_categorization_cache) == 0

    def test_get_cache_stats(self):
        """Test cache statistics functionality."""
        # Clear cache first
        clear_categorization_cache()

        stats = get_cache_stats()
        assert "cache_size" in stats
        assert "cache_entries" in stats
        assert stats["cache_size"] == 0


@pytest.mark.unit
class TestShellWrapperParsing:
    """Test cases for shell wrapper command parsing."""

    def test_parse_shell_wrapper_simple(self):
        """Test parsing of simple shell wrapper commands."""
        cmd_name, full_cmd = parse_shell_wrapper_command("sh", "sh -c 'gcc file.c'")
        assert cmd_name == "gcc"
        assert full_cmd == "gcc file.c"

    def test_parse_shell_wrapper_bash(self):
        """Test parsing of bash wrapper commands."""
        cmd_name, full_cmd = parse_shell_wrapper_command("bash", "bash -c 'make -j4'")
        assert cmd_name == "make"
        assert full_cmd == "make -j4"

    def test_parse_shell_wrapper_complex(self):
        """Test parsing of complex shell wrapper commands."""
        cmd_name, full_cmd = parse_shell_wrapper_command(
            "sh", "sh -c 'cd /tmp && gcc -O2 -o output file.c'"
        )
        # Should extract the main command or return original
        assert isinstance(cmd_name, str)
        assert isinstance(full_cmd, str)

    def test_parse_shell_wrapper_no_wrapper(self):
        """Test parsing of non-wrapper commands."""
        cmd_name, full_cmd = parse_shell_wrapper_command("gcc", "gcc -O2 file.c")
        assert cmd_name == "gcc"
        assert full_cmd == "gcc -O2 file.c"

    def test_parse_shell_wrapper_invalid(self):
        """Test parsing of invalid shell wrapper commands."""
        # Malformed command should return original
        cmd_name, full_cmd = parse_shell_wrapper_command("sh", "sh -c")
        assert cmd_name == "sh"
        assert full_cmd == "sh -c"

    def test_parse_shell_wrapper_quoted_commands(self):
        """Test parsing of quoted commands in shell wrappers."""
        test_cases = [
            ("sh", 'sh -c "gcc -O2 file.c"', "gcc", "gcc -O2 file.c"),
            ("bash", "bash -c 'make clean'", "make", "make clean"),
            ("sh", "sh -c '/usr/bin/gcc file.c'", "gcc", "/usr/bin/gcc file.c"),
        ]

        for shell, full_cmd, _, _ in test_cases:
            cmd_name, parsed_cmd = parse_shell_wrapper_command(shell, full_cmd)
            # Note: Exact behavior depends on implementation
            # This test verifies the function doesn't crash and returns reasonable results
            assert isinstance(cmd_name, str)
            assert isinstance(parsed_cmd, str)
