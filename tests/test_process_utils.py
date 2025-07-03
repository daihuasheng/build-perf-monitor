import pytest
from pathlib import Path

from mymonitor.models import AppConfig, RuleConfig, MonitorConfig
from mymonitor.classification import get_process_category
from mymonitor.system import plan_cpu_allocation

# --- Fixture for Mocking Configuration ---


@pytest.fixture
def mock_rules_config(monkeypatch):
    """
    Mocks the application configuration for process utility tests.
    This fixture creates a controlled set of rules in memory and uses monkeypatch
    to make `config.get_config()` return this controlled config. This isolates
    the test from the actual .toml files on disk.
    """
    # Define a minimal set of rules needed for these specific tests
    test_rules = [
        RuleConfig(
            major_category="CPP_Compile",
            category="GCCInternalCompiler",
            priority=160,
            match_field="current_cmd_name",
            match_type="in_list",
            patterns=["cc1", "cc1plus"],
        ),
        RuleConfig(
            major_category="CPP_Driver",
            category="Driver_Compile",
            priority=125,
            match_field="current_cmd_full",
            match_type="regex",
            patterns=r"^(?:.*/)?(clang\+\+|clang|gcc|g\+\+|cc)\b.*\s-c(\s|$)",
        ),
        RuleConfig(
            major_category="CPP_Link",
            category="DirectLinker",
            priority=155,
            match_field="current_cmd_name",
            match_type="in_list",
            patterns=["ld", "collect2", "lld", "ld.lld"],
        ),
        RuleConfig(
            major_category="BuildSystem",
            category="Ninja",
            priority=190,
            match_field="current_cmd_name",
            match_type="exact",
            patterns="ninja",
        ),
        RuleConfig(
            major_category="Scripting",
            category="ShellScriptFile",
            priority=62,
            match_field="current_cmd_full",
            match_type="regex",
            patterns=r"\.sh(\s|$|')",
        ),
        RuleConfig(
            major_category="Scripting",
            category="ShellInteractiveOrDirect",
            priority=61,
            match_field="current_cmd_name",
            match_type="in_list",
            patterns=["bash", "sh", "zsh"],
        ),
        RuleConfig(
            major_category="Scripting",
            category="ShellScriptFile",
            priority=45,
            match_field="current_cmd_name",
            match_type="exact",
            patterns=".sh",
        ),

    ]
    test_rules.sort(key=lambda r: r.priority, reverse=True)

    # Create a mock AppConfig object. Monitor and Projects can be empty for this test.
    mock_app_config = AppConfig(
        monitor=MonitorConfig(
            # [monitor.general]
            default_jobs=[],
            skip_plots=True,
            log_root_dir=Path("/tmp"),
            categorization_cache_size=1000,
            # [monitor.collection]
            interval_seconds=1.0,
            metric_type="pss_psutil",
            pss_collector_mode="full_scan",
            # [monitor.scheduling]
            scheduling_policy="adaptive",
            monitor_core=0,
            manual_build_cores="",
            manual_monitoring_cores="",
        ),
        projects=[],
        rules=test_rules,
    )

    # Use monkeypatch to directly set the internal _CONFIG object in the config module.
    # This ensures that any call to config.get_config() during this test will return our mock object.
    # We also set it to None after the test to ensure clean state for other tests.
    monkeypatch.setattr("mymonitor.config.manager._CONFIG", mock_app_config)
    yield
    monkeypatch.setattr("mymonitor.config.manager._CONFIG", None)


# --- Tests for get_process_category ---


@pytest.mark.parametrize(
    "cmd_name, cmd_full, expected_major, expected_minor",
    [
        # Test case 1: Simple C++ compiler call
        (
            "clang++",
            "/usr/bin/clang++ -c main.cpp -o main.o",
            "CPP_Driver",
            "Driver_Compile",
        ),
        # Test case 2: Internal compiler process
        (
            "cc1plus",
            "/usr/lib/gcc/cc1plus -quiet main.cpp",
            "CPP_Compile",
            "GCCInternalCompiler",
        ),
        # Test case 3: Linker call
        ("ld", "/usr/bin/ld -o my_app main.o lib.a", "CPP_Link", "DirectLinker"),
        # Test case 4: A command wrapped in sh -c
        ("sh", 'sh -c "gcc -c test.c -o test.o"', "CPP_Driver", "Driver_Compile"),
        # NEW: Add a test case for sh -c with a quoted path containing spaces.
        # This validates that the shlex-based parsing is working correctly.
        (
            "sh",
            "sh -c \"'/path with spaces/my_script.sh' --arg1\"",
            "Scripting",
            "ShellScriptFile",
        ),
        # Test case 5: A build system tool
        ("ninja", "ninja -C out/Release", "BuildSystem", "Ninja"),
        # Test case 6: An uncategorized command
        (
            "my_unknown_script",
            "/usr/local/bin/my_unknown_script",
            "Other",
            "Other_my_unknown_script",
        ),
        # Test case 7: A shell script file
        ("my_script.sh", "./my_script.sh --all", "Scripting", "ShellScriptFile"),
    ],
)
def test_get_process_category(
    mock_rules_config, cmd_name, cmd_full, expected_major, expected_minor
):
    """
    Tests the get_process_category function with various command inputs.
    The 'mock_rules_config' fixture ensures that a controlled set of rules is used.
    """
    major_cat, minor_cat = get_process_category(cmd_name, cmd_full)
    assert major_cat == expected_major
    assert minor_cat == expected_minor


# --- Tests for plan_cpu_allocation (Updated from determine_build_cpu_affinity) ---


def test_plan_cpu_allocation_adaptive_policy():
    """
    Tests the 'adaptive' policy for CPU allocation.
    """
    plan = plan_cpu_allocation(
        policy="adaptive",
        j_level=4,
        manual_build_cores_str="",
        manual_monitor_cores_str="",
        main_monitor_core=0,
    )
    # The exact cores will depend on the system, but we can check basic structure
    assert plan.build_command_prefix.startswith("taskset -c") or plan.build_command_prefix == ""
    assert "Adaptive" in plan.build_cores_desc
    assert "Adaptive" in plan.monitoring_cores_desc or "Shared" in plan.monitoring_cores_desc


def test_plan_cpu_allocation_manual_policy():
    """
    Tests the 'manual' policy for CPU allocation.
    """
    plan = plan_cpu_allocation(
        policy="manual",
        j_level=4,
        manual_build_cores_str="2,4-6",
        manual_monitor_cores_str="7",
        main_monitor_core=0,
    )
    assert plan.build_command_prefix == "taskset -c 2,4-6 "
    assert "Manual: cores 2,4-6" in plan.build_cores_desc
    assert plan.monitoring_cores == [7]
    assert "Manual: cores 7" in plan.monitoring_cores_desc


def test_plan_cpu_allocation_unknown_policy():
    """
    Tests behavior with an unknown policy.
    """
    plan = plan_cpu_allocation(
        policy="unknown_policy",
        j_level=4,
        manual_build_cores_str="",
        manual_monitor_cores_str="",
        main_monitor_core=0,
    )
    assert plan.build_command_prefix == ""
    assert "unknown policy" in plan.build_cores_desc
    assert "unknown policy" in plan.monitoring_cores_desc
