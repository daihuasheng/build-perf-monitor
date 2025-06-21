import pytest
from pathlib import Path

from mymonitor.data_models import AppConfig, RuleConfig, MonitorConfig
from mymonitor.process_utils import get_process_category, determine_build_cpu_affinity

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
        RuleConfig(major_category="CPP_Compile", category="GCCInternalCompiler", priority=160, match_field="current_cmd_name", match_type="in_list", patterns=["cc1", "cc1plus"]),
        RuleConfig(major_category="CPP_Driver", category="Driver_Compile", priority=125, match_field="current_cmd_full", match_type="regex", pattern=r"^(?:.*/)?(clang\+\+|clang|gcc|g\+\+|cc)\b.*\s-c(\s|$)"),
        RuleConfig(major_category="CPP_Link", category="DirectLinker", priority=155, match_field="current_cmd_name", match_type="in_list", patterns=["ld", "collect2", "lld", "ld.lld"]),
        RuleConfig(major_category="BuildSystem", category="Ninja", priority=190, match_field="current_cmd_name", match_type="exact", pattern="ninja"),
        RuleConfig(major_category="Scripting", category="ShellScriptFile", priority=45, match_field="current_cmd_name", match_type="endswith", pattern=".sh"),
    ]
    test_rules.sort(key=lambda r: r.priority, reverse=True)

    # Create a mock AppConfig object. Monitor and Projects can be empty for this test.
    mock_app_config = AppConfig(
        monitor=MonitorConfig(interval_seconds=1, default_jobs=[], metric_type="", monitor_core=-1, build_cores_policy="", specific_build_cores="", skip_plots=True, log_root_dir=Path("/tmp")),
        projects=[],
        rules=test_rules
    )

    # Use monkeypatch to directly set the internal _CONFIG object in the config module.
    # This ensures that any call to config.get_config() during this test will return our mock object.
    # We also set it to None after the test to ensure clean state for other tests.
    monkeypatch.setattr("mymonitor.config._CONFIG", mock_app_config)
    yield
    monkeypatch.setattr("mymonitor.config._CONFIG", None)


# --- Tests for get_process_category ---

@pytest.mark.parametrize(
    "cmd_name, cmd_full, expected_major, expected_minor",
    [
        # Test case 1: Simple C++ compiler call
        ("clang++", "/usr/bin/clang++ -c main.cpp -o main.o", "CPP_Driver", "Driver_Compile"),

        # Test case 2: Internal compiler process
        ("cc1plus", "/usr/lib/gcc/cc1plus -quiet main.cpp", "CPP_Compile", "GCCInternalCompiler"),

        # Test case 3: Linker call (This one was already passing)
        ("ld", "/usr/bin/ld -o my_app main.o lib.a", "CPP_Link", "DirectLinker"),

        # Test case 4: A command wrapped in sh -c
        ("sh", 'sh -c "gcc -c test.c -o test.o"', "CPP_Driver", "Driver_Compile"),

        # Test case 5: A build system tool (Passing)
        ("ninja", "ninja -C out/Release", "BuildSystem", "Ninja"),

        # Test case 6: An uncategorized command (Passing)
        ("my_unknown_script", "/usr/local/bin/my_unknown_script", "Other", "Other_my_unknown_script"),

        # Test case 7: A shell script file (Passing)
        ("my_script.sh", "./my_script.sh --all", "Scripting", "ShellScriptFile"),
    ],
)
def test_get_process_category(mock_rules_config, cmd_name, cmd_full, expected_major, expected_minor):
    """
    Tests the get_process_category function with various command inputs.
    The 'mock_rules_config' fixture ensures that a controlled set of rules is used.
    """
    major_cat, minor_cat = get_process_category(cmd_name, cmd_full)
    assert major_cat == expected_major
    assert minor_cat == expected_minor


# --- Tests for determine_build_cpu_affinity (No changes needed here) ---

def test_determine_cpu_affinity_all_others_policy():
    """
    Tests the 'all_others' policy for CPU affinity.
    """
    prefix, desc = determine_build_cpu_affinity(
        build_cpu_cores_policy="all_others",
        specific_build_cores_str=None,
        monitor_core_id=0,
        taskset_available=True,
        total_cores_available=8,
    )
    assert prefix == "taskset -c 1-7 "
    assert "All Other Cores (cores: 1-7)" in desc

def test_determine_cpu_affinity_specific_policy():
    """
    Tests the 'specific' policy for CPU affinity.
    """
    prefix, desc = determine_build_cpu_affinity(
        build_cpu_cores_policy="specific",
        specific_build_cores_str="2,4-6",
        monitor_core_id=0,
        taskset_available=True,
        total_cores_available=8,
    )
    assert prefix == "taskset -c 2,4-6 "
    assert "Specific (cores: 2,4-6)" in desc

def test_determine_cpu_affinity_none_policy():
    """
    Tests the 'none' policy for CPU affinity.
    """
    prefix, desc = determine_build_cpu_affinity(
        build_cpu_cores_policy="none",
        specific_build_cores_str=None,
        monitor_core_id=0,
        taskset_available=True,
        total_cores_available=8,
    )
    assert prefix == ""
    assert "All Available" in desc

def test_determine_cpu_affinity_taskset_unavailable():
    """
    Tests behavior when taskset command is not available.
    """
    prefix, desc = determine_build_cpu_affinity(
        build_cpu_cores_policy="all_others",
        specific_build_cores_str=None,
        monitor_core_id=0,
        taskset_available=False,
        total_cores_available=8,
    )
    assert prefix == ""
    assert "All Available (taskset not available)" in desc
