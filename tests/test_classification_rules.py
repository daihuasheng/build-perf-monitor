import pytest
from pathlib import Path
import tomllib

# Import the function to be tested and the necessary data models
from mymonitor.process_utils import get_process_category
from mymonitor.data_models import AppConfig, RuleConfig, MonitorConfig
from mymonitor import config


@pytest.fixture
def app_config_real_rules(monkeypatch):
    """
    Fixture that loads the REAL rules from conf/rules.toml and provides them
    in a controlled AppConfig object.

    This approach isolates the test from the main config.toml and projects.toml,
    making it robust against changes in those files. It focuses solely on
    testing the categorization rules.
    """
    project_root = Path(__file__).parent.parent
    rules_file_path = project_root / "conf" / "rules.toml"

    with open(rules_file_path, "rb") as f:
        rules_data = tomllib.load(f)

    # Load and sort rules just like the real config loader does.
    rules_config = []
    for rule_data in rules_data.get("rules", []):
        # Manually construct RuleConfig to handle pattern/patterns compatibility
        patterns_value = rule_data.get("patterns", rule_data.get("pattern", ""))
        pattern_value = patterns_value if isinstance(patterns_value, str) else None
        
        rule = RuleConfig(
            priority=rule_data.get("priority", 0),
            major_category=rule_data.get("major_category", ""),
            category=rule_data.get("category", ""),
            match_field=rule_data.get("match_field", ""),
            match_type=rule_data.get("match_type", ""),
            patterns=patterns_value,
            pattern=pattern_value,
            comment=rule_data.get("comment", "")
        )
        rules_config.append(rule)
    
    rules_config.sort(key=lambda r: r.priority, reverse=True)

    # Create a minimal, mock AppConfig. We only need the 'rules' part to be real.
    mock_app_config = AppConfig(
        monitor=MonitorConfig(  # Dummy data for MonitorConfig
            default_jobs=[],
            skip_plots=True,
            log_root_dir=Path("/tmp"),
            categorization_cache_size=1,
            interval_seconds=1.0,
            metric_type="pss_kb",
            pss_collector_mode="full_scan",
            scheduling_policy="adaptive",
            monitor_core=0,
            manual_build_cores="",
            manual_monitoring_cores="",
        ),
        projects=[],  # Projects are not needed for this test
        rules=rules_config,  # Use the real rules
    )

    # Inject the controlled config object, bypassing the file loader.
    monkeypatch.setattr(config, "_CONFIG", mock_app_config)
    yield mock_app_config
    # Clean up after the test.
    monkeypatch.setattr(config, "_CONFIG", None)


# A comprehensive list of test cases to validate the rules in conf/rules.toml
# Each tuple is: (command_name, full_command, expected_major_category, expected_minor_category)
CLASSIFICATION_TEST_CASES = [
    # --- High Priority: Specific Internals & Build Systems ---
    (
        "cc1plus",
        "/usr/lib/gcc/x86_64-linux-gnu/11/cc1plus ...",
        "CPP_Compile",
        "GCCInternalCompiler",
    ),
    ("clang", "clang -cc1 -O2 ...", "CPP_Compile", "ClangInternalCompiler"),
    ("go", "/usr/bin/go tool compile -o ...", "Go_Compile", "GoInternalCompiler"),
    ("go", "/usr/bin/go tool link -o ...", "Go_Link", "GoInternalLinker"),
    ("ninja", "ninja -C out/Default chrome", "BuildSystem", "Ninja"),
    (
        "soong_build",
        "/usr/bin/soong_build --soong-out ...",
        "BuildSystem",
        "SoongBuild",
    ),
    (
        "sbox",
        "/path/to/out/host/linux-x86/bin/sbox --sandbox-path ...",
        "BuildSystem",
        "SoongSandbox",
    ),
    ("ld", "/usr/bin/ld -o my_app main.o", "CPP_Link", "DirectLinker"),
    ("as", "/usr/bin/as -o main.o main.s", "CPP_Assemble", "DirectAssembler"),
    # --- Medium Priority: Driver-based actions ---
    ("gcc", "gcc -E source.c", "CPP_Preprocess", "Driver_Preprocessing"),
    ("g++", "g++ -S source.cpp", "CPP_Assemble", "Driver_SourceToAsm"),
    ("cc", "cc -c main.c -o main.o", "CPP_Compile", "Driver_Compile"),
    ("gcc", "gcc main.o utils.o -o my_program", "CPP_Link", "Driver_Link"),
    (
        "compile",
        "/some/wrapper/compile gcc -c main.c",
        "CPP_Compile",
        "GenericCompileWrapper",
    ),
    # --- Low Priority: Fallback Drivers ---
    ("gcc", "gcc --version", "CPP_Driver", "Driver_GCC_Fallback"),
    ("clang", "clang --help", "CPP_Driver", "Driver_Clang_Fallback"),
    ("go", "go version", "Go_Compile", "Go_Compiler_Fallback"),
    # --- Java and Rust ---
    (
        "java",
        "/usr/bin/java -jar compiler.jar ...",
        "Java_Compile",
        "Java_CompileAndDex",
    ),
    ("javac", "/path/to/jdk/bin/javac MyClass.java", "Java_Compile", "Javac"),
    (
        "rustc",
        "/home/.rustup/toolchains/stable/bin/rustc --crate-name my_lib src/lib.rs",
        "Rust_Compile",
        "Rust_Compiler",
    ),
    # --- Scripting ---
    ("python3.12", "python3.12 my_script.py --arg", "Scripting", "Python"),
    ("bash", "bash /path/to/my_script.sh --arg1", "Scripting", "ShellScriptFile"),
    ("my_script.sh", "./my_script.sh", "Scripting", "ShellScriptFile"),
    (
        "sh",
        "sh -c 'gcc -c main.c'",
        "CPP_Compile",
        "Driver_Compile",
    ),  # Test sh -c unwrapping
    # --- Utilities and Other Tools (with full paths) ---
    ("ar", "/usr/bin/ar rcs libmylib.a file1.o", "DevelopmentTools", "ArchiveAR"),
    ("zip", "/usr/bin/zip -r archive.zip my_folder", "DevelopmentTools", "ArchiveZIP"),
    (
        "aapt2",
        "/path/to/android/sdk/aapt2 link ...",
        "DevelopmentTools",
        "AndroidSDKTools",
    ),
    ("cp", "/bin/cp file1.txt file2.txt", "OSUtilities", "Generic_Path"),
    ("_mkdir", "_mkdir /tmp/foo", "OSUtilities", "Generic_Path"),  # Test with prefix
    # --- Testing tools ---
    ("gotestmain", "/path/to/gotestmain -test.v", "Testing", "GoTestRunner"),
    ("mytest.test", "./mytest.test -test.run TestMyFeature", "Testing", "GoTestBinary"),
    # --- Ignored and Uncategorized ---
    ("code", "/usr/share/code/bin/../code --some-arg", "Ignored", "VSCodeServer"),
    ("my_custom_tool", "/opt/bin/my_custom_tool", "Other", "Other_my_custom_tool"),
]


@pytest.mark.parametrize(
    "cmd_name, cmd_full, expected_major, expected_minor", CLASSIFICATION_TEST_CASES
)
def test_real_classification_rules(
    app_config_real_rules, cmd_name, cmd_full, expected_major, expected_minor
):
    """
    Tests the get_process_category function against the REAL rules from conf/rules.toml.

    This test is crucial for validating that the rule priorities and patterns are correct
    and that changes to the rules.toml file don't break existing classifications.
    """
    # The app_config_real_rules fixture ensures that config.get_config() returns
    # the real, loaded configuration for this test.
    major_cat, minor_cat = get_process_category(cmd_name, cmd_full)

    assert major_cat == expected_major
    assert minor_cat == expected_minor
