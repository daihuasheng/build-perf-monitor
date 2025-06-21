import pytest
from pathlib import Path

# Import the function to be tested and the config loader
from mymonitor.process_utils import get_process_category
from mymonitor import config


@pytest.fixture
def app_config_real_rules(monkeypatch):
    """
    Fixture to load the real application configuration from conf/rules.toml.
    It uses monkeypatch to ensure that the config is reloaded for each test,
    providing isolation.
    """
    # The root of the project where conf/ is located.
    # This assumes tests are run from the project root.
    project_root = Path(__file__).parent.parent
    config_file_path = project_root / "conf" / "config.toml"

    # 1. Force the config module to use the real config.toml file path.
    monkeypatch.setattr("mymonitor.config._CONFIG_FILE_PATH", config_file_path)
    # 2. IMPORTANT: Reset the global _CONFIG singleton to None so that get_config()
    #    is forced to reload from the real path.
    monkeypatch.setattr(config, "_CONFIG", None)

    # Yield the loaded config to the test function.
    yield config.get_config()

    # Teardown: Reset the singleton again after the test to ensure no state
    # leaks to other test files.
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
    ("ninja", "ninja -C out/Default chrome", "BuildSystem", "Ninja"),
    (
        "soong_build",
        "/usr/bin/soong_build --soong-out ...",
        "BuildSystem",
        "SoongBuild",
    ),
    ("ld", "/usr/bin/ld -o my_app main.o", "CPP_Link", "DirectLinker"),
    ("as", "/usr/bin/as -o main.o main.s", "CPP_Assemble", "DirectAssembler"),
    # --- Medium Priority: Driver-based actions ---
    ("gcc", "gcc -E source.c", "CPP_Driver", "Driver_Preprocessing"),
    ("g++", "g++ -S source.cpp", "CPP_Driver", "Driver_SourceToAsm"),
    ("clang++", "clang++ -c my_file.s -o my_file.o", "CPP_Driver", "Driver_AsmToObj"),
    ("cc", "cc -c main.c -o main.o", "CPP_Driver", "Driver_Compile"),
    ("gcc", "gcc main.o utils.o -o my_program", "CPP_Driver", "Driver_Link"),
    # --- Low Priority: Fallback Drivers ---
    ("gcc", "gcc --version", "CPP_Driver", "Driver_GCC_Fallback"),
    ("clang", "clang --help", "CPP_Driver", "Driver_Clang_Fallback"),
    # --- Java and Rust ---
    (
        "java",
        "/usr/bin/java -jar compiler.jar ...",
        "Java_Compile",
        "Java_CompileAndDex",
    ),
    ("rustc", "rustc --crate-name my_lib src/lib.rs", "Rust_Compile", "Rust_Compiler"),
    # --- Scripting ---
    ("python3.12", "python3.12 my_script.py --arg", "Scripting", "Python"),
    ("bash", "bash /path/to/my_script.sh --arg1", "Scripting", "ShellScriptFile"),
    ("my_script.sh", "./my_script.sh", "Scripting", "ShellScriptFile"),
    (
        "bash",
        "bash -c 'echo hello'",
        "OSUtilities",
        "Generic",
    ),  # This should NOT match sh -c
    (
        "sh",
        "sh -c 'gcc -c main.c'",
        "CPP_Driver",
        "Driver_Compile",
    ),  # Test sh -c unwrapping
    # --- Utilities and Other Tools ---
    ("ar", "ar rcs libmylib.a file1.o file2.o", "DevelopmentTools", "ArchiveAR"),
    ("zip", "zip -r archive.zip my_folder", "DevelopmentTools", "ArchiveZIP"),
    ("aapt2", "aapt2 link ...", "DevelopmentTools", "AndroidResource"),
    ("cp", "cp file1.txt file2.txt", "OSUtilities", "Generic"),
    # --- Ignored and Uncategorized ---
    (
        "code",
        "/usr/share/code/bin/../code --some-arg",
        "Ignored",
        "VSCodeServer",
    ),  # Example based on contains rule
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
