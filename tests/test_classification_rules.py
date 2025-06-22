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
