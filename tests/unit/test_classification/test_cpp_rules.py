"""
Comprehensive tests for CPP classification rules.

Tests the CPP_COMPILE and CPP_LINK classification rules to ensure
they correctly identify different types of C/C++ build operations.
"""

import pytest
from unittest.mock import patch

from mymonitor.classification.classifier import get_process_category


@pytest.mark.unit
class TestCPPCompileRules:
    """Test cases for CPP_COMPILE classification rules."""

    def test_gcc_frontend_internal(self):
        """Test classification of GCC internal frontend processes."""
        test_cases = [
            ("cc1", "cc1 -quiet -v main.c", "CPP_COMPILE", "Frontend_GCC"),
            ("cc1plus", "cc1plus -quiet -v main.cpp", "CPP_COMPILE", "Frontend_GCC"),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"

    def test_clang_frontend_internal(self):
        """Test classification of Clang internal frontend processes."""
        test_cases = [
            (
                "clang",
                "clang -cc1 -triple x86_64-unknown-linux-gnu main.c",
                "CPP_COMPILE",
                "Frontend_Clang",
            ),
            (
                "clang++",
                "clang++ -cc1 -emit-obj main.cpp",
                "CPP_COMPILE",
                "Frontend_Clang",
            ),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"

    def test_clang_full_compilation(self):
        """Test classification of Clang full compilation processes."""
        test_cases = [
            (
                "clang++",
                "clang++ -c -O2 -std=c++17 src/module.cpp -o build/module.o",
                "CPP_COMPILE",
                "Full_Clang",
            ),
            (
                "clang",
                "clang -c -O3 -march=native main.c -o main.o",
                "CPP_COMPILE",
                "Full_Clang",
            ),
            (
                "clang++",
                "clang++ -c -g -O0 -std=c++14 widget.cpp -o widget.o",
                "CPP_COMPILE",
                "Full_Clang",
            ),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"

    def test_gcc_full_compilation(self):
        """Test classification of GCC full compilation processes."""
        test_cases = [
            (
                "gcc",
                "gcc -c -O2 -Wall src/utils.c -o build/utils.o",
                "CPP_COMPILE",
                "Full_GCC",
            ),
            (
                "g++",
                "g++ -c -O3 -std=c++17 main.cpp -o main.o",
                "CPP_COMPILE",
                "Full_GCC",
            ),
            (
                "g++-9",
                "g++-9 -c -g -O0 -std=c++11 test.cpp -o test.o",
                "CPP_COMPILE",
                "Full_Versioned",
            ),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"

    def test_cross_compilation(self):
        """Test classification of cross-compilation processes."""
        test_cases = [
            (
                "arm-linux-gnueabihf-gcc",
                "arm-linux-gnueabihf-gcc -c -O2 embedded.c -o embedded.o",
                "CPP_COMPILE",
                "Full_Cross",
            ),
            (
                "aarch64-linux-gnu-g++",
                "aarch64-linux-gnu-g++ -c -O3 driver.cpp -o driver.o",
                "CPP_COMPILE",
                "Full_Cross",
            ),
            (
                "x86_64-w64-mingw32-gcc",
                "x86_64-w64-mingw32-gcc -c -O2 windows_port.c -o windows_port.o",
                "CPP_COMPILE",
                "Full_Cross",
            ),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"

    def test_specialized_compilers(self):
        """Test classification of specialized compiler processes."""
        test_cases = [
            (
                "nvcc",
                "nvcc -c -O3 -arch=sm_75 kernel.cu -o kernel.o",
                "CPP_COMPILE",
                "Full_Specialized",
            ),
            (
                "icpc",
                "icpc -c -O3 -xHost optimized.cpp -o optimized.o",
                "CPP_COMPILE",
                "Full_Specialized",
            ),
            (
                "pgcc",
                "pgcc -c -O4 -acc parallel_code.c -o parallel_code.o",
                "CPP_COMPILE",
                "Full_Specialized",
            ),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"

    def test_preprocessing(self):
        """Test classification of preprocessing operations."""
        test_cases = [
            (
                "gcc",
                "gcc -E -I./include -DDEBUG=1 main.c",
                "CPP_COMPILE",
                "Preprocess_Any",
            ),
            (
                "clang++",
                "clang++ -E -std=c++17 -I./src template_heavy.cpp",
                "CPP_COMPILE",
                "Preprocess_Any",
            ),
            (
                "cpp",
                "cpp -I./include -D_GNU_SOURCE source.c",
                "CPP_COMPILE",
                "Preprocess_CPP",
            ),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"

    def test_backend_code_generation(self):
        """Test classification of backend code generation operations."""
        test_cases = [
            (
                "gcc",
                "gcc -S -O3 -march=native math_intensive.c -o math_intensive.s",
                "CPP_COMPILE",
                "Backend_Any",
            ),
            (
                "clang++",
                "clang++ -S -O2 -std=c++17 algorithm.cpp -o algorithm.s",
                "CPP_COMPILE",
                "Backend_Any",
            ),
            (
                "g++",
                "g++ -S -Og -g debug_version.cpp -o debug_version.s",
                "CPP_COMPILE",
                "Backend_Any",
            ),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"

    def test_assembly_stage(self):
        """Test classification of assembly stage operations."""
        test_cases = [
            ("as", "as --64 -o main.o main.s", "CPP_COMPILE", "Assembly_Any"),
            (
                "gas",
                "gas --32 --march=i686 -o legacy.o legacy.s",
                "CPP_COMPILE",
                "Assembly_Any",
            ),
            (
                "arm-linux-gnueabihf-as",
                "arm-linux-gnueabihf-as -march=armv7-a -o embedded.o embedded.s",
                "CPP_COMPILE",
                "Assembly_Any",
            ),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"


@pytest.mark.unit
class TestCPPLinkRules:
    """Test cases for CPP_LINK classification rules."""

    def test_lto_optimizer(self):
        """Test classification of LTO optimizer processes."""
        test_cases = [
            (
                "lto-wrapper",
                "lto-wrapper --plugin-opt=-fresolution=/tmp/ccXXXXXX.res",
                "CPP_LINK",
                "LTO_Optimizer",
            ),
            (
                "lto1",
                "lto1 -fltrans-output-list=/tmp/ccXXXXXX.ltrans.out",
                "CPP_LINK",
                "LTO_Optimizer",
            ),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"

    def test_executable_linking(self):
        """Test classification of executable linking operations."""
        test_cases = [
            ("gcc", "gcc -o myapp main.o utils.o -lm", "CPP_LINK", "Executable_Driver"),
            (
                "clang++",
                "clang++ -o server main.o network.o -lboost_system",
                "CPP_LINK",
                "Executable_Driver",
            ),
            (
                "g++",
                "g++ -o game_engine core.o graphics.o -lSDL2 -lGL",
                "CPP_LINK",
                "Executable_Driver",
            ),
            (
                "/usr/bin/gcc",
                "/usr/bin/gcc -o myapp main.o",
                "CPP_LINK",
                "Executable_Driver",
            ),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"

    def test_shared_library_linking(self):
        """Test classification of shared library linking operations."""
        test_cases = [
            (
                "gcc",
                "gcc -shared -fPIC -o libutils.so.1.0 utils.o",
                "CPP_LINK",
                "SharedLib_Driver",
            ),
            (
                "clang++",
                "clang++ -shared -fPIC -Wl,-soname,libmylib.so.1 -o libmylib.so.1.2.3 obj1.o",
                "CPP_LINK",
                "SharedLib_Driver",
            ),
            (
                "g++",
                "g++ -shared -fPIC -O2 -o libgraphics.so graphics.o -lGL",
                "CPP_LINK",
                "SharedLib_Driver",
            ),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"

    def test_static_library_creation(self):
        """Test classification of static library creation operations."""
        test_cases = [
            (
                "ar",
                "ar rcs libutils.a utils.o string_ops.o",
                "CPP_LINK",
                "StaticLib_Archiver",
            ),
            (
                "llvm-ar",
                "llvm-ar rcs libcore.a core.o memory.o",
                "CPP_LINK",
                "StaticLib_Archiver",
            ),
            (
                "arm-linux-gnueabihf-ar",
                "arm-linux-gnueabihf-ar cr libembedded.a sensor.o",
                "CPP_LINK",
                "StaticLib_Archiver",
            ),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"

    def test_direct_linker_invocation(self):
        """Test classification of direct linker invocations."""
        test_cases = [
            (
                "ld",
                "ld -dynamic-linker /lib64/ld-linux-x86-64.so.2 -o myapp main.o",
                "CPP_LINK",
                "Direct_Linker",
            ),
            (
                "ld.lld",
                "ld.lld --hash-style=gnu -o server main.o -lc",
                "CPP_LINK",
                "Direct_Linker",
            ),
            (
                "ld.gold",
                "ld.gold --threads --thread-count=4 -o optimized_app main.o",
                "CPP_LINK",
                "Direct_Linker",
            ),
        ]

        for cmd_name, cmd_full, expected_major, expected_minor in test_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == expected_major
            ), f"Failed for {cmd_name}: got {major}, expected {expected_major}"
            assert (
                minor == expected_minor
            ), f"Failed for {cmd_name}: got {minor}, expected {expected_minor}"


@pytest.mark.unit
class TestCPPRuleEdgeCases:
    """Test edge cases and boundary conditions for CPP rules."""

    def test_compilation_vs_linking_distinction(self):
        """Test that compilation and linking are correctly distinguished."""
        # These should be classified as compilation, NOT linking
        compilation_cases = [
            ("gcc", "gcc -c -O2 main.c -o main.o"),  # Has -c flag
            (
                "g++",
                "g++ -c -std=c++17 src/module.cpp -o build/module.o",
            ),  # Has -c flag
            ("clang", "clang -c -O3 -march=native main.c -o main.o"),  # Has -c flag
        ]

        for cmd_name, cmd_full in compilation_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == "CPP_COMPILE"
            ), f"Compilation incorrectly classified as {major}: {cmd_full}"
            assert (
                "Full_" in minor
            ), f"Compilation should be Full_* category, got {minor}: {cmd_full}"

        # These should be classified as linking, NOT compilation
        linking_cases = [
            ("gcc", "gcc -o myapp main.o utils.o -lm"),  # No -c flag, has -o
            (
                "clang++",
                "clang++ -o server main.o network.o -lboost_system",
            ),  # No -c flag, has -o
        ]

        for cmd_name, cmd_full in linking_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == "CPP_LINK"
            ), f"Linking incorrectly classified as {major}: {cmd_full}"

    def test_shared_library_vs_executable_linking(self):
        """Test distinction between shared library and executable linking."""
        # Shared library linking should have higher priority
        shared_lib_cases = [
            ("gcc", "gcc -shared -fPIC -o libutils.so utils.o"),
            ("clang++", "clang++ -shared -o libtest.so test.o"),
        ]

        for cmd_name, cmd_full in shared_lib_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == "CPP_LINK"
            ), f"Shared lib linking incorrectly classified as {major}: {cmd_full}"
            assert (
                minor == "SharedLib_Driver"
            ), f"Shared lib should be SharedLib_Driver, got {minor}: {cmd_full}"

    def test_preprocessing_vs_compilation(self):
        """Test distinction between preprocessing and compilation."""
        # Preprocessing should be identified correctly
        preprocess_cases = [
            ("gcc", "gcc -E -I./include main.c"),
            ("clang++", "clang++ -E -std=c++17 template.cpp"),
        ]

        for cmd_name, cmd_full in preprocess_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == "CPP_COMPILE"
            ), f"Preprocessing incorrectly classified as {major}: {cmd_full}"
            assert (
                minor == "Preprocess_Any"
            ), f"Preprocessing should be Preprocess_Any, got {minor}: {cmd_full}"

    def test_assembly_vs_compilation(self):
        """Test distinction between assembly generation and compilation."""
        # Assembly generation should be identified correctly
        assembly_cases = [
            ("gcc", "gcc -S -O2 main.c -o main.s"),
            ("clang++", "clang++ -S -std=c++17 algorithm.cpp"),
        ]

        for cmd_name, cmd_full in assembly_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == "CPP_COMPILE"
            ), f"Assembly generation incorrectly classified as {major}: {cmd_full}"
            assert (
                minor == "Backend_Any"
            ), f"Assembly generation should be Backend_Any, got {minor}: {cmd_full}"

    def test_fallback_generic_rules(self):
        """Test fallback to generic rules for unspecified operations."""
        # These should fall back to generic rules
        generic_cases = [
            ("gcc", "gcc"),  # Just compiler name
            ("clang++", "clang++"),  # Just compiler name
        ]

        for cmd_name, cmd_full in generic_cases:
            major, minor = get_process_category(cmd_name, cmd_full)
            assert (
                major == "CPP_COMPILE"
            ), f"Generic case incorrectly classified as {major}: {cmd_full}"
            assert (
                minor == "Full_Generic"
            ), f"Generic case should be Full_Generic, got {minor}: {cmd_full}"
