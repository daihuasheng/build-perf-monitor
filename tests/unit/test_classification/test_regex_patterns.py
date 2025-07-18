"""
Tests for regex patterns used in classification rules.

This module tests the regex patterns in isolation to ensure they
work correctly before being integrated into the classification system.
"""

import pytest
import re
from typing import List, Tuple


@pytest.mark.unit
class TestCPPLinkRegexPatterns:
    """Test regex patterns for CPP linking rules."""

    def test_executable_link_pattern(self):
        """Test the regex pattern for executable linking."""
        # The optimized pattern from our analysis
        pattern = r"^(?:.*/)?(?:gcc|g\+\+|clang|clang\+\+)\s+(?!.*-c\s)(?!.*-S\s)(?!.*-E\s)(?!.*-shared\s).*-o\s"

        # Test cases: (command, should_match)
        test_cases = [
            # Should match (linking operations)
            ("gcc -o myapp main.o utils.o -lm", True),
            ("g++ -o server main.o network.o -lboost_system", True),
            ("clang++ -o game_engine core.o graphics.o -lSDL2", True),
            ("/usr/bin/gcc -o myapp main.o", True),
            ("gcc -O2 -o optimized_app main.o lib1.o -lc", True),
            # Should NOT match (compilation operations)
            ("gcc -c -O2 main.c -o main.o", False),
            ("g++ -c -std=c++17 src/module.cpp -o build/module.o", False),
            ("clang -c -O3 main.c -o main.o", False),
            # Should NOT match (other operations)
            ("gcc -E -I./include main.c", False),
            ("gcc -S -O2 main.c -o main.s", False),
            ("gcc -shared -fPIC -o libutils.so utils.o", False),
            # Edge cases
            ("gcc -omyapp main.o utils.o -lm", False),  # No space after -o
            ("gcc", False),  # Just compiler name
            ("make -j4", False),  # Different tool
        ]

        self._test_pattern(pattern, test_cases, "Executable Link Pattern")

    def test_shared_library_link_pattern(self):
        """Test the regex pattern for shared library linking."""
        # Pattern for shared library linking
        pattern = r"^(?:.*/)?(?:gcc|g\+\+|clang|clang\+\+|[^/]*-(?:linux-)?gnu[^/]*-g(?:cc|\+\+)|[^/]*-clang(?:\+\+)?)\s.*-shared\s"

        test_cases = [
            # Should match (shared library linking)
            ("gcc -shared -fPIC -o libutils.so.1.0 utils.o", True),
            (
                "clang++ -shared -fPIC -Wl,-soname,libmylib.so.1 -o libmylib.so.1.2.3 obj1.o",
                True,
            ),
            ("g++ -shared -fPIC -O2 -o libgraphics.so graphics.o", True),
            ("arm-linux-gnueabihf-gcc -shared -o libembedded.so embedded.o", True),
            # Should NOT match (other operations)
            ("gcc -o myapp main.o utils.o -lm", False),  # Executable linking
            ("gcc -c -O2 main.c -o main.o", False),  # Compilation
            ("gcc -E main.c", False),  # Preprocessing
        ]

        self._test_pattern(pattern, test_cases, "Shared Library Link Pattern")

    def test_static_library_pattern(self):
        """Test the regex pattern for static library creation."""
        pattern = r"^(?:.*/)?(?:ar|llvm-ar|[^/]*-(?:linux-)?gnu[^/]*-ar)(?:\s+.*?)?\s+(?:rcs|cr|cru)(?:\s|$)"

        test_cases = [
            # Should match (static library creation)
            ("ar rcs libutils.a utils.o string_ops.o", True),
            ("llvm-ar rcs libcore.a core.o memory.o", True),
            ("arm-linux-gnueabihf-ar cr libembedded.a sensor.o", True),
            ("/usr/bin/ar rcs libtest.a test.o", True),
            # Should NOT match (other operations)
            ("gcc -o myapp main.o", False),
            ("ar --help", False),  # No archive operation
            ("tar -czf archive.tar.gz files/", False),  # Different tool
        ]

        self._test_pattern(pattern, test_cases, "Static Library Pattern")

    def _test_pattern(
        self, pattern: str, test_cases: List[Tuple[str, bool]], pattern_name: str
    ):
        """Helper method to test a regex pattern against test cases."""
        print(f"\n🧪 Testing {pattern_name}")
        print(f"Pattern: {pattern}")
        print("-" * 80)

        all_correct = True

        for command, should_match in test_cases:
            try:
                match = bool(re.search(pattern, command))
                status = "✅" if match == should_match else "❌"
                expected = "SHOULD MATCH" if should_match else "SHOULD NOT MATCH"
                actual = "MATCHED" if match else "NO MATCH"

                print(f"{status} {expected:15} | {actual:10} | {command}")

                if match != should_match:
                    all_correct = False

            except re.error as e:
                print(f"❌ REGEX ERROR: {e} | {command}")
                all_correct = False

        assert all_correct, f"{pattern_name} failed some test cases"


@pytest.mark.unit
class TestCPPCompileRegexPatterns:
    """Test regex patterns for CPP compilation rules."""

    def test_gcc_full_compile_pattern(self):
        """Test the regex pattern for GCC full compilation."""
        pattern = r"^(?:.*/)?(?:gcc|g\+\+|[^/]*-(?:linux-)?gnu[^/]*-g(?:cc|\+\+))(?=\s)(?!.*\s+-E(?:\s|$))(?!.*\s+-S(?:\s|$)).*\s+-c(?:\s|$)"

        test_cases = [
            # Should match (GCC compilation)
            ("gcc -c -O2 -Wall src/utils.c -o build/utils.o", True),
            ("g++ -c -O3 -std=c++17 main.cpp -o main.o", True),
            ("/usr/bin/gcc -c -g test.c -o test.o", True),
            ("arm-linux-gnueabihf-gcc -c embedded.c -o embedded.o", True),
            # Should NOT match (other operations)
            ("gcc -o myapp main.o utils.o -lm", False),  # Linking
            ("gcc -E -I./include main.c", False),  # Preprocessing
            ("gcc -S -O2 main.c -o main.s", False),  # Assembly generation
            ("clang -c main.c -o main.o", False),  # Clang (different rule)
        ]

        self._test_pattern(pattern, test_cases, "GCC Full Compile Pattern")

    def test_clang_full_compile_pattern(self):
        """Test the regex pattern for Clang full compilation."""
        pattern = r"^(?:.*/)?(?:clang|clang\+\+|[^/]*-clang(?:\+\+)?)(?=\s)(?!.*\s+-E(?:\s|$))(?!.*\s+-S(?:\s|$)).*\s+-c(?:\s|$)"

        test_cases = [
            # Should match (Clang compilation)
            ("clang++ -c -O2 -std=c++17 src/module.cpp -o build/module.o", True),
            ("clang -c -O3 -march=native main.c -o main.o", True),
            ("/usr/bin/clang++ -c -g widget.cpp -o widget.o", True),
            # Should NOT match (other operations)
            ("clang++ -o server main.o network.o", False),  # Linking
            ("clang++ -E -std=c++17 template.cpp", False),  # Preprocessing
            ("clang++ -S algorithm.cpp", False),  # Assembly generation
            ("gcc -c main.c -o main.o", False),  # GCC (different rule)
        ]

        self._test_pattern(pattern, test_cases, "Clang Full Compile Pattern")

    def test_preprocessing_pattern(self):
        """Test the regex pattern for preprocessing operations."""
        pattern = r"^(?:.*/)?(?:cpp|gcc|g\+\+|clang|clang\+\+|[^/]*-(?:linux-)?gnu[^/]*-g(?:cc|\+\+)|[^/]*-clang(?:\+\+)?)(?:\s+.*?)?\s+-E(?:\s|$)"

        test_cases = [
            # Should match (preprocessing)
            ("gcc -E -I./include -DDEBUG=1 main.c", True),
            ("clang++ -E -std=c++17 -I./src template_heavy.cpp", True),
            ("cpp -E -I./include -D_GNU_SOURCE source.c", True),
            ("arm-linux-gnueabihf-gcc -E embedded.c", True),
            # Should NOT match (other operations)
            ("gcc -c -O2 main.c -o main.o", False),  # Compilation
            ("gcc -o myapp main.o", False),  # Linking
            ("gcc -S main.c", False),  # Assembly generation
        ]

        self._test_pattern(pattern, test_cases, "Preprocessing Pattern")

    def test_assembly_generation_pattern(self):
        """Test the regex pattern for assembly generation."""
        pattern = r"^(?:.*/)?(?:gcc|g\+\+|clang|clang\+\+|[^/]*-(?:linux-)?gnu[^/]*-g(?:cc|\+\+)|[^/]*-clang(?:\+\+)?)(?:\s+.*?)?\s+-S(?:\s|$)"

        test_cases = [
            # Should match (assembly generation)
            ("gcc -S -O3 -march=native math_intensive.c -o math_intensive.s", True),
            ("clang++ -S -O2 -std=c++17 algorithm.cpp -o algorithm.s", True),
            ("g++ -S -Og -g debug_version.cpp -o debug_version.s", True),
            # Should NOT match (other operations)
            ("gcc -c -O2 main.c -o main.o", False),  # Compilation
            ("gcc -o myapp main.o", False),  # Linking
            ("gcc -E main.c", False),  # Preprocessing
        ]

        self._test_pattern(pattern, test_cases, "Assembly Generation Pattern")

    def _test_pattern(
        self, pattern: str, test_cases: List[Tuple[str, bool]], pattern_name: str
    ):
        """Helper method to test a regex pattern against test cases."""
        print(f"\n🧪 Testing {pattern_name}")
        print(f"Pattern: {pattern}")
        print("-" * 80)

        all_correct = True

        for command, should_match in test_cases:
            try:
                match = bool(re.search(pattern, command))
                status = "✅" if match == should_match else "❌"
                expected = "SHOULD MATCH" if should_match else "SHOULD NOT MATCH"
                actual = "MATCHED" if match else "NO MATCH"

                print(f"{status} {expected:15} | {actual:10} | {command}")

                if match != should_match:
                    all_correct = False

            except re.error as e:
                print(f"❌ REGEX ERROR: {e} | {command}")
                all_correct = False

        assert all_correct, f"{pattern_name} failed some test cases"


@pytest.mark.unit
class TestRegexPatternEdgeCases:
    """Test edge cases and boundary conditions for regex patterns."""

    def test_path_handling(self):
        """Test that patterns correctly handle full paths."""
        patterns_and_commands = [
            # (pattern, command, should_match)
            (r"^(?:.*/)?(?:gcc|g\+\+)\s.*-o\s", "/usr/bin/gcc -o myapp main.o", True),
            (
                r"^(?:.*/)?(?:gcc|g\+\+)\s.*-o\s",
                "/opt/toolchain/bin/g++ -o server main.o",
                True,
            ),
            (r"^(?:.*/)?(?:gcc|g\+\+)\s.*-o\s", "gcc -o myapp main.o", True),
        ]

        for pattern, command, should_match in patterns_and_commands:
            match = bool(re.search(pattern, command))
            assert match == should_match, f"Path handling failed for: {command}"

    def test_cross_compiler_patterns(self):
        """Test patterns for cross-compilation toolchains."""
        cross_pattern = r"^(?:.*/)?[^/]*-(?:linux-)?gnu[^/]*-g(?:cc|\+\+)(?=\s)(?!.*\s+-E(?:\s|$))(?!.*\s+-S(?:\s|$)).*\s+-c(?:\s|$)"

        test_cases = [
            ("arm-linux-gnueabihf-gcc -c embedded.c -o embedded.o", True),
            ("aarch64-linux-gnu-g++ -c driver.cpp -o driver.o", True),
            (
                "x86_64-w64-mingw32-gcc -c windows.c -o windows.o",
                False,
            ),  # Not gnu toolchain
            ("mips-linux-gnu-gcc -c mips_code.c -o mips_code.o", True),
            ("gcc -c main.c -o main.o", False),  # Not cross-compiler
        ]

        for command, should_match in test_cases:
            match = bool(re.search(cross_pattern, command))
            assert (
                match == should_match
            ), f"Cross-compiler pattern failed for: {command}"

    def test_flag_boundary_conditions(self):
        """Test boundary conditions for flag matching."""
        # Test that -c flag is properly detected with word boundaries
        compile_pattern = r".*\s+-c\s"

        test_cases = [
            ("gcc -c main.c", True),
            ("gcc -c -O2 main.c", True),
            ("gcc -O2 -c main.c", True),
            (
                "gcc -include file.h main.c",
                False,
            ),  # -include contains 'c' but is not -c
            ("gcc -fPIC main.c", False),  # -fPIC contains 'c' but is not -c
        ]

        for command, should_match in test_cases:
            match = bool(re.search(compile_pattern, command))
            assert match == should_match, f"Flag boundary test failed for: {command}"
