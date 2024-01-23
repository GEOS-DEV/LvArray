from dataclasses import dataclass
import logging
import os
import subprocess
from typing import (
    Any,
    Iterable,
)

import pytest


@dataclass(frozen=True)
class TestCase:
    __test__ = False
    executable: str
    cpp_var_name: str
    expected: str
    breakpoint: int | str = "breakpoint_helper"
    timeout: int = 1


def build_output_str(array_type: str, value_type: str, dimensions: tuple[Any, ...], values: tuple[Any, ...],
                     permutation: tuple[Any, ...], buffer="MallocBuffer") -> str:
    if array_type in ("ArrayOfArrays", "ArrayOfArraysView"):
        shape_args = f"size {dimensions[0]}"
        if array_type == "ArrayOfArrays":
            template_args = f"{value_type}, int, LvArray::{buffer}"
        elif array_type == "ArrayOfArraysView":
            template_args = f"{value_type}, int const, {'true' if 'const' in value_type else 'false'}, LvArray::{buffer}"
    else:
        if "View" in array_type:
            p = str(permutation[-1])
        elif "Slice" in array_type:
            p = len(dimensions) - 1
        else:
            p = f"camp::int_seq<long, {', '.join(map(str, permutation))}>"
        template_args = f"{value_type}, {len(dimensions)}, {p}, int{', LvArray::' + buffer if not 'Slice' in array_type else ''}"
        shape_args = f"shape [{', '.join(map(str, dimensions))}]"
    result = f"LvArray::{array_type}<{template_args}> of {shape_args}"
    if values:
        v = str(values).replace('(', '{').replace(')', '}')
        result += " = " + v
    return result


def generate_data() -> Iterable[TestCase]:
    # Passing cli arguments to pytest is a bit complicated for the little use we make of it.
    # Therefore, running the tests with `LVARRAY_SRC_DIR=/... LVARRAY_BUILD_DIR=/... python -m pytest`
    # looks like a good compromise.
    build_dir: str = os.environ["LVARRAY_BUILD_DIR"]
    exe: str = os.path.join(build_dir, "tests/testPrettyPrinters")
    # Empty vector
    yield TestCase(exe, "v0", build_output_str("Array", "int", (0,), (), (0,)))
    # 1d vectors
    d, v, p = (2,), (1, 2), (0,)
    yield TestCase(exe, "v1", build_output_str("Array", "int", d, v, p))
    yield TestCase(exe, "v1v", build_output_str("ArrayView", "int", d, v, p))
    yield TestCase(exe, "v1vc", build_output_str("ArrayView", "int const", d, v, p))
    # 2d vectors
    d, v, p = (2, 3), ((1, 2, 3), (4, 5, 6)), (0, 1)
    yield TestCase(exe, "v2", build_output_str("Array", "int", d, v, p))
    yield TestCase(exe, "v2v", build_output_str("ArrayView", "int", d, v, p))
    yield TestCase(exe, "v2s", build_output_str("ArraySlice", "int", d[1:], v[0], p[:1]))
    yield TestCase(exe, "v2sc", build_output_str("ArraySlice", "int const", d[1:], v[0], p[:1]))
    # 3d vectors
    d, p = (2, 3, 4), (0, 1, 2)
    v = (((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)), ((12, 13, 14, 15), (16, 17, 18, 19), (20, 21, 22, 23)))
    yield TestCase(exe, "v3", build_output_str("Array", "int", d, v, p))
    yield TestCase(exe, "v3v", build_output_str("ArrayView", "int", d, v, p))
    yield TestCase(exe, "v3vc", build_output_str("ArrayView", "int const", d, v, p))
    yield TestCase(exe, "v3s", build_output_str("ArraySlice", "int", d[1:], v[0], p[:1]))
    yield TestCase(exe, "v3s2", build_output_str("ArraySlice", "int", d[2:], v[0][0], p[:2]))
    # array of arrays
    d, v, p = (2,), ((1, 2, 3), (7, 8)), (0,)
    yield TestCase(exe, "aoa0", build_output_str("ArrayOfArrays", "int", d, v, p))
    yield TestCase(exe, "aoa0v", build_output_str("ArrayOfArraysView", "int", d, v, p))
    yield TestCase(exe, "aoa0vc", build_output_str("ArrayOfArraysView", "int const", d, v, p))
    yield TestCase(exe, "aoa0s", build_output_str("ArraySlice", "int", d, v[1], p))


@pytest.mark.parametrize("test_case", generate_data())
def test_pretty_printer(test_case: TestCase):
    src_dir: str = os.environ["LVARRAY_SRC_DIR"]  # Skip tedious CLI pytest configuration, see comments above.
    cli = ["/usr/bin/gdb",
           "-iex", f"source {os.path.join(src_dir, 'scripts/gdb-printers-shallow.py')}",
           "-ex", "set width 300",  # I do not wan t the output to be polluted with carriage returns.
           "-ex", "set style enabled off",
           "-ex", "set verbose off",
           "-ex", "set confirm off",
           "-ex", f"break {test_case.breakpoint}",
           "-ex", "run",
           "-ex", "finish",
           "-ex", f"print {test_case.cpp_var_name}",
           "-ex", "quit",
           test_case.executable]
    try:
        # I'm using a simple `subprocess.run` because I can use define all the interactions I need
        # through the `-ex` options of `gdb`. In the case something more complex would be needed,
        # it's possible to create a `subprocess.Popen`, to manage the `std{in,out,err}` as `subprocess.PIPE`,
        # and to send the information using `process.stdin.write("run\n")`.
        # Another more flexible approach would be to use Pexpect (https://github.com/pexpect/pexpect).
        process = subprocess.run(cli, timeout=1, check=True, capture_output=True, text=True)
        logging.debug(process.stdout, process.stderr)
        assert test_case.expected in process.stdout
    except subprocess.TimeoutExpired as e:
        logging.error(e.stdout, e.stderr)
        raise e
    except subprocess.CalledProcessError as e:
        logging.error(f"Process \"{' '.join(cli)}\" exited with error code {process.returncode}.")
        logging.error(e.stdout, e.stderr)
        raise e
