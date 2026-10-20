from pathlib import Path

import pytest

from mutant_gate.mutants import KernelFormatError
from mutant_gate.script import extract_code, full_program, has_check, split_script

SRC = (Path(__file__).parent.parent / "examples" / "kernels" / "vecadd.py").read_text()


def test_split_script_separates_helpers_from_sanity_check():
    head, t1 = split_script(SRC)
    assert "def check(size)" in head and "tier = sys.argv" not in head
    assert t1.strip() == "check(98432)                      # tutorial shape"


def test_split_script_rejects_a_file_that_does_not_parse():
    with pytest.raises(KernelFormatError):
        split_script("def broken(:\n")


@pytest.mark.parametrize("code,ok", [
    ("assert 1 == 1", True),
    ("check(5)", True),
    ("# check(5) only in a comment\nx = 1", False),
    ("s = 'check(5)'", False),
    ("", False),
    ("def f(:\n  pass", False),
    ("def t():\n    assert True\nt()", True),
])
def test_has_check_requires_an_executed_assert_or_check_call(code, ok):
    assert has_check(code) is ok


def test_extract_code_tolerates_unclosed_fence():
    assert extract_code("text\n```python\nassert x\n").strip() == "assert x"
    assert extract_code("no code") == ""


def test_full_program_runs_sanity_check_then_test():
    prog = full_program(SRC, "check(7)")
    assert prog.index("check(98432)") < prog.index("check(7)") and prog.rstrip().endswith('print("PASS")')
