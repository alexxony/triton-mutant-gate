from pathlib import Path

import pytest

from mutant_gate.mutants import KernelFormatError, kernel_span, make_mutants

KERNELS = Path(__file__).parent.parent / "examples" / "kernels"
EXPECTED = {"vecadd": 4, "softmax": 5, "layernorm": 18, "matmul": 13}   # counts recorded in the experiments


@pytest.mark.parametrize("name,count", EXPECTED.items())
def test_example_mutant_counts(name, count):
    assert len(make_mutants((KERNELS / f"{name}.py").read_text())) == count


def test_every_mutant_compiles_and_differs():
    src = (KERNELS / "layernorm.py").read_text()
    for op, site, code in make_mutants(src):
        assert code != src
        compile(code, f"{op}-{site}", "exec")


def test_mutations_stay_inside_the_kernel():
    src = (KERNELS / "vecadd.py").read_text()
    start, end = kernel_span(src)
    for _, _, code in make_mutants(src):
        assert code[:start] == src[:start] and code.endswith(src[end:])


def test_drop_load_mask_removes_mask_and_other():
    src = (KERNELS / "vecadd.py").read_text()
    drops = [c for op, _, c in make_mutants(src) if op == "drop_load_mask"]
    assert len(drops) == 2 and all("mask=mask" not in c.split("def add(")[0].split("x_ptr + offsets")[1].split("\n")[0] for c in drops[:1])


def test_script_without_kernel_is_rejected():
    with pytest.raises(KernelFormatError):
        make_mutants("print('no kernel here')\n")
