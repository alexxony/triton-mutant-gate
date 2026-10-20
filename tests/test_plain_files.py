"""Kernel files that do not follow the examples' layout (no `tier = sys.argv[1]`, no wrapper after two blank lines)."""
import pytest

from mutant_gate.cli import main
from mutant_gate.mutants import KernelFormatError, kernel_spans, make_mutants
from mutant_gate.script import cpuize, full_program, shown_script, split_script

MAIN_GUARD = '''import torch
import triton
import triton.language as tl

@triton.autotune(configs=[], key=["n"])
@triton.jit
def scale(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m, other=0.0) * 2, mask=m)
def run(x):
    out = torch.empty_like(x)
    scale[(1,)](x, out, x.numel(), BLOCK=128)
    return out
@triton.jit
def shift(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=offs < n, other=0.0) + 1, mask=offs < n)

if __name__ == "__main__":
    x = torch.rand(100, device="cuda")
    assert torch.allclose(run(x), x * 2)
    print("ok")
'''

TRAILING = '''import triton
import triton.language as tl

@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=offs < n))

def go(x):
    k[(1,)](x, x.numel(), BLOCK=64)

go(None)
'''

NO_CHECK = TRAILING.split("go(None)")[0]


def test_main_guard_body_is_the_sanity_check_and_definitions_are_the_head():
    head, t1 = split_script(MAIN_GUARD)
    assert "def run(x)" in head and "__main__" not in head and "x = torch.rand" not in head
    assert t1.startswith("x = torch.rand(100") and "assert torch.allclose" in t1 and not t1.startswith(" ")


def test_trailing_statements_after_the_last_def_are_the_sanity_check():
    head, t1 = split_script(TRAILING)
    assert head.rstrip().endswith("k[(1,)](x, x.numel(), BLOCK=64)") and t1.strip() == "go(None)"


def test_file_with_only_definitions_has_an_empty_sanity_check():
    head, t1 = split_script(NO_CHECK)
    assert t1 == "" and "def go" in head
    assert "existing sanity check" not in shown_script(NO_CHECK)


def test_full_program_runs_the_check_unconditionally_after_the_definitions():
    prog = full_program(MAIN_GUARD, "assert True")
    assert "__main__" not in prog and prog.index("x = torch.rand") > prog.index("def run(x)") and prog.rstrip().endswith('print("PASS")')


def test_every_jit_kernel_is_mutated_and_decorators_are_kept():
    names = [n for n, _, _ in kernel_spans(MAIN_GUARD)]
    assert names == ["scale", "shift"]
    muts = make_mutants(MAIN_GUARD)
    sites = {site for _, site, _ in muts}
    assert any(s.startswith("scale:") for s in sites) and any(s.startswith("shift:") for s in sites)
    for _, _, code in muts:
        assert "@triton.autotune" in code and code.count("@triton.jit") == 2


def test_single_kernel_sites_have_no_prefix():
    assert all(":" not in site for _, site, _ in make_mutants(TRAILING))


def test_file_without_a_jit_kernel_is_rejected():
    with pytest.raises(KernelFormatError):
        make_mutants("def f():\n    pass\n")


def test_cpuize_points_cuda_at_cpu():
    new, n = cpuize("x = torch.rand(3, device='cuda')\ny = z.cuda()\nw = torch.zeros(1, device=\"cuda:0\")\n")
    assert n == 3 and "cuda" not in new and new.count('"cpu"') == 2


def test_cli_mutants_accepts_a_plain_file(tmp_path, capsys):
    f = tmp_path / "plain.py"
    f.write_text(TRAILING)
    assert main(["mutants", str(f)]) == 0
    assert "drop_load_mask" in capsys.readouterr().out
