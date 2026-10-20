import asyncio

from mutant_gate.cli import build_parser
from mutant_gate.pipeline import compile_survivors
from mutant_gate.report import markdown
from mutant_gate.script import split_script

SRC = "import sys\n@triton.jit\ndef k(): pass\n\ntier = sys.argv[1]\ncheck()\n"


class FakeSandbox:
    """compile_check answers from a table keyed by the kernel head; no network."""

    def __init__(self, verdicts):
        self.verdicts, self.calls = verdicts, []

    async def compile_check(self, head, code):
        self.calls.append(head)
        return self.verdicts[head]


def _muts():
    return [("acc_dtype_fp16", "acc#0", SRC.replace("pass", "x = 1")), ("drop_load_mask", "load#0", SRC.replace("pass", "y = 2"))]


def _survivors():
    return [{"op": op, "site": site, "T3B": "SURVIVED"} for op, site, _ in _muts()]


def test_cli_flag_defaults_off():
    p = build_parser()
    assert p.parse_args(["run", "k.py"]).compile_check is False
    assert p.parse_args(["run", "k.py", "--compile-check"]).compile_check is True


def test_compile_error_survivor_is_marked_and_others_are_not():
    heads = {split_script(c)[0]: ("COMPILE_ERROR", "AssertionError: Loop-carried variable") if op == "acc_dtype_fp16" else ("COMPILES", "")
             for op, _, c in _muts()}
    sb = FakeSandbox({split_script(SRC)[0]: ("COMPILES", ""), **heads})
    surv, logs = _survivors(), []
    assert asyncio.run(compile_survivors(sb, SRC, "", _muts(), surv, logs.append)) is True
    assert [m["compile"]["status"] for m in surv] == ["COMPILE_ERROR", "COMPILES"]
    assert "1 of 2 survivors do not compile" in logs[-1]


def test_check_is_skipped_when_the_original_does_not_compile():
    sb = FakeSandbox({split_script(SRC)[0]: ("COMPILE_ERROR", "boom")})
    surv, logs = _survivors(), []
    assert asyncio.run(compile_survivors(sb, SRC, "", _muts(), surv, logs.append)) is False
    assert all("compile" not in m for m in surv) and "skipped" in logs[-1]
    assert len(sb.calls) == 1   # only the original was tried


def test_report_says_a_non_compiling_survivor_is_not_a_test_gap():
    report = {"kernel": "k.py", "model": "m", "mutants_planted": 1, "accepted": True, "tokens": {"generate": 0, "close": 0},
              "mutants": [{"op": "acc_dtype_fp16", "site": "acc#0", "T3": "SURVIVED", "T3B": "SURVIVED",
                           "compile": {"status": "COMPILE_ERROR", "detail": "Loop-carried variable"}}]}
    out = markdown(report)
    assert "survived, but does not compile" in out and "not a test gap" in out
