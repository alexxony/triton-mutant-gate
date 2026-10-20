import asyncio

from mutant_gate.pipeline import diff_survivors
from mutant_gate.report import markdown, oracle_text

SRC = "import sys\n@triton.jit\ndef k(): pass\n\ntier = sys.argv[1]\ncheck()\n"


class FakeSandbox:
    """diff_check answers from a table keyed by whether the program contains the mutation marker; no network."""

    def __init__(self, answers, boom=False):
        self.answers, self.boom, self.calls = answers, boom, []

    async def diff_check(self, original, mutant):
        self.calls.append((original, mutant))
        if self.boom:
            raise RuntimeError("HTTP 500")
        return self.answers["mut" if "x = 1" in mutant else "other"]


def _muts():
    return [("acc_dtype_fp16", "acc#0", SRC.replace("pass", "x = 1")), ("drop_load_mask", "load#0", SRC.replace("pass", "y = 2"))]


def _survivors():
    return [{"op": op, "site": site, "T3B": "SURVIVED"} for op, site, _ in _muts()]


def test_diff_survivors_sets_oracle_per_survivor():
    sb = FakeSandbox({"mut": {"class": "IDENTICAL", "launches": 3}, "other": {"class": "DIFFERS", "worst": {"arg": "y", "launch": 1, "max_abs": 0.5, "max_rel": 0.1}}})
    surv, logs = _survivors(), []
    asyncio.run(diff_survivors(sb, SRC, "check(1)", _muts(), surv, logs.append))
    assert surv[0]["oracle"]["class"] == "IDENTICAL" and surv[1]["oracle"]["class"] == "DIFFERS"
    assert len(sb.calls) == 2 and "check(1)" in sb.calls[0][0]
    assert logs == ["output oracle: 1 DIFFERS, 1 IDENTICAL"]


def test_a_sandbox_error_is_recorded_not_raised():
    surv = _survivors()
    asyncio.run(diff_survivors(FakeSandbox({}, boom=True), SRC, "check(1)", _muts(), surv, lambda s: None))
    assert [m["oracle"]["class"] for m in surv] == ["ERROR", "ERROR"]


def test_report_shows_oracle_next_to_the_model_claim():
    rep = {"kernel": "k.py", "model": "m", "mutants_planted": 1, "tokens": {"generate": 1, "close": 1}, "accepted": True,
           "mutants": [{"op": "load_other_changed", "site": "zero#1", "T3B": "SURVIVED",
                        "stage2": {"outcome": "claimed_equivalent"}, "oracle": {"class": "IDENTICAL", "launches": 9}}]}
    assert "Nemotron says equivalent (not proven); outputs identical on all 9 launches" in markdown(rep)


def test_oracle_text_differs_names_the_tensor_and_the_size():
    t = oracle_text({"class": "DIFFERS", "worst": {"arg": "Rstd", "launch": 1, "max_abs": 316.0, "max_rel": 0.999}})
    assert "`Rstd`" in t and "316" in t and "the test passes anyway" in t and "tolerance" not in t


def test_oracle_text_bits_only_does_not_claim_a_value_difference():
    t = oracle_text({"class": "DIFFERS", "first": {"arg": "y", "launch": 0, "max_abs": 0.0, "max_rel": 0.0, "bits_only": True}})
    assert "bit patterns differ" in t and "max abs" not in t
