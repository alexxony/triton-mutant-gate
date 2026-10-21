import asyncio

import pytest

from mutant_gate import sandbox
from mutant_gate.sandbox import Sandbox


class Flaky:
    """img.run that fails with the given exceptions first, then returns 'ok'."""

    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    async def run(self, **kw):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def fast(_):
        return None

    monkeypatch.setattr(sandbox.asyncio, "sleep", fast)


def test_a_transient_500_is_retried():
    img = Flaky([RuntimeError("ApiStatusCodeError(... status=500, error='Internal Server Error')")] * 2)
    assert asyncio.run(Sandbox(img)._run(shell="x")) == "ok" and img.calls == 3


def test_a_non_transient_error_is_raised_at_once():
    img = Flaky([RuntimeError("status=404 not found")])
    with pytest.raises(RuntimeError):
        asyncio.run(Sandbox(img)._run(shell="x"))
    assert img.calls == 1


def test_gives_up_after_the_attempts():
    img = Flaky([RuntimeError("status=503")] * 9)
    with pytest.raises(RuntimeError):
        asyncio.run(Sandbox(img)._run(attempts=3, shell="x"))
    assert img.calls == 3
