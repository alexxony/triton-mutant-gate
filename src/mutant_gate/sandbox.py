"""Run kernel scripts in Nebius Sandboxes (CPU, Triton interpreter) through the contree SDK."""
import asyncio
import json
import os
from pathlib import Path

from contree_sdk import Contree

CONFIG = Path.home() / ".config" / "mutant-gate" / "image"
BOUNDCHECK = (Path(__file__).parent / "payload" / "boundcheck.py").read_text()
COMPILECHECK = (Path(__file__).parent / "payload" / "compilecheck.py").read_text()
DIFFCHECK = (Path(__file__).parent / "payload" / "diffcheck.py").read_text()
SHIMS = (Path(__file__).parent / "payload" / "interp_shims.py").read_text()   # @triton.autotune without a GPU driver
INSTALL = ("pip install --no-cache-dir triton==3.8.0 torch numpy "
           "--extra-index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3")


def image_id():
    """Image id from $MUTANT_GATE_IMAGE or the file written by `mutant-gate setup-image`."""
    if os.environ.get("MUTANT_GATE_IMAGE"):
        return os.environ["MUTANT_GATE_IMAGE"]
    return CONFIG.read_text().strip() if CONFIG.exists() else None


async def build_image():
    """Build python:3.12-slim + triton + torch(cpu) once; remember the image id. Takes about a minute."""
    client = Contree()
    base = await client.images.oci("docker.io/library/python:3.12-slim")
    res = await base.run(shell=INSTALL, disposable=False, timeout=900)
    if res.exit_code != 0 or res.uuid is None:
        raise RuntimeError(f"image build failed (exit {res.exit_code}): {(res.stderr or res.stdout or '')[-300:]}")
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(str(res.uuid))
    return str(res.uuid)


def _transient(e):
    """The sandbox API now and then answers HTTP 500/502/503 (file lookup, image lookup); those are worth retrying."""
    return any(f"status={c}" in repr(e) or f"HTTP {c}" in str(e) for c in (500, 502, 503, 504))


class Sandbox:
    def __init__(self, img):
        self.img = img

    async def _run(self, attempts=4, **kw):
        """img.run with a short backoff on transient server errors; anything else is raised at once."""
        for n in range(attempts):
            try:
                return await self.img.run(**kw)
            except Exception as e:
                if n == attempts - 1 or not _transient(e):
                    raise
                await asyncio.sleep(2 * (n + 1))

    @classmethod
    async def open(cls, image=None):
        image = image or image_id()
        if not image:
            raise RuntimeError("no sandbox image: run `mutant-gate setup-image` or set MUTANT_GATE_IMAGE")
        for n in range(4):
            try:
                return cls(await Contree().images.use(image, strict=True))
            except Exception as e:
                if n == 3 or not _transient(e):
                    raise
                await asyncio.sleep(2 * (n + 1))

    async def run(self, code, bound=True):
        """Run a script with TRITON_INTERPRET=1. Returns (exit_code, last stderr line).

        bound=True adds out-of-bounds checks for tl.load/tl.store (the interpreter otherwise reads raw host memory).
        """
        shell = "cd /w && TRITON_INTERPRET=1 python " + ("-c \"import interp_shims, boundcheck, runpy; runpy.run_path('k.py', run_name='__main__')\"" if bound
                                                       else "-c \"import interp_shims, runpy; runpy.run_path('k.py', run_name='__main__')\"")
        files = {"/w/k.py": code.encode(), "/w/boundcheck.py": BOUNDCHECK.encode(), "/w/interp_shims.py": SHIMS.encode()}
        r = await self._run(shell=shell, files=files, disposable=True, timeout=240)
        err = (r.stderr or "").strip().splitlines()
        return r.exit_code, (err[-1][:200] if err else "")

    async def compile_check(self, head, code):
        """Compile every kernel launch of a script (head = kernels and helpers, code = the full program) for a CUDA target (no GPU, nothing executes). Returns (status, detail).

        status: COMPILES / COMPILE_ERROR / NO_LAUNCH (the script never launched a kernel) / CHECK_FAILED (the check itself broke).
        The script runs under the interpreter so all its launches happen; each distinct launch is compiled (sm_75 target).
        """
        shell = ("cd /w && TRITON_INTERPRET=1 python -c \"import interp_shims, compilecheck; compilecheck.record('k.py', 'l.json')\" >/dev/null 2>&1; "
                 "python -c \"import compilecheck; compilecheck.compile_all('h.py', 'l.json')\"")
        files = {"/w/k.py": code.encode(), "/w/h.py": head.encode(), "/w/compilecheck.py": COMPILECHECK.encode(), "/w/interp_shims.py": SHIMS.encode()}
        r = await self._run(shell=shell, files=files, disposable=True, timeout=240)
        line = next((l for l in reversed((r.stdout or "").splitlines()) if l.startswith("@@COMPILECHECK")), "")
        detail = line.partition("first=")[2][:150]
        status = {0: "COMPILES", 3: "COMPILE_ERROR", 4: "NO_LAUNCH", 5: "CHECK_FAILED"}.get(r.exit_code, "CHECK_FAILED")
        return status, detail if line else ((r.stderr or "").strip().splitlines() or [""])[-1][:150]

    async def diff_check(self, original, mutant):
        """Run the original twice and the mutant once (same seeds, bounds checks on) and compare every kernel launch's tensors. Returns the oracle's result dict.

        class: IDENTICAL / DIFFERS (with where and by how much) / MUTANT_FAILED / UNSTABLE (the original disagrees with itself) /
        ORIGINAL_FAILED / ERROR (the oracle itself broke). Costs one sandbox invocation, no model.
        """
        rec = "python -c \"import interp_shims, boundcheck, diffcheck; diffcheck.record('{src}', '{out}')\" >/dev/null 2>&1; "
        shell = ("cd /w && export TRITON_INTERPRET=1 PYTHONHASHSEED=0; " + rec.format(src="orig.py", out="o1") + rec.format(src="orig.py", out="o2")
                 + rec.format(src="mut.py", out="m") + "python diffcheck.py o1 o2 m")
        files = {"/w/orig.py": original.encode(), "/w/mut.py": mutant.encode(), "/w/boundcheck.py": BOUNDCHECK.encode(),
                 "/w/diffcheck.py": DIFFCHECK.encode(), "/w/interp_shims.py": SHIMS.encode()}
        r = await self._run(shell=shell, files=files, disposable=True, timeout=600)
        line = next((l for l in reversed((r.stdout or "").splitlines()) if l.startswith("@@DIFFCHECK ")), "")
        if not line:
            return {"class": "ERROR", "why": ((r.stderr or "").strip().splitlines() or [""])[-1][:150]}
        return json.loads(line[len("@@DIFFCHECK "):])
