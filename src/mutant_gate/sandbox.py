"""Run kernel scripts in Nebius Sandboxes (CPU, Triton interpreter) through the contree SDK."""
import os
from pathlib import Path

from contree_sdk import Contree

CONFIG = Path.home() / ".config" / "mutant-gate" / "image"
BOUNDCHECK = (Path(__file__).parent / "payload" / "boundcheck.py").read_text()
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


class Sandbox:
    def __init__(self, img):
        self.img = img

    @classmethod
    async def open(cls, image=None):
        image = image or image_id()
        if not image:
            raise RuntimeError("no sandbox image: run `mutant-gate setup-image` or set MUTANT_GATE_IMAGE")
        return cls(await Contree().images.use(image, strict=True))

    async def run(self, code, bound=True):
        """Run a script with TRITON_INTERPRET=1. Returns (exit_code, last stderr line).

        bound=True adds out-of-bounds checks for tl.load/tl.store (the interpreter otherwise reads raw host memory).
        """
        shell = "cd /w && TRITON_INTERPRET=1 python " + ("-c \"import boundcheck, runpy; runpy.run_path('k.py', run_name='__main__')\"" if bound else "k.py")
        files = {"/w/k.py": code.encode(), "/w/boundcheck.py": BOUNDCHECK.encode()}
        r = await self.img.run(shell=shell, files=files, disposable=True, timeout=240)
        err = (r.stderr or "").strip().splitlines()
        return r.exit_code, (err[-1][:200] if err else "")
