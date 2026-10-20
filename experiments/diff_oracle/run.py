"""Run the output-difference oracle over the 44 programs of experiments/compiled_mode/bundle.json (see PREREGISTRATION.md).

For every mutant: original x2 + mutant x1 in one sandbox invocation, no model. Writes results.json next to this file.
Usage: uv run python experiments/diff_oracle/run.py [--only softmax]
"""
import asyncio
import json
import sys
from pathlib import Path

from mutant_gate.sandbox import Sandbox

HERE = Path(__file__).parent
BUNDLE = HERE.parent / "compiled_mode" / "bundle.json"


async def main(only=None, parallel=6):
    bundle = json.loads(BUNDLE.read_text())
    for attempt in range(4):   # the sandbox API answers HTTP 500 now and then (image lookup, file fetch); nothing has been measured yet
        try:
            sb = await Sandbox.open()
            break
        except Exception as e:
            print(f"Sandbox.open failed ({type(e).__name__}), retry {attempt + 1}", flush=True)
            await asyncio.sleep(5 * (attempt + 1))
    else:
        raise SystemExit("sandbox unavailable")
    sem = asyncio.Semaphore(parallel)

    async def one(kernel, orig, m):
        async with sem:
            for attempt in range(3):   # the sandbox API answered HTTP 500 once under load; a retry is not a result change
                try:
                    r = await sb.diff_check(orig, m["program"])
                    break
                except Exception as e:
                    r = {"class": "ERROR", "why": f"{type(e).__name__}: {str(e)[:100]}"}
                    await asyncio.sleep(3 * (attempt + 1))
        print(f"{kernel:9s} {m['op']:22s} {m['site']:12s} recorded={m['sandbox_interp']:9s} oracle={r['class']}", flush=True)
        return {"kernel": kernel, "op": m["op"], "site": m["site"], "recorded": m["sandbox_interp"], "oracle": r}

    jobs = [one(k, v["original"], m) for k, v in bundle["kernels"].items() if only in (None, k) for m in v["mutants"]]
    rows = await asyncio.gather(*jobs)
    (HERE / ("results.json" if only is None else f"results_{only}.json")).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--only" else None))
