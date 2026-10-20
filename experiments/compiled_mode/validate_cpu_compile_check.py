"""Compare the CPU compile check (sandbox, triton 3.8.0) with the recorded T4 compiled runs (triton 3.6.0)."""
import asyncio, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mutant_gate.sandbox import Sandbox

EXP = str(ROOT / "experiments" / "compiled_mode") + "/"
bundle = json.load(open(EXP + "bundle.json"))["kernels"]
res = json.load(open(EXP + "results.json"))["kernels"]


async def main():
    sb = await Sandbox.open()
    sem = asyncio.Semaphore(8)
    jobs = []
    from mutant_gate.script import split_script
    for k, v in bundle.items():
        t1 = split_script(open(ROOT / "examples" / "kernels" / f"{k}.py").read())[1]
        progs = [("ORIGINAL", "-", v["original"])] + [(m["op"], m["site"], m["program"]) for m in v["mutants"]]
        truth = {(r["op"], r["site"]): r["colab_compiled"]["result"] for r in res[k]}
        for op, site, prog in progs:
            head = prog.partition("\n" + t1 + "\n")[0]
            assert head != prog, (k, op, site)
            jobs.append((k, op, site, truth[(op, site)], prog, head))

    async def one(j):
        async with sem:
            return j, await sb.compile_check(j[5], j[4])

    out = await asyncio.gather(*[one(j) for j in jobs])
    bad = 0
    rows = []
    for (k, op, site, truth, _, __), (status, detail) in out:
        t4_err = truth == "COMPILE_ERROR"
        mine_err = status == "COMPILE_ERROR"
        flag = "" if t4_err == mine_err else "  <-- MISMATCH"
        bad += t4_err != mine_err
        rows.append(f"{k:10} {op:18} {site:8} T4={truth:13} cpu={status:13}{flag} {detail if mine_err or flag else ''}")
    print("\n".join(rows))
    print(f"total={len(out)} compile-error mismatches={bad}")

asyncio.run(main())
