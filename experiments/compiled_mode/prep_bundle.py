"""Build bundle.json: for each example kernel, the original and every mutant as a standalone program that already contains the
recorded Nemotron test, plus the recorded sandbox-interpreter verdict. Offline, reads files only.

Usage: uv run python experiments/compiled_mode/prep_bundle.py
"""
import json
from pathlib import Path

from mutant_gate.mutants import make_mutants
from mutant_gate.script import full_program

ROOT = Path(__file__).resolve().parents[2]
DEMO = json.loads((ROOT / "demo" / "data.json").read_text())
RESULTS = ROOT / "recorded" / "results"


def main():
    bundle = {"meta": {"source": "demo/data.json runs, recorded sandbox verdicts = T3B (Triton 3.8.0, bounds checks on)"}, "kernels": {}}
    for k, v in DEMO["kernels"].items():
        tag = v["run"]
        src = (ROOT / "examples" / "kernels" / f"{k}.py").read_text()
        test = (ROOT / "recorded" / "generated" / f"run_{tag}" / f"{k}_T3.py").read_text()
        assert test.strip() == v["test"].strip(), f"{k}: recorded test differs from the demo's test"
        rec = {(m["op"], m["site"]): m["T3B"] for m in json.loads((RESULTS / f"results_generated_{tag}.json").read_text())["kernels"][k]["mutants"]}
        muts = []
        for op, site, code in make_mutants(src):
            muts.append({"op": op, "site": site, "program": full_program(code, test), "sandbox_interp": rec[(op, site)]})
        bundle["kernels"][k] = {"run": tag, "original": full_program(src, test), "mutants": muts}
        print(k, tag, len(muts), "mutants", sum(m["sandbox_interp"] == "SURVIVED" for m in muts), "survived in the sandbox interpreter")
    out = Path(__file__).parent / "bundle.json"
    out.write_text(json.dumps(bundle))
    print("bundle.json", out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
