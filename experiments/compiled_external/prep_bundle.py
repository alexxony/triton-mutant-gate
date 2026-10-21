"""Build bundle.json for the compiled run on files not written for this tool (see PREREGISTRATION.md). Offline, reads files only.

Tutorial 03 is read from a path you give (fetched at the pinned commit, e.g. with
`gh api "repos/triton-lang/triton/contents/python/tutorials/03-matrix-multiplication.py?ref=<commit>" --jq .content | base64 -d`);
its sha256 is checked. It is not stored in this repository, and neither is the generated bundle.

Usage: uv run python experiments/compiled_external/prep_bundle.py <path to 03-matrix-multiplication.py>
"""
import hashlib
import json
import sys
from pathlib import Path

from mutant_gate.mutants import make_mutants
from mutant_gate.script import full_program

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
TUTORIAL03_SHA256_PREFIX = "f88a9506694c"   # tutorial 03 at triton-lang/triton 16491796407ec76ef223f8227e7d9a96c3723363


def one(name, src, rec_dir):
    test = (rec_dir / "generated_test.py").read_text()
    test = test.replace("device='cpu'", "device='cuda'")   # Amendment A.1: the recorded test was written against the CPU-pointed script
    report = json.loads((rec_dir / "report.json").read_text())
    rec = {(m["op"], m["site"]): m["T3B"] for m in report["mutants"]}
    muts = make_mutants(src)
    got = {(op, site) for op, site, _ in muts}
    assert got == set(rec), f"{name}: regenerated mutants differ from the recorded run: {sorted(got ^ set(rec))}"
    return {"original": full_program(src, test),
            "mutants": [{"op": op, "site": site, "program": full_program(code, test), "sandbox_interp": rec[(op, site)]} for op, site, code in muts]}


def main(tutorial_path):
    tut = Path(tutorial_path).read_text()
    digest = hashlib.sha256(tut.encode()).hexdigest()
    assert digest.startswith(TUTORIAL03_SHA256_PREFIX), f"tutorial 03 sha256 {digest[:12]} is not the pinned file"
    bundle = {"meta": {"tutorial03": "triton-lang/triton 16491796407ec76ef223f8227e7d9a96c3723363, sha256 " + digest[:16],
                       "sandbox_interp": "recorded T3B from recorded/*/report.json (Nebius sandbox, Triton 3.8.0, bounds checks on)"},
              "kernels": {
                  "tutorial03": one("tutorial03", tut, HERE / "recorded" / "tutorial03"),
                  "autotune_axpy": one("autotune_axpy", (ROOT / "examples" / "plain" / "autotune_axpy.py").read_text(), HERE / "recorded" / "autotune_axpy")}}
    for k, v in bundle["kernels"].items():
        print(k, len(v["mutants"]), "mutants,", sum(m["sandbox_interp"] == "SURVIVED" for m in v["mutants"]), "survived in the sandbox interpreter")
    out = HERE / "bundle.json"
    out.write_text(json.dumps(bundle))
    print("bundle.json", out.stat().st_size, "bytes (not committed)")


if __name__ == "__main__":
    main(sys.argv[1])
