"""Read results.json (from the Colab run) and evaluate the pre-registered predictions P1-P4 mechanically. Offline.

Usage: uv run python experiments/compiled_mode/analyze.py experiments/compiled_mode/results.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

FP16 = {("layernorm", "acc_dtype_fp16"), ("matmul", "acc_dtype_fp16")}


def detected(r):
    return r != "PASS"


def main(path):
    d = json.loads(Path(path).read_text())
    print("versions:", d["versions"])
    orig_ok = {}
    rows = []
    for k, entries in d["kernels"].items():
        for e in entries:
            if e["op"] == "ORIGINAL":
                orig_ok[k] = (e["colab_compiled"]["result"], e["colab_interp"]["result"])
            else:
                rows.append((k, e))
    print("\nOriginal kernel with the test (compiled, colab_interp):")
    for k, v in orig_ok.items():
        print(f"  {k:10} {v}")
    p1 = sum(1 for v in orig_ok.values() if v[0] == "PASS")
    print(f"P1 (original passes compiled for >=3 of 4): {p1}/4 -> {'holds' if p1 >= 3 else 'FAILS'}")

    usable = [(k, e) for k, e in rows if orig_ok.get(k, ("", ""))[0] == "PASS"]
    print(f"\nmutants with a usable compiled baseline: {len(usable)} of {len(rows)}")

    print("\nresult categories per mode (all mutants):")
    for mode in ("colab_compiled", "colab_interp"):
        print(" ", mode, dict(Counter(e[mode]["result"] for _, e in rows)))

    print("\nagreement of detection (detected = not PASS), usable mutants:")
    pairs = {"sandbox_interp vs colab_interp": ("sandbox", "colab_interp"), "colab_interp vs colab_compiled": ("colab_interp", "colab_compiled"),
             "sandbox_interp vs colab_compiled": ("sandbox", "colab_compiled")}

    def det(e, m):
        return e["sandbox_interp"] == "KILLED" if m == "sandbox" else detected(e[m]["result"])

    for name, (a, b) in pairs.items():
        same = sum(det(e, a) == det(e, b) for _, e in usable)
        print(f"  {name}: {same}/{len(usable)} same")

    print("\nmutants whose detection differs between modes (usable):")
    for k, e in usable:
        s, i, c = det(e, "sandbox"), det(e, "colab_interp"), det(e, "colab_compiled")
        if len({s, i, c}) > 1:
            print(f"  {k:10} {e['op']:20} {e['site']:8} sandbox={'det' if s else 'PASS'} colab_interp={e['colab_interp']['result']} compiled={e['colab_compiled']['result']} {e['colab_compiled']['detail'][:70]}")

    surv = [(k, e) for k, e in usable if e["sandbox_interp"] == "SURVIVED"]
    print(f"\nsandbox-interpreter survivors with a usable baseline: {len(surv)}")
    for k, e in surv:
        print(f"  {k:10} {e['op']:20} {e['site']:8} colab_interp={e['colab_interp']['result']:12} compiled={e['colab_compiled']['result']:13} {e['colab_compiled']['detail'][:60]}")

    other = [e for k, e in surv if e["op"] == "load_other_changed"]
    p2 = sum(e["colab_compiled"]["result"] == "PASS" for e in other)
    print(f"\nP2 (load_other_changed stays PASS compiled, >=90%): {p2}/{len(other)}")
    fp = [(k, e) for k, e in usable if (k, e["op"]) in FP16]
    p3 = [(k, e["site"]) for k, e in fp if detected(e["colab_compiled"]["result"])]
    print(f"P3 (>=1 fp16-accumulator mutant detected compiled): {len(p3)} of {len(fp)} -> {p3}")
    mask = [e for _, e in usable if e["op"] in ("drop_load_mask", "drop_store_mask")]
    by = defaultdict(Counter)
    for e in mask:
        by[e["op"]][e["colab_compiled"]["result"]] += 1
    print("P4 (dropped-mask mutants, compiled categories):", {k: dict(v) for k, v in by.items()})


if __name__ == "__main__":
    main(sys.argv[1])
