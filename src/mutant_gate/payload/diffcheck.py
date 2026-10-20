"""Output-difference oracle for the Triton CPU interpreter, runs inside the sandbox.

record(script, out_prefix): run the script (seeded, under TRITON_INTERPRET=1) and, after every kernel launch, keep a copy of
every tensor argument (inputs too: kernels can write in place). Writes <out_prefix>.npz and <out_prefix>.json.

classify(a, b): compare two recordings of the same script. IDENTICAL (same launches, every tensor bit-identical) or DIFFERS
(launch count, shape, dtype or value differs; with the first differing launch/argument and the largest abs/rel difference).

compare_cli(orig1, orig2, mutant): the sandbox entry point. The two runs of the original must agree with each other first
(UNSTABLE otherwise). Last stdout line: `@@DIFFCHECK {json}`.
Only numpy is needed for classify, so it is unit-tested offline; record needs torch and triton.
"""
import json
import random
import runpy
import sys

import numpy as np

SEED = 0


def record(path, prefix):
    import inspect
    import torch
    from triton.runtime import interpreter as I

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    launches, arrays = [], {}
    orig = I.InterpretedFunction.run

    def run(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        sig = list(inspect.signature(self.fn).parameters)
        named = {**dict(zip(sig, args)), **{k: v for k, v in kwargs.items() if k not in ("grid", "warmup")}}
        i = len(launches)
        meta = {"fn": self.fn.__name__, "args": {}}
        for k, v in named.items():
            if isinstance(v, torch.Tensor):
                t = v.detach().cpu().contiguous()
                meta["args"][k] = {"shape": list(t.shape), "dtype": str(t.dtype)}
                arrays[f"L{i}_{k}_raw"] = t.view(torch.uint8).numpy().copy() if t.numel() else np.zeros(0, np.uint8)
                arrays[f"L{i}_{k}_f32"] = t.float().numpy().copy()
        launches.append(meta)
        return out

    I.InterpretedFunction.run = run
    status = "ok"
    try:
        runpy.run_path(path, run_name="__main__")
    except BaseException as e:   # a failing assert or bounds check: the run is not a survivor, the comparison is moot
        status = f"{type(e).__name__}: {str(e)[:120]}"
    np.savez_compressed(prefix + ".npz", **arrays)
    with open(prefix + ".json", "w") as f:
        json.dump({"status": status, "launches": launches}, f)


def _load(prefix):
    meta = json.load(open(prefix + ".json"))
    return meta, dict(np.load(prefix + ".npz"))


def classify(a, b):
    """a, b = (meta, arrays) of two recordings. Returns a dict with the class and, for DIFFERS, where and by how much."""
    (ma, xa), (mb, xb) = a, b
    la, lb = ma["launches"], mb["launches"]
    if len(la) != len(lb):
        return {"class": "DIFFERS", "why": f"launch count {len(la)} vs {len(lb)}"}
    worst = None
    first = None
    for i, (p, q) in enumerate(zip(la, lb)):
        if p["fn"] != q["fn"] or p["args"] != q["args"]:
            return {"class": "DIFFERS", "launch": i, "why": "kernel, argument names, shapes or dtypes differ"}
        for k in p["args"]:
            if np.array_equal(xa[f"L{i}_{k}_raw"], xb[f"L{i}_{k}_raw"]):
                continue
            u, v = xa[f"L{i}_{k}_f32"].astype(np.float64), xb[f"L{i}_{k}_f32"].astype(np.float64)
            bad = ~((u == v) | (np.isnan(u) & np.isnan(v)))
            d = np.where(bad, np.abs(u - v), 0.0)
            d = np.where(bad & (np.isnan(u) ^ np.isnan(v)), np.inf, d)
            abs_d = float(np.nanmax(d)) if d.size else 0.0
            denom = np.maximum(np.abs(u), 1e-30)
            rel_d = float(np.nanmax(np.where(bad, d / denom, 0.0))) if d.size else 0.0
            rec = {"launch": i, "fn": p["fn"], "arg": k, "n_diff": int(bad.sum()), "max_abs": abs_d, "max_rel": rel_d,
                   "bits_only": bool(not bad.any())}   # bit patterns differ but float32 values are equal (e.g. -0.0 vs 0.0, NaN payload)
            if first is None:
                first = rec
            if worst is None or abs_d > worst["max_abs"]:
                worst = rec
    if first is None:
        return {"class": "IDENTICAL", "launches": len(la)}
    return {"class": "DIFFERS", "first": first, "worst": worst}


def compare_cli(o1, o2, m):
    a1, a2, mm = _load(o1), _load(o2), _load(m)
    if a1[0]["status"] != "ok":
        res = {"class": "ORIGINAL_FAILED", "why": a1[0]["status"]}
    elif classify(a1, a2)["class"] != "IDENTICAL":
        res = {"class": "UNSTABLE", "why": "two runs of the original differ", "detail": classify(a1, a2)}
    elif mm[0]["status"] != "ok":
        res = {"class": "MUTANT_FAILED", "why": mm[0]["status"]}
    else:
        res = classify(a1, mm)
    print("@@DIFFCHECK " + json.dumps(res))


if __name__ == "__main__":
    compare_cli(*sys.argv[1:4])
