"""Runs on a Colab T4 (`colab exec -f`). Reads /content/bundle.json, runs every program in its own subprocess in two modes
(Triton interpreter on the Colab machine, normal compiled execution on the GPU) and prints the results as JSON between markers.
Same classification as experiments/compiled_mode/run_remote.py; differences are listed in PREREGISTRATION.md: perf_report benchmarks are
skipped in both modes, the interpreter mode loads interp_shims.py (autotuner keeps its first config), timeout 400 s.
"""
import json
import os
import subprocess
import sys
import time

BUNDLE = "/content/bundle.json"
STATUS = "/content/status.json"
WORK = "/content/work"
TIMEOUT = 400
SKIP_BENCH = "from triton import testing; testing.Mark.run = lambda self, *a, **k: None"
# Amendment A.1: cuda is the default device in both modes. A.3: print the autotuner's chosen config after the program ran.
BEST = "[print('@@BEST', k, v.best_config) for k, v in g.items() if hasattr(v, 'best_config')]"
COMPILED = "import torch, runpy; torch.set_default_device('cuda'); " + SKIP_BENCH + "; g = runpy.run_path('{p}', run_name='__main__'); " + BEST
INTERP = ("import sys; sys.path.insert(0, '/content'); import torch, interp_shims, runpy; torch.set_default_device('cuda'); "
          "g = runpy.run_path('{p}', run_name='__main__'); " + BEST)
ONLY_ORIGINALS = os.path.exists("/content/only_originals")   # Amendment A.4: first exec runs the two originals only


def versions():
    out = {"python": sys.version.split()[0]}
    for mod in ("torch", "triton"):
        r = subprocess.run([sys.executable, "-c", f"import {mod}; print({mod}.__version__)"], capture_output=True, text=True)
        out[mod] = r.stdout.strip() or r.stderr.strip()[-80:]
    r = subprocess.run([sys.executable, "-c", "import torch; print(torch.version.cuda, torch.cuda.get_device_name(0))"], capture_output=True, text=True)
    out["cuda_and_gpu"] = r.stdout.strip() or r.stderr.strip()[-80:]
    return out


def classify(rc, out, err, timed_out):
    if timed_out:
        return "TIMEOUT", ""
    last = (err.strip().splitlines() or [""])[-1][:200]
    if rc == 0 and "PASS" in out:
        return "PASS", ""
    low = err.lower()
    if "compilationerror" in low or "mlir" in low or "incompatibletype" in low or "triton.compiler" in low:
        return "COMPILE_ERROR", last
    if "cuda error" in low or "illegal memory" in low or "device-side assert" in low or "cudaerror" in low:
        return "CUDA_ERROR", last
    if "assertionerror" in low:
        return "FAIL_ASSERT", last
    return "OTHER", last


def run(path, mode):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    if mode == "colab_interp":
        env["TRITON_INTERPRET"] = "1"
        cmd = [sys.executable, "-c", INTERP.format(p=path)]
    else:
        env.pop("TRITON_INTERPRET", None)
        cmd = [sys.executable, "-c", COMPILED.format(p=path)]
    t = time.time()
    stdout = ""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=env, cwd="/content")
        stdout = r.stdout
        cat, last = classify(r.returncode, r.stdout, r.stderr, False)
    except subprocess.TimeoutExpired:
        cat, last = classify(0, "", "", True)
    best = [l[7:].strip()[:160] for l in stdout.splitlines() if l.startswith("@@BEST")]
    return {"result": cat, "detail": last, "seconds": round(time.time() - t, 1), "best_config": best}


def main():
    os.makedirs(WORK, exist_ok=True)
    bundle = json.load(open(BUNDLE))
    res = {"versions": versions(), "kernels": {}}
    print("versions", res["versions"], flush=True)
    for k, v in bundle["kernels"].items():
        entries = [{"op": "ORIGINAL", "site": "-", "program": v["original"], "sandbox_interp": "n/a"}] + ([] if ONLY_ORIGINALS else v["mutants"])
        out = []
        for i, e in enumerate(entries):
            p = f"{WORK}/{k}_{i}.py"
            open(p, "w").write(e["program"])
            row = {"op": e["op"], "site": e["site"], "sandbox_interp": e["sandbox_interp"]}
            for mode in ("colab_compiled", "colab_interp"):
                row[mode] = run(p, mode)
            out.append(row)
            res["kernels"][k] = out
            json.dump(res, open(STATUS, "w"))   # progress file, survives a client-side stall
            print(k, e["op"], e["site"], "compiled=", row["colab_compiled"]["result"], "interp=", row["colab_interp"]["result"], flush=True)
    print("@@RESULT_BEGIN@@")
    print(json.dumps(res))
    print("@@RESULT_END@@")


main()
