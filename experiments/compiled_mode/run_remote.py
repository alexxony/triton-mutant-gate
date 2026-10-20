"""Runs on a Colab T4 (executed with `colab exec -f`). Reads /content/bundle.json, runs every program in its own subprocess in
two modes (Triton interpreter on CPU, normal compiled execution on the GPU) and prints the results as JSON between markers.
"""
import json
import os
import subprocess
import sys
import time

BUNDLE = "/content/bundle.json"
STATUS = "/content/status.json"
WORK = "/content/work"
TIMEOUT = 180
COMPILED_WRAPPER = "import torch, runpy; torch.set_default_device('cuda'); runpy.run_path('{p}', run_name='__main__')"


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
    if mode == "colab_interp":
        env["TRITON_INTERPRET"] = "1"
        cmd = [sys.executable, path]
    else:
        env.pop("TRITON_INTERPRET", None)
        cmd = [sys.executable, "-c", COMPILED_WRAPPER.format(p=path)]
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=env, cwd="/content")
        cat, last = classify(r.returncode, r.stdout, r.stderr, False)
    except subprocess.TimeoutExpired:
        cat, last = classify(0, "", "", True)
    return {"result": cat, "detail": last, "seconds": round(time.time() - t, 1)}


def main():
    os.makedirs(WORK, exist_ok=True)
    bundle = json.load(open(BUNDLE))
    res = {"versions": versions(), "kernels": {}}
    print("versions", res["versions"], flush=True)
    for k, v in bundle["kernels"].items():
        entries = [{"op": "ORIGINAL", "site": "-", "program": v["original"], "sandbox_interp": "n/a"}] + v["mutants"]
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
