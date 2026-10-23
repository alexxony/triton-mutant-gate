"""Runs on the Colab VM (Amendment C): tutorial 03 original, compiled, 900 s cap, Python stack dumped every 120 s.
Started detached; writes /content/diag_result.json and /content/diag_done when finished. Reads /content/bundle.json."""
import json
import os
import subprocess
import sys
import time

prog = json.load(open("/content/bundle.json"))["kernels"]["tutorial03"]["original"]
open("/content/t03_orig.py", "w").write(prog)
code = ("import faulthandler; faulthandler.dump_traceback_later(120, repeat=True, file=open('/content/diag_stack.txt', 'w')); "
        "import torch, runpy; torch.set_default_device('cuda'); "
        "from triton import testing; testing.Mark.run = lambda self, *a, **k: None; "
        "runpy.run_path('/content/t03_orig.py', run_name='__main__')")
env = dict(os.environ)
env.pop("TRITON_INTERPRET", None)
env["PYTHONHASHSEED"] = "0"
t = time.time()
try:
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900, env=env, cwd="/content")
    res = {"rc": r.returncode, "seconds": round(time.time() - t, 1), "stdout_tail": r.stdout[-600:], "stderr_tail": r.stderr[-1500:]}
except subprocess.TimeoutExpired:
    res = {"rc": "TIMEOUT", "seconds": round(time.time() - t, 1)}
json.dump(res, open("/content/diag_result.json", "w"))
open("/content/diag_done", "w").write("1")
