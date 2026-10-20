"""Compile-only check for Triton kernel scripts, runs on CPU in two processes.

The interpreter accepts programs the real compiler rejects (e.g. a loop-carried fp16 accumulator that picks up fp32).

1. record(script, launches.json): run the script under TRITON_INTERPRET=1 (so the script's own asserts keep working and every
   launch is reached) and write down each kernel launch: kernel name, scalar values, tensor dtype and 16-byte alignment.
2. compile_all(head.py, launches.json): without the interpreter, rebuild a stand-in argument for each recorded launch, let
   Triton specialize it the way it does for a real launch, and compile it for a CUDA target (explicit GPUTarget, no GPU).
   Only the head of the script (kernels and helpers, without the tests) is executed here.

Exit code of compile_all: 0 all launches compiled, 3 at least one compile error, 4 no launch was recorded, 5 the check itself
broke. Last stdout line: `@@COMPILECHECK launches=N unique=U errors=M internal=K first=<message>`.
"""
import json
import runpy
import sys


def record(path, out):
    from triton.runtime import interpreter as I
    import inspect
    import torch

    launches = []
    orig = I.InterpretedFunction.run

    def describe(v):
        if isinstance(v, torch.Tensor):
            return {"t": str(v.dtype).split(".")[-1], "al": v.data_ptr() % 16 == 0}
        if v is None or isinstance(v, (bool, int, float, str)):
            return {"v": v}
        return {"x": repr(v)[:80]}

    def run(self, *args, **kwargs):
        sig = list(inspect.signature(self.fn).parameters)
        kw = {k: v for k, v in kwargs.items() if k not in ("grid", "warmup")}
        named = {**dict(zip(sig, args)), **kw}
        launches.append({"fn": self.fn.__name__, "args": {k: describe(v) for k, v in named.items()}})
        return orig(self, *args, **kwargs)

    I.InterpretedFunction.run = run
    try:
        runpy.run_path(path, run_name="__main__")
    except BaseException:   # a failing assert in the script says nothing about compilation
        pass
    with open(out, "w") as f:
        json.dump(launches, f)


def _msg(e):
    lines = str(e).strip().splitlines()
    return f"{type(e).__name__}: {lines[-1][:150] if lines else ''}"


def compile_all(head_path, launches_json):
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler import ASTSource
    from triton.runtime import jit as J

    target = GPUTarget("cuda", 75, 32)
    backend = CUDABackend(target)
    launches = json.load(open(launches_json))
    ns = {"__name__": "kernel_head", "__file__": head_path}
    exec(compile(open(head_path).read(), head_path, "exec"), ns)

    def standin(d):
        if "t" in d:
            base = torch.empty(64 + 1, dtype=getattr(torch, d["t"]))
            return base[: 64] if d["al"] else base[1:]
        if "v" in d:
            return d["v"]
        raise TypeError(f"unsupported argument {d['x']}")

    seen, errors, internal = set(), [], []
    for launch in launches:
        jf = ns.get(launch["fn"])
        if not isinstance(jf, J.JITFunction):
            internal.append(f"kernel {launch['fn']} is not defined before the tier marker")
            continue
        try:
            given = {k: standin(v) for k, v in launch["args"].items()}
            names = [p.name for p in jf.params]
            kw = {k: v for k, v in given.items() if k not in names}   # num_warps etc.
            args = {p.name: given.get(p.name, p.default) for p in jf.params}
            binder = J.create_function_from_signature(jf.signature, jf.params, backend)
            bound_args, specialization, options = binder(**args, **kw)
            options, signature, constexprs, attrs = jf._pack_args(backend, kw, bound_args, specialization, options)
        except Exception as e:   # could not derive the launch signature: not a verdict about the kernel
            internal.append(_msg(e))
            continue
        key = (launch["fn"], repr(signature), repr(constexprs), repr(attrs), repr(options.__dict__))
        if key in seen:
            continue
        seen.add(key)
        try:
            triton.compile(ASTSource(jf, signature, constexprs, attrs), target=target, options=options.__dict__)
        except Exception as e:
            errors.append(_msg(e))
    first = (errors or internal or [""])[0]
    print(f"@@COMPILECHECK launches={len(launches)} unique={len(seen)} errors={len(errors)} internal={len(internal)} first={first}")
    sys.exit(3 if errors else 5 if internal else (0 if launches else 4))
