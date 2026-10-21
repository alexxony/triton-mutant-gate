"""Shims that let ordinary Triton files run under the CPU interpreter in the sandbox. Imported before the script runs.

A CPU-only machine has no GPU driver, and many Triton files touch one outside the kernels. Three things are smoothed over:

- @triton.autotune benchmarks every config through `Autotuner.do_bench`, which asks for a driver ("0 active drivers"). Timing a config
  under the interpreter means nothing, so the benchmarker reports the same time for every config: the autotuner keeps the first one
  and the number of launches no longer depends on timing (the output oracle needs that).
- `triton.testing.perf_report` benchmarks (`Mark.run`, usually the last statement of a tutorial) are skipped.
- Module-level device queries (`triton.runtime.driver.active.get_active_torch_device()`, `get_current_target()`, `utils.get_device_properties`)
  get CPU answers when there is no real driver. This does not make code that compiles for the GPU work (for example tutorial 02's
  `softmax_kernel.warmup(...)`); such files still fail, and the report says so.

`tabulate`, which `perf_report` imports for printing, is stubbed when missing since the printing is skipped.
"""
import importlib.util
import sys
import types


def _fake_bench(fn, *args, **kwargs):
    return [1.0, 1.0, 1.0]   # (median, p20, p80) of a benchmark that did not run


def _skip_benchmarks():
    try:
        from triton import testing
    except ImportError:
        return
    testing.Mark.run = lambda self, *args, **kwargs: None
    if importlib.util.find_spec("tabulate") is None:   # a plain-text stand-in: only the table formatting is lost
        stub = types.ModuleType("tabulate")
        stub.tabulate = lambda rows, *args, **kwargs: "\n".join(" | ".join(map(str, r)) for r in rows)
        sys.modules["tabulate"] = stub


def _cpu_driver():
    try:
        from triton.runtime import driver
    except ImportError:
        return
    try:
        driver.active
        return   # a real driver exists, leave it alone
    except Exception:
        pass
    import torch
    from triton.backends.compiler import GPUTarget

    class Utils:
        def get_device_properties(self, index):
            return {"multiprocessor_count": 1, "max_shared_mem": 65536, "max_num_regs": 65536, "warpSize": 32}

    class CpuDriver:
        utils = Utils()

        def get_active_torch_device(self):
            return torch.device("cpu")

        def get_current_target(self):
            return GPUTarget("cuda", 75, 32)

        def get_current_device(self):
            return 0

        def get_benchmarker(self):
            return _fake_bench

    driver._active = CpuDriver()


def install():
    try:
        from triton.runtime import autotuner
    except ImportError:
        return
    autotuner.Autotuner.do_bench = property(lambda self: _fake_bench)   # a data descriptor beats the class's cached_property
    _skip_benchmarks()
    _cpu_driver()


install()
