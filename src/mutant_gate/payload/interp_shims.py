"""Shims that let ordinary Triton files run under the CPU interpreter in the sandbox. Imported before the script runs.

@triton.autotune benchmarks every config on first use through `Autotuner.do_bench`, which asks for a GPU driver; a CPU-only
machine has none ("0 active drivers"). Timing a config under the interpreter means nothing anyway, so the benchmarker is
replaced by one that reports the same time for every config: the autotuner then keeps the first config and runs it, and
the number of kernel launches no longer depends on timing (the output oracle needs that to be stable).
"""


def _fake_bench(fn, *args, **kwargs):
    return [1.0, 1.0, 1.0]   # (median, p20, p80) of a benchmark that did not run


def install():
    try:
        from triton.runtime import autotuner
    except ImportError:
        return
    autotuner.Autotuner.do_bench = property(lambda self: _fake_bench)   # a data descriptor beats the class's cached_property


install()
