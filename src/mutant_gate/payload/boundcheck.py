"""Bounds instrumentation for the Triton CPU interpreter.

The interpreter reads and writes raw host memory for every active lane of tl.load / tl.store,
so a kernel that touches memory past its tensors can still produce correct output. This
module raises AssertionError when an active lane falls outside the storage of every tensor
passed to the kernel. Import it before running a kernel script.

Limit: the check is per storage, not per row, so an out-of-row access that stays inside the
same tensor (e.g. reading the next row) is not flagged.
"""
import numpy as np
from triton.runtime import interpreter as I

_ranges: list[tuple[int, int]] = []

_orig_init = I.GridExecutor._init_args_hst


def _init_args_hst(self, args_dev, kwargs):
    res = _orig_init(self, args_dev, kwargs)
    args_hst, kwargs_hst = res
    _ranges.clear()
    for a in list(args_hst) + list(kwargs_hst.values()):
        if hasattr(a, "data_ptr") and hasattr(a, "untyped_storage"):
            st = a.untyped_storage()
            _ranges.append((st.data_ptr(), st.data_ptr() + st.nbytes()))
    return res


def _check(ptrs, mask, kind):
    itemsize = I._get_np_dtype(ptrs.get_element_ty()).itemsize
    p = ptrs.data.astype(np.uint64)
    active = np.broadcast_to(mask.data.astype(bool), p.shape)
    ok = np.zeros(p.shape, dtype=bool)
    for lo, hi in _ranges:
        ok |= (p >= np.uint64(lo)) & (p + np.uint64(itemsize) <= np.uint64(hi))
    bad = int((active & ~ok).sum())
    if bad:
        raise AssertionError(f"out-of-bounds {kind}: {bad} active lane(s) outside every kernel tensor")


_orig_load = I.InterpreterBuilder.create_masked_load
_orig_store = I.InterpreterBuilder.create_masked_store


def _masked_load(self, ptrs, mask, other, *rest):
    _check(ptrs, mask, "load")
    return _orig_load(self, ptrs, mask, other, *rest)


def _masked_store(self, ptrs, value, mask, *rest):
    _check(ptrs, mask, "store")
    return _orig_store(self, ptrs, value, mask, *rest)


I.GridExecutor._init_args_hst = _init_args_hst
I.InterpreterBuilder.create_masked_load = _masked_load
I.InterpreterBuilder.create_masked_store = _masked_store
