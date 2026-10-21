"""An ordinary Triton file with a @triton.autotune kernel, a second kernel and several launches of different sizes.

Written for this repo to exercise the cases a one-kernel tutorial file does not (autotune re-launches, two kernels, a host
loop over shapes). It is not taken from an external project.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": 64}), triton.Config({"BLOCK_SIZE": 128}), triton.Config({"BLOCK_SIZE": 256})],
    key=["n"],
)
@triton.jit
def axpy_kernel(x_ptr, y_ptr, out_ptr, a, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a * x + y, mask=mask)


@triton.jit
def rowsum_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * n_cols + cols, mask=cols < n_cols, other=0.0)
    tl.store(out_ptr + row, tl.sum(x, axis=0))


def axpy(x, y, a):
    out = torch.empty_like(x)
    n = x.numel()
    axpy_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)](x, y, out, a, n)
    return out


def rowsum(x):
    rows, cols = x.shape
    out = torch.empty(rows, device=x.device, dtype=x.dtype)
    rowsum_kernel[(rows,)](x, out, cols, BLOCK=triton.next_power_of_2(cols))
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    for n in (1000, 4097):
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        torch.testing.assert_close(axpy(x, y, 2.5), 2.5 * x + y)
    m = torch.randn(7, 33, device="cuda")
    torch.testing.assert_close(rowsum(m), m.sum(dim=1), rtol=1e-4, atol=1e-4)
    print("OK")
