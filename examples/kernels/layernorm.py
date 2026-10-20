import sys
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


def layer_norm(x, w, b, eps=1e-5):
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((M,), dtype=torch.float32)
    rstd = torch.empty((M,), dtype=torch.float32)
    _layer_norm_fwd_fused[(M,)](x, y, w, b, mean, rstd, x.stride(0), N, eps, BLOCK_SIZE=1024)
    return y


def check(M, N, eps=1e-5, shift=0.0):
    torch.manual_seed(0)
    x = -2.3 + 0.5 * torch.randn(M, N) + shift
    w = torch.rand(N)
    b = torch.rand(N)
    ref = torch.nn.functional.layer_norm(x, (N,), w, b, eps)
    assert torch.allclose(layer_norm(x, w, b, eps), ref, atol=1e-2, rtol=0), f"mismatch {M}x{N}"


tier = sys.argv[1]
check(8, 8192)                    # tutorial N=8192, M reduced for CPU speed
if tier == "T2":
    check(4, 1000)
    check(4, 1025)
    check(4, 64, eps=1e-1)        # large eps makes eps visible
print("PASS")
