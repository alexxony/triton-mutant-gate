"""An ordinary Triton file (no special layout): a leaky-ReLU kernel, a wrapper, and a __main__ self-test on "cuda".

mutant-gate reads it as is: the @triton.jit function is mutated, the self-test is the existing check, and 'cuda' is pointed
at the CPU inside the sandbox.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(x_ptr, out_ptr, n_elements, slope, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x >= 0, x, x * slope)
    tl.store(out_ptr + offsets, y, mask=mask)


def leaky_relu(x, slope=0.01):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    leaky_relu_kernel[grid](x, out, n, slope, BLOCK_SIZE=256)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1000, device="cuda")
    assert torch.allclose(leaky_relu(x), torch.nn.functional.leaky_relu(x, 0.01))
    print("ok")
