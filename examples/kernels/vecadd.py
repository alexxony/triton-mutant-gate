import sys
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x, y):
    output = torch.empty_like(x)
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output


def check(size):
    torch.manual_seed(0)
    x = torch.rand(size)
    y = torch.rand(size)
    assert torch.allclose(add(x, y), x + y), f"mismatch size={size}"


tier = sys.argv[1]
check(98432)                      # tutorial shape
if tier == "T2":
    for s in (1, 1000, 1025, 2047):
        check(s)
print("PASS")
