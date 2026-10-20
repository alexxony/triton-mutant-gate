import sys
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y


def check(m, n, scale=1.0):
    torch.manual_seed(0)
    x = torch.randn(m, n) * scale
    assert torch.allclose(softmax(x), torch.softmax(x, axis=1), atol=1e-5), f"mismatch {m}x{n} scale={scale}"


tier = sys.argv[1]
check(181, 781)                   # tutorial shape 1823x781, rows reduced for CPU speed
if tier == "T2":
    check(8, 1000)
    check(8, 33)
    check(8, 128, scale=60.0)     # large logits: needs the max subtraction
print("PASS")
