def _run_one_test(M, N, K, *, seed=0, a=None, b=None):
    torch.manual_seed(seed)
    if a is None:
        a = torch.randn((M, K), dtype=torch.float16)
    if b is None:
        b = torch.randn((K, N), dtype=torch.float16)
    ref = torch.matmul(a, b)
    out = matmul(a, b)
    assert torch.allclose(out, ref, atol=0.125, rtol=0), \
        f"matmul mismatch for shape ({M},{N},{K})"

# 1. Non‑multiple dimensions in all axes
_run_one_test(70, 70, 70, seed=1)

# 2. M not multiple of BLOCK_SIZE_M, N not multiple of BLOCK_SIZE_N,
#    K not multiple of BLOCK_SIZE_K
_run_one_test(63, 100, 50, seed=2)

# 3. K not multiple of BLOCK_SIZE_K while M,N are multiples
_run_one_test(128, 128, 33, seed=3)

# 4. Minimal problem size (1x1x1)
_run_one_test(1, 1, 1, seed=4)

# 5. Slightly over the block size in each dimension
_run_one_test(65, 65, 65, seed=5)

# 6. Non‑contiguous inputs (transposed views) to stress stride handling
torch.manual_seed(6)
a_base = torch.randn((70, 70), dtype=torch.float16)   # base for (M,K) after transpose
b_base = torch.randn((70, 70), dtype=torch.float16)   # base for (K,N) after transpose
a = a_base.t()   # shape (70,70), non‑contiguous
b = b_base.t()   # shape (70,70), non‑contiguous
_run_one_test(70, 70, 70, seed=6, a=a, b=b)
