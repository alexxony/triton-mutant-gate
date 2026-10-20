def _run_test(M, N, K, a=None, b=None, tag=""):
    """Helper that runs matmul and compares to torch.matmul."""
    torch.manual_seed(0 if a is None else 1234)  # deterministic seed for random data
    if a is None:
        a = torch.randn((M, K), dtype=torch.float16)
    if b is None:
        b = torch.randn((K, N), dtype=torch.float16)
    ref = torch.matmul(a, b)
    got = matmul(a, b)
    # tolerance matches the original check
    assert torch.allclose(got, ref, atol=0.125, rtol=0), \
        f"{tag} failure for shape ({M},{N},{K})"

# ---- Additional test cases -------------------------------------------------

# 1. Minimal size
_run_test(1, 1, 1, tag="minimal")

# 2. Small non‑power‑of‑two dimensions
_run_test(2, 3, 5, tag="small_odd")

# 3. Mixed tail cases (M,N,K not multiples of block sizes)
_run_test(33, 65, 33, tag="mixed_tails")

# 4. All dimensions just over a block boundary (stress tail handling)
_run_test(129, 129, 129, tag="just_over_block")

# 5. Transpose A (non‑contiguous stride)
M, N, K = 64, 64, 64
a_t = torch.randn((K, M), dtype=torch.float16).t()  # shape (M,K) but transposed stride
b = torch.randn((K, N), dtype=torch.float16)
ref = torch.matmul(a_t, b)
got = matmul(a_t, b)
assert torch.allclose(got, ref, atol=0.125, rtol=0), \
    "transpose_A failure"

# 6. Transpose B (non‑contiguous stride)
a = torch.randn((M, K), dtype=torch.float16)
b_t = torch.randn((N, K), dtype=torch.float16).t()  # shape (K,N) transposed
ref = torch.matmul(a, b_t)
got = matmul(a, b_t)
assert torch.allclose(got, ref, atol=0.125, rtol=0), \
    "transpose_B failure"

# 7. Extreme values – large numbers (testing fp16 overflow path)
M, N, K = 64, 64, 2
a_large = torch.full((M, K), 6e3, dtype=torch.float16)   # near fp16 max
b_ones  = torch.ones((K, N), dtype=torch.float16)
ref = torch.matmul(a_large, b_ones)
got = matmul(a_large, b_ones)
assert torch.allclose(got, ref, atol=0.125, rtol=0), \
    "large_values failure"

# 8. Extreme values – very small numbers (testing subnormal / underflow)
M, N, K = 64, 64, 64
a_small = torch.full((M, K), 1e-4, dtype=torch.float16)
b_small = torch.full((K, N), 1e-4, dtype=torch.float16)
ref = torch.matmul(a_small, b_small)
got = matmul(a_small, b_small)
assert torch.allclose(got, ref, atol=0.125, rtol=0), \
    "small_values failure"
