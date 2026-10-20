def _run_test(M, N, K, tag=""):
    torch.manual_seed(1234 + hash(tag))
    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    ref = torch.matmul(a, b)
    got = matmul(a, b)
    # tolerance matches the original check
    assert torch.allclose(got, ref, atol=0.125, rtol=0), \
        f"{tag} shape {M}x{N}x{K} failed"

# 1) All dimensions not multiples of block sizes
_run_test(63, 65, 61, "non-multiple")

# 2) M multiple of BLOCK_SIZE_M, others not
_run_test(64, 65, 63, "M-multiple")

# 3) Very small M (should trigger wrap‑around in A loads)
_run_test(1, 128, 64, "tiny-M")

# 4) Very small N (should trigger wrap‑around in B loads)
_run_test(128, 1, 64, "tiny-N")

# 5) Very small K (inner dimension wrap‑around)
_run_test(64, 64, 1, "tiny-K")

# 6) Non‑contiguous inputs (stride not unit)
a_base = torch.randn((64, 128), dtype=torch.float16)   # K=64, M=128
b_base = torch.randn((128, 64), dtype=torch.float16)   # K=128, N=64
a = a_base.t()   # shape (128,64) but stride changed
b = b_base.t()   # shape (64,128) but stride changed
# Now a is (128,64) i.e. M=128,K=64 ; b is (64,128) i.e. K=64,N=128
ref = torch.matmul(a, b)
got = matmul(a, b)
assert torch.allclose(got, ref, atol=0.125, rtol=0), "non-contiguous test failed"

# 7) Large but still under ~100k elements per tensor
#    Choose M*K < 100k and K*N < 100k
_run_test(200, 200, 20, "medium-square")   # 200*20=4k, 20*200=4k

# 8) Odd prime dimensions to maximize wrap‑effects
_run_test(97, 101, 103, "prime-sizes")
