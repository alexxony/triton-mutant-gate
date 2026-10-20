def _run_extra_tests():
    def test_case(m, n, scale=1.0):
        torch.manual_seed(0)
        x = torch.randn(m, n, dtype=torch.float32) * scale
        y = softmax(x)
        ref = torch.softmax(x, dim=1)
        assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5), \
            f"softmax mismatch: m={m}, n={n}, scale={scale}"

    # Edge cases: very small column counts
    test_case(20, 1)          # n_cols = 1
    test_case(20, 2)          # n_cols = 2
    test_case(20, 3)          # n_cols = 3 (non‑power‑of‑two)

    # Power‑of‑two and just‑over sizes
    test_case(20, 128)        # n_cols = power of two
    test_case(20, 129)        # n_cols = just over power of two

    # Larger row count to stress row‑loop
    test_case(150, 50)        # moderate size

    # Different scales to test numerical stability
    test_case(30, 40, scale=0.1)
    test_case(30, 40, scale=10.0)

# Run the additional tests
_run_extra_tests()
