def test_constant_row_tol(M, N, eps=1e-5):
    """Layer norm of a constant row should match the bias (allowing for accumulation round‑off)."""
    torch.manual_seed(0)
    x = torch.full((M, N), 3.14, dtype=torch.float32)
    w = torch.rand(N, dtype=torch.float32)
    b = torch.rand(N, dtype=torch.float32)
    y = layer_norm(x, w, b, eps)
    expected = b.expand_as(x)  # bias broadcast to each row
    # Tolerate larger error due to round‑off in the mean/variance accumulations
    assert torch.allclose(y, expected, atol=1e-2, rtol=0), \
        f"constant row mismatch M={M}, N={N}, eps={eps}"

# Exercise boundary/tail handling with constant inputs
test_constant_row_tol(1, 1)          # smallest tensor
test_constant_row_tol(2, 1024)       # exact block size
test_constant_row_tol(4, 1025)       # just over block size → tail in second iteration
