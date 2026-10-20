def _extra_check_fixed(m, n, scale=1.0, dtype=torch.float32, noncontig_rows=False):
    torch.manual_seed(0)
    # create a contiguous base tensor
    base = torch.randn(m * (2 if noncontig_rows else 1), n, dtype=dtype) * scale
    if noncontig_rows:
        # take every other row -> row stride becomes 2*n, column stride stays 1
        x = base[::2, :m]  # shape (m, n)
    else:
        x = base[:m, :n]   # shape (m, n), fully contiguous
    y = softmax(x)
    y_ref = torch.softmax(x, dim=1)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5), \
        f"m={m}, n={n}, dtype={dtype}, noncontig_rows={noncontig_rows}, scale={scale}"

# original contiguous cases
_extra_check_fixed(128, 1, scale=1.0)
_extra_check_fixed(20, 781, scale=0.5)
_extra_check_fixed(5, 1024, scale=2.0)
_extra_check_fixed(16, 128, scale=1.0, dtype=torch.float64)
_extra_check_fixed(8, 256, scale=1e4)
_extra_check_fixed(8, 256, scale=1e-4)

# non‑contiguous in the row dimension (stride(0) != n, stride(1) == 1)
_extra_check_fixed(50, 30, scale=1.0, noncontig_rows=True)
