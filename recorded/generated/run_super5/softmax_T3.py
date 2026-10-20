# Additional tests that stress boundary, tail, and stride handling
def _run_test(m, n, scale=1.0, pad=0):
    """Test with optional padding to create non‑unit row stride while keeping row‑major layout."""
    torch.manual_seed(0)
    # Allocate a slightly wider matrix and take a view to get a strided row stride
    wide = torch.randn(m, n + pad, dtype=torch.float32) * scale
    x = wide[:, :n]                     # shape (m, n), stride[0] = n+pad, stride[1] = 1
    y = softmax(x)
    y_ref = torch.softmax(x, dim=1)
    assert torch.allclose(y, y_ref, atol=1e-5), \
        f"softmax mismatch: m={m}, n={n}, pad={pad}, scale={scale}"

# ---- Exercise various sizes and strides ----
# Minimal width
_run_test(2, 1, pad=0)
_run_test(2, 1, pad=7)   # row stride > width
# Small power‑of‑two widths
_run_test(2, 2, pad=0)
_run_test(2, 2, pad=3)
# Just over a small power‑of‑two (tail handling)
_run_test(2, 3, pad=0)
_run_test(2, 3, pad=4)
# Large power‑of‑two width
_run_test(2, 1024, pad=0)
_run_test(2, 1024, pad=128)
# Just over a large power‑of‑two (tail handling + stride)
_run_test(2, 1025, pad=0)
_run_test(2, 1025, pad=200)
# Medium width with substantial padding (non‑unit stride)
_run_test(2, 1000, pad=500)
