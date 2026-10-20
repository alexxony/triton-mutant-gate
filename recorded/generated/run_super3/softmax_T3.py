def _test_shape(m, n, scale=1.0):
    torch.manual_seed(0)
    x = torch.randn(m, n) * scale
    assert torch.allclose(softmax(x), torch.softmax(x, axis=1), atol=1e-5), \
        f"mismatch {m}x{n} scale={scale}"

# Additional tests to catch boundary/tail and numerical issues
_test_shape(1, 1)                                 # single element
_test_shape(2, 1024, 1.0)                         # power-of-two width
_test_shape(3, 1023, 1.0)                         # non-power-of-two width (tail handling)
_test_shape(128, 128, 1.0)                        # square matrix
_test_shape(1000, 10, 0.1)                        # many rows, small values
_test_shape(10, 1000, 0.1)                        # many columns, small values
