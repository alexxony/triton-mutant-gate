# Additional tests to stress boundary/tail handling and edge cases
torch.manual_seed(123)

# 1. axpy with empty tensor (n=0)
n = 0
x = torch.empty(n, device='cpu')
y = torch.empty(n, device='cpu')
out = axpy(x, y, 3.14)
assert out.shape == torch.Size([0]), "axpy output shape mismatch for n=0"

# 2. axpy with single element (n=1)
n = 1
x = torch.randn(n, device='cpu')
y = torch.randn(n, device='cpu')
a = -2.5
out = axpy(x, y, a)
torch.testing.assert_close(out, a * x + y, rtol=1e-6, atol=1e-6,
                           msg="axpy failed for n=1")

# 3. axpy with size equal to a typical block size (n=64)
n = 64
x = torch.randn(n, device='cpu')
y = torch.randn(n, device='cpu')
a = 1.5
out = axpy(x, y, a)
torch.testing.assert_close(out, a * x + y, rtol=1e-6, atol=1e-6,
                           msg="axpy failed for n=64 (block-aligned)")

# 4. axpy with zero scalar (a=0) – should return y unchanged
n = 123
x = torch.randn(n, device='cpu')
y = torch.randn(n, device='cpu')
a = 0.0
out = axpy(x, y, a)
torch.testing.assert_close(out, y, rtol=1e-6, atol=1e-6,
                           msg="axpy with a=0 should return y")

# 5. rowsum with a single column (cols=1)
rows, cols = 57, 1
m = torch.randn(rows, cols, device='cpu')
out = rowsum(m)
torch.testing.assert_close(out, m.sum(dim=1), rtol=1e-5, atol=1e-5,
                           msg="rowsum failed for single column")
