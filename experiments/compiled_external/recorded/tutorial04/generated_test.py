# Additional tests for boundary/tail handling, seeding, and numerical correctness
# Total kernel launches: 8

import torch

# Test 1: p=0, size=1 -> identity
x = torch.randn(1, device=DEVICE, dtype=torch.float32)
out = seeded_dropout(x, 0.0, seed=123)
assert torch.allclose(out, x, atol=1e-6), "p=0 failed for size=1"

# Test 2: p=1, size=1 -> zero
x = torch.randn(1, device=DEVICE, dtype=torch.float32)
out = seeded_dropout(x, 1.0, seed=123)
assert torch.allclose(out, torch.zeros_like(x), atol=1e-6), "p=1 failed for size=1"

# Test 3: p=0.5, size=1023, seed=123 -> scaling and zero mask
x = torch.randn(1023, device=DEVICE, dtype=torch.float32)
out = seeded_dropout(x, 0.5, seed=123)
nz = out != 0
if nz.any():
    expected = x[nz] / (1.0 - 0.5)
    assert torch.allclose(out[nz], expected, atol=1e-5, rtol=0), "scaling failed for size=1023 seed=123"
zm = out == 0
assert torch.allclose(out[zm], torch.zeros_like(out[zm]), atol=1e-6), "zero mask failed for size=1023 seed=123"

# Test 4: p=0.5, size=1023, seed=456 -> scaling and different from seed 123
x = torch.randn(1023, device=DEVICE, dtype=torch.float32)
out2 = seeded_dropout(x, 0.5, seed=456)
nz2 = out2 != 0
if nz2.any():
    expected2 = x[nz2] / (1.0 - 0.5)
    assert torch.allclose(out2[nz2], expected2, atol=1e-5, rtol=0), "scaling failed for size=1023 seed=456"
assert not torch.allclose(out, out2, atol=1e-6), "different seeds produced identical output for size=1023"

# Test 5: p=0.5, size=1024 (exact block) -> scaling
x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
out = seeded_dropout(x, 0.5, seed=123)
nz = out != 0
if nz.any():
    expected = x[nz] / (1.0 - 0.5)
    assert torch.allclose(out[nz], expected, atol=1e-5, rtol=0), "scaling failed for size=1024"
zm = out == 0
assert torch.allclose(out[zm], torch.zeros_like(out[zm]), atol=1e-6), "zero mask failed for size=1024"

# Test 6: p=0.5, size=1025 (one over block) -> scaling
x = torch.randn(1025, device=DEVICE, dtype=torch.float32)
out = seeded_dropout(x, 0.5, seed=123)
nz = out != 0
if nz.any():
    expected = x[nz] / (1.0 - 0.5)
    assert torch.allclose(out[nz], expected, atol=1e-5, rtol=0), "scaling failed for size=1025"
zm = out == 0
assert torch.allclose(out[zm], torch.zeros_like(out[zm]), atol=1e-6), "zero mask failed for size=1025"

# Test 7: p=0, size=2049 (multiple blocks + tail) -> identity
x = torch.randn(2049, device=DEVICE, dtype=torch.float32)
out = seeded_dropout(x, 0.0, seed=123)
assert torch.allclose(out, x, atol=1e-6), "p=0 failed for size=2049"

# Test 8: p=1, size=2049 -> zero
x = torch.randn(2049, device=DEVICE, dtype=torch.float32)
out = seeded_dropout(x, 1.0, seed=123)
assert torch.allclose(out, torch.zeros_like(x), atol=1e-6), "p=1 failed for size=2049"
