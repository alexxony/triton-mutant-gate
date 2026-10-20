# Additional tests to catch boundary, tail, numeric edge‑case, dtype and memory‑safety bugs

def _tensors_equal(a, b):
    """Element‑wise equality that handles NaNs correctly."""
    if a.is_floating_point():
        return torch.all(torch.isnan(a) & torch.isnan(b) | (a == b))
    else:
        return torch.all(a == b)

def _check_concat(x_list, y_list):
    """Run one add kernel on the concatenated inputs and verify each segment."""
    x_cat = torch.cat(x_list)
    y_cat = torch.cat(y_list)
    out_cat = add(x_cat, y_cat)               # one kernel launch
    out_list = torch.split(out_cat, [t.numel() for t in x_list])
    for i, (x, y, out) in enumerate(zip(x_list, y_list, out_list)):
        expected = x + y
        assert _tensors_equal(out, expected), \
            f"segment {i} failed (size={x.numel()}, dtype={x.dtype})"

# 1. Various sizes – includes tiny, non‑multiple of BLOCK_SIZE, and large but <100k
sizes = [1, 2, 3, 15, 1023, 1024, 1025, 2047, 4095, 50000, 98304]
xs = [torch.randn(s, dtype=torch.float32) for s in sizes]
ys = [torch.randn(s, dtype=torch.float32) for s in sizes]
_check_concat(xs, ys)

# 2. Special floating‑point values (inf, -inf, NaN, large, tiny, subnormal)
special_vals = [
    float('inf'), float('-inf'), float('nan'),
    1e30, -1e30,
    1e-30,
    torch.finfo(torch.float32).tiny,                 # smallest normal
    torch.finfo(torch.float32).tiny * 0.5,           # subnormal
    -torch.finfo(torch.float32).tiny * 0.5
]
# Build tensors by repeating the pattern to get a reasonable length
pattern_len = len(special_vals)
repeat = 20   # gives 20*pattern_len elements (<100k)
x_special = torch.tensor(special_vals * repeat, dtype=torch.float32)
y_special = torch.tensor([v * 2.0 if not torch.isnan(torch.tensor(v)) else float('nan')
                         for v in special_vals] * repeat, dtype=torch.float32)
_check_concat([x_special], [y_special])

# 3. Integer dtype (int32) – small values to avoid overflow
xi = torch.randint(-1000, 1000, (12345,), dtype=torch.int32)
yi = torch.randint(-1000, 1000, (12345,), dtype=torch.int32)
_check_concat([xi], [yi])
