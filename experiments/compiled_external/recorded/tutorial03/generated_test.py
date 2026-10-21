# Additional tests to expose boundary, numerical, and edge-case bugs
def torch_ref(a, b, activation=""):
    a_float = a.to(torch.float32)
    b_float = b.to(torch.float32)
    res = torch.matmul(a_float, b_float)
    if activation == "leaky_relu":
        res = torch.where(res >= 0, res, 0.01 * res)
    return res.to(torch.float16)

test_cases = [
    # (description, a_fill, b_fill, activation, expect_nan)
    ("zeros_no_act", 0.0, 0.0, "", False),
    ("zeros_act", 0.0, 0.0, "leaky_relu", False),
    ("ones_no_act", 1.0, 1.0, "", False),
    ("ones_act", 1.0, 1.0, "leaky_relu", False),
    ("neg_one_no_act", -1.0, 1.0, "", False),
    ("neg_one_act", -1.0, 1.0, "leaky_relu", False),
    ("inf_zero_no_act", float('inf'), 0.0, "", True),
    ("nan_one_no_act", float('nan'), 1.0, "", True),
]

for desc, aval, bval, act, is_nan in test_cases:
    a = torch.full((10, 10), aval, dtype=torch.float16, device=DEVICE)
    b = torch.full((10, 10), bval, dtype=torch.float16, device=DEVICE)
    triton_out = matmul(a, b, act)
    torch_out = torch_ref(a, b, act)
    if is_nan:
        assert torch.all(torch.isnan(triton_out)), f"{desc}: Triton output not all NaN"
        assert torch.all(torch.isnan(torch_out)), f"{desc}: Reference output not all NaN"
    else:
        assert torch.allclose(triton_out, torch_out, atol=1e-2, rtol=0), f"{desc}: Output mismatch"
