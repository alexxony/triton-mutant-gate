import numpy as np

from mutant_gate.payload.diffcheck import classify

A = {"fn": "k", "args": {"x": {"shape": [3], "dtype": "torch.float32"}, "y": {"shape": [3], "dtype": "torch.float32"}}}


def rec(xs, ys, status="ok", launches=None):
    """A recording with one launch of kernel k: tensors x (input) and y (output)."""
    arrays = {}
    for i, (x, y) in enumerate(zip(xs, ys)):
        for name, v in (("x", x), ("y", y)):
            v = np.asarray(v, dtype=np.float32)
            arrays[f"L{i}_{name}_raw"] = v.view(np.uint8)
            arrays[f"L{i}_{name}_f32"] = v
    return {"status": status, "launches": launches or [A] * len(xs)}, arrays


def test_identical():
    a, b = rec([[1, 2, 3]], [[4, 5, 6]]), rec([[1, 2, 3]], [[4, 5, 6]])
    assert classify(a, b) == {"class": "IDENTICAL", "launches": 1}


def test_differs_reports_where_and_how_much():
    a, b = rec([[1, 2, 3]], [[4, 5, 6]]), rec([[1, 2, 3]], [[4, 5, 6.5]])
    r = classify(a, b)
    assert r["class"] == "DIFFERS" and r["first"]["launch"] == 0 and r["first"]["arg"] == "y"
    assert r["first"]["n_diff"] == 1 and r["first"]["max_abs"] == 0.5 and abs(r["first"]["max_rel"] - 0.5 / 6) < 1e-9


def test_input_changed_in_place_counts():
    a, b = rec([[1, 2, 3]], [[4, 5, 6]]), rec([[1, 2, 9]], [[4, 5, 6]])
    assert classify(a, b)["first"]["arg"] == "x"


def test_launch_count_differs():
    a, b = rec([[1, 2, 3]], [[4, 5, 6]]), rec([[1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6]])
    assert classify(a, b)["class"] == "DIFFERS" and "launch count" in classify(a, b)["why"]


def test_shape_or_dtype_change():
    other = {"fn": "k", "args": {"x": {"shape": [3], "dtype": "torch.float16"}, "y": A["args"]["y"]}}
    a, b = rec([[1, 2, 3]], [[4, 5, 6]]), rec([[1, 2, 3]], [[4, 5, 6]], launches=[other])
    assert classify(a, b)["class"] == "DIFFERS"


def test_nan_vs_nan_is_identical_nan_vs_number_is_not():
    nan = float("nan")
    assert classify(rec([[1, 2, 3]], [[nan, 5, 6]]), rec([[1, 2, 3]], [[nan, 5, 6]]))["class"] == "IDENTICAL"
    r = classify(rec([[1, 2, 3]], [[nan, 5, 6]]), rec([[1, 2, 3]], [[0, 5, 6]]))
    assert r["class"] == "DIFFERS" and r["first"]["max_abs"] == float("inf")


def test_negative_zero_is_flagged_as_bits_only():
    r = classify(rec([[1, 2, 3]], [[0.0, 5, 6]]), rec([[1, 2, 3]], [[-0.0, 5, 6]]))
    assert r["class"] == "DIFFERS" and r["first"]["bits_only"] and r["first"]["n_diff"] == 0
