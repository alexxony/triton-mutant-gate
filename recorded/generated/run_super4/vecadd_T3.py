# Additional tests – limited to ≤8 kernel launches, each tensor <100k elements
BLOCK = 1024

def _run_add_test(x, y, msg):
    out = add(x, y)
    ref = x + y
    if x.is_floating_point():
        assert torch.allclose(out, ref, equal_nan=True), f"{msg}: floating-point mismatch"
    else:
        assert torch.equal(out, ref), f"{msg}: integer/bool mismatch"

# 1. Empty tensor
torch.manual_seed(0)
x = torch.empty(0, dtype=torch.float32)
y = torch.empty(0, dtype=torch.float32)
_run_add_test(x, y, "size=0")

# 2. Single element
torch.manual_seed(1)
x = torch.rand(1, dtype=torch.float32)
y = torch.rand(1, dtype=torch.float32)
_run_add_test(x, y, "size=1")

# 3. Just under a block
torch.manual_seed(2)
x = torch.rand(BLOCK - 1, dtype=torch.float32)
y = torch.rand(BLOCK - 1, dtype=torch.float32)
_run_add_test(x, y, f"size={BLOCK-1}")

# 4. Exact block
torch.manual_seed(3)
x = torch.rand(BLOCK, dtype=torch.float32)
y = torch.rand(BLOCK, dtype=torch.float32)
_run_add_test(x, y, f"size={BLOCK}")

# 5. One over a block
torch.manual_seed(4)
x = torch.rand(BLOCK + 1, dtype=torch.float32)
y = torch.rand(BLOCK + 1, dtype=torch.float32)
_run_add_test(x, y, f"size={BLOCK+1}")

# 6. Just under two blocks
torch.manual_seed(5)
x = torch.rand(2 * BLOCK - 1, dtype=torch.float32)
y = torch.rand(2 * BLOCK - 1, dtype=torch.float32)
_run_add_test(x, y, f"size={2*BLOCK-1}")

# 7. Exact two blocks
torch.manual_seed(6)
x = torch.rand(2 * BLOCK, dtype=torch.float32)
y = torch.rand(2 * BLOCK, dtype=torch.float32)
_run_add_test(x, y, f"size={2*BLOCK}")

# 8. One over two blocks (also exercises tail handling)
torch.manual_seed(7)
x = torch.rand(2 * BLOCK + 1, dtype=torch.float32)
y = torch.rand(2 * BLOCK + 1, dtype=torch.float32)
_run_add_test(x, y, f"size={2*BLOCK+1}")

# ---- Additional property tests (reuse same launches by varying dtype/values) ----
# We'll perform a few more checks using the same launch count by
# temporarily overriding the global BLOCK_SIZE in the kernel call via a new wrapper.
# However, to stay within the launch limit we instead run a few quick checks
# on already allocated tensors without launching new kernels by calling add()
# which will launch a kernel each time. To keep total launches ≤8 we stop here.
# The above eight launches already cover boundary/tail cases.
# If further tests are needed, they would exceed the limit, so we end.
