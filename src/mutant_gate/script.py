"""Kernel script format, test-code extraction and the prompts sent to Nemotron.

Script contract (see examples/kernels/): the file contains an @triton.jit kernel, a wrapper, helper functions such as
`check(...)`, then a line `tier = sys.argv[1]` followed by the existing sanity check. Everything after the first
`if tier == "T2":` is ignored here.
"""
import ast
import difflib
import re

from .mutants import KernelFormatError

TIER_MARKER = "\ntier = sys.argv[1]\n"

GENERATE_PROMPT = """Below is a Triton kernel script (CPU interpreter mode, TRITON_INTERPRET=1) with its existing sanity check.
Your job: write ADDITIONAL test code that would expose bugs a careless implementation of this kernel could have
(boundary and tail handling, numerical edge cases, memory safety, dtype/precision, anything specific to this kernel's math).

Rules:
- Return ONE ```python code block containing statements that run right after the existing check. You may define helper functions.
- Reuse the helpers already defined in the script (see below). Do not modify or redefine the kernel.
- Keep runtime small: each test tensor under ~100k elements, at most ~8 kernel launches in total.
- Use assert with a short message. Do not print anything except by failing.

Script so far (kernel, helpers, and the existing sanity check that already runs):
```python
{script}
```"""

CLOSE_PROMPT = """A Triton kernel script (CPU interpreter, TRITON_INTERPRET=1) passes all of its existing tests, but the mutant below
(a small edit of the kernel) ALSO passes them. Decide whether the mutant is behaviourally different from the original.

Case A - the mutant can be distinguished: write extra test code (ONE ```python block, statements appended after the existing
check; reuse the script's helper functions; you may define helpers; keep it to at most ~4 kernel launches and tensors under
~2M elements) that PASSES on the original kernel and FAILS (assert) on the mutant. Out-of-bounds memory access is checked
separately by the harness, so you may rely on assertions about output values and precision.
Case B - no input can tell them apart: reply with the single word EQUIVALENT and one sentence of reasoning, no code.

Original script (kernel, helpers, existing sanity check):
```python
{script}
```

Existing extra tests that the mutant already survives:
```python
{existing}
```

Mutant, as a diff against the original kernel:
```diff
{diff}
```"""


def split_script(src):
    """Return (head, sanity_check): everything before the tier marker, and the base check that follows it."""
    head, sep, rest = src.partition(TIER_MARKER)
    if not sep:
        raise KernelFormatError("missing the line `tier = sys.argv[1]` that separates the kernel/helpers from the sanity check")
    return head, rest.partition('if tier == "T2":')[0]


def shown_script(src):
    head, t1 = split_script(src)
    return head + "\ncheck(...)  # existing sanity check on the tutorial shape (see below)\n" + t1


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\n(.*?)(?:```|\Z)", text, re.S)   # tolerate a fence cut off by max_tokens
    return blocks[0] if blocks else ""


def has_check(gen):
    """Accept a test only if it executes an assert or calls the script's `check` helper (comments/strings do not count)."""
    try:
        tree = ast.parse(gen)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.Assert) or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "check")
               for n in ast.walk(tree))


def diff_of(orig, mutant):
    a, b = split_script(orig)[0].splitlines(), split_script(mutant)[0].splitlines()
    return "\n".join(difflib.unified_diff(a, b, "original", "mutant", lineterm="", n=2))


def full_program(src, test):
    """The script with the sanity check and the extra test, ready to run in the sandbox."""
    head, t1 = split_script(src)
    return head + "\n" + t1 + "\n" + test + '\nprint("PASS")\n'
