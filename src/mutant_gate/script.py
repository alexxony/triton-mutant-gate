"""Kernel script format, test-code extraction and the prompts sent to Nemotron.

Any Triton file works: the @triton.jit functions are mutated, the imports/defs/classes are the "head", and what runs after
them (an `if __name__ == "__main__":` body, or the top-level statements after the last def) is the existing sanity check.
The examples in examples/kernels/ use an older layout that is still read: a line `tier = sys.argv[1]` separates the head
from the sanity check, and everything after the first `if tier == "T2":` is ignored.
"""
import ast
import difflib
import re
import textwrap

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


DEFINITIONS = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _is_main_guard(node):
    t = node.test if isinstance(node, ast.If) else None
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
            and len(t.comparators) == 1 and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == "__main__")


def _split_plain(src):
    """Any other kernel file: the definitions are the head, whatever runs at the end (an `if __name__ == "__main__":` body or
    the top-level statements after the last import/def/class) is the existing sanity check."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise KernelFormatError(f"the kernel file does not parse: {e.msg} (line {e.lineno})") from None
    lines = src.splitlines(keepends=True)
    guard = next((n for n in tree.body if _is_main_guard(n)), None)
    if guard is not None:
        body = "".join(lines[guard.body[0].lineno - 1:guard.end_lineno])
        return "".join(lines[:guard.lineno - 1]), textwrap.dedent(body)
    last_def = max((i for i, n in enumerate(tree.body) if isinstance(n, DEFINITIONS)), default=-1)
    if last_def + 1 >= len(tree.body):
        return src, ""
    cut = tree.body[last_def + 1].lineno - 1
    return "".join(lines[:cut]), "".join(lines[cut:])


def is_marker_format(src):
    return TIER_MARKER in src


def split_script(src):
    """Return (head, sanity_check): the kernels and helpers, and the existing check that runs after them.

    Two layouts: the examples' (a `tier = sys.argv[1]` line separates them) and any ordinary file (see _split_plain).
    """
    if is_marker_format(src):
        head, _, rest = src.partition(TIER_MARKER)
        return head, rest.partition('if tier == "T2":')[0]
    return _split_plain(src)


def shown_script(src):
    head, t1 = split_script(src)
    if is_marker_format(src):
        return head + "\ncheck(...)  # existing sanity check on the tutorial shape (see below)\n" + t1
    return head + ("\n# existing sanity check that runs after the definitions:\n" + t1 if t1.strip() else "")


def cpuize(src):
    """The sandbox is CPU-only: point 'cuda' device strings and .cuda() calls at the CPU. Returns (new_src, number_of_edits)."""
    new, n1 = re.subn(r"""(['"])cuda(?::\d+)?\1""", '"cpu"', src)
    new, n2 = re.subn(r"\.cuda\(\)", "", new)
    return new, n1 + n2


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
