"""Mutation operators for Triton kernel scripts: small, local source rewrites inside the @triton.jit function.

Generic operators (mask/other removal, `<` to `<=`, fp32 accumulator to fp16, ...) apply to any kernel. Operators tied to a
specific pattern (softmax max-subtract, layernorm eps, matmul K loop, ...) only fire when that pattern is present.
"""
import ast
import re


class KernelFormatError(ValueError):
    pass


def split_args(s):
    """Split a call's argument string on top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


CALL = {name: re.compile(r"tl\.%s\(" % name) for name in ("load", "store")}


def calls(src, name):
    """Yield (start, end, args_str) of every tl.<name>( ... ) call."""
    for m in CALL[name].finditer(src):
        i, depth = m.end(), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        yield m.start(), i, src[m.end():i - 1]


def strip_kw(src, name, kws):
    out = []
    for n, (s, e, args) in enumerate(calls(src, name)):
        parts = split_args(args)
        keep = [p for p in parts if not any(p.strip().startswith(k + "=") for k in kws)]
        if len(keep) != len(parts):
            out.append((f"{name}#{n}", src[:s] + f"tl.{name}(" + ",".join(keep) + ")" + src[e:]))
    return out


def regex_sites(src, pattern, repl, label):
    out = []
    for n, m in enumerate(re.finditer(pattern, src)):
        out.append((f"{label}#{n}", src[:m.start()] + m.expand(repl) + src[m.end():]))
    return out


def operators(kernel_src):
    ops = {}
    ops["drop_load_mask"] = strip_kw(kernel_src, "load", ["mask", "other"])
    ops["drop_store_mask"] = strip_kw(kernel_src, "store", ["mask"])
    ops["mask_lt_to_le"] = regex_sites(kernel_src, r"(?<![<-])(\w+(?:\[[^\]]*\])?) < ((?:\w|\()[^,):]*)", r"\1 <= \2", "lt")
    ops["load_other_changed"] = (regex_sites(kernel_src, r"other=-float\('inf'\)", "other=0.0", "inf")
                                 + regex_sites(kernel_src, r"other=0\.0?(?!\d)", "other=1.0", "zero"))
    ops["drop_max_subtract"] = regex_sites(kernel_src, r"row - tl\.max\(row, axis=0\)", "row", "max")
    ops["drop_eps"] = regex_sites(kernel_src, r"var \+ eps", "var", "eps")
    ops["divisor_off_by_one"] = regex_sites(kernel_src, r"/ N\b", "/ (N - 1)", "div")
    ops["acc_dtype_fp16"] = regex_sites(kernel_src, r"dtype=tl\.float32", "dtype=tl.float16", "acc")
    ops["k_loop_floor_div"] = regex_sites(kernel_src, r"tl\.cdiv\(K, BLOCK_SIZE_K\)", "K // BLOCK_SIZE_K", "kloop")
    ops["drop_modulo_wrap"] = regex_sites(kernel_src, r"\) % [MN]\b", ")", "mod")
    return {k: v for k, v in ops.items() if v}


def _is_jit(decorator):
    d = decorator.func if isinstance(decorator, ast.Call) else decorator   # @triton.jit and @triton.jit(...)
    return (isinstance(d, ast.Attribute) and d.attr == "jit") or (isinstance(d, ast.Name) and d.id == "jit")


def kernel_spans(src):
    """[(name, start, end)] character spans of every @triton.jit function, decorators included."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise KernelFormatError(f"the kernel file does not parse: {e.msg} (line {e.lineno})") from None
    lines = src.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    spans = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(_is_jit(d) for d in node.decorator_list):
            first = min(d.lineno for d in node.decorator_list)
            spans.append((node.name, starts[first - 1], starts[node.end_lineno] - len(lines[node.end_lineno - 1]) +
                          len(lines[node.end_lineno - 1].rstrip("\r\n"))))
    if not spans:
        raise KernelFormatError("no top-level @triton.jit function found")
    return spans


def kernel_span(src):
    """Span of the first kernel."""
    _, start, end = kernel_spans(src)[0]
    return start, end


def make_mutants(src):
    """Return [(operator, site, mutated_source)] for every applicable, syntactically valid mutation.

    With more than one @triton.jit kernel in the file, sites are prefixed with the kernel name (`k2:load#0`).
    """
    spans = kernel_spans(src)
    mutants = []
    for name, s, e in spans:
        head, kern, tail = src[:s], src[s:e], src[e:]
        for op, sites in operators(kern).items():
            for site, new_kern in sites:
                code = head + new_kern + tail
                if code == src:
                    continue
                try:
                    compile(code, "mutant", "exec")
                except SyntaxError:
                    continue
                mutants.append((op, f"{name}:{site}" if len(spans) > 1 else site, code))
    return mutants
