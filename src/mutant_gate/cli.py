"""mutant-gate command line interface."""
import argparse
import asyncio
import sys
from pathlib import Path

from . import __version__
from .llm import DEFAULT_MODEL, LLMError
from .mutants import KernelFormatError, make_mutants
from .script import diff_of, split_script


def cmd_mutants(args):
    src = Path(args.kernel).read_text()
    split_script(src)   # validates the script format early
    muts = make_mutants(src)
    for op, site, code in muts:
        print(f"{op:20} {site}")
        if args.diff:
            print("\n".join("    " + line for line in diff_of(src, code).splitlines()[2:]))
    print(f"{len(muts)} mutants", file=sys.stderr)
    return 0


def cmd_run(args):
    from .pipeline import analyze   # imported here so `mutants` works without the sandbox SDK configured

    report = asyncio.run(analyze(args.kernel, model=args.model, tries=args.tries, close=not args.no_close,
                                 out_dir=args.out, log=lambda s: print(s, file=sys.stderr), image=args.image,
                                 compile_check=args.compile_check))
    return 0 if report["accepted"] else 2


def cmd_setup_image(args):
    from .sandbox import build_image

    print("building python:3.12-slim + triton + torch(cpu) in Nebius Sandboxes (about a minute)...", file=sys.stderr)
    print(asyncio.run(build_image()))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="mutant-gate", description="Plant bugs in a Triton kernel, check whether your tests catch them, "
                                "and let Nemotron close the gaps. Every verdict comes from running code in a Nebius sandbox.")
    p.add_argument("--version", action="version", version=f"mutant-gate {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mutants", help="list the mutants planted in a kernel script (offline, no API key needed)")
    m.add_argument("kernel", help="kernel script, see examples/kernels/")
    m.add_argument("--diff", action="store_true", help="also print each change")
    m.set_defaults(fn=cmd_mutants)

    r = sub.add_parser("run", help="run the full gate on a kernel script (needs NEBIUS_API_KEY and a sandbox image)")
    r.add_argument("kernel")
    r.add_argument("--model", default=DEFAULT_MODEL, help="Token Factory model id (default: %(default)s)")
    r.add_argument("--tries", type=int, default=4, help="attempts for the first test (default: %(default)s)")
    r.add_argument("--no-close", action="store_true", help="skip stage 2 (targeting the survivors)")
    r.add_argument("--compile-check", action="store_true", help="also compile each survivor for CUDA (CPU only, no GPU needed) and "
                   "set aside survivors that are not valid GPU programs")
    r.add_argument("--out", help="output directory (default: mutant-gate-out/<kernel name>)")
    r.add_argument("--image", help="sandbox image id (default: $MUTANT_GATE_IMAGE or the one from setup-image)")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("setup-image", help="build the sandbox image once (Python + Triton + CPU torch) and remember its id")
    s.set_defaults(fn=cmd_setup_image)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (KernelFormatError, LLMError, RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
