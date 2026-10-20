# Pre-registration: interpreter vs. compiled GPU execution of the same generated tests

Written 2026-10-20 before any compiled-mode run. Nothing below was edited after results existed; a result section is appended below the line at the end.

## Question

The study's verdicts come from Triton's CPU interpreter. Does the same Nemotron-written test suite give the same verdict
(pass / fail) on a compiled GPU run (Colab T4)? In particular, are the survivors that the interpreter let through
(fp16 accumulators, `other=` changes, dropped masks) really undetectable, or only undetectable in the interpreter?

## Design

- Programs: for each of the four example kernels, the test shown in the replay demo (recorded run: vecadd super1, softmax super2,
  layernorm super1, matmul super1). Each is run against the original kernel and all its mutants (4 + 5 + 18 + 13 = 40 mutants).
- Three verdict columns per mutant: (a) `sandbox_interp`: recorded T3B result (Nebius sandbox, Triton 3.8.0, bounds checks on);
  (b) `colab_interp`: Triton interpreter on the Colab machine, plain (no bounds checks); (c) `colab_compiled`: normal compiled
  execution on the Colab T4 with `torch.set_default_device("cuda")`.
  (b) exists so that (a) vs (c) is not confounded by a different Triton version or machine.
- Every program runs in its own subprocess (a bad memory access can kill a CUDA context), 180 s timeout.
- Result categories: `PASS` (exit 0), `FAIL_ASSERT` (AssertionError), `COMPILE_ERROR` (Triton/MLIR compilation or type error),
  `CUDA_ERROR` (illegal address, launch failure), `TIMEOUT`, `OTHER`. A mutant counts as **detected** if its result is not `PASS`.
- Software versions (triton, torch, CUDA, GPU name) are recorded. The Colab Triton version may differ from 3.8.0; that is
  reported, not hidden.

## Predictions (stated before running)

- P1. The original kernel passes the test in compiled mode for at least 3 of 4 kernels. The risk is tolerance (matmul on tensor cores).
- P2. The `load_other_changed` mutants (outputs were bit-identical on all sampled interpreter inputs) stay `PASS` in compiled mode (at least 90% of them).
- P3. At least one fp16-accumulator mutant (layernorm acc#0/acc#1, matmul acc#0) is detected in compiled mode, either as a
  compile/type error or as a numeric failure. My confidence is about 60%.
- P4. Dropped-mask mutants are not systematically *easier* to detect compiled than in the sandbox interpreter with bounds checks:
  an out-of-bounds read on a GPU usually returns garbage rather than faulting, so many are expected to be `PASS`.

## How each outcome will be read

| outcome | reading | consequence for the project |
|---|---|---|
| fp16-accumulator survivors detected compiled but `PASS` in colab_interp | confirmed interpreter blind spot for those mutants | add an optional compiled verification step; state it as a finding, still one kernel family, one GPU |
| fp16-accumulator survivors `PASS` in both | genuine test gaps (or equivalent at the test tolerance); the interpreter is not the cause | drop the "interpreter artifact" hypothesis from the docs |
| original fails compiled | the test is not portable (interpreter tolerances) | list those kernels as not comparable; do not count their mutants |
| detection differs between sandbox_interp and colab_interp | Triton version or machine effect | report separately; do not attribute to compilation |
| T4 unavailable or two failed attempts of the same kind | stop and report the cause | no result claimed |

## Not claimed whatever the outcome

One GPU model, one Triton build, four kernels, one test per kernel. No statement about other hardware or kernels.

---

## Results

(appended after the run)
