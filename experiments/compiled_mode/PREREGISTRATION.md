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

Run 2026-10-20 11:24 to 11:30 KST (6 minutes, 44 programs x 2 modes), Colab T4: `{python 3.13.15, torch 2.11.0+cu130, triton 3.6.0, CUDA 13.0, Tesla T4}`.
The sandbox verdicts were recorded with Triton 3.8.0. Raw data: `results.json`, `run_log.txt`; mechanical evaluation: `analysis.txt`
(produced by `analyze.py`, written before the run).

Categories over the 40 mutants: compiled = 22 PASS, 15 FAIL_ASSERT, 3 COMPILE_ERROR; Colab interpreter = 20 PASS, 15 FAIL_ASSERT, 5 OTHER
(OTHER = the interpreter without bounds checks crashed on an out-of-bounds store).

| prediction | result |
|---|---|
| P1 original passes compiled for >= 3 of 4 kernels | holds: 4 of 4 |
| P2 `load_other_changed` stays PASS compiled (>= 90%) | holds: 4 of 4 |
| P3 at least one fp16-accumulator mutant detected compiled | holds: 3 of 3, all as COMPILE_ERROR (none as a numeric mismatch) |
| P4 dropped-mask mutants not systematically easier to detect compiled | holds: `drop_load_mask` 8 of 10 PASS compiled, `drop_store_mask` 1 of 4 PASS compiled |

Reading by the pre-registered table:

- layernorm acc#0 and acc#1: `PASS` in the Colab interpreter, `COMPILE_ERROR` compiled, with the message `Loop-carried variable _mean has
  initial type <['1024'], fp16> but is re-assigned to <['1024'], fp32> in loop!` (acc#1: `_var`). This is the row "detected compiled but
  PASS in colab_interp": **confirmed interpreter blind spot for these two mutants** (the interpreter has no loop-carried type check).
- matmul acc#0: `COMPILE_ERROR` compiled, but `FAIL_ASSERT` in the Colab interpreter, while the sandbox interpreter (Triton 3.8.0) let it
  survive. So for this mutant the difference between sandbox and Colab interpreter is a version/machine effect; per the table it is reported
  separately. The stored compile-error detail is only the last stderr line (a caret), so its cause was not read.
- Detection differences between `sandbox_interp` and `colab_interp` (27 of 40 same) are dominated by the bounds checks that only the sandbox
  run had; they are not a compilation effect.

Post-hoc observations (not pre-registered):

- Of the 9 mutants that survived in the sandbox interpreter, 3 are not valid Triton programs on a compiled run (the fp16 accumulators), 4 stay
  PASS in every mode (`load_other_changed`, consistent with the earlier bit-identical evidence of equivalence), and 2 stay PASS in every mode
  and are not equivalent-looking (softmax `drop_max_subtract`, layernorm `mask_lt_to_le` lt#2): the real test gaps.
- One mutant is detected only compiled (layernorm `mask_lt_to_le` lt#3).

Limits: one GPU model, Triton 3.6.0 (sandbox: 3.8.0), four kernels, one test per kernel, single run per cell. No statement about other
hardware, versions or kernels.
