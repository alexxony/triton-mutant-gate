# Recorded results

All numbers come from files in `recorded/results/` (raw JSON) and were produced with this pipeline's earlier experiment
scripts. Model temperature 0.2. Kernels: the four in `examples/kernels/` (vecadd, softmax, layernorm, matmul).
Execution is Triton's CPU interpreter in Nebius Sandboxes, with out-of-bounds checks on.

## Survival of planted bugs under Nemotron-written tests (stage 1)

Each model wrote tests for each kernel 5 times (without seeing any mutant). Attempts that an early, too-strict acceptance
rule had rejected were rerun with the fixed rule and are included. Cells are `survived / planted`.

| model | vecadd | softmax | layernorm | matmul | all |
|---|---|---|---|---|---|
| `nvidia/nemotron-3-super-120b-a12b` | 0/20 | 3/20 | 22/90 | 15/65 | 40/195 = 20.5% |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | 0/20 | 2/25 | 21/90 | 14/65 | 37/200 = 18.5% |
| `nvidia/Nemotron-3-Ultra-550b-a55b` | 0/20 | 0/25 | 33/90 | 15/65 | 48/200 = 24.0% |

Reading it: the same weak spots recur across runs (the `other=` change on masked-out lanes and fp16 accumulators in
layernorm and matmul). Ultra's layernorm figure is driven by one run in which 17 of 18 bugs survived. The data does
**not** support ranking the models: samples are small and denominators differ.

## Stage 2: targeting the survivors

For each survivor the model sees the diff and either writes a test that must pass the original and fail the mutant
(checked by execution) or answers EQUIVALENT. Counts below are for the five original repeats per model (before the rerun):

| model | killed by a new, verified test | model said equivalent | no accepted test |
|---|---|---|---|
| Super | 7 | 22 | 4 |
| Nano | 2 | 19 | 7 |
| Ultra | 6 | 26 | 8 |

The mutants the models called equivalent were checked by differential testing (original vs. mutant outputs compared
bit for bit on 4 to 6 inputs): the `other=` changes were identical on every input. That is evidence, not proof.

## Control: can a model judge without running anything?

Super was shown the kernel, a generated test and a mutant diff (no execution) and asked whether the test catches the
mutant. 160 mutants from 16 accepted Super test suites; the executed result is the answer key.

| | value |
|---|---|
| accuracy | 0.869 |
| always answering "caught" | 0.794 |
| survivors it found | 16 of 33 (precision 16 of 20) |

A first attempt used `max_tokens=600`; 103 of 160 replies came back empty because the reasoning budget ate the limit.
That run is invalid and kept only as `results_control_noexec_INVALID_maxtok600.json`. The valid run used 4000.

What this does and does not show: a no-execution judge beats the baseline by 7.5 points and finds most of the
equivalent-looking mutants, so **these numbers do not prove that execution is necessary**. Execution matters most
where numeric behaviour decides the outcome (for example fp16 accumulators: 12 survivors, the judge found 1). The key is
the interpreter's own verdict, and the compiled-run check below shows it is wrong for some of those survivors.

## Compiled GPU run (Colab T4)

Pre-registered in `experiments/compiled_mode/PREREGISTRATION.md` (committed before the run). The same Nemotron-written tests used in the
replay demo were run against the original and all 40 mutants, both in the Triton interpreter and compiled on a Tesla T4
(Triton 3.6.0; the sandbox runs used 3.8.0, so the Colab interpreter was run too, to separate version effects).

- The original kernel passes its test compiled for all 4 kernels.
- The 9 mutants that survived in the sandbox interpreter split three ways: **3 fp16-accumulator mutants do not compile** (layernorm:
  `Loop-carried variable _mean has initial type fp16 but is re-assigned to fp32`; matmul: compile error, cause not read), 4 `other=`
  mutants pass in every mode (consistent with the equivalence evidence), and 2 pass in every mode without looking equivalent
  (softmax `drop_max_subtract`, layernorm `mask_lt_to_le` at lt#2): the real test gaps.
- Dropped-mask mutants are mostly **not** detected compiled (8 of 10 loads pass): an out-of-bounds GPU read returns garbage instead of
  failing. The sandbox's bounds checks are what catch them there.
- For matmul fp16 the Colab interpreter detects the mutant numerically while the sandbox interpreter did not: a version/machine
  effect, not a compilation effect.

So the interpreter-only verdict is wrong in both directions: it misses ill-typed mutants (no loop-carried type check) and a compiled
run misses out-of-bounds reads that the instrumented interpreter catches. One GPU model, one Triton build, four kernels.

## CPU compile check (`mutant-gate run --compile-check`)

Triton compiles for a CUDA target without a GPU (`triton.compile` with an explicit `GPUTarget`, sm_75), so the compile errors
above can be found inside the CPU sandbox. The script first runs under the interpreter to record every kernel launch, then each
distinct launch is specialized the way Triton does for a real launch and compiled.

Checked against the 44 programs of the T4 study (40 mutants + 4 originals; script `experiments/compiled_mode/`): 43 of 44 agree
on "compile error or not". The two layernorm fp16-accumulator mutants fail with the same message as on the T4. The one
disagreement is matmul `acc_dtype_fp16`, which gave a compile error on the T4 (Triton 3.6.0) and compiles in the sandbox
(Triton 3.8.0). The same check with `triton==3.6.0` installed in the sandbox reproduces the error, so the difference comes from
the Triton version, not from the check. A compile verdict therefore holds for the sandbox's Triton build only. (Mutants that
crash the interpreter, such as dropped store masks, report `CHECK_FAILED`; they are caught mutants, never survivors.)

## Output oracle (`--diff-oracle`)

Until now the "equivalent" verdicts rested on the model's word, plus a one-off offline differential study (8 survivors, 4 to 6
hand-written inputs each, `recorded/results/results_survivor_classification.json`) that was not part of the tool. `--diff-oracle` moves
that step into the tool and makes it generic: it hooks `InterpretedFunction.run`, keeps a copy of every tensor argument after every
launch, runs the original twice and the survivor once with the same seeds, and compares bit for bit. No model is called.

Pre-registered in `experiments/diff_oracle/PREREGISTRATION.md` (committed before the first run), validated on the 44 programs of
the T4 study. Result (run 2): all 31 killed mutants fail their run (`MUTANT_FAILED`); the 9 survivors split into 6 `IDENTICAL` and
3 `DIFFERS`. Of 5 predictions made in advance, 4 were right; the wrong one is layernorm `mask_lt_to_le` lt#2 (guessed identical, it is not).

| survivor | oracle | what moved |
|---|---|---|
| layernorm `load_other_changed` zero#1, zero#2; matmul zero#0, zero#1 | identical | |
| layernorm `acc_dtype_fp16` acc#0, acc#1 | identical | (these do not compile for CUDA, see above) |
| softmax `drop_max_subtract` | differs | output: max abs 1.2e-7, plausibly inside the test's tolerance (the oracle does not see the tolerance) |
| matmul `acc_dtype_fp16` | differs | `c_ptr`: 176,378 elements, max abs 0.125 |
| layernorm `mask_lt_to_le` lt#2 | differs | `Rstd`: one element moves by 316 and the test still passes, so no tolerance explains it; the test most likely never checks that output. Stage 2 found no accepted test either |

The useful column is `differs`: it is what surfaced layernorm lt#2, a real gap that the model's stage 2 did not close. Against the
recorded stage-2 outcomes: the 2 survivors Nemotron killed with a verified test are both `DIFFERS`, and the 5 it called equivalent
are all `IDENTICAL`, so no `EQUIVALENT` claim is contradicted. That check is one-sided: `IDENTICAL` is measured on the same
launches that already failed to kill the mutant, so it only means "no contradiction on these inputs"; the earlier hand-written-input
study, which used different inputs, is still the stronger evidence for the `other=` cases. 9 survivors on 4 kernels, no claim about
kernels in general.

Two things went wrong on the way and are kept in the pre-registration: run 1 left the matmul original unstable (the generated test seeds
itself with `hash(str)`, which changes per process; the sandbox now sets `PYTHONHASHSEED=0`), and the sandbox API returned HTTP 500 once
before any measurement (retried). `DIFFERS` is reported on its own line and is not counted as a kill.

## A file with `@triton.autotune`, two kernels and several launches

`examples/plain/autotune_axpy.py` (written for this repo, not taken from a project; 9 mutants over two kernels, the first kernel
autotuned over three block sizes). Trying it found three things that a one-kernel tutorial file never hit, all now fixed:

- The unmodified file failed in the sandbox before anything was judged: `Autotuner.do_bench` asks for a GPU driver
  (`0 active drivers`). The sandbox now loads a small shim (`payload/interp_shims.py`) that replaces the benchmarker, so the autotuner
  keeps its first config. Consequence to know: **only the first config is exercised**, the other block sizes are never run.
- `--compile-check` skipped the file (`CHECK_FAILED`) because it only accepted a bare `JITFunction`; it now unwraps the autotuner.
- The sandbox API answered HTTP 500 three times while this was being tried (image lookup, file lookup); sandbox calls now retry transient 5xx.

Result of one full run (`--diff-oracle --compile-check`, 3,275 + 3,873 tokens): 7 of 9 mutants caught, 2 survived (`other=` on the
masked lanes of `axpy_kernel`); the survivors compile for CUDA; the oracle reports their outputs identical on all 8 launches
and Nemotron answers equivalent for both. One run on one file, so this shows that such a file now goes through the pipeline, not how
well the tests it gets are.

## Known limits

- The interpreter is not a compiler. Triton's own documentation lists interpreter limitations, and upstream issues
  #10658 and #10291 report JIT/interpreter divergences. A compiled T4 run (above) showed that the fp16-accumulator
  survivors are not valid Triton programs. `--compile-check` finds compile errors only; whether a compiled run would pass or
  fail the test (for example dropped masks, which the GPU does not flag) is still not checked.
- Mutation operators are a fixed set of rewrite rules, not real bug reports.
- "Equivalent" is the model's judgement; `--diff-oracle` adds a check that the outputs are bit-identical on the launches the test makes.
  That is evidence on those inputs only, and a different input shape can still tell the two apart.
- Four tutorial-style kernels plus two small plain files (`leaky_relu`, `autotune_axpy`); no claim about kernels in general. With
  `@triton.autotune` only the first config runs in the sandbox.
- Cost of the full study: Super 5 repeats used 343,276 tokens (about $0.17 at list price).
