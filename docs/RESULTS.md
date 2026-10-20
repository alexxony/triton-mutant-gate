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

## Known limits

- The interpreter is not a compiler. Triton's own documentation lists interpreter limitations, and upstream issues
  #10658 and #10291 report JIT/interpreter divergences. A compiled T4 run (above) showed that the fp16-accumulator
  survivors are not valid Triton programs. `--compile-check` finds compile errors only; whether a compiled run would pass or
  fail the test (for example dropped masks, which the GPU does not flag) is still not checked.
- Mutation operators are a fixed set of rewrite rules, not real bug reports.
- "Equivalent" is the model's judgement plus sampled inputs.
- Four kernels, one tutorial-style kernel each; no claim about kernels in general.
- Cost of the full study: Super 5 repeats used 343,276 tokens (about $0.17 at list price).
