# Pre-registration: output-difference oracle for surviving mutants

Written 2026-10-20 (KST), before any run of the oracle. Nothing above the final line is edited after results exist.

## Question

Today a surviving mutant is only "the generated test's asserts passed". Stage 2 then asks Nemotron, which may answer
`EQUIVALENT` with no execution behind it. Can a deterministic step that runs the original and the mutant side by side and
compares the kernel outputs split the survivors into (a) outputs identical on everything the test launches and (b)
outputs that differ but stay inside the test's tolerance, without any model?

What exists already: `recorded/results/results_survivor_classification.json` holds a one-off offline differential study
(8 survivors, 4 to 6 hand-written inputs each). It is not part of the pipeline. This experiment moves that step into the tool
and makes it generic: it uses the launches the test itself makes, not hand-written inputs.

## Design

- Hook: `triton.runtime.interpreter.InterpretedFunction.run` (the same hook `compilecheck.record` uses). After each launch, a
  copy of every tensor argument is stored (inputs too, kernels can write in place).
- One sandbox invocation per mutant: run the original twice and the mutant once, same seeds (`random`, `numpy`, `torch` all
  seeded to 0 before the script starts), bounds checks on, `TRITON_INTERPRET=1`, Triton 3.8.0 image.
- Classes (decided per mutant, rules fixed here):
  - `IDENTICAL`: same launch count, same shapes/dtypes, every tensor bit-identical in every launch.
  - `DIFFERS`: launch count, shape, dtype or any value differs. Reported with first launch, argument name, max abs diff,
    max rel diff.
  - `MUTANT_FAILED`: the mutant run exits non-zero (an assert or bounds check fired). Not a survivor, not classified.
  - `UNSTABLE`: the two original runs differ from each other. The oracle is skipped for that kernel (determinism gate).
- A `DIFFERS` result is **not** counted as a kill. It is reported on its own line: "outputs move by X, the test's tolerance hides it".
  Nemotron's verified killer tests remain the only "killed" column.
- No model is called. Cost: 3 sandbox runs per program.

## Validation set (no mutants regenerated)

The 44 programs in `experiments/compiled_mode/bundle.json` (4 originals + 40 mutants, each with the demo's generated test).
Recorded sandbox verdicts: 9 survivors (T3B), 31 killed.

## Controls

- Positive control: the 4 `load_other_changed` survivors (layernorm zero#1, zero#2, matmul zero#0, zero#1) were bit-identical on every
  input of the earlier study. If any is `DIFFERS`, the harness is wrong (seeding or snapshot timing), not the mutant. Stop and fix.
- Determinism gate: all 4 originals must come out stable. If one does not, that kernel is excluded and reported.
- Negative control: killed mutants must come out `MUTANT_FAILED`, or `DIFFERS` if their failing assert comes after the first
  divergence. A killed mutant classed `IDENTICAL` is impossible by construction (a failed assert means the run exited non-zero), so
  seeing one means the comparison is broken.

## Predictions for the other 5 survivors (stated before running)

| survivor | prediction | basis | confidence |
|---|---|---|---|
| layernorm `acc_dtype_fp16` acc#0 | IDENTICAL | earlier study: identical on 6 inputs; mutant does not compile for CUDA | 70% |
| layernorm `acc_dtype_fp16` acc#1 | IDENTICAL | same | 70% |
| matmul `acc_dtype_fp16` acc#0 | DIFFERS | earlier study: differs on all 5 inputs | 80% |
| softmax `drop_max_subtract` max#0 | DIFFERS | earlier study: differs on all 4 inputs | 85% |
| layernorm `mask_lt_to_le` lt#2 | IDENTICAL | not in the earlier study; guess, no data | 50% |

Disagreements are reported, not tuned away.

## What counts as success

1. Positive control 4/4 `IDENTICAL`, 4/4 originals stable.
2. All 31 killed mutants are `MUTANT_FAILED` or `DIFFERS`; none `IDENTICAL`.
3. The 9 survivors are fully classified (no `UNSTABLE`), and the table above is filled in with predicted vs observed.

If 1 or 2 fails the oracle is not shipped; the README keeps the current wording ("the model's opinion, not a proof").

## Limits stated in advance

- `IDENTICAL` means identical on the launches the test makes. It is not a proof of equivalence.
- Only the Triton CPU interpreter, Triton 3.8.0, four kernels, one generated test per kernel.
- The oracle sees only values, not the test's tolerance; the report shows the observed magnitude and the user compares it with their own tolerance.

---
Results are appended below this line.
