# Pre-registration: compiled T4 run of surviving mutants on files that were not written for this tool

Written 2026-10-21 (KST) before any T4 run of these programs. Nothing above the final line is edited after results exist.

## Question

`experiments/compiled_mode/` ran the demo's four tutorial-style kernels (adapted for the CPU interpreter) on a Colab T4 and found that
the interpreter's survivors are not all alike: some do not compile, some are equivalent, some are real gaps. Does the same hold on code
that was not adapted: the unmodified Triton tutorial 03 (`triton-lang/triton`, commit `16491796407ec76ef223f8227e7d9a96c3723363`, MIT) and
`examples/plain/autotune_axpy.py`? In particular, for the 5 survivors below (the sandbox interpreter's verdict, see `docs/RESULTS.md`),
what does a compiled run on a T4 say?

## Design

- Programs: for each file, the original and every mutant, each as one standalone program = (kernel file, its existing sanity check, the
  recorded Nemotron test). The tests are in `recorded/*/generated_test.py` (accepted in the sandbox run that produced `recorded/*/report.json`).
  Tutorial 03 has 13 mutants, `autotune_axpy` has 9; mutants are regenerated from the unmodified source with `make_mutants` and must
  match the recorded (operator, site) set exactly, else the prep script stops. The tutorial source is fetched at the pinned commit and its
  sha256 checked; it is not copied into this repository.
- Columns per program: `sandbox_interp` (recorded T3B: Nebius sandbox, Triton 3.8.0, bounds checks on, device shims); `colab_interp` (Triton
  interpreter on the Colab machine, no bounds checks; autotuner benchmark replaced by the repo's `interp_shims.py` so it keeps the first
  config, as in the sandbox); `colab_compiled` (normal compiled execution on the T4, `torch.set_default_device("cuda")`, real autotuning).
- Harness decision made before the run: `triton.testing.perf_report` benchmarks (`Mark.run`, the last statement of tutorial 03) are skipped in
  both Colab modes. Otherwise the original's "sanity check" would be a full benchmark sweep and every program would time out. Nothing else in the
  program is changed.
- Each program runs in its own subprocess, timeout 400 s (tutorial 03 compiles up to 11 autotune configs per program). Results:
  `PASS` (exit 0), `FAIL_ASSERT`, `COMPILE_ERROR`, `CUDA_ERROR`, `TIMEOUT`, `OTHER`. A mutant is **detected** in a column if the result is not `PASS`.
- Software versions (triton, torch, CUDA, GPU) are recorded. The Colab Triton may differ from 3.8.0; that is reported.

## Controls

- Both originals must be `PASS` in `colab_compiled` and `colab_interp`. A file whose original does not pass in a column is excluded from that column
  and reported as such (no reading of its mutants).
- The 7 + 4 mutants that the sandbox interpreter caught are run too. They are not the question; they show what the compiled column does to mutants
  already known to be caught (the earlier study found most dropped-mask mutants undetected when compiled).

## Predictions for the 5 survivors (stated before running)

| survivor | sandbox oracle (earlier) | predicted `colab_compiled` | confidence |
|---|---|---|---|
| tutorial 03 `load_other_changed` zero#0 | outputs identical on 9 launches | PASS | 85% |
| tutorial 03 `load_other_changed` zero#1 | outputs identical on 9 launches | PASS | 85% |
| tutorial 03 `acc_dtype_fp16` acc#0 | outputs differ (c_ptr, max abs 0.0078) | COMPILE_ERROR (accumulator type mismatch, as for the earlier matmul on Triton 3.6.0) | 60% |
| `autotune_axpy` `load_other_changed` zero#0 | outputs identical on 8 launches | PASS | 90% |
| `autotune_axpy` `load_other_changed` zero#1 | outputs identical on 8 launches | PASS | 90% |

Further: `colab_interp` equals `sandbox_interp` (PASS) for all 5 (80%); if not, the Colab Triton version is the first suspect. Of the 17 mutants
caught in the sandbox, at least half of the dropped-mask ones (`drop_load_mask`, `drop_store_mask`, `mask_lt_to_le`) are `PASS` in `colab_compiled`
(70%), repeating the earlier finding that the GPU does not fault on out-of-bounds reads.

## How each outcome will be read

| outcome | reading |
|---|---|
| `other=` survivors `PASS` compiled, outputs identical in the sandbox | consistent with equivalence on this hardware as well; still not a proof |
| fp16-accumulator survivor `COMPILE_ERROR` compiled | not a test gap: the interpreter accepted an invalid program. `--compile-check` could have said so, but cannot run on tutorial 03 (head launches kernels), so this is the case where a GPU step would add information |
| fp16-accumulator survivor `PASS` compiled | a real gap on this GPU too (the oracle saw outputs differ) |
| fp16-accumulator survivor detected numerically compiled (`FAIL_ASSERT`) | the interpreter's survival was a numeric-version or interpreter effect |
| original fails compiled | tutorial 03 / the generated test is not portable to this run; nothing about its mutants is read |
| `colab_interp` differs from `sandbox_interp` | Triton version effect, reported before anything else |

## Limits stated in advance

2 files, 5 survivors, 1 GPU model, 1 Triton build, one generated test per file. The result is an illustration for those files, not a rate.
The Colab session is development tooling, not part of `mutant-gate run`.

---
Results are appended below this line.
