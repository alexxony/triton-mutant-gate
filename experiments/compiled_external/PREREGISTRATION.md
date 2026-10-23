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

## Amendment A (2026-10-21, after an independent review of this document, before any T4 run; the predictions above stay as written)

1. **Device.** `torch.set_default_device("cuda")` is set in both Colab modes (it was only in the compiled one), so tensors a test creates without a device
   come from the same RNG stream in both. The recorded `autotune_axpy` test hard-codes `device='cpu'` in 9 places (it was written against the CPU-pointed
   script); for the Colab programs only, `device='cpu'` in that test is replaced by `device='cuda'`. The bundle records this substitution.
   Tutorial 03's test uses the file's own `DEVICE` and needs none.
2. **What the columns can be compared on.** `colab_interp` and `colab_compiled` share the machine, the inputs and the shim-free driver, so every
   reading below compares those two. `sandbox_interp` ran on CPU RNG with a faked device and a different Triton build; it is context, not a control.
   The row "`colab_interp` differs from `sandbox_interp` -> Triton version effect" in the table above is withdrawn: inputs, device path, shims and the
   version cannot be told apart here.
3. **Autotune.** Compiled mode autotunes for real and may pick another block size than the interpreter column (which keeps the first config). The
   runner prints `best_config` for every autotuned kernel after each compiled program, so a verdict that differs between the two columns can be checked
   against the configuration actually used.
4. **Order of the run.** First only the two ORIGINAL programs in both modes (4 results). The full batch runs only if all four are `PASS`. Tutorial 03's
   fp8 block (`TORCH_HAS_FP8 and is_cuda()`, no compute-capability check) runs on a T4 and may make the original fail compiled; if so, a second
   amendment (hide `torch.float8_e5m2` in both Colab modes) is written and committed before the full batch.
5. `interp_shims.py` is uploaded next to the runner (the interpreter mode imports it).

---
Results are appended below this line.

## Run 0 (2026-10-21 KST, Amendment A.4 step 1: the two ORIGINAL programs only, both modes)

Colab T4 session, `nvidia-smi`: Tesla T4, compute capability 7.5. Versions: Python 3.13.15, torch 2.11.0+cu130, **triton 3.6.0** (the sandbox used 3.8.0), CUDA 13.0.

| program | `colab_compiled` | `colab_interp` |
|---|---|---|
| `autotune_axpy` original | PASS | PASS |
| tutorial 03 original | **TIMEOUT (400 s)** | PASS |

By the rule above, tutorial 03 is excluded from the full batch until its original passes compiled; none of its mutants are read. The full batch was **not run**.

Why 03 times out is **not known**. A diagnostic (stack dump at 200 s from inside the program) was started in the same session; the Colab connection dropped
(`RuntimeError: Connection was lost.`) and its output never reached the client, so nothing was learned. The two suspects named before the run are untested: the
fp8 block (`TORCH_HAS_FP8 and is_cuda()` with no compute-capability check, so it runs on sm_75) and compiling up to 11 autotune configs. After the drop the VM was
idle (GPU 0 %, no program running). The session was stopped; no further T4 time was used.

Open if resumed: diagnose 03 with the program started under `nohup` on the VM and its output written to a file read by a separate short call, then either amend
(hide `torch.float8_e5m2` in both Colab modes) before the full batch, or run `autotune_axpy` alone (9 mutants, 2 survivors in the sandbox). Nothing in the
predictions table has been tested.

## Amendment B (2026-10-23 KST, after run 0, before any further T4 run; predictions and Amendment A stay as written)

1. **Tutorial 03 is dropped from the compiled batch.** Its original timed out compiled (400 s) in run 0 and the cause is unknown. Under the control rule above
   ("A file whose original does not pass in a column is excluded ... no reading of its mutants") it cannot contribute. No attempt is made here to diagnose it
   (fp8 block and 11 autotune configs remain untested suspects) and no second harness amendment is written: the time cap for this experiment is 1.5 hours.
2. **What runs:** `autotune_axpy` only (original + 9 mutants, 2 survivors in the sandbox interpreter), both modes, same harness, same 400 s timeout.
   `run_remote.py` gets a filter: if `/content/only_kernels` exists, only the kernels named in it are run (one name per line). No other change to the runner.
3. **Predictions still standing:** the 2 `autotune_axpy` rows of the table (PASS, 90% each) and the dropped-mask claim, restricted to `autotune_axpy`'s caught mutants.
   The 3 tutorial-03 rows are **untested and stay untested**; they are not evaluated by this run.
4. **Hard stop:** two Colab connection drops, or 1.5 hours of wall clock from session start. A partial run is reported as partial.
5. **Limits:** now 1 file, 2 survivors, 1 GPU model, 1 Triton build (3.6.0 on Colab), one generated test. An illustration, not a rate.

---
Results are appended below this line (run 0 above remains the record for tutorial 03).

## Run 1 (2026-10-23 KST 13:46-13:54, `autotune_axpy` only, per Amendment B; raw: `recorded/run1_autotune_axpy.json`)

Colab T4 (Tesla T4, sm_75), Python 3.13.15, torch 2.11.0+cu130, **triton 3.6.0** (sandbox: 3.8.0). One session, no connection drop, 10 programs x 2 modes.
The original passed in both columns, so the file was read.

| mutant | sandbox_interp | colab_compiled | colab_interp |
|---|---|---|---|
| `drop_load_mask` axpy:load#0 | KILLED | PASS | PASS |
| `drop_load_mask` axpy:load#1 | KILLED | PASS | PASS |
| `drop_store_mask` axpy:store#0 | KILLED | PASS | OTHER (`double free or corruption (!prev)`) |
| `mask_lt_to_le` axpy:lt#0 | KILLED | PASS | PASS |
| `load_other_changed` axpy:zero#0 | **SURVIVED** | PASS | PASS |
| `load_other_changed` axpy:zero#1 | **SURVIVED** | PASS | PASS |
| `drop_load_mask` rowsum:load#0 | KILLED | FAIL_ASSERT | FAIL_ASSERT |
| `mask_lt_to_le` rowsum:lt#0 | KILLED | FAIL_ASSERT | FAIL_ASSERT |
| `load_other_changed` rowsum:zero#0 | KILLED | FAIL_ASSERT | FAIL_ASSERT |

Against the predictions that stand (Amendment B.3):
- The 2 `autotune_axpy` survivors are `PASS` compiled and in the Colab interpreter: **as predicted** (90% each). Reading from the table above: consistent with equivalence on this
  hardware as well, outputs identical in the sandbox; not a proof.
- Dropped-mask family on this file (`drop_load_mask`, `drop_store_mask`, `mask_lt_to_le`, 6 mutants, all caught in the sandbox): 4 of 6 are `PASS` compiled (the 4 on `axpy_kernel`), 2 detected
  (`rowsum_kernel`, numeric). **As predicted** (at least half). The sandbox caught the 4 `axpy_kernel` ones through its interpreter bounds checks; the Colab interpreter has none,
  and the compiled GPU run does not fault either. In the Colab interpreter 3 of the 4 are also `PASS`; the 4th (`drop_store_mask`) died with a heap-corruption message, which
  by the rule above counts as "detected" in that column only.
- Autotune: every compiled `axpy_kernel` run chose BLOCK_SIZE 128 or 256 (4 warps, 3 stages); the interpreter keeps the first config. The compiled verdicts do not depend on which one was chosen
  here (the same verdict appears for both sizes).
- Not tested, still untested: the 3 tutorial-03 rows of the predictions table.

Limit: 1 file, 2 survivors, 1 GPU model, 1 Triton build, one generated test. The `rowsum_kernel` mutants were detected in every column; the Colab-vs-sandbox difference
is confined to `axpy_kernel`'s mask mutants. Why those pass compiled was not investigated.

## Amendment C (2026-10-23 KST, diagnosis only, before the run; Amendment B and the predictions stay as written)

Purpose: find out where tutorial 03's original spends its time when compiled on the T4 (run 0: TIMEOUT at 400 s, cause unknown). This is not a mutant run and adds no
result to the predictions table.

- One run, no variants: tutorial 03 **original** (from `bundle.json`, same harness as run 0: `torch.set_default_device('cuda')`, `perf_report` benchmark skipped),
  timeout **900 s**. `faulthandler.dump_traceback_later(120, repeat=True)` writes the Python stack every 120 s to a file, so the stuck frame is visible.
- Hypotheses, fixed now: **H1** the time is spent in the module-level fp8 `matmul` call (an fp8 block with no compute-capability check, running on sm_75);
  **H2** it is spent compiling the autotuned fp16 `matmul` (11 configs); **H3** it simply finishes between 400 s and 900 s (slow compile, no hang).
- How it is read: the stack frames (script line numbers, mapped to the program text in `bundle.json`) say which call; a normal exit before 900 s says H3. Several
  hypotheses can hold at once; one run cannot rank them.
- Stop: 900 s of program time or one Colab disconnect, then the session is stopped. Free T4, no approval needed.
- If the cause is found, any rerun of tutorial 03's mutants needs its own amendment before it starts.

## Run 2 (diagnosis, 2026-10-23 KST 16:33-16:45; Colab T4, triton 3.6.0, torch 2.11.0+cu130; raw: `recorded/run2_tutorial03_diagnosis.txt`)

- Result: the original **exited 0 after 699.9 s** (cap 900 s). **H3 holds:** run 0's TIMEOUT at 400 s was a cap that was too short, not a hang.
- Stack tail (the last 3500 of 7451 bytes of the dump file were retrieved; earlier dumps were not kept): the program is inside `autotuner.py _bench -> do_bench ->
  kernel_call -> jit compile` (`make_llir`, `make_ptx`) at script line 383 (`matmul(a, b)` with `float8_e5m2` inputs, the fp8 block) in the earliest retained
  dump, and at line 469 (`matmul(a, b, act)` in the test-case loop) in the two later ones. So the time goes to compiling autotune configs (**H2**), including in the
  fp8 block (**H1**, partly) and again in the later loop. The order of the dumps is inferred from their order in the file. One run on one T4; the host speed was not controlled.
- Consequence, extrapolated and not measured: if each of the 13 mutants took as long as the original (about 700 s), compiled runs of tutorial 03's mutants would need
  about 2.5 hours here. They were not run. A rerun needs its own amendment before it starts (for example fewer autotune configs, which changes the program).
- The predictions table is unchanged; the 3 tutorial-03 rows are still untested.
