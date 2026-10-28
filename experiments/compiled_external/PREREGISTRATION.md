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

## Amendment D (2026-10-28 KST, before any T4 run of tutorial 03's mutants; the predictions table, Amendments A to C and the row-reading table stay as written)

Purpose: run tutorial 03's 13 mutants compiled on a T4, the run that Run 2 estimated at about 2.5 hours. The program text is not changed.

1. **Timeout.** `colab_compiled` gets 1500 s per program (read from `/content/timeout`; Run 0 and Run 2 show the original needs about 700 s). The `colab_interp` column keeps 400 s.
2. **Control in the same session.** The tutorial 03 ORIGINAL runs first, in both columns. Run 2's 699.9 s (rc 0, `PASS` in stdout) is the earlier reference, not the control, because the Colab image and the host speed may differ. If the original is not `PASS` compiled, none of its mutants are read (control rule above).
3. **Order.** After the original, the 3 sandbox survivors first (`load_other_changed` zero#0 and zero#1, `acc_dtype_fp16` acc#0: the only rows with a prediction), then the 10 caught mutants in recorded order. If the cap below cuts the run, it cuts caught mutants, not survivors.
4. **Caps.** Wall cap 4 h (14400 s) from the runner's start: no new program starts if elapsed time plus 1500 s plus 400 s would pass the cap; the result is then marked `partial`. The run is detached (own session on the VM, progress in `/content/status.json`, which I pull about every 20 to 30 minutes). If the VM is lost, the result is whatever was last pulled, reported as partial. A client disconnect alone does not stop the run and is not a stop condition.
5. **Predictions.** The 3 tutorial-03 rows of the table stand as written (PASS 85%, PASS 85%, COMPILE_ERROR 60%). Added before running: of the 10 caught mutants, at least half of the dropped-mask ones (`drop_load_mask`, `drop_store_mask`, `mask_lt_to_le`; 7 here) are `PASS` compiled (60%; the `autotune_axpy` finding was 4 of 6, and the lower confidence reflects a different kernel and no basis to transfer the rate). A `TIMEOUT` at 1500 s is reported as `TIMEOUT`, not as detected-by-test.
6. **Software.** Versions are recorded by the runner. If triton or torch differs from Run 2 (triton 3.6.0, torch 2.11.0+cu130), that is reported before any result is read.
7. **Hardware.** One T4 session. No A100 run was requested for this experiment; none is made.
8. **Limits.** 1 file, 3 survivors, 1 GPU model, 1 Triton build, one generated test, one run per program (no repeats, so no noise estimate). An illustration, not a rate.

## Amendment E (2026-10-28 KST 14:58, after run 3a was lost, before the restart; Amendment D otherwise stands)

- **Run 3a:** started 14:30:47 KST on Colab session `mg-t03` (T4, triton 3.6.0, torch 2.11.0+cu130, runner pid 1633). The first pull of `/content/status.json` at 14:57 found the session gone: `colab sessions` reported "No active sessions found on server" and the kernel endpoint returned 404 (Colab log 14:57:11). Nothing had been pulled before that (no earlier check), so **run 3a produced no recorded result**. Why the VM went away is not known (a reclaimed runtime, a crash of the VM during the run, or something else; no evidence for any of them).
- **Restart (one):** same settings as Amendment D (T4, 1500 s compiled cap, original first, survivors next, 4 h wall cap), a new session with its own colab `--config` file, and `status.json` pulled to the local disk about every 10 minutes. D.4's rule applies: if the VM is lost again, the result is whatever was last pulled, reported as partial, and no second restart is made.
- A lost VM is an infrastructure failure, not a result; it is not read as TIMEOUT or as any outcome of the program.

## Run 3 (tutorial 03 mutants, partial: 8 of 14 programs; 2026-10-28 KST 14:58 start, last pull 16:23; raw: `recorded/run3_tutorial03_partial.json`)

Colab T4 (Tesla T4, sm_75), Python 3.13.16, torch 2.11.0+cu130, **triton 3.6.0**: the same versions as Run 2. Session `mg-t03b`, run detached; the status file was pulled every 10 to 25 minutes.
**The run is partial.** At the pull at 16:49 the session was gone (`colab sessions`: "No active sessions found on server"; the CLI reported 404/401). The last good pull is 16:23 (8 of 14 programs done). Anything the VM finished after 16:23 was not retrieved.
This is the second lost VM (run 3a was lost before 14:57). Why is not known. Per Amendment E there was no second restart. The 6 programs not run: `mask_lt_to_le` lt#1, lt#2, lt#3, `k_loop_floor_div` kloop#0, `drop_modulo_wrap` mod#0, mod#1 (all caught in the sandbox).

Control first: the ORIGINAL is `PASS` compiled (691.8 s) and in the Colab interpreter (4.2 s), in the same session (Run 2: 699.9 s). The mutants are read.

| mutant | sandbox_interp | colab_compiled | colab_interp |
|---|---|---|---|
| `load_other_changed` zero#0 | SURVIVED | PASS (687.8 s) | PASS |
| `load_other_changed` zero#1 | SURVIVED | PASS (686.9 s) | PASS |
| `acc_dtype_fp16` acc#0 | SURVIVED | COMPILE_ERROR (3.7 s) | FAIL_ASSERT (InterpreterError) |
| `drop_load_mask` load#0 | KILLED | PASS (681.9 s) | FAIL_ASSERT (`zeros_no_act: Output mismatch`) |
| `drop_load_mask` load#1 | KILLED | PASS (683.3 s) | FAIL_ASSERT (`zeros_no_act: Output mismatch`) |
| `drop_store_mask` store#0 | KILLED | PASS (686.8 s) | OTHER (`free(): invalid pointer`) |
| `mask_lt_to_le` lt#0 | KILLED | PASS (708.6 s) | PASS |

Against the predictions that stand:
- The 3 survivors: zero#0 and zero#1 `PASS` compiled (predicted PASS, 85% each) and acc#0 `COMPILE_ERROR` (predicted COMPILE_ERROR, 60%): **all three as predicted**. Reading from the table above: the two zero rows are consistent with equivalence on this hardware (outputs identical in the sandbox), not a proof; the acc#0 row is not a test gap, since the compiler rejects a program the interpreter accepted (the sandbox survival came from the interpreter accepting an invalid program).
- Dropped-mask family (7 mutants: `drop_load_mask` x2, `drop_store_mask`, `mask_lt_to_le` x4; all caught in the sandbox): 4 of the 4 run are `PASS` compiled, so at least 4 of 7 are `PASS` (**as predicted**, at least half; the prediction holds whatever the 3 untested `mask_lt_to_le` rows would have given). The 3 untested rows and the 3 other caught mutants have no compiled result.
- `colab_interp` differs from `sandbox_interp` for acc#0 (Colab: FAIL_ASSERT, sandbox: SURVIVED); the columns differ in RNG, device path and Triton version (Amendment A.2), so this is context, not a control.
- Autotune: every compiled tutorial 03 program that ran chose the same configuration (BLOCK_SIZE_M 32, N 64, K 32, GROUP_SIZE_M 8, num_warps 2, num_stages 5), so the compiled verdicts do not depend on which configuration was chosen.

Limits: 1 file, 3 survivors, 8 of 14 programs, 1 GPU model, 1 Triton build, one generated test, one run per program (no noise estimate). Two VM losses of unknown cause. An illustration, not a rate.
