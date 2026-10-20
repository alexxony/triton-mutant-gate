# Gate experiment: mutation survival of Triton kernel tests on CPU interpreter (pre-registered 2026-10-19 before first run)

Kernels: vecadd, softmax, layernorm_fwd, matmul — written in the style of Triton tutorials 01/02/05/03 (MIT, triton-lang/triton),
single-kernel simplifications; tests use the tutorial's own shapes where the interpreter can run them (layernorm M and matmul block sizes reduced for CPU speed — deviation, to be reported).

Mutation operators (applied one site at a time): drop_load_mask, drop_store_mask, mask_lt_to_le, load_other_changed,
drop_max_subtract, drop_eps, divisor_off_by_one, acc_dtype_fp16, k_loop_floor_div.

Test tiers: T1 = tutorial-style test (tutorial shapes). T2 = T1 + stress shapes (non-multiples of block).
Verdict per mutant: KILLED (exit != 0, incl. crash exit<0) or SURVIVED (exit 0). Controls: original must pass T1 and T2.
Equivalent mutants are NOT filtered automatically; survivors are inspected by hand and labelled equivalent / real gap.

GO if: (a) >= 20% of applicable mutants survive T1, AND (b) at least one survivor class is a real gap (non-equivalent) that is not killed by T2 either
(i.e. needs a stronger check such as bounds instrumentation, or an LLM-written test).
NO-GO if: < 5% of applicable mutants survive T1 (the tutorial tests are already strong), or all T1 survivors are equivalent mutants.
Between 5% and 20%: ambiguous, reconsider framing before building anything.
