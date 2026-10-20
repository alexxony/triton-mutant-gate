# Control: no-execution LLM judge (written 2026-10-20 before the run)

Question: how often does Nemotron, WITHOUT running anything, mispredict whether a generated test suite kills a mutant?

- Items: the 16 accepted Super kernel attempts of runs super1..5 (results_generated_super{1..5}.json), 160 mutants. Ground truth = executed T3B outcome (KILLED / SURVIVED) recorded in those files. Reruns under the new acceptance rule are NOT in this set.
- Judge: nvidia/nemotron-3-super-120b-a12b, temperature 0.2, one call per item, no retries. Input: original script, the generated test, the mutant as a unified diff. Output: JSON {"verdict": "KILLED"|"SURVIVES", "reason": short}.
- Metrics (fixed now): accuracy; baseline accuracy of always answering KILLED (= 1 - survival rate); recall and precision for SURVIVES; per-mutant-class (op) confusion.
- Reading rule: the judge shows no value of execution unless it is clearly worse than execution on SURVIVES recall. Accuracy near the always-KILLED baseline means the judge adds nothing. Ground truth is itself an execution result (T3B), so this measures judge error against execution, not truth about equivalence.
- Threats noted in advance: ground truth for OOB-read mutants may flip between runs (T3B reduces this); the judge sees the test code that was produced by the same model family; 16 attempts only; parse failures counted as wrong, reported separately.
