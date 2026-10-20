# Prior art

Every component of this project exists in earlier work, so it makes no claim of a new method. What is here is a
combination built around Nebius Token Factory, Nemotron and Nebius Sandboxes. Entries below were checked against the
abstract (or README / issue text) of the cited page; quoted sentences were compared literally with the source.
Only abstracts and short page sections were read, not full papers.

| work | what it covers | link |
|---|---|---|
| Mutation-Guided LLM-based Test Generation at Meta (ACH) | LLM writes tests that kill mutants; LLM-based equivalent-mutant detection | arXiv 2501.12862 |
| Mutation-Guided Unit Test Generation with a Large Language Model (MUTGEN) | mutation feedback in the prompt for LLM test generation | arXiv 2506.02954 |
| Effective Test Generation Using Pre-trained LLMs and Mutation Testing (MuTAP) | prompts augmented with surviving mutants | arXiv 2308.16557 |
| Large Language Models for Equivalent Mutant Detection: How Far Are We? | LLMs judging equivalent mutants (Java) | arXiv 2408.01760 |
| Applying Mutation Testing to GPU Programs (MUTGPU, ICST 2020) | GPU-specific mutation operators for CUDA | TU Delft repository record |
| Measuring the Checker: Mutation Analysis for GPU-Kernel Benchmark Oracles | mutation analysis of KernelBench-style oracles (CUDA) | arXiv 2609.22220 |
| The Correctness Illusion in LLM-Generated GPU Kernels | seeded bugs in Triton and CPU stand-in kernels, fuzzing against a reference | arXiv 2606.20128 |
| Predicting Test Results without Execution (Hora, FSE 2024) | LLM predicts test results without running; GPT-4 reported at 81% accuracy | DOI 10.1145/3663529.3663794 |
| Triton debugging guide (interpreter) | documented interpreter limitations | triton-lang.org, programming guide ch. 3 |
| triton-lang/triton issues #10658, #10291 | JIT vs. interpreter result differences | GitHub |

The raw survey outputs (an arXiv abstract classification by Nemotron, and a separate web search whose citations were
verified by script) are in `recorded/prior_art/`.
