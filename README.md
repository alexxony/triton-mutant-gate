# triton-mutant-gate (Mutant Replay)

**Does your test suite really protect your Triton kernel?** This tool plants small bugs ("mutants") in a kernel, runs
the tests against every one of them in a Nebius sandbox, and asks Nemotron to write tests for the bugs that slipped
through. A test is accepted only if it **passes your original kernel and fails the buggy copy**. Nothing a model says
is trusted without running the code.

Built for the Nebius x NVIDIA Global AI Hackathon, track *Coding and Agentic Engineering*.

- **Replay demo:** `demo/index.html` (static page of recorded runs; public URL to be added at publication)
- **Recorded results and limits:** [docs/RESULTS.md](docs/RESULTS.md) · **Prior art:** [docs/PRIOR_ART.md](docs/PRIOR_ART.md)

## How it works

```mermaid
flowchart LR
  K[your kernel script] --> M[plant mutants<br/>mask, other=, fp16 accumulator, ...]
  K --> G["stage 1: Nemotron writes extra tests<br/>(never sees the mutants)"]
  G --> A{"passes your original kernel<br/>in the sandbox?"}
  A -- yes --> R[run the test against every mutant]
  M --> R
  R --> S[survivors]
  S --> C["stage 2: Nemotron sees one survivor's diff<br/>writes a killer test, or says EQUIVALENT"]
  C --> V{"passes original AND<br/>fails the mutant?"}
  V -- yes --> OK[verified killer test]
```

## Where Nemotron, Token Factory and Nebius tools are used

- **Nemotron on Token Factory** (`https://api.tokenfactory.nebius.com/v1`): writes the extra tests (stage 1) and, for each
  surviving mutant, either a killer test or an `EQUIVALENT` verdict (stage 2). Default model
  `nvidia/nemotron-3-super-120b-a12b`; `Nano-30B-A3B` and `Ultra-550b-a55b` were also run (see docs/RESULTS.md). Every call
  checks that the response names the requested model.
- **Nebius Sandboxes** (through the `contree-sdk`): every test execution (original and each mutant) runs in a disposable
  sandbox with the Triton CPU interpreter (`TRITON_INTERPRET=1`) plus out-of-bounds checks, so no generated code runs on
  your machine.
- Why Token Factory helps: the whole study (every model, five repeats each) used about 1.5M tokens. The Super-only part
  (343,276 tokens) cost about $0.17 at list price, which is what makes repeating each experiment five times practical.

## Quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), a Nebius Token Factory API key and Nebius Sandboxes access.

```bash
git clone <this repo> && cd triton-mutant-gate
uv sync
export NEBIUS_API_KEY=...            # Token Factory key (never commit it)

uv run mutant-gate mutants examples/kernels/layernorm.py --diff   # offline: list the planted bugs
uv run mutant-gate setup-image                                    # once, ~1 min: Python + Triton + CPU torch in a sandbox image
uv run mutant-gate run examples/kernels/layernorm.py              # the full gate (about 30k tokens for layernorm)
```

Output goes to `mutant-gate-out/<kernel>/`: `report.md` (summary table), `report.json`, `generated_test.py` and `raw/`
(every model reply). Options: `--model`, `--tries`, `--no-close` (skip stage 2), `--image`, `--out`.
Tests (offline): `uv run pytest`.

Example output for `examples/kernels/layernorm.py` (one real run): 18 bugs planted, 13 caught by the generated test,
5 survived; stage 2 produced one verified killer test (`drop_eps`), called three survivors equivalent and found no
accepted test for one. Results vary between runs (temperature 0.2).

## Your own kernel: script format

A kernel script is one Python file shaped like `examples/kernels/*.py`: an `@triton.jit` kernel, then two blank lines and
the wrapper `def`, any helpers (for example `check(...)`, which generated tests may call), then the line
`tier = sys.argv[1]` followed by your existing sanity check. Generic mutants (dropping `mask`/`other`, `<` to `<=`,
fp32 to fp16 accumulators) apply to any kernel; softmax, layernorm and matmul specific ones fire only when the matching
code pattern is present.

## Limits (read before trusting a verdict)

- The Triton **CPU interpreter is not a compiler**. In a pre-registered check on a Colab T4 (Triton 3.6.0), the three fp16
  accumulator survivors turned out not to compile ("loop-carried variable ... re-assigned to fp32"), so the interpreter let
  invalid mutants through; in the other direction, a compiled run missed most dropped-mask bugs that the interpreter's bounds
  checks catch. This compiled check is not part of `mutant-gate run` yet. One GPU, one Triton build
  ([details](docs/RESULTS.md#compiled-gpu-run-colab-t4)).
- Mutants come from a fixed set of rewrite rules, not real bug reports. "Equivalent" is the model's opinion plus sampled
  inputs, not a proof.
- Four tutorial-style kernels only. The data does not support ranking the three Nemotron models.
- A model that only *reads* the test and the diff (no execution) predicted catch/survive correctly 86.9% of the time
  versus a 79.4% always-"caught" baseline, so execution is not shown to be strictly necessary; it matters most for numeric
  behaviour. Details in docs/RESULTS.md.
- Every part of the idea (LLM test generation against mutants, GPU-kernel mutation analysis, equivalent-mutant
  detection) exists in earlier work; this repository is a combination, not a new method.

## Repository layout

| path | contents |
|---|---|
| `src/mutant_gate/` | CLI, mutation operators, prompts, sandbox runner, the two-stage pipeline |
| `examples/kernels/` | the four kernels used in the study |
| `demo/` | replay page (`index.html`), its data and the builder (`uv run python demo/build_demo_data.py`) |
| `recorded/` | raw results of the study (JSON, model replies, prior-art survey outputs) |
| `tests/` | offline unit tests |

## License

MIT, see [LICENSE](LICENSE). The example kernels adapt the Triton tutorials (MIT, OpenAI and the Triton contributors);
see [THIRD_PARTY.md](THIRD_PARTY.md).
