"""Markdown summary of a pipeline report."""

OUTCOME_TEXT = {
    "killed": "new test kills it (verified by execution)",
    "claimed_equivalent": "Nemotron says equivalent (not proven)",
    "failed": "no accepted test",
}


def markdown(report):
    lines = [f"# mutant-gate report: {report['kernel']}", "",
             f"Model `{report['model']}`, {report['mutants_planted']} mutants planted, "
             f"tokens: generate {report['tokens']['generate']}, close {report['tokens']['close']}.", ""]
    if not report["accepted"]:
        return "\n".join(lines + ["No generated test passed the original kernel, so no mutant was run. See raw/ for the replies."]) + "\n"
    muts = report["mutants"]
    surv = [m for m in muts if m["T3B"] == "SURVIVED"]
    lines += [f"**{len(muts) - len(surv)} of {len(muts)} planted bugs were caught by the generated test; {len(surv)} survived.**", "",
              "| mutant | verdict | follow-up |", "|---|---|---|"]
    for m in muts:
        s = m.get("stage2")
        follow = OUTCOME_TEXT.get(s["outcome"], s["outcome"]) if s else ""
        verdict = "survived" if m["T3B"] == "SURVIVED" else "caught"
        if m.get("compile", {}).get("status") == "COMPILE_ERROR":
            verdict, follow = "survived, but does not compile", f"not a test gap: {m['compile']['detail']}"
        lines.append(f"| {m['op']} @ {m['site']} | {verdict} | {follow} |")
    compiled = ("Survivors were also compiled for CUDA (sandbox Triton build, no GPU); a compiled run's pass/fail is not checked."
                if report.get("compile_check") else "Compilation for a GPU is not checked (use --compile-check).")
    lines += ["", "Verdicts come from running the kernel in the Triton CPU interpreter with out-of-bounds checks on. "
              f"'Equivalent' answers are the model's opinion, not a proof. {compiled}"]
    return "\n".join(lines) + "\n"
