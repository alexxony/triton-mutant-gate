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
        lines.append(f"| {m['op']} @ {m['site']} | {'survived' if m['T3B'] == 'SURVIVED' else 'caught'} | {follow} |")
    lines += ["", "Verdicts come from running the kernel in the Triton CPU interpreter with out-of-bounds checks on. "
              "'Equivalent' answers are the model's opinion, not a proof; behaviour on a compiled GPU run is not checked."]
    return "\n".join(lines) + "\n"
