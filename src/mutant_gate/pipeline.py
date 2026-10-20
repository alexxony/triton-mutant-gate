"""The gate: Nemotron writes a hidden-mutant test, the sandbox runs it against every mutant, Nemotron targets the survivors.

Nothing a model says is trusted without execution: a generated test is accepted only if it passes the original kernel
(with bounds checks on); a killer test for a survivor is accepted only if it also fails the mutant.
"""
import asyncio
import json
import time
from pathlib import Path

from . import llm
from .mutants import make_mutants
from .sandbox import Sandbox
from .script import CLOSE_PROMPT, GENERATE_PROMPT, diff_of, extract_code, full_program, has_check, shown_script, split_script


def _save_raw(raw_dir, name, data, text):
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / name).write_text(f"finish_reason={data['choices'][0].get('finish_reason')}\nusage={data['usage']}\n---\n{text}")


async def generate_test(sb, src, model, tries, raw_dir, run_id, log):
    """Stage 1: ask for extra tests (without showing any mutant). Returns {accepted, tries, test, tokens}."""
    msgs = [{"role": "user", "content": GENERATE_PROMPT.format(script=shown_script(src))}]
    tokens, gen, ok = 0, "", False
    for n in range(1, tries + 1):
        data = await llm.chat(msgs, model=model)
        tokens += data["usage"]["total_tokens"]
        text = data["choices"][0]["message"]["content"] or ""
        _save_raw(raw_dir, f"generate_try{n}_{run_id}.txt", data, text)
        gen = extract_code(text)
        if not has_check(gen):   # a test with no assert and no check() call would "pass" every mutant
            log(f"  try {n}: no usable test code (finish_reason={data['choices'][0].get('finish_reason')})")
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "No usable test code found. Return one complete ```python block with assert statements or calls to check(...)."}]
            continue
        code, tail = await sb.run(full_program(src, gen), bound=True)   # must pass on the original, even under bounds checks
        if code == 0:
            ok = True
            log(f"  try {n}: test accepted (passes the original kernel)")
            break
        log(f"  try {n}: test fails on the original kernel ({tail[:80]})")
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": f"Your test failed on the correct kernel: {tail}\nFix the test (the kernel is correct). Return one python block."}]
    return {"accepted": ok, "tries": n, "test": gen if ok else "", "tokens": tokens}


async def run_mutants(sb, src, test, muts, parallel=12):
    """Run the accepted test against every mutant, with and without bounds checks. Survival under bounds checks is the verdict."""
    sem = asyncio.Semaphore(parallel)

    async def one(code, bound):
        async with sem:
            return await sb.run(full_program(code, test), bound=bound)

    res = await asyncio.gather(*[one(c, b) for (_, _, c) in muts for b in (False, True)])
    out = []
    for i, (op, site, _) in enumerate(muts):
        (c0, t0), (c1, t1) = res[2 * i], res[2 * i + 1]
        out.append({"op": op, "site": site, "T3": "SURVIVED" if c0 == 0 else "KILLED", "T3B": "SURVIVED" if c1 == 0 else "KILLED",
                    "detail": (t1 or t0)[:120]})
    return out


async def compile_survivors(sb, src, test, muts, survivors, log, parallel=8):
    """Optional step: drop survivors that are not valid GPU programs. The interpreter accepts code the compiler rejects
    (e.g. an fp16 accumulator that picks up fp32), so such a survivor is not a test gap. Sets m['compile'] = {status, detail}.

    The unmodified kernel is checked first; if it does not compile here either, the check is meaningless and is skipped.
    """
    base, detail = await sb.compile_check(split_script(src)[0], full_program(src, test))
    if base != "COMPILES":
        log(f"compile check skipped: the original kernel gives {base} ({detail})")
        return False
    codes = {(op, site): c for op, site, c in muts}
    sem = asyncio.Semaphore(parallel)

    async def one(m):
        async with sem:
            status, detail = await sb.compile_check(split_script(codes[(m["op"], m["site"])])[0], full_program(codes[(m["op"], m["site"])], test))
            m["compile"] = {"status": status, "detail": detail}

    await asyncio.gather(*[one(m) for m in survivors])
    bad = [m for m in survivors if m["compile"]["status"] == "COMPILE_ERROR"]
    log(f"compile check: {len(bad)} of {len(survivors)} survivors do not compile for CUDA, so they are not test gaps")
    return True


async def close_survivor(sb, src, test, op, site, mut_code, model, tries, raw_dir, run_id, sem):
    """Stage 2: show Nemotron the survivor's diff. Outcome: killed (verified by execution) / claimed_equivalent / failed."""
    mhead = split_script(mut_code)[0]
    head, t1 = split_script(src)
    msgs = [{"role": "user", "content": CLOSE_PROMPT.format(script=shown_script(src), existing=test, diff=diff_of(src, mut_code))}]
    rec = {"outcome": "failed", "tries": 0, "syntax_errors": 0, "tokens": 0, "log": []}
    for n in range(1, tries + 1):
        rec["tries"] = n
        data = await llm.chat(msgs, model=model)
        rec["tokens"] += data["usage"]["total_tokens"]
        text = data["choices"][0]["message"]["content"] or ""
        _save_raw(raw_dir, f"close_{op}_{site.replace('#', '')}_try{n}_{run_id}.txt", data, text)
        gen = extract_code(text)
        if not gen.strip() and "EQUIVALENT" in text.upper():
            rec["outcome"] = "claimed_equivalent"
            rec["log"].append("model answered EQUIVALENT")
            break
        feedback = None
        if not has_check(gen):
            feedback = "No usable test code (need one ```python block with assert, or the word EQUIVALENT)."
        else:
            try:
                compile(gen, "generated_test", "exec")   # syntax check only, nothing runs locally
            except SyntaxError as e:
                rec["syntax_errors"] += 1
                feedback = f"Your code has a SyntaxError: {e.msg} at line {e.lineno}: {(e.text or '').strip()}"
        if feedback is None:
            async with sem:
                oc, otail = await sb.run(head + "\n" + t1 + "\n" + gen + '\nprint("PASS")\n', bound=True)
                if oc != 0:
                    feedback = f"Your test FAILS on the original kernel ({otail}). It must pass on the original."
                else:
                    mc, mtail = await sb.run(mhead + "\n" + t1 + "\n" + gen + '\nprint("PASS")\n', bound=True)
                    if mc == 0:
                        feedback = "Your test PASSES on the mutant too, so it does not tell them apart."
                    else:
                        rec.update(outcome="killed", test=gen, mutant_failure=mtail)
                        rec["log"].append("accepted")
                        break
        rec["log"].append(feedback)
        msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": feedback + " Try again: one python block, or EQUIVALENT."}]
    return rec


async def analyze(kernel_path, model=llm.DEFAULT_MODEL, tries=4, close=True, out_dir=None, log=print, image=None, compile_check=False):
    """Full pipeline for one kernel script. Writes report.json, report.md and generated_test.py into out_dir."""
    kernel_path = Path(kernel_path)
    src = kernel_path.read_text()
    muts = make_mutants(src)
    run_id = int(time.time())
    out_dir = Path(out_dir or Path("mutant-gate-out") / kernel_path.stem)
    raw_dir = out_dir / "raw"
    log(f"{kernel_path.name}: {len(muts)} mutants planted")
    sb = await Sandbox.open(image)
    report = {"kernel": kernel_path.name, "model": model, "run_id": run_id, "mutants_planted": len(muts), "accepted": False,
              "tokens": {"generate": 0, "close": 0}, "mutants": []}

    log("stage 1: Nemotron writes extra tests (it never sees the mutants)")
    gen = await generate_test(sb, src, model, tries, raw_dir, run_id, log)
    report.update(accepted=gen["accepted"], tries=gen["tries"], test=gen["test"])
    report["tokens"]["generate"] = gen["tokens"]
    if gen["accepted"]:
        (out_dir / "generated_test.py").parent.mkdir(parents=True, exist_ok=True)
        (out_dir / "generated_test.py").write_text(gen["test"])
        log(f"running the test against {len(muts)} mutants in the sandbox")
        report["mutants"] = await run_mutants(sb, src, gen["test"], muts)
        survivors = [m for m in report["mutants"] if m["T3B"] == "SURVIVED"]
        log(f"{len(report['mutants']) - len(survivors)} caught, {len(survivors)} survived")
        if compile_check and survivors:
            report["compile_check"] = await compile_survivors(sb, src, gen["test"], muts, survivors, log)
        # stage 2 only for survivors that are valid programs (all of them when the compile check is off or was skipped)
        targets = [m for m in survivors if m.get("compile", {}).get("status") != "COMPILE_ERROR"]
        if close and targets:
            log(f"stage 2: Nemotron targets the {len(targets)} survivors (a killer test must pass the original and fail the mutant)")
            codes = {(op, site): c for op, site, c in muts}
            sem = asyncio.Semaphore(8)
            res = await asyncio.gather(*[close_survivor(sb, src, gen["test"], m["op"], m["site"], codes[(m["op"], m["site"])], model, 3, raw_dir, run_id, sem)
                                         for m in targets])
            for m, r in zip(targets, res):
                m["stage2"] = r
                report["tokens"]["close"] += r["tokens"]
                log(f"  {m['op']} @ {m['site']}: {r['outcome']} (tries={r['tries']})")
    else:
        log("no test was accepted; nothing to run against the mutants")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    from .report import markdown
    (out_dir / "report.md").write_text(markdown(report))
    log(f"report: {out_dir / 'report.md'}")
    return report
