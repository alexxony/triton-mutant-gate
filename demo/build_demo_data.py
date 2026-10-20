"""Bundle the recorded runs in recorded/ into demo/data.json and demo/index.html. Reads files only; runs nothing.

Usage: uv run python demo/build_demo_data.py
"""
import difflib
import glob
import json
import re
from pathlib import Path

from mutant_gate.mutants import make_mutants

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "recorded" / "results"
GENERATED = ROOT / "recorded" / "generated"
KERNEL_DIR = ROOT / "examples" / "kernels"
KERNELS = ["vecadd", "softmax", "layernorm", "matmul"]


def kill_for(tag):
    """Latest stage-2 result file of a repeat run, with its run id."""
    fs = sorted(glob.glob(str(RESULTS / f"results_kill_{tag}_*.json")))
    if not fs:
        return None, {}
    d = json.loads(Path(fs[-1]).read_text())
    return d["run_id"], {(r["kernel"], r["op"], r["site"]): r for r in d["results"]}


def code_of(text):
    blocks = re.findall(r"```(?:python)?\n(.*?)(?:```|\Z)", text, re.S)
    return blocks[0] if blocks else ""


def stage2(tag, run_id, k, op, site, r):
    out = {"outcome": r["outcome"], "tries": r["tries"], "tokens": r["tokens"]}
    raw = GENERATED / f"run_{tag}" / "raw" / f"kill_{k}_{op}_{site.replace('#', '')}_try{r['tries']}_{run_id}.txt"
    if raw.exists():
        text = raw.read_text().split("---\n", 1)[-1]
        out["reply_head"] = re.sub(r"\s+", " ", text.split("```")[0]).strip()[:400]
        out["test"] = code_of(text)
    return out


def classification():
    p = RESULTS / "results_survivor_classification.json"
    rows = json.loads(p.read_text()) if p.exists() else []
    return {(c["kernel"], c["op"], c["site"]): c["rows"] for c in rows}


def main():
    cls = classification()
    data = {"kernels": {}, "source": "model nvidia/nemotron-3-super-120b-a12b, temperature 0.2, one recorded run per kernel"}
    for k in KERNELS:
        src = (KERNEL_DIR / f"{k}.py").read_text()
        codes = {(op, site): c for op, site, c in make_mutants(src)}
        best = None
        for i in range(1, 6):   # pick the repeat run whose survivors have the most stage-2 follow-ups
            tag = f"super{i}"
            g = json.loads((RESULTS / f"results_generated_{tag}.json").read_text())["kernels"][k]
            if not g["accepted"]:
                continue
            run_id, kr = kill_for(tag)
            n = sum(1 for x in g["mutants"] if (k, x["op"], x["site"]) in kr)
            if best is None or n > best[0]:
                best = (n, tag, g, run_id, kr)
        _, tag, g, run_id, kr = best
        muts = []
        for x in g["mutants"]:
            key = (k, x["op"], x["site"])
            diff = "\n".join(difflib.unified_diff(src.splitlines(), codes[(x["op"], x["site"])].splitlines(), lineterm="", n=1))
            e = {"op": x["op"], "site": x["site"], "T3B": x["T3B"], "detail": x["detail"][:160], "diff": diff}
            if key in kr:
                e["stage2"] = stage2(tag, run_id, k, x["op"], x["site"], kr[key])
            if key in cls:
                e["differential"] = {"cases": len(cls[key]), "identical": sum(1 for r in cls[key] if r["identical"])}
            muts.append(e)
        data["kernels"][k] = {"run": tag, "source": src, "test": g["test"], "mutants": muts}
    ctl = json.loads((RESULTS / "results_control_noexec.json").read_text())["results"]
    surv = [r for r in ctl if r["truth"] == "SURVIVED"]
    data["control"] = {"items": len(ctl), "accuracy": round(sum(r["pred"] == ("SURVIVES" if r["truth"] == "SURVIVED" else "KILLED") for r in ctl) / len(ctl), 3),
                       "baseline": round(1 - len(surv) / len(ctl), 3), "survives_recall": f"{sum(r['pred'] == 'SURVIVES' for r in surv)}/{len(surv)}"}
    out = Path(__file__).parent
    (out / "data.json").write_text(json.dumps(data, ensure_ascii=False))
    page = (out / "template.html").read_text().replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    (out / "index.html").write_text(page)
    for k, v in data["kernels"].items():
        print(k, v["run"], "mutants", len(v["mutants"]), "survived", sum(x["T3B"] == "SURVIVED" for x in v["mutants"]),
              "stage2", sum("stage2" in x for x in v["mutants"]))
    print(data["control"])


if __name__ == "__main__":
    main()
