"""Prior-art survey: arXiv abstracts fetched by this script, classified by Nemotron (Token Factory).

The model sees only the fetched abstract. Every quote it returns is checked to be a literal substring of
that abstract (whitespace-normalised); quotes that fail the check are dropped and counted.
"""
import asyncio
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from probe import llm  # noqa: E402

HERE = Path(__file__).parent
MODEL = "nvidia/nemotron-3-super-120b-a12b"
NS = {"a": "http://www.w3.org/2005/Atom"}
QUERIES = {
    "T1": ['all:"mutation testing" AND all:"large language model" AND all:"test generation"',
           'all:"mutation-guided" AND all:"test generation"',
           'all:"mutant" AND all:LLM AND all:"killing" AND all:tests'],
    "T2": ['all:"equivalent mutant" AND all:"large language model"',
           'all:"mutation testing" AND all:LLM AND all:"mutant generation"'],
    "T3": ['all:"mutation testing" AND all:GPU AND all:kernel',
           'all:"mutation testing" AND all:CUDA',
           'all:"mutation" AND all:Triton AND all:kernel'],
    "T4": ['all:KernelBench', 'all:TritonBench',
           'all:LLM AND all:"GPU kernel" AND all:"correctness" AND all:"test"',
           'all:LLM AND all:"kernel" AND all:"test case generation"'],
}
PROMPT = """You classify one paper from its abstract only. Do not use outside knowledge.
Our project: inject mutants into Triton GPU kernels, run the test suite on a CPU interpreter to see which mutants survive, and have an LLM write tests that kill surviving mutants (accepted only if it passes the original and fails the mutant when executed).

Title: {title}
Abstract: {abstract}

Reply with ONE JSON object only:
{{"relevant": true|false,
 "covers": ["llm_test_generation","mutation_testing","equivalent_mutants","gpu_or_kernel","executes_to_verify"] (only those the abstract states),
 "quote": "one sentence copied EXACTLY from the abstract that best shows its relevance, or empty",
 "executes_to_verify_quote": "sentence copied EXACTLY from the abstract showing generated tests/mutants are verified by execution, or empty"}}"""


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def arxiv(q):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": q, "max_results": 8, "sortBy": "relevance"})
    with urllib.request.urlopen(url, timeout=60) as r:
        root = ET.fromstring(r.read())
    out = []
    for e in root.findall("a:entry", NS):
        out.append({"id": e.find("a:id", NS).text.strip(), "title": norm(e.find("a:title", NS).text),
                    "abstract": norm(e.find("a:summary", NS).text), "published": e.find("a:published", NS).text[:10]})
    return out


async def classify(sem, p):
    async with sem:
        d = await llm.chat([{"role": "user", "content": PROMPT.format(title=p["title"], abstract=p["abstract"])}], None, model=MODEL, max_tokens=4000)
    text = d["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        j = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        j = None
    p = {**p, "tokens": d["usage"]["total_tokens"], "parsed": j is not None}
    if j:
        for key in ("quote", "executes_to_verify_quote"):
            q = norm(j.get(key) or "")
            if q and q not in p["abstract"]:
                p[key + "_rejected"] = q
                q = ""
            j[key] = q
        p["cls"] = j
    return p


async def main():
    papers = {}
    for topic, qs in QUERIES.items():
        for q in qs:
            for e in arxiv(q):
                papers.setdefault(e["id"], {**e, "topics": set()})["topics"].add(topic)
            await asyncio.sleep(3)   # arXiv API etiquette
    plist = [{**p, "topics": sorted(p["topics"])} for p in papers.values()]
    print("papers fetched", len(plist), flush=True)
    sem = asyncio.Semaphore(6)
    res = await asyncio.gather(*[classify(sem, p) for p in plist])
    (HERE / "results_prior_art.json").write_text(json.dumps({"model": MODEL, "papers": res}, indent=1, ensure_ascii=False))
    rel = [r for r in res if r.get("cls", {}).get("relevant")]
    print("parsed", sum(r["parsed"] for r in res), "relevant", len(rel), "quote_rejected",
          sum(1 for r in res for k in ("quote_rejected", "executes_to_verify_quote_rejected") if k in r), "tokens", sum(r["tokens"] for r in res))
    for r in sorted(rel, key=lambda r: -len(r["cls"].get("covers", []))):
        print(f"- {r['published']} {r['title'][:90]} | {r['id']} | {','.join(r['cls'].get('covers', []))}")


if __name__ == "__main__":
    asyncio.run(main())
