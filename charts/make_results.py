"""Static charts for the demo video and the README, computed from demo/data.json (the recorded runs).

    uv run --with plotly --with kaleido python charts/make_results.py

Writes charts/before_after.{svg,png} and charts/survivor_fate.{svg,png}. Nothing here is measured anew: every number is
read from the recorded runs, one run per kernel.
"""
import json
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
GREEN, RED, AMBER, GREY, INK = "#1f8a5b", "#c0392b", "#b7791f", "#9aa3b2", "#1f2937"
FONT = dict(family="Inter, Helvetica, Arial, sans-serif", color=INK)

data = json.loads((ROOT / "demo" / "data.json").read_text())
muts = [m for k in data["kernels"].values() for m in k["mutants"]]
planted = len(muts)
survivors = [m for m in muts if m["T3B"] == "SURVIVED"]
caught = planted - len(survivors)
outcome = [(m.get("stage2") or {}).get("outcome") for m in survivors]
killed, equiv, failed = outcome.count("killed"), outcome.count("claimed_equivalent"), outcome.count("failed")
assert killed + equiv + failed == len(survivors), outcome
no_compile = [m for m in survivors if (m.get("compiled") or {}).get("result") == "COMPILE_ERROR"]
print(f"planted={planted} caught={caught} survivors={len(survivors)} killed={killed} equivalent={equiv} unresolved={failed} "
      f"survivors_not_compiling_on_T4={len(no_compile)}")


def save(fig, name, width=1600, height=900):
    fig.update_layout(width=width, height=height, font=FONT, paper_bgcolor="white", plot_bgcolor="white")
    fig.write_image(OUT / f"{name}.svg")
    fig.write_image(OUT / f"{name}.png", scale=1)


# 1. share of planted bugs the tests catch, before and after Nemotron's verified tests
before, after = caught / planted, (caught + killed) / planted
fig = go.Figure(go.Bar(
    x=["Tests you had", "+ tests Nemotron wrote\n(each verified by execution)"], y=[before * 100, after * 100],
    marker_color=[GREY, GREEN], text=[f"{caught} of {planted}<br>{before:.1%}", f"{caught + killed} of {planted}<br>{after:.1%}"],
    textposition="outside", textfont=dict(size=34), width=0.5))
fig.update_yaxes(range=[0, 100], title="planted bugs caught (%)", ticksuffix="%", gridcolor="#e5e7eb", title_font_size=24, tickfont_size=20)
fig.update_xaxes(tickfont_size=26)
fig.update_layout(title=dict(text="Planted bugs your tests catch, before and after", font_size=40, x=0.04),
                  margin=dict(l=110, r=60, t=120, b=190))
fig.add_annotation(
    x=0, y=-0.27, xref="paper", yref="paper", xanchor="left", showarrow=False, align="left", font=dict(size=19, color="#4b5563"),
    text=(f"Recorded runs, one per kernel (vecadd, softmax, layernorm, matmul), {planted} planted bugs, Triton CPU interpreter in a Nebius sandbox.<br>"
          f"Of the {killed} new verified tests, one targets an fp16-accumulator mutant that does not compile on Triton 3.6.0 (T4) but does on 3.8.0.<br>"
          "Not a proof that the remaining survivors are harmless."))
save(fig, "before_after")

# 2. what happened to the survivors
labels = ["Closed by a new test<br>(passes original, fails the bug)", "Nemotron says equivalent<br>(its opinion, not proven)", "No accepted test"]
vals, colors = [killed, equiv, failed], [GREEN, AMBER, RED]
fig = go.Figure(go.Bar(y=labels[::-1], x=vals[::-1], orientation="h", marker_color=colors[::-1], text=[str(v) for v in vals[::-1]],
                       textposition="outside", textfont=dict(size=34), width=0.55))
fig.update_xaxes(range=[0, max(vals) + 1.2], dtick=1, title="surviving bugs", gridcolor="#e5e7eb", title_font_size=24, tickfont_size=20)
fig.update_yaxes(tickfont_size=24)
fig.update_layout(title=dict(text=f"What happened to the {len(survivors)} bugs the first tests missed", font_size=40, x=0.04),
                  margin=dict(l=420, r=80, t=120, b=150))
fig.add_annotation(x=0, y=-0.2, xref="paper", yref="paper", xanchor="left", showarrow=False, font=dict(size=19, color="#4b5563"),
                   text="Same recorded runs. 'Equivalent' answers are the model's judgement plus sampled inputs, not a proof.")
save(fig, "survivor_fate")
