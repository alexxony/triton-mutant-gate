"""Check worker quotes against the cited pages: fetch each item's first URL, strip tags, test literal substring."""
import html
import re
import sys
import urllib.request
from pathlib import Path


def norm(s):
    s = html.unescape(re.sub(r"<[^>]+>", " ", s)).replace("\\`", "`").replace("**", "").replace("`", "").replace('\\"', '"').replace("\\n", " ").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def main(path):
    items = re.split(r"\n### ", Path(path).read_text())[1:]
    ok = bad = unreachable = 0
    for it in items:
        title = it.splitlines()[0][:70]
        urls = re.findall(r"https?://[^\s)\]]+", it)
        quotes = [norm(q) for q in re.findall(r'"([^"\n]{25,}?)"(?=\s|$|\.|\)|/)', it)]
        page = None
        used = None
        last = ""
        for u in urls[:4]:   # concatenate every reachable cited URL of the item
            try:
                page = (page or "") + " " + norm(fetch(u)); used = u
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__} {e}"[:60]
        if page is None:
            unreachable += len(quotes)
            print(f"UNREACHABLE {title} ({last}) quotes={len(quotes)}")
            continue
        for q in quotes:
            q2 = q.replace("`", "").rstrip(". ")
            hit = q2 in page
            ok += hit; bad += not hit
            print(("OK  " if hit else "MISS"), title, "|", q[:70])
    print("ok", ok, "miss", bad, "unreachable_quotes", unreachable)


if __name__ == "__main__":
    main(sys.argv[1])
