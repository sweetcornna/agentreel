"""Wikimedia Commons helper: search audio files and show license metadata.

Usage:
  python music/sampler/tools/commons.py search "<query>" [limit]
  python music/sampler/tools/commons.py cat "Category:..."
  python music/sampler/tools/commons.py info "File:Title.ogg" ...
  python music/sampler/tools/commons.py wikitext "File:Title.ogg" ...
  python music/sampler/tools/commons.py get "File:Title.ogg" <dest_path>
找候选野外录音用（search / cat 只列出 CC0 / PD / CC BY 的文件）。选定的文件加进 fetch_samples.py 的 COMMONS 表和
CREDITS.md；fetch_samples.py 下载前会再读一次页面的 LicenseShortName。
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = "agentreel-sampler/0.1 (field-recording license check)"


def _get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:  # 429: back off politely, then retry
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):  # flaky local proxy / TLS EOF
            if attempt == 4:
                raise
            time.sleep(3)


OK_LICENSE = re.compile(r"^(CC0|Public domain|PD|CC BY \d(\.\d)?( [a-z]{2})?$)", re.I)


def license_ok(lic):
    return bool(lic) and bool(OK_LICENSE.match(lic)) and "SA" not in lic and "NC" not in lic and "ND" not in lic


def _strip(html):
    return re.sub(r"<[^>]+>", "", html or "").strip()


def info(titles):
    out = []
    for i in range(0, len(titles), 40):
        d = _get({
            "action": "query", "titles": "|".join(titles[i:i + 40]), "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata|mediatype", "format": "json",
        })
        for p in d["query"]["pages"].values():
            ii = (p.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata", {})
            out.append({
                "title": p.get("title"),
                "license": em.get("LicenseShortName", {}).get("value"),
                "license_url": em.get("LicenseUrl", {}).get("value"),
                "artist": _strip(em.get("Artist", {}).get("value")),
                "credit": _strip(em.get("Credit", {}).get("value"))[:120],
                "desc": _strip(em.get("ImageDescription", {}).get("value"))[:160],
                "duration": ii.get("duration"),
                "size": ii.get("size"),
                "mime": ii.get("mime"),
                "url": ii.get("url"),
                "page": ii.get("descriptionurl"),
            })
    return out


def search(q, limit=40):
    d = _get({"action": "query", "list": "search", "srsearch": q + " filetype:audio",
              "srnamespace": 6, "srlimit": limit, "format": "json"})
    return [x["title"] for x in d["query"]["search"]]


def category(cat, limit=200):
    d = _get({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmlimit": limit,
              "cmtype": "file|subcat", "format": "json"})
    return [x["title"] for x in d["query"]["categorymembers"]]


def _show(rows):
    for r in rows:
        if not license_ok(r["license"]) or "LL-Q" in r["title"]:
            continue
        print(f"{r['duration'] or 0:7.1f}s {r['license']!s:14} {r['size']!s:>9} | {r['title']} | {r['artist'][:40]!r}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "search":
        lim = int(sys.argv[3]) if len(sys.argv) > 3 else 40
        for r in info(search(sys.argv[2], lim)):
            if not license_ok(r["license"]) or "LL-Q" in r["title"]:
                continue
            print(f"{r['duration'] or 0:7.1f}s ", end="")
            print(f"{r['license']!s:18} {r['size']!s:>9} {r['mime']!s:12} | {r['title']} | {r['artist'][:40]!r}")
    elif cmd == "cat":
        titles = category(sys.argv[2])
        for t in titles:
            if t.startswith("Category:"):
                print("   [subcat]", t)
        _show(info([t for t in titles if t.startswith("File:")]))
    elif cmd == "wikitext":
        for t in sys.argv[2:]:
            d = _get({"action": "query", "titles": t, "prop": "revisions", "rvprop": "content",
                      "rvslots": "main", "format": "json"})
            pg = list(d["query"]["pages"].values())[0]
            print("===", t)
            print(pg["revisions"][0]["slots"]["main"]["*"][:1200])
    elif cmd == "info":
        for r in info(sys.argv[2:]):
            print(json.dumps(r, ensure_ascii=False, indent=1))
    elif cmd == "get":
        r = info([sys.argv[2]])[0]
        print(r["license"], r["url"])
        req = urllib.request.Request(r["url"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as resp, open(sys.argv[3], "wb") as f:
            f.write(resp.read())
