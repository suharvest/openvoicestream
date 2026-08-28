"""One scorer for every platform: the repo's own compute_error_rate."""
import json, glob, sys, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/perf, for runners.py
from runners import compute_error_rate
# Hard requirements, not optional niceties. Without OpenCC the t2s column
# silently becomes a copy of the raw CER; without cn2an, Chinese number
# normalisation silently switches off ("十五米" then scores 66.7% against
# "15米"). The point of scoring off-device is that every platform goes through
# ONE scorer, and a scorer that changes behaviour with its environment defeats
# that more thoroughly than a second implementation would.
try:
    from opencc import OpenCC
    T2S = OpenCC("t2s")
except Exception as exc:
    raise SystemExit(
        f"opencc is required: without it the t2s column silently duplicates "
        f"the raw CER. Install it (uv run --with opencc ...). Error: {exc}"
    )
try:
    import cn2an  # noqa: F401  — used by runners.compute_error_rate
except Exception as exc:
    raise SystemExit(
        f"cn2an is required: without it Chinese number normalisation silently "
        f"switches off. Install it (uv run --with cn2an ...). Error: {exc}"
    )

def score(path):
    d = json.load(open(path))
    out = {}
    seen = set()
    for r in d["rows"]:
        # A duplicate id weights one file twice in the mean; both it and a
        # short row set are reported rather than absorbed into a plausible n.
        if r["id"] in seen:
            raise SystemExit(f"{path}: duplicate row id {r['id']!r}")
        seen.add(r["id"])
        lang, cat = r["lang"], r["category"]
        hyp = r["hyp"]
        e = compute_error_rate(r["ref"], hyp, lang)
        es = compute_error_rate(r["ref"], T2S.convert(hyp), lang) if (lang == "zh" and T2S) else e
        out.setdefault(f"{lang}_{cat}", []).append(
            # ttft is nullable in the schema (a chunk that produced no tokens);
            # passing None to np.mean raises and takes the whole batch down.
            dict(err=e, err_t2s=es, rtf=r["rtf"],
                 ttft=r["ttft_ms"] if r["ttft_ms"] is not None else float("nan"),
                 enc=r["encoder_ms"], dec=r["decoder_ms"], tok=r.get("n_tokens", 0)))
    return d.get("label", os.path.basename(path)), out

rows = []
for p in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results_rk/*.json")):
    label, groups = score(p)
    for g in ["en_short", "en_long", "zh_short", "zh_long"]:
        if g not in groups: continue
        v = groups[g]
        m = lambda k: float(np.nanmean([x[k] for x in v]))
        rows.append((label, g, len(v), m("err")*100, m("err_t2s")*100, m("rtf"), m("ttft"), m("enc"), m("dec"), m("tok")))
print(f'{"config":<32}{"group":<10}{"n":>3}{"err":>8}{"t2s":>8}{"RTF":>7}{"TTFT":>8}{"enc":>7}{"dec":>8}{"tok":>6}')
for r in rows:
    print(f"{r[0]:<32}{r[1]:<10}{r[2]:>3}{r[3]:>7.2f}%{r[4]:>7.2f}%{r[5]:>7.3f}{r[6]:>7.0f}ms{r[7]:>7.0f}{r[8]:>8.0f}{r[9]:>6.0f}")
