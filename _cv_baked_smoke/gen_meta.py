#!/usr/bin/env python3
"""Generate .meta.json sidecars for the baked CV engines so engine_resolver
treats them as a local cache hit (no HF fetch). Run INSIDE the baked image with
host CUDA/TRT mounts so detect_host_signature() yields the real sm87-... key.
Writes sidecars next to each engine AND copies them to /out for re-baking."""
import os, shutil, sys
sys.path.insert(0, "/opt/speech")
from pathlib import Path
from server.core.engine_resolver import detect_host_signature, _write_meta, _meta_path, _meta_matches

host = detect_host_signature()
print("HOST KEY:", host.key)
print("HOST DICT:", host.to_dict())

ENGINES = [
    "/opt/models/qwen3-tts-customvoice/talker_assembled_dir/llm.engine",
    "/opt/models/qwen3-tts-customvoice/code_predictor/llm.engine",
    "/opt/models/qwen3-tts-customvoice/code2wav/code2wav.engine",
]
out = Path("/out"); out.mkdir(parents=True, exist_ok=True)
for e in ENGINES:
    ep = Path(e)
    assert ep.exists(), f"missing {ep}"
    _write_meta(ep, host, "cache", None)
    mp = _meta_path(ep)
    ok = _meta_matches(ep, host)
    print(f"WROTE {mp} matches={ok}")
    # mirror to /out preserving the engine-relative subpath so we can COPY back
    rel = ep.relative_to("/opt/models/qwen3-tts-customvoice")
    dst = out / rel.parent
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mp, dst / mp.name)

print("DONE-META")
