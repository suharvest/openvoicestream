# Kokoro RK ZH G2P regression after P7(b) container recreate — 2026-05-23

## TL;DR

After the P7(b) bucket-8 static-INT8 ship (`a7b4575`) recreated the
`openvoicestream-kokoro` container on radxa, the ZH path silently broke:
`POST /tts/stream {"text":"你好。"}` returned a 4-byte WAV header with zero
audio. EN bucket-8 path remained healthy. Root cause: the recreated container
was missing the `misaki` PyPI package (and `cn2an`). The Kokoro tokenizer's
ZH branch (`_MisakiG2P.load()`) caught the ImportError, logged a warning, and
silently fell back to char-level lookup. Chinese codepoints are not in
`tokens.txt`, so every char was dropped → 0 tokens → `ValueError: zero
phoneme tokens`. Fixed by `pip install 'misaki[zh]==0.9.4'` inside the
container, and version-pinning all ZH G2P deps in `deploy/docker/Dockerfile.rk`
so future image rebuilds are reproducible.

## Symptoms

```
POST /tts/stream {"text":"你好。"}  → 4 bytes (SR header only), HTTP 200
```

Container log:
```
[W] kokoro_rknn: Kokoro tokenizer dropped 3 char(s) not in tokens.txt:
    '你好。' (lang=zh, encoded='你好。')
[E] app.main: tts/stream synthesis failed for sentence='你好。'
ValueError: Kokoro tokenizer produced zero phoneme tokens for text='你好。'
    (language='zh', encoded='你好。'); check G2P and tokens.txt coverage
```

Reproduced for all ZH inputs (short and mid), all 3 quantization variants
(FP32 / dynamic INT8 / static INT8). EN inputs unaffected. Confirms the bug is
upstream of the vocoder — at the G2P/tokenizer layer.

## Root cause

`third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py`
(md5 `0dbb0314...`) line 106-158 defines `_MisakiG2P`, a lazy loader for
`misaki.zh.ZHG2P(version="1.1")`. On import failure it logs:

```
misaki not available; Chinese text will fall back to char-level lookup
(silent drops expected). Install with: pip install 'misaki[zh]'.
```

The fallback path encodes the raw Chinese chars into `tokens.txt`, which only
covers Bopomofo + tone digits + Latin — so every CJK glyph is dropped, n_tokens
becomes 0, and the encode call raises `ValueError`.

`pip3 show misaki` inside the container returned `Package(s) not found:
misaki`. All other ZH deps (jieba, pypinyin, etc.) were present — only the
`misaki` package itself was missing (plus `cn2an` which only `misaki[zh]`
extra pulls).

The Dockerfile.rk at HEAD (commit `ca9332c`, dated before P7(b)) does include
`misaki` in its `pip install` block, but the running container appears to have
been built or rebuilt from a layer that pre-dates that commit, or the layer
cache was missing the package. Either way the dep was unpinned so future
rebuilds could silently regress again.

## Fix

### Hot patch (production container, immediate)

```bash
docker exec openvoicestream-kokoro pip3 install --no-cache-dir 'misaki[zh]==0.9.4'
docker restart openvoicestream-kokoro
```

This pulled `misaki-0.9.4` + `cn2an-0.5.24`; all other ZH deps were already
satisfied at the expected versions (jieba 0.42.1, pypinyin 0.55.0,
pypinyin_dict 0.9.0, ordered_set 4.1.0, proces 0.1.7, addict 2.4.0, regex
2026.5.9).

After restart, container log emits:
```
[I] kokoro_rknn: misaki ZH G2P (v1.1) loaded for Kokoro RKNN
```

### Persistent patch (Dockerfile, future-proof)

`deploy/docker/Dockerfile.rk` ZH G2P block changed to **pinned versions** so
future `docker build` cannot pull a newer misaki that breaks the Bopomofo
mapping documented in `kokoro-rk-zh-fix-misaki.md`:

```dockerfile
'misaki[zh]==0.9.4'  \
'jieba==0.42.1'      \
'pypinyin==0.55.0'   \
'pypinyin_dict==0.9.0' \
'ordered_set==4.1.0' \
'cn2an==0.5.24'      \
'proces==0.1.7'      \
'addict==2.4.0'      \
regex                \
```

Note `misaki[zh]` (extras) instead of bare `misaki` — guarantees `cn2an` is
pulled as well.

## Verification

| Test | Before | After |
|---|---|---|
| `POST /tts/stream {"text":"你好。"}` × 10 | 4 bytes, HTTP 200 | 53252 bytes, HTTP 200, TTFA ~5 ms (cached) |
| `POST /tts/stream {"text":"abc."}` × 10  | 43012 bytes (P7b OK) | 43012 bytes (P7b unchanged) |
| `POST /tts/stream {"text":"你好，今天天气真好。"}` | 0 bytes / ValueError | 251908 bytes, HTTP 200 |
| container `tts.py` md5 | `4961497d910cac5531ceafe35e4f1713` | unchanged |
| container `kokoro_rknn.py` md5 | `0dbb03149b1ee2b587abd2f0b4cf821b` | unchanged |
| log `misaki ZH G2P (v1.1) loaded` | absent | present at startup |
| log `zero phoneme` / `tokens.txt drop` | per-request | absent after fix |

EN bucket-8 INT8 static path (`KOKORO_RKNN_BUCKET8_TAIL_REST_INT8STATIC_PATH=
/opt/kokoro-bucket-8/kokoro-vocoder-tail-rest-cpu-bucket8.int8static.onnx`)
remained active and uncompromised by the fix.

## Lessons

1. **Silent fallback masking failure** — `_MisakiG2P.load()` logs a warning
   but continues with raw-char fallback. That fallback is doomed to fail with
   `zero phoneme tokens`, but the warning was buried beneath rknn startup
   noise. Consider promoting the warning to ERROR or refusing to start the
   service when ZH is configured but misaki is missing.
2. **Image dep persistence is not enough — pin versions** — even though
   commit `ca9332c` documented and added misaki to Dockerfile.rk, an unpinned
   line still drifted in some rebuild flow. ZH G2P deps are tightly coupled
   to a specific misaki Bopomofo notation; any minor version bump can break
   the tokens.txt mapping.
3. **Validate ZH after every container recreate** — single curl with `你好。`
   would catch this in 0.5 s. Add to deploy smoke test.
