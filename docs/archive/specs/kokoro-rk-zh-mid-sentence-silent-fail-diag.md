# Kokoro RK — Chinese mid-length silent fail (4-byte response) diagnostic

**Date:** 2026-05-23
**Status:** Root cause identified — fix deferred to follow-up task
**Author:** diagnostic-only

## TL;DR

The "production bug" is **not** a 4-stage pipeline bug, not a vocoder front-half
RKNN issue, and not a misaki/G2P middleware glitch. The `kokoro_rknn` backend
shipped in the radxa image **has no Chinese text-to-phoneme front-end at all**.
Its tokenizer (`_KokoroTokenizer.encode`) walks the input string **character by
character**, looks each codepoint up in `tokens.txt` (171 entries, mostly Latin
+ English phoneme symbols, only 21 CJK chars), and silently drops anything not
found (`unk_id` is `None` because `tokens.txt` has no `<unk>` row).

For `"你好，今天天气真好。"`:

- The Chinese full stop `。` is consumed by `_split_sentences` as the segment
  delimiter.
- The remaining characters are `你 好 ， 今 天 天 气 真 好` — **none** of them
  exist in `tokens.txt`, and the comma is U+FF0C (Chinese comma) which is also
  absent.
- `encode()` returns `n_tokens=0`, `_infer_segment` short-circuits to
  `np.zeros(0)`, `synthesize_stream`'s `if audio.size == 0: continue` skips the
  segment, and the generator exits with zero chunks yielded.
- The HTTP wrapper still emits its 4-byte sample-rate preamble (`0x00005DC0`
  = 24000 LE) → client sees `Content-Length: 4`, audio length 0.

For the "long ZH" case `"你好，今天天气怎么样？我感觉很棒。"`, this is **not** a
real success either: it splits into two sentences, the first yields 0 tokens
(skipped), the second yields **1 token only** (`棒` at id 168 — the only mapped
character in the whole string), and the synthesized 251908 bytes is just that
single token padded out. Audibly this is essentially noise, not the requested
sentence.

The 4-stage hot-promotion (commit `c7302f9`) did not introduce this — the same
behaviour exists on the 3-stage path. It only became user-visible because
production traffic now hits the Chinese sentence.

## Reproduction

Container state at time of diagnosis:

```
$ fleet exec radxa -- docker ps | grep kokoro
openvoicestream-kokoro   openvoicestream:rk-kokoro-2026-05-23   Up 40 minutes

$ docker exec openvoicestream-kokoro md5sum \
    /opt/speech/app/backends/rk/tts.py \
    /opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py
4961497d910cac5531ceafe35e4f1713  /opt/speech/app/backends/rk/tts.py
beff1378356c30d06cc0462d4149fe66  /opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py
```

HTTP repro:

```
=== TEST 1: short EN "abc." ===
size=251908 http=200
=== TEST 2: ZH mid (BUG) "你好，今天天气真好。" ===
size=4 http=200
=== TEST 3: ZH "long" "你好，今天天气怎么样？我感觉很棒。" ===
size=251908 http=200      # actually 1-token garbled, NOT real audio
```

Response of TEST 2 (the bug), full hex dump:

```
$ od -An -tx1 -N 32 /tmp/t2.wav
 c0 5d 00 00
```

`0x00005DC0` LE = `24000` → the SR preamble Starlette `StreamingResponse` writes
before any audio chunks. Zero PCM chunks ever yielded.

## Stage-by-stage instrumentation (in-process)

Direct tokenizer call inside the container (skipping HTTP):

| Text | `_split_sentences` output | Per-segment `n_tokens` | Resulting ids |
|---|---|---|---|
| `"abc."` | `['abc.']` | `4` | `[43, 44, 45, 4]` |
| `"你好，今天天气真好。"` | `['你好，今天天气真好']` | **`0`** | all zeros (pad) |
| `"你好，今天天气怎么样？我感觉很棒。"` | `['你好，今天天气怎么样？', '我感觉很棒']` | `0, 1` | seg2: `[168, ...]` |

Per-character lookup inside the production `tokens.txt` (171 entries total):

```
'你' -> None     '好' -> None    '今' -> None    '天' -> None
'天' -> None     '气' -> None    '真' -> None    '好' -> None
'a'  -> 43       'b'  -> 44      'c'  -> 45      '.'  -> 4
'，' -> None     '。' -> None
```

- `unk_id`: **`None`** (no `<unk>` row in `tokens.txt`)
- `bos_id`: **`None`** (no `<s>`/`<bos>`/`^` row)
- `eos_id`: **`None`**
- `pad_id`: `0`
- CJK characters present: **21 total**, none of the ones in the failing
  sentence. `棒` (id 168) is one of the lucky ones — hence test 3's
  "non-zero-but-wrong" output.

`_split_sentences` regex is `(?<=[.!?;。！？；\n])\s*` — note that **U+FF0C
Chinese comma `，` is NOT a sentence delimiter**, so it stays inside the
segment, but is also not in the tokens map, so it gets silently dropped during
encode (because `unk_id=None`).

## Root cause

`KokoroRKNNBackend._tokenizer` is a character-level lookup against the kokoro
phoneme-symbol table. The exported model expects **phoneme tokens** (Kokoro
upstream uses `misaki` G2P to convert raw text → phonemes → ids before this
stage). The radxa image ships:

- the prefix/decoder/vocoder/tail-rest model artifacts ✓
- the tokens.txt phoneme map ✓
- but **no G2P pre-processor** (no `misaki`, no `phonemizer`, no
  language-specific tokenizer wrapper)

So:

- English ASCII text "happens to work" only because Latin letters and a handful
  of punctuation marks share ids with the phoneme symbols in `tokens.txt`. The
  audio it produces for `"abc."` is also not real English speech — it is the
  phonemes whose symbols happen to be `a`, `b`, `c`, `.` — which is fine as a
  smoke test but not real synthesis.
- Chinese text has effectively zero overlap with phoneme symbols, so it
  collapses to 0–1 tokens depending on which CJK characters happen to be in
  the table.

This is a **missing-G2P bug**, not a 4-stage pipeline bug. The 4-stage
RKNN/CPU split (commit `c7302f9` promoted to default) is unrelated and
correctly executes when `n_tokens > 0`.

### Why this was not caught earlier

- Earlier perf reports (`docs/specs/kokoro-rk-34pct-*.md`) measured RTF and
  audio bytes, not audio intelligibility — and used short ASCII strings.
- The BERT A/B report (`kokoro-bert-ab-audio-report.md`) likely used English.
- "Long Chinese works" was a false positive: it returned ≥1 segment with ≥1
  matched CJK char, so it produced non-empty bytes and was assumed to be
  working audio.

### Why no exception / error log

`_infer_segment` early-returns `np.zeros(0)` when `n_tokens == 0` (kokoro_rknn.py
line 418). `synthesize_stream` then does `if audio.size == 0: continue`. The
HTTP wrapper sees an empty generator and closes the stream cleanly. There is
**no warning, no error, no log line** anywhere in the path. This is a silent
fail by construction.

## Suggested fix paths (not implemented)

Listed cheapest first. **All require a follow-up task.**

### Option A — Reject upstream (cheapest, safest)

Add a guard in `app/backends/rk/tts.py`'s `generate_streaming` / `synthesize`:
after `detect_zh_en` runs, if `language == "zh"` raise a clear error (or fall
back to a different backend). This stops silent-fail but disables Chinese
TTS on RK entirely until a proper fix lands.

### Option B — Add explicit `<unk>` handling + log

Patch `_KokoroTokenizer.encode` to log a WARNING when ≥1 character drops out,
and propagate a `meta["dropped_chars"]` counter. Doesn't fix the audio, but
makes the failure mode visible in logs and gives ops a hook to alert on.

### Option C — Wire `misaki` (the real fix)

Add `misaki` (Kokoro upstream's G2P) into the radxa image (`pip install
misaki[zh,en]` adds CJK pypinyin + jieba dependencies) and call it before
`_tokenizer.encode`. This is the production-grade fix but adds ~50MB to the
image and needs CJK dictionary models. The kokoro_rknn.py file's own docstring
hints at this:

```python
# Production-quality G2P should live in the exported model package and
# provide a matching tokens file.
```

The exported model package (`/opt/kokoro-rknn`) doesn't ship a G2P, so this
must be added at the Python wrapper layer.

### Option D — Re-export model with character-token vocab

Re-train/re-export Kokoro with a Chinese-character-aware token table. Out of
scope; mentioned for completeness.

## Recommended next step

1. Ship Option A (reject ZH at the RK adapter) as an immediate hotfix so
   silent fails stop reaching users.
2. Ship Option B (log dropped chars) so we have telemetry.
3. Plan Option C (misaki integration) as a separate spec — needs image-size
   review and CJK dictionary licensing check.

## File references

- Backend wrapper: `/opt/speech/app/backends/rk/tts.py` (container);
  source: `app/backends/rk/tts.py` (repo). md5 `4961497d910cac5531ceafe35e4f1713`.
- Inner backend: `/opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py`. md5 `beff1378356c30d06cc0462d4149fe66`.
  - `_split_sentences` — line 78
  - `_KokoroTokenizer.encode` — line 161
  - `_infer_segment` — line 412 (`n_tokens == 0` early-return at 418)
  - `synthesize_stream` — line 600 (`if audio.size == 0: continue` at 612)
- Tokens map: `/opt/kokoro-rknn/tokens.txt` (171 entries, 21 CJK, no `<unk>`).
- Image: `openvoicestream:rk-kokoro-2026-05-23`.
- Promoted-as-default commit: `c7302f9` (4-stage hybrid; unrelated to this bug).
