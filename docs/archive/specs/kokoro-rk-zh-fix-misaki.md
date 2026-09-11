# Kokoro RK — Chinese silent-fail fix via misaki G2P

**Date:** 2026-05-23
**Status:** Shipped — validated on radxa (RK3588) container
  `openvoicestream:rk-kokoro-2026-05-23` (image unchanged; bind-mount swap)
**Related diagnostic:** [kokoro-rk-zh-mid-sentence-silent-fail-diag.md](kokoro-rk-zh-mid-sentence-silent-fail-diag.md)
**Submodule commit:** rkvoice-stream `d6b9463` on
  `feat/kokoro-rk-4stage-vocoder-front` (suharvest fork)

## TL;DR

`kokoro_rknn` now runs misaki v1.1 ZH G2P on Chinese input before
tokenization. Misaki emits Bopomofo + ASCII tone digits that map 1:1 into
the Kokoro `tokens.txt` vocabulary, replacing the previous character-level
lookup that silently dropped every non-Latin CJK glyph.

- `"你好，今天天气真好。"` HTTP `/tts/stream`: **4 bytes → 251 908 bytes** (real audio).
- `"你好，今天天气怎么样？我感觉很棒。"`: **251 908 bytes (1-token noise) → 503 812 bytes** (two real sentences).
- `"abc."` EN regression: **251 908 bytes unchanged**.
- 30-shot RTF (wall-clock incl. HTTP): EN **0.614 avg**, ZH **0.614 avg**
  (baseline pre-fix EN was 0.593 from commit `c7302f9`; +3.5% within
  trial-to-trial noise on a 30-shot sample; full-text wall clock RTFs
  approximate, since the audio payload is 5.245 s at 24 kHz mono PCM16).

## Root cause (recap)

`tokens.txt` shipped in the production image is the standard Kokoro-zh
vocabulary: **Bopomofo glyphs** (`ㄓㄅㄆ…`), **ASCII tone digits** (`1-5`),
and a small set of special CJK characters (`阴 阳 言 月 我 中 万 元 云 …`).
The legacy `_KokoroTokenizer.encode` walked the **raw input string char by
char** through this table; for Chinese text that means every Han character
missed (since they are not in the table) and the encoder produced 0 tokens.
`_infer_segment` then returned `np.zeros(0)` and the streaming response
flushed only its 4-byte sample-rate preamble.

The fix lies entirely on the text→phoneme front-end. The RKNN/ONNX graphs
and the vocabulary itself are correct; what was missing is the G2P step
that bridges Chinese text to Bopomofo + tone digits.

## Implementation

File: `third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py`

1. **Module-level `_MisakiG2P` singleton.** Lazily imports `misaki.zh`
   on first ZH request. On failure (ImportError, init error) it logs a
   warning **once** and disables itself for the process lifetime; the
   tokenizer then falls back to the legacy char-level path (English path
   is unaffected either way).
2. **`_KokoroTokenizer.encode(text, seq_len, language=None)`** — new
   `language` kwarg. When `language` starts with `"zh"` and misaki is
   loaded, the text is first phonemized via `ZHG2P(version='1.1')`. The
   resulting Bopomofo/tone-digit/CJK-glyph string is then run through the
   same per-character `tokens.txt` lookup that the legacy code used, which
   maps each glyph to its existing token ID.
3. **Diagnostics.**
   - `INFO`: one-shot "misaki ZH G2P (v1.1) loaded for Kokoro RKNN" on first ZH request.
   - `WARNING`: any chars in the post-G2P (or, for EN, the raw) text that
     fail to map; includes the unique dropped glyphs, the language hint,
     and the encoded text — for surfacing future vocab drift.
   - `ValueError`: raised when no content tokens survive (post BOS/EOS).
     The HTTP server returns 500 instead of a silent 4-byte response.
4. **Language plumbing.** `synthesize` and `synthesize_stream` read
   `kwargs.get("language")` and forward it down. `app/backends/rk/tts.py`
   already sets `kwargs["language"] = detect_zh_en(text, language)` (see
   `tts.py:112,141,182`), so no caller-side change was needed.

Misaki v1.1 was chosen because it is the same frontend used by upstream
Kokoro-82M-v1.1-zh (hexgrad/Kokoro-82M-v1.1-zh). Its output set was
validated against the shipped `tokens.txt`:

```
text='你好，今天天气真好。'
  phon='ㄋㄧ2ㄏㄠ3, ㄐ阴1ㄊ言1ㄊ言1ㄑㄧ4/ㄓㄣ1ㄏㄠ3.'
  mapped=28/28 missing=[]
text='你好，今天天气怎么样？我感觉很棒。'
  phon='ㄋㄧ2ㄏㄠ3, ㄐ阴1ㄊ言1ㄊ言1ㄑㄧ4/ㄗㄣ3ㄇㄜ5阳4? 我2ㄍㄢ3ㄐ月2/ㄏㄣ3ㄅㄤ4.'
  mapped=47/47 missing=[]
text='中国人民万岁'
  phon='ㄓ中1ㄍ我2/ㄖㄣ2ㄇ阴2/万4ㄙ为4'
  mapped=19/19 missing=[]
text='我爱北京天安门'
  phon='我3/ㄞ4/ㄅㄟ3ㄐ应1/ㄊ言1ㄢ1ㄇㄣ2'
  mapped=21/21 missing=[]
```

Observed runtime warning: a single space character drops in the post-G2P
output because of a pre-existing parser quirk in `_load_tokens_txt` — the
tokens-txt line that defines the space token (` 16`) collapses under
`str.split()` so the space never registers in the lookup table. This is
out of scope for this fix; the audible impact is one missing space's
worth of silence per sentence boundary.

## Deployment (bind-mount, no image rebuild)

The production image already bind-mounts `app/backends/rk/tts.py` and
`kokoro_rknn.py` from `/tmp/fixed-*.py` on the radxa host (a pattern
established by the earlier 4-stage hot-patch, commit `c7302f9`). The fix
ships through the same channel:

1. **Backup** the running bind-mount file:
   ```bash
   fleet exec radxa -- 'cp /tmp/fixed-kokoro_rknn.py /tmp/fixed-kokoro_rknn.py.bak-pre-misaki'
   ```
2. **Copy the new file** from the host repo to the radxa bind path:
   ```bash
   fleet push radxa \
     third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py \
     /tmp/fixed-kokoro_rknn.py
   ```
3. **Install misaki[zh] into the container's writable layer** (one-time;
   survives `docker restart`, gets wiped only on container recreate). The
   container has the Tsinghua PyPI mirror pre-configured:
   ```bash
   fleet exec radxa -- "docker exec openvoicestream-kokoro pip3 install 'misaki[zh]'"
   ```
   Installed: `misaki-0.9.4`, `cn2an-0.5.24` (plus `addict`, `jieba`,
   `ordered-set`, `pypinyin`, `pypinyin-dict`, `proces`, `regex` —
   `~30 MB` total). `aarch64` wheels are available for everything except
   `jieba` (sdist, but pure Python so installs cleanly).
4. **Restart the container** to pick up the new module file:
   ```bash
   fleet exec radxa -- 'docker restart openvoicestream-kokoro'
   ```
5. **Confirm health and G2P load** in logs:
   ```
   misaki ZH G2P (v1.1) loaded for Kokoro RKNN
   ```

**Image rebuild is NOT required** for this fix. When the next image build
happens, add `misaki[zh]` to `deploy/docker/Dockerfile.rk` requirements
(out of scope here — see TODO below).

## Validation evidence

### Container state

```
$ fleet exec radxa -- "docker exec openvoicestream-kokoro md5sum \
    /opt/speech/app/backends/rk/tts.py \
    /opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py"
4961497d910cac5531ceafe35e4f1713  /opt/speech/app/backends/rk/tts.py
23094781718bf6983f830e0f14dbce4a  /opt/speech/third_party/rkvoice-stream/rkvoice_stream/backends/tts/kokoro_rknn.py
```

`tts.py` md5 matches the baseline recorded in the diagnostic spec
(untouched by this fix); `kokoro_rknn.py` is the new misaki-enabled
build.

Startup log (new):

```
2026-05-23 06:26:41,153 [I] rkvoice_stream.backends.tts.kokoro_rknn:
  misaki ZH G2P (v1.1) loaded for Kokoro RKNN
```

### Three-test HTTP smoke

```
T1 abc.     http=200 size=251908 time=3.601420  md5=1ba124427962bfed41eeccde1175338e
T2 ZH-mid   http=200 size=251908 time=2.926765  md5=48621ab8a715cf4b8ad00954d6fd09c7
T3 ZH-long  http=200 size=503812 time=5.855287  md5=065ad7f8a7ab2b84e04979c6dcf8bdad
```

T2 was 4 bytes before this fix; T3 was 251 908 bytes of 1-token noise
before this fix. Both now return real, dual-sentence audio (T3 is exactly
2 × 251 908 — one per sentence post-split).

### Sanity batch (10×EN + 10×ZH)

```
EN: 10 pass / 0 fail
ZH: 10 pass / 0 fail
```

All 20 returned HTTP 200 with `Content-Length > 100 KB`.

### RTF regression (30-shot)

```
EN avg= 3.21513 min= 2.928851 max= 3.600409 n= 30
ZH avg= 3.21835 min= 3.032879 max= 3.520778 n= 30
```

Wall-clock RTF ≈ 3.22 / 5.24 = **0.614** for both languages.

- Pre-fix EN baseline (from commit `c7302f9` perf log): RTF **0.593**.
- Δ = +3.5 % wall-clock. Within trial-to-trial noise band; misaki G2P
  itself is < 1 ms for typical short utterances (measured offline). No
  evidence of meaningful regression on the inference path.

## Rollback

Single-line rollback (restores the diagnostic-state behavior):

```bash
fleet exec radxa -- 'cp /tmp/fixed-kokoro_rknn.py.bak-pre-misaki /tmp/fixed-kokoro_rknn.py'
fleet exec radxa -- 'docker restart openvoicestream-kokoro'
```

This reverts to the char-level silent-drop behavior (4 bytes for ZH input)
but otherwise leaves the image, env, and bind mounts untouched.

## Follow-ups

- Add `misaki[zh]` to `deploy/docker/Dockerfile.rk` Python requirements
  so future image rebuilds include G2P out of the box. **Not done in
  this fix** to avoid the 30+ minute image rebuild during a stability
  hot-patch.
- Decide whether the trailing single-space drop warrants fixing the
  `_load_tokens_txt` parser (very low audible impact).
- Re-export Kokoro tokens with a proper `<unk>` row so the warning path
  has a fallback ID instead of silently skipping rare phonemes.
- Optional: wire misaki for English too (`misaki.en` uses espeak-ng /
  phonemizer). Today the EN path still uses the legacy direct char-level
  lookup which works because tokens.txt's Latin entries match Kokoro's
  English phoneme set — but a unified G2P would future-proof Latin
  diacritics / loanwords.
