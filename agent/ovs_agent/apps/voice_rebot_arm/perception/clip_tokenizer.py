# Pure-Python (stdlib-only) vendored OpenAI CLIP BPE tokenizer.
#
# Why this exists
# ---------------
# The YOLOE text-encoder ONNX (`text_encoder_pe.onnx`) takes `tokens [1,77]
# int32` and expects the OpenAI CLIP BPE tokenization. The production device
# image has NO `clip` / `torch` / `regex` packages, so we vendor the tokenizer
# with ZERO third-party deps (numpy is a repo dependency and is allowed).
#
# Output contract (byte-identical to `clip.clip.tokenize`):
#   tokenize(texts, context_length=77, truncate=True) -> np.ndarray int32 [N,77]
#   row = [49406, <bpe ids...>, 49407, 0, 0, ...]  (SOT=49406, EOT=49407)
#   truncate=True clamps to context_length keeping EOT as the last token.
#
# Parity notes (vs upstream clip.simple_tokenizer.SimpleTokenizer)
# ---------------------------------------------------------------
# * Upstream `basic_clean` calls `ftfy.fix_text(text)` before HTML-unescaping.
#   ftfy is a mojibake / encoding repairer. For well-formed Unicode input
#   (which the grasp class vocabulary always is -- lowercase ASCII words such
#   as "yellow banana", "cardboard box") ftfy is the identity function, so we
#   omit it. The ONLY way to observe a divergence is to feed deliberately
#   mojibaked / malformed text (e.g. "Ã©" that ftfy would "fix" to "é"). The
#   grasp vocabulary never contains such input, so parity is exact there.
#
# * Upstream uses the third-party `regex` module with `\p{L}` / `\p{N}` Unicode
#   property classes. Stdlib `re` does not support `\p{...}`. We reproduce the
#   exact same token boundaries with a hand-written scanner whose character
#   predicates use Python's `str.isalpha()` / `str.isdigit()`, which follow the
#   Unicode General_Category L / N classes. The scanner applies the same
#   alternation priority as the upstream pattern:
#       <|startoftext|> | <|endoftext|> | 's|'t|'re|'ve|'m|'ll|'d
#       | [letters]+ | [single digit] | [non-space non-letter non-digit]+
#   (the upstream pattern is compiled IGNORECASE; the contractions are matched
#    case-insensitively to mirror that).

from __future__ import annotations

import gzip
import html
import os
from functools import lru_cache

import numpy as np

__all__ = ["tokenize", "SimpleTokenizer", "default_bpe"]

_VOCAB_FILENAME = "clip_bpe_vocab_16e6.txt.gz"

# Contraction suffixes, in the order the upstream pattern lists them. They are
# matched case-insensitively (upstream compiles the pattern with re.IGNORECASE).
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")

_SPECIAL_TOKENS = ("<|startoftext|>", "<|endoftext|>")


@lru_cache()
def default_bpe() -> str:
    """Path to the vendored BPE vocab/merges gzip, resolved next to this file."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _VOCAB_FILENAME)


@lru_cache()
def bytes_to_unicode():
    """Reversible map: utf-8 byte value -> printable unicode char.

    Identical to the upstream OpenAI/GPT-2 table.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Set of adjacent symbol pairs in a word (tuple of symbol strings)."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text: str) -> str:
    """HTML-unescape (twice, as upstream) and strip.

    Upstream also runs `ftfy.fix_text` first; for well-formed input that is the
    identity, and the grasp vocabulary is always well-formed -- see module
    docstring for the documented divergence on deliberately mojibaked input.
    """
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    """Collapse runs of whitespace to single spaces, then strip.

    Mirrors upstream `re.sub(r"\\s+", " ", text).strip()`. Stdlib `re` is not
    needed: `str.split()` (no arg) splits on arbitrary Unicode whitespace runs,
    exactly matching `\\s+` under the `regex`/`re` UNICODE default, and joining
    on a single space reproduces the substitution.
    """
    return " ".join(text.split())


def _findall(text: str):
    """Reproduce upstream `regex.findall(pat, text)` token boundaries.

    Upstream pattern (IGNORECASE):
        <|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d
        |[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+

    We scan left-to-right, at each position taking the first alternative that
    matches (regex alternation is ordered / leftmost-first).
    """
    tokens = []
    i = 0
    n = len(text)
    lower = text.lower()
    while i < n:
        ch = text[i]

        # 1. special tokens (literal)
        matched_special = False
        for sp in _SPECIAL_TOKENS:
            if text.startswith(sp, i):
                tokens.append(sp)
                i += len(sp)
                matched_special = True
                break
        if matched_special:
            continue

        # 2. contractions (case-insensitive, like IGNORECASE)
        matched_contraction = False
        for c in _CONTRACTIONS:
            if lower.startswith(c, i):
                tokens.append(text[i : i + len(c)])
                i += len(c)
                matched_contraction = True
                break
        if matched_contraction:
            continue

        # 3. [\p{L}]+  -- one or more letters
        if ch.isalpha():
            j = i + 1
            while j < n and text[j].isalpha():
                j += 1
            tokens.append(text[i:j])
            i = j
            continue

        # 4. [\p{N}]  -- exactly one numeric char
        if ch.isdigit():
            tokens.append(ch)
            i += 1
            continue

        # 5. whitespace: not captured by any alternative -> skip it
        if ch.isspace():
            i += 1
            continue

        # 6. [^\s\p{L}\p{N}]+  -- run of non-space, non-letter, non-digit
        j = i + 1
        while j < n:
            cj = text[j]
            if cj.isspace() or cj.isalpha() or cj.isdigit():
                break
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


class SimpleTokenizer:
    """Stdlib-only re-implementation of clip.simple_tokenizer.SimpleTokenizer."""

    def __init__(self, bpe_path: str | None = None):
        bpe_path = bpe_path or default_bpe()
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.sot_token_id = self.encoder["<|startoftext|>"]
        self.eot_token_id = self.encoder["<|endoftext|>"]
        self.context_length = 77

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in _findall(text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens


@lru_cache()
def _get_tokenizer() -> SimpleTokenizer:
    return SimpleTokenizer()


def tokenize(
    texts: list[str] | str,
    context_length: int = 77,
    truncate: bool = True,
) -> np.ndarray:
    """Tokenize input string(s) to CLIP token-id rows.

    Byte-identical to ``clip.clip.tokenize(texts, context_length, truncate)``.

    Returns
    -------
    np.ndarray of dtype int32, shape ``[len(texts), context_length]``.
    Each row is ``[49406, <bpe ids>, 49407, 0, 0, ...]``. When the BPE id
    sequence is too long, ``truncate=True`` clamps the row to
    ``context_length`` keeping EOT (49407) as the final token; ``truncate=False``
    raises ``RuntimeError`` (matching upstream).
    """
    if isinstance(texts, str):
        texts = [texts]

    tok = _get_tokenizer()
    sot_token = tok.sot_token_id
    eot_token = tok.eot_token_id
    all_tokens = [[sot_token] + tok.encode(text) + [eot_token] for text in texts]

    result = np.zeros((len(all_tokens), context_length), dtype=np.int32)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]!r} is too long for context length {context_length}"
                )
        result[i, : len(tokens)] = np.asarray(tokens, dtype=np.int32)
    return result
