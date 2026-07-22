"""Self-contained golden test for the vendored CLIP BPE tokenizer.

Needs NO ``clip`` / ``torch`` / ``regex`` dependency. It imports only the
vendored ``clip_tokenizer`` module and asserts its output equals FROZEN
expected id rows that were captured from ``clip.clip.tokenize(text, 77,
truncate=True)`` in the probe venv (which has the real ``clip`` package).

If this test ever fails, the vendored tokenizer has diverged from upstream
OpenAI CLIP BPE -- the YOLOE text encoder would then receive wrong token rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from ovs_agent.apps.voice_rebot_arm.perception import clip_tokenizer

CONTEXT_LENGTH = 77
SOT = 49406
EOT = 49407


def _nz(row):
    """Non-zero (real-token) prefix of a 77-wide row, as a plain list."""
    arr = np.asarray(row)
    return arr[arr != 0].tolist()


# Frozen non-zero prefixes captured from clip.clip.tokenize(..., 77, truncate=True).
# Key = input string, value = the leading non-zero token ids (the rest is 0-pad
# out to context_length=77). EOT (49407) is always the last non-zero id.
GOLDEN_PREFIX = {
    # --- the 5 grasp classes ---
    "box": [SOT, 2063, EOT],
    "cardboard box": [SOT, 19845, 2063, EOT],
    "carton": [SOT, 41812, EOT],
    "package": [SOT, 7053, EOT],
    "yellow banana": [SOT, 4481, 8922, EOT],
    # --- broad coverage ---
    "Yellow Banana": [SOT, 4481, 8922, EOT],            # mixed case -> lowercased
    "box 12": [SOT, 2063, 272, 273, EOT],               # digits (each digit one token)
    "red, green & blue!": [SOT, 736, 267, 1901, 261, 1746, 256, EOT],  # punctuation
    "a cup of coffee": [SOT, 320, 1937, 539, 2453, EOT],               # multi-word
    "12345": [SOT, 272, 273, 274, 275, 276, EOT],       # all digits
    "apple": [SOT, 3055, EOT],                          # generic word
    "table": [SOT, 2175, EOT],                          # generic word
    "don't drop it": [SOT, 847, 713, 3387, 585, EOT],   # contraction
    # --- embin-vocab edge cases (captured from probe venv clip.clip.tokenize,
    #     2026-06-15): hyphen, trailing/leading/multi space, apostrophe, mixed
    #     alnum+hyphen identifier (B601-DM) ---
    "cardboard-box": [SOT, 19845, 268, 2063, EOT],
    "box   ": [SOT, 2063, EOT],                          # trailing spaces stripped
    "  yellow   banana  ": [SOT, 4481, 8922, EOT],       # leading/multi/trailing
    "worker's glove": [SOT, 8098, 568, 16078, EOT],      # apostrophe contraction
    "B601-DM box": [SOT, 321, 277, 271, 272, 268, 4667, 2063, EOT],
}

GRASP_CLASSES = ["box", "cardboard box", "carton", "package", "yellow banana"]

# A long input (>75 bpe tokens) that triggers truncate=True clamping to 77,
# keeping EOT last. 100 repeats of "banana" -> 100 bpe tokens, clamped to 75
# real tokens after SOT, with EOT forced into the final slot.
LONG_INPUT = " ".join(["banana"] * 100)
# Expected full row: [SOT, 8922 x 75, EOT] (length exactly 77, no padding).
LONG_EXPECTED = [SOT] + [8922] * 75 + [EOT]


def test_output_shape_and_dtype():
    out = clip_tokenizer.tokenize(["box", "yellow banana"])
    assert out.dtype == np.int32
    assert out.shape == (2, CONTEXT_LENGTH)


@pytest.mark.parametrize("text", list(GOLDEN_PREFIX.keys()))
def test_golden_prefix_matches(text):
    out = clip_tokenizer.tokenize([text], context_length=CONTEXT_LENGTH, truncate=True)
    assert out.shape == (1, CONTEXT_LENGTH)
    assert out.dtype == np.int32
    # full row is the golden prefix followed by zero padding
    expected_prefix = GOLDEN_PREFIX[text]
    assert _nz(out[0]) == expected_prefix
    # padding region is all zeros
    assert out[0, len(expected_prefix):].tolist() == [0] * (CONTEXT_LENGTH - len(expected_prefix))
    # EOT is the last real token
    assert expected_prefix[-1] == EOT


@pytest.mark.parametrize("text", GRASP_CLASSES)
def test_grasp_classes_structure(text):
    """Every grasp class must start with SOT and end its real tokens with EOT."""
    row = clip_tokenizer.tokenize(text)[0]
    prefix = _nz(row)
    assert prefix[0] == SOT
    assert prefix[-1] == EOT
    assert prefix == GOLDEN_PREFIX[text]


def test_string_input_equals_singleton_list():
    a = clip_tokenizer.tokenize("yellow banana")
    b = clip_tokenizer.tokenize(["yellow banana"])
    assert np.array_equal(a, b)
    assert a.shape == (1, CONTEXT_LENGTH)


def test_batch_rows_independent():
    texts = GRASP_CLASSES
    out = clip_tokenizer.tokenize(texts)
    assert out.shape == (len(texts), CONTEXT_LENGTH)
    for i, t in enumerate(texts):
        assert _nz(out[i]) == GOLDEN_PREFIX[t]


def test_truncate_true_clamps_with_eot_last():
    out = clip_tokenizer.tokenize([LONG_INPUT], context_length=CONTEXT_LENGTH, truncate=True)
    row = out[0]
    assert row.tolist() == LONG_EXPECTED
    assert int(row[-1]) == EOT          # EOT kept as final token
    assert int((row != 0).sum()) == CONTEXT_LENGTH  # fully packed, no padding


def test_truncate_false_raises_on_overflow():
    with pytest.raises(RuntimeError):
        clip_tokenizer.tokenize([LONG_INPUT], context_length=CONTEXT_LENGTH, truncate=False)


def test_special_token_ids():
    # SOT / EOT must be the canonical CLIP ids the YOLOE encoder expects.
    tok = clip_tokenizer._get_tokenizer()
    assert tok.sot_token_id == SOT
    assert tok.eot_token_id == EOT
