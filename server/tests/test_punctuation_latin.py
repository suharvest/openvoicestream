"""Latin-context normalization of CT-Transformer punctuation output.

The model (ct_transformer_zh_en_vocab272727) has a CJK-only label set, so it
emits full-width marks even for all-English text, and it tokenizes a
pre-existing ASCII terminal mark as its own token. Both are corrected in the
product layer. These tests exercise the pure helpers, so no model is loaded.
"""

from __future__ import annotations

import pytest

from server.core.punctuation import _localize_punct, _TRAILING_PUNCT


@pytest.mark.parametrize(
    "raw, expected",
    [
        # All-English output: full-width marks become ASCII with spacing.
        (
            "Television reports show white smoke coming from the plant。",
            "Television reports show white smoke coming from the plant.",
        ),
        (
            "However，due to the slow communication，channels，styles in the "
            "West could lag behind by 25 to 30 years。",
            "However, due to the slow communication, channels, styles in the "
            "West could lag behind by 25 to 30 years.",
        ),
        # Pure Chinese is left alone.
        ("周二，他在大阪去世。", "周二，他在大阪去世。"),
        ("适当使用博客，可以使学生变得更善于分析。", "适当使用博客，可以使学生变得更善于分析。"),
        # Mixed script: a Latin word followed by CJK keeps the full-width mark,
        # because converting it would read wrong in the Chinese sentence.
        ("他说 hello，世界。", "他说 hello，世界。"),
        ("价格是 25，含税。", "价格是 25，含税。"),
        # Closing quote / bracket before the mark still counts as Latin.
        ('He said "no"。', 'He said "no".'),
    ],
)
def test_localize_punct(raw: str, expected: str) -> None:
    assert _localize_punct(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        # SenseVoice already punctuates English; feeding the trailing period
        # back into the model is what produced "plant .。".
        ("Television reports show white smoke coming from the plant.",
         "Television reports show white smoke coming from the plant"),
        ("周二，他在大阪去世，", "周二，他在大阪去世"),
        # Internal marks survive; only the tail is trimmed.
        ("a, b, c.", "a, b, c"),
    ],
)
def test_trailing_punct_is_trimmed(raw: str, expected: str) -> None:
    assert raw.strip().rstrip(_TRAILING_PUNCT) == expected


def test_localize_punct_is_idempotent() -> None:
    once = _localize_punct("However，due to the slow communication。")
    assert _localize_punct(once) == once


def test_localize_punct_leaves_unmarked_text_untouched() -> None:
    for text in ("", "no punctuation here", "纯中文没有标点"):
        assert _localize_punct(text) == text
