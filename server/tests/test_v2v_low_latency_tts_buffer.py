from server.core.v2v import LowLatencyTTSBuffer, SentenceBuffer


def test_low_latency_tts_buffer_emits_cjk_clause_before_full_sentence():
    buf = LowLatencyTTSBuffer(language="zh", min_chars=8, target_chars=12, max_chars=16)

    assert list(buf.add("从前有个小狐狸，")) == ["从前有个小狐狸，"]
    assert list(buf.add("它特别喜欢冒险。")) == ["它特别喜欢冒险。"]
    assert buf.is_empty()


def test_low_latency_tts_buffer_emits_bounded_cjk_run_without_punctuation():
    buf = LowLatencyTTSBuffer(language="zh", target_chars=10, max_chars=14)

    chunks = list(buf.add("这是一个没有标点但应该尽快开口的中文回复"))

    assert chunks == ["这是一个没有标点但应该尽快开口"]
    assert list(buf.flush()) == ["的中文回复"]


def test_low_latency_tts_buffer_default_cjk_waits_for_sentence_end():
    buf = LowLatencyTTSBuffer(language="zh")

    text = "从前有个小狐狸，它特别喜欢收集各种小玩意儿。"
    out = []
    for ch in text:
        out.extend(buf.add(ch))

    assert out == [text]
    assert list(buf.flush()) == []


def test_low_latency_tts_buffer_keeps_short_soft_break_until_useful():
    buf = LowLatencyTTSBuffer(language="zh", min_chars=8, target_chars=12, max_chars=16)

    assert list(buf.add("你好，")) == []
    assert list(buf.add("我现在可以")) == []
    assert list(buf.add("继续处理")) == ["你好，我现在可以继续处理"]
    assert list(buf.flush()) == []


def test_low_latency_tts_buffer_does_not_flush_on_short_early_comma():
    buf = LowLatencyTTSBuffer(language="zh", min_chars=15, target_chars=24, max_chars=40)

    assert list(buf.add("从前有个小狐狸，")) == []
    assert list(buf.add("它特别")) == []
    assert list(buf.add("喜欢收集")) == []
    assert list(buf.add("各种小玩意儿。")) == ["从前有个小狐狸，它特别喜欢收集各种小玩意儿。"]


def test_low_latency_tts_buffer_flushes_remainder():
    buf = LowLatencyTTSBuffer(language="zh")

    assert list(buf.add("再给我讲个")) == []
    assert list(buf.flush()) == ["再给我讲个"]
    assert buf.is_empty()


def test_low_latency_tts_buffer_drops_punctuation_only_tail_after_length_cut():
    buf = LowLatencyTTSBuffer(language="zh", min_chars=4, target_chars=6, max_chars=6)

    assert list(buf.add("这是六个汉字")) == ["这是六个汉字"]
    assert list(buf.add("。")) == []
    assert list(buf.flush()) == []
    assert buf.is_empty()


def test_low_latency_tts_buffer_keeps_short_speakable_final_with_punctuation():
    buf = LowLatencyTTSBuffer(language="zh")

    assert list(buf.add("好。")) == ["好。"]
    assert list(buf.flush()) == []


def test_sentence_buffer_still_waits_for_pysbd_or_flush():
    buf = SentenceBuffer(language="zh")

    assert list(buf.add("从前有个小狐狸，")) == []
    assert list(buf.flush()) == ["从前有个小狐狸，"]


def test_sentence_buffer_drops_punctuation_only_flush():
    buf = SentenceBuffer(language="zh")

    assert list(buf.add("。")) == []
    assert list(buf.flush()) == []
