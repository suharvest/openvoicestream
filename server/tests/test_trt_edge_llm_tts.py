import json
import subprocess
import io
import wave
import types


def _make_wav_bytes(frame_count: int, sample_rate: int = 24000) -> bytes:
    payload = b"\x00\x00" * frame_count
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(payload)
    return out.getvalue()


# NOTE: the non-worker one-shot binary synth path (--codePredictorEngineDir +
# talker sampling + min_audio_length) moved to voxedge with the env-free
# migration (TTS_BINARY/PLUGIN_PATH module constants are now config fields). It is
# re-covered in voxedge/tests/test_tts_oneshot_and_product_segment.py.


def test_split_tts_text_handles_cjk_and_latin(monkeypatch):
    import voxedge.backends.jetson.trt_edge_llm_tts as tts_mod

    zh = "你好，很高兴认识你。今天我们来测试一下语音合成的稳定性，看看这段稍微长一点的中文是不是能清楚自然地读出来。"
    zh_parts = tts_mod._split_tts_text(zh, max_chars=24)

    assert len(zh_parts) > 1
    assert "".join(zh_parts) == zh
    assert max(len(part) for part in zh_parts) <= 25
    assert all(part not in "。！？!?；;，,、：" for part in zh_parts)

    no_punctuation = "这是一个没有任何标点符号的很长中文单句我们要验证它会不会被切短"
    no_punctuation_parts = tts_mod._split_tts_text(no_punctuation, max_chars=16)
    assert "".join(no_punctuation_parts) == no_punctuation
    assert max(len(part) for part in no_punctuation_parts) <= 16

    punctuated = "真的吗？可以的，请继续！不过，逗号也要保留。"
    punctuated_parts = tts_mod._split_tts_text(punctuated, max_chars=8)

    assert "".join(punctuated_parts) == punctuated
    assert max(len(part) for part in punctuated_parts) <= 8
    assert any(part.endswith("？") for part in punctuated_parts)
    assert any(part.endswith("！") for part in punctuated_parts)
    assert any("，" in part for part in punctuated_parts)
    assert all(part not in "。！？!?；;，,、：" for part in punctuated_parts)

    en = "Hello, this is a longer text for validating that product-side segmentation also works for English input without relying on Chinese punctuation."
    en_parts = tts_mod._split_tts_text(en, max_chars=48)

    assert len(en_parts) > 1
    assert " ".join(en_parts).replace("  ", " ") == en
    assert max(len(part) for part in en_parts) <= 48


def test_split_tts_text_preserves_common_punctuation_and_grammar():
    import voxedge.backends.jetson.trt_edge_llm_tts as tts_mod

    cases = [
        ("中文", "真的吗？可以的，请继续！不过，逗号、顿号、冒号：都要保留。", 8, ""),
        ("中文引号", "他说：“今天很好，可以继续。”然后停了一下。", 10, ""),
        ("英文", "Really? Yes, please continue! However, commas, semicolons; and colons: must stay.", 28, " "),
        ("英文缩写", "Dr. Smith said, \"Let's test TTS, ASR, and V2V.\" It worked.", 32, " "),
        ("混合", "EdgeLLM 可以跑 TTS/ASR，对吗？Yes, it can.", 12, ""),
    ]

    punctuation = set("。！？!?；;，,、：:.\"'“”‘’()（）")
    for _, text, max_chars, joiner in cases:
        parts = tts_mod._split_tts_text(text, max_chars=max_chars)
        reconstructed = joiner.join(parts).replace("  ", " ") if joiner else "".join(parts)

        assert reconstructed == text
        assert len(parts) > 1
        assert all(part.strip() for part in parts)
        assert all(not set(part).issubset(punctuation) for part in parts)

    zh_parts = tts_mod._split_tts_text(cases[0][1], max_chars=8)
    assert any(part.endswith("？") for part in zh_parts)
    assert any(part.endswith("！") for part in zh_parts)
    assert any("，" in part for part in zh_parts)

    en_parts = tts_mod._split_tts_text(cases[2][1], max_chars=28)
    assert any(part.endswith("?") for part in en_parts)
    assert any(part.endswith("!") for part in en_parts)
    assert any("," in part for part in en_parts)

    abbrev_parts = tts_mod._split_tts_text(cases[3][1], max_chars=32)
    assert all(part != "Dr." for part in abbrev_parts)
    assert "Dr. Smith" in " ".join(abbrev_parts)

    decimal = "Version 3.14 works. Version 4.0 also works!"
    decimal_parts = tts_mod._split_tts_text(decimal, max_chars=24)
    assert "3.14" in " ".join(decimal_parts)
    assert "4.0" in " ".join(decimal_parts)


# NOTE: the segmented one-shot synth + concat-with-segment-pauses path moved to
# voxedge — re-covered in voxedge/tests/test_tts_oneshot_and_product_segment.py.


def test_cjk_default_segmentation_prefers_sentence_boundary(monkeypatch):
    import voxedge.backends.jetson.trt_edge_llm_tts as tts_mod

    monkeypatch.delenv("EDGE_LLM_TTS_CJK_SEGMENT_MAX_CHARS", raising=False)
    text = "你好，今天我们继续验证语音合成的稳定性。这个版本应该保持清晰自然，不应该出现逐渐变沙、吞音或者明显的噪声积累。"
    parts = tts_mod._split_tts_text(text)

    assert parts == [
        "你好，今天我们继续验证语音合成的稳定性。",
        "这个版本应该保持清晰自然，不应该出现逐渐变沙、吞音或者明显的噪声积累。",
    ]


# NOTE: the qwen3 product-segmentation synthesize orchestration and
# product_explicit_kv backend selection moved to voxedge — re-covered in
# voxedge/tests/test_tts_oneshot_and_product_segment.py. backend selection is
# now purely config.backend_mode (env→config covered in test_voxedge_backend_config.py).


# NOTE: the config → worker-env / worker-request behavior these 11
# ``test_edgellm_worker_*`` tests asserted (perf-profile first_chunk_frames,
# stateful/official env dict, v2v streaming window, fixed-seed segment reuse,
# base64 chunk decode) is now config-driven in voxedge and re-covered in
# voxedge/tests/test_trt_edge_llm_tts_worker_behavior.py. The env→config half
# (OVS_TTS_* aliases, perf_profile, seed, talker_*) is covered product-side in
# test_voxedge_backend_config.py. The OVS_TTS_SPEAKERS_JSON registry lookups
# (speaker-name / embedding-by-id) are dropped product behavior — voxedge
# resolve_speaker_kwargs is registry-free. The explicit-CP-groups env passthrough
# test below still exercises a product os.environ passthrough and stays.


def test_edgellm_worker_stateful_respects_explicit_cp_groups(monkeypatch):
    import voxedge.backends.jetson.trt_edge_llm_tts as tts_mod

    monkeypatch.setenv("EDGE_LLM_TTS_STATEFUL_CODE2WAV", "1")
    monkeypatch.setenv("EDGE_LLM_TTS_PERF_PROFILE", "balanced")
    monkeypatch.setenv("QWEN3_TTS_ACTIVE_CP_GROUPS", "14")

    backend = tts_mod.TRTEdgeLLMTTSBackend()
    env = backend._worker_env()

    assert env["QWEN3_TTS_ACTIVE_CP_GROUPS"] == "14"
