from ovs_agent.audio_io import AudioIO


def test_playback_position_counts_only_bytes_handed_to_output_callback() -> None:
    audio = AudioIO(output_sr=16000)
    audio._ensure_playback_buffer()
    audio.begin_response_playback("resp_1")
    with audio._playback_lock:
        audio._playback_buffer.extend(b"\x01\x00" * 160)

    out = bytearray(640)
    audio._output_callback(out, 320, None, None)

    # Only 160 actual PCM samples were available; callback zero-fill does not
    # count as assistant audio heard by the user.
    assert audio.playback_position_ms("resp_1") == 10
    assert audio.playback_position_ms("another_response") == 0
