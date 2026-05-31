"""Behavior-equivalence smoke test for the thin-shell translator service.

No real model / no httpx needed: fake ``ctranslate2`` + ``sentencepiece``
modules are injected (mirroring voxedge's own unit test), and the FastAPI
lifespan + ``/translate`` handler are driven directly via asyncio.

Asserts:
  * lifespan constructs + preloads the voxedge NLLBTranslatorBackend from env,
  * the ``/translate`` handler returns the unchanged response schema,
  * the token path matches the pre-extraction ``_translate_sync`` exactly:
    EncodeAsPieces, ``</s>``+src appended (not prefixed), tgt_lang prefix,
    position-0 tgt token stripped before decode.

Run:  uv run python test_server_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import types


# ── fakes (same shape as voxedge/tests/test_nllb_translator.py) ──────────────
class _FakeSP:
    def __init__(self):
        self.load_path = None
        self.encode_calls = []
        self.decode_calls = []

    def Load(self, path):  # noqa: N802
        self.load_path = path

    def EncodeAsPieces(self, text):  # noqa: N802
        self.encode_calls.append(text)
        return [f"▁{text}", "piece2"]

    def DecodePieces(self, pieces):  # noqa: N802
        self.decode_calls.append(list(pieces))
        return " ".join(pieces)


class _FakeHyp:
    def __init__(self, tokens):
        self.hypotheses = [tokens]


class _FakeTranslator:
    last = None

    def __init__(self, *args, **kwargs):
        type(self).last = {"args": args, "kwargs": kwargs}
        self.batch_calls = []

    def translate_batch(self, batch, **kwargs):
        self.batch_calls.append({"batch": batch, "kwargs": kwargs})
        prefix = kwargs.get("target_prefix", [["xxx"]])
        return [_FakeHyp([prefix[i][0], "hello", "world"]) for i in range(len(batch))]


def _install_fakes():
    ct2 = types.ModuleType("ctranslate2")
    ct2.Translator = _FakeTranslator
    ct2.get_cuda_device_count = lambda: 1
    sp = types.ModuleType("sentencepiece")
    sp.SentencePieceProcessor = _FakeSP
    sys.modules["ctranslate2"] = ct2
    sys.modules["sentencepiece"] = sp


def main() -> int:
    _install_fakes()
    os.environ["TRANSLATOR_MODEL_PATH"] = "/models/nllb-test"
    os.environ["TRANSLATOR_DEVICE"] = "cuda"
    os.environ["TRANSLATOR_DEVICE_INDEX"] = "0"

    import server

    async def run():
        # Drive the lifespan: should preload the backend.
        async with server.lifespan(server.app):
            assert server._backend is not None and server._backend.is_ready(), \
                "lifespan did not preload backend"
            assert server._executor is not None, "executor not created"

            # tokenizer loaded the bpe model from model_path
            assert server._backend._tokenizer.load_path == \
                "/models/nllb-test/sentencepiece.bpe.model"
            # device_index coerced to int, device/compute_type passed through
            ctor = _FakeTranslator.last["kwargs"]
            assert ctor["device"] == "cuda"
            assert ctor["device_index"] == 0 and isinstance(ctor["device_index"], int)

            # health
            health = await server.health_check()
            assert health.status == "ok"
            assert health.model == "nllb-200-distilled-600M"
            assert health.device == "cuda"

            # /translate
            req = server.TranslateRequest(text="你好")  # 你好, default langs
            resp = await server.translate(req)
            assert resp.translation == "hello world", resp.translation
            assert resp.src_lang == "zho_Hans"
            assert resp.tgt_lang == "eng_Latn"
            assert resp.model == "nllb-200-distilled-600M"

            # token equivalence to old _translate_sync
            call = server._backend._translator.batch_calls[0]
            sent = call["batch"][0]
            assert server._backend._tokenizer.encode_calls == ["你好"], \
                "must use EncodeAsPieces on raw text"
            assert sent == ["▁你好", "piece2", "</s>", "zho_Hans"], sent
            assert sent[-2:] == ["</s>", "zho_Hans"], "src lang appended, not prefixed"
            assert call["kwargs"]["target_prefix"] == [["eng_Latn"]]
            # position-0 tgt token stripped before decode
            decoded = server._backend._tokenizer.decode_calls[0]
            assert "eng_Latn" not in decoded
            assert decoded == ["hello", "world"]

            # explicit lang override flows through
            req2 = server.TranslateRequest(
                text="x", src_lang="fra_Latn", tgt_lang="deu_Latn"
            )
            resp2 = await server.translate(req2)
            assert (resp2.src_lang, resp2.tgt_lang) == ("fra_Latn", "deu_Latn")
            sent2 = server._backend._translator.batch_calls[1]["batch"][0]
            assert sent2[-2:] == ["</s>", "fra_Latn"]

            # empty text → 400
            from fastapi import HTTPException
            try:
                await server.translate(server.TranslateRequest(text="   "))
                raise AssertionError("empty text should 400")
            except HTTPException as e:
                assert e.status_code == 400

        # after lifespan exit, backend unloaded
        assert not server._backend.is_ready(), "backend should be unloaded on shutdown"
        print("SMOKE PASS: lifespan + /translate + /health + token equivalence")

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
