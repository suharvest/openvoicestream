"""TranslatorApp: Pure translation without LLM (sentence-level interpreter mode)."""
from __future__ import annotations

from ovs_agent import BaseApp


class TranslatorApp(BaseApp):
    """Voice translation app.

    ASR → translate → TTS, no LLM.
    Waits for ASRFinal (VAD silence), translates the sentence, streams to TTS.
    """

    async def on_user_utterance(
        self, text: str, detected_language: str | None = None
    ) -> None:
        """Translate user's utterance and send to TTS."""
        translated = await self.translator.translate(
            text,
            self.config.translator_src_lang,
            self.config.translator_tgt_lang,
        )
        await self.slv.send_text(translated)
        await self.slv.flush_tts()
