"""Compile human-readable Chinese/English phrases for sherpa-onnx KWS."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CompiledKeywords:
    phrases: tuple[str, ...]
    keywords: str


class PhraseCompiler:
    """Safe wrapper around ``sherpa-onnx-cli text2token``.

    The CLI, lexicon and token table are deployment assets.  No phrase is
    interpolated into a shell command: phrases travel only through a private
    input file and subprocess is invoked with an argv list.
    """

    def __init__(
        self,
        *,
        tokens: str,
        lexicon: str,
        cli: str = "sherpa-onnx-cli",
        tokens_type: str = "phone+ppinyin",
        timeout_s: float = 10.0,
    ) -> None:
        self.tokens = str(tokens)
        self.lexicon = str(lexicon)
        self.cli = str(cli)
        self.tokens_type = str(tokens_type)
        self.timeout_s = float(timeout_s)

    @staticmethod
    def normalise(phrases: Sequence[str]) -> tuple[str, ...]:
        clean: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            value = " ".join(unicodedata.normalize("NFKC", str(phrase)).strip().split())
            if not value:
                continue
            if len(value) > 64:
                raise ValueError("each wake phrase must be at most 64 characters")
            if value not in seen:
                clean.append(value)
                seen.add(value)
            if len(clean) > 8:
                raise ValueError("at most 8 wake phrases are supported")
        if not clean:
            raise ValueError("at least one non-empty wake phrase is required")
        return tuple(clean)

    def compile(self, phrases: Sequence[str]) -> CompiledKeywords:
        normalised = self.normalise(phrases)
        with tempfile.TemporaryDirectory(prefix="ovs-kws-") as tmp:
            os.chmod(tmp, 0o700)
            raw_path = Path(tmp) / "keywords.raw.txt"
            out_path = Path(tmp) / "keywords.txt"
            # Preserve a display label after @.  text2token compiles only the
            # phrase side and sherpa returns the label on a match.
            raw_path.write_text(
                # sherpa requires the @ display label to be one token; the
                # phrase being compiled may still contain normal word spaces.
                "".join(f"{phrase} @{phrase.replace(' ', '_')}\n" for phrase in normalised),
                encoding="utf-8",
            )
            os.chmod(raw_path, 0o600)
            out_path.touch(mode=0o600)
            argv = [
                self.cli,
                "text2token",
                "--tokens",
                self.tokens,
                "--tokens-type",
                self.tokens_type,
                "--lexicon",
                self.lexicon,
                str(raw_path),
                str(out_path),
            ]
            try:
                proc = subprocess.run(
                    argv,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"wake phrase compilation timed out after {self.timeout_s:.1f}s"
                ) from exc
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "unknown error").strip()
                raise ValueError(f"wake phrase compilation failed: {detail}")
            keywords = out_path.read_text(encoding="utf-8").strip()
            if not keywords:
                raise ValueError("wake phrase compiler produced no supported tokens")
            return CompiledKeywords(normalised, keywords + "\n")
