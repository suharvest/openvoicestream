"""Shared env-var coercion helpers (truthy / int / float).

Single source for the boolean / int / float environment parsing that several
opt-in capability modules (``diarization``, ``speaker_embedding``,
``punctuation``) previously each copied verbatim. Behavior is byte-identical to
those copies:

  * ``truthy`` matches the ``1/true/yes/on`` set (case- and space-insensitive),
  * ``env_int`` / ``env_float`` return ``default`` on a missing or unparseable
    value (never raise).

``env`` defaults to ``os.environ`` but is injectable for tests.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

_TRUTHY = ("1", "true", "yes", "on")


def truthy(value) -> bool:
    """True when ``value`` is one of ``1/true/yes/on`` (case/space-insensitive)."""
    return str(value).strip().lower() in _TRUTHY


def env_int(key: str, default, env: Optional[Mapping[str, str]] = None):
    """``int`` from ``env[key]``, or ``default`` if missing/unparseable."""
    src = env if env is not None else os.environ
    try:
        return int(src.get(key, default))
    except (TypeError, ValueError):
        return default


def env_float(key: str, default, env: Optional[Mapping[str, str]] = None):
    """``float`` from ``env[key]``, or ``default`` if missing/unparseable."""
    src = env if env is not None else os.environ
    try:
        return float(src.get(key, default))
    except (TypeError, ValueError):
        return default
