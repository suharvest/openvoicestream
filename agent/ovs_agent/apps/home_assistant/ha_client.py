"""Home Assistant REST client + spoken-name → entity_id resolver.

WHY THE RESOLVER IS THE HARD PART
---------------------------------
Home Assistant derives ``entity_id`` from the entity name, and for non-Latin
names that derivation is a TRANSLITERATION. A device a user calls 客厅灯 lands
as ``light.ke_ting_deng``; 客厅窗帘 becomes ``cover.ke_ting_chuang_lian``.
(Verified against a live HA instance — these are real ids from a test setup.)

Two consequences that shape this module:

1. **Never parse ``entity_id`` to figure out what a device is.** The only
   reliable human-facing label is the ``friendly_name`` attribute. All matching
   here is done against that.
2. **The LLM must be handed the names, not the ids.** Tools therefore take a
   spoken ``device`` string and resolve it here; the tool schema stays stable
   while the device set is discovered at runtime.

Resolution is deliberately forgiving (users say 把客厅的灯 rather than 客厅灯)
but never guesses between two plausible devices — an ambiguous query returns
candidates so the caller can ask, which for an LLM tool loop means it can
re-call with a better name instead of switching off the wrong room.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Domains worth exposing to a voice assistant. Everything else in a typical HA
# install (sensor, sun, zone, person, weather, update, …) is either read-only
# noise or not something a user asks to "turn on", and including it would bloat
# the device list the LLM has to reason over.
#
# `input_boolean` is deliberately NOT here even though it is controllable.
# Helpers are usually PLUMBING: template lights and switches are commonly backed
# by an input_boolean, so including the domain lists every such device twice —
# once as `light.x` and once as the helper behind it — and the LLM then has two
# equally-plausible targets for "turn on the living room light". Setups that
# really do drive helpers directly (a "vacation mode" toggle, say) can add the
# domain back via ``extra_domains``.
CONTROLLABLE_DOMAINS = (
    "light", "switch", "fan", "cover", "climate",
    "media_player", "lock", "scene", "script",
)

# States that mean "there is nothing here to control right now". Offering these
# to the LLM produces confident-sounding actions that silently do nothing.
_DEAD_STATES = ("unavailable", "unknown")

# Filler that shows up in spoken Chinese device references but never in a
# friendly_name. Stripped before matching so 把客厅的灯 matches 客厅灯.
_ZH_FILLER = ("的", "把", "请", "帮我", "一下", "那个", "这个")
_LATIN_FILLER = ("the", "a", "an", "please")

_CACHE_TTL_S = 30.0


def _normalize(s: str) -> str:
    """Lowercase, strip whitespace/punctuation and drop filler words."""
    s = (s or "").strip().lower()
    s = re.sub(r"[\s,，。.、!！?？:：;；\"'“”‘’()（）\[\]【】]+", "", s)
    for f in _ZH_FILLER:
        s = s.replace(f, "")
    for f in _LATIN_FILLER:
        s = re.sub(rf"\b{f}\b", "", s)
    return s


@dataclass
class Device:
    """One controllable HA entity, as the voice layer sees it."""

    entity_id: str
    name: str          # friendly_name — what the user says
    state: str

    @property
    def domain(self) -> str:
        return self.entity_id.split(".", 1)[0]


class ResolveError(Exception):
    """Raised when a spoken name matches nothing or is ambiguous.

    Carries ``candidates`` so the tool layer can return them to the LLM: a
    wrong-room action is much worse than a follow-up question.
    """

    def __init__(self, message: str, candidates: Optional[list[str]] = None) -> None:
        super().__init__(message)
        self.candidates = candidates or []


class HAClient:
    """Minimal HA REST client. One instance per app; not thread-safe."""

    def __init__(self, base_url: str, token: str, timeout_s: float = 8.0,
                 extra_domains: tuple[str, ...] = (),
                 exclude_entity_ids: tuple[str, ...] = ()) -> None:
        """``extra_domains`` widens the exposed set (e.g. ``("input_boolean",)``
        for a setup that drives helpers directly); ``exclude_entity_ids`` hides
        specific entities that would otherwise be offered to the LLM."""
        self.base_url = base_url.rstrip("/")
        self.domains = CONTROLLABLE_DOMAINS + tuple(extra_domains)
        self.exclude_entity_ids = frozenset(exclude_entity_ids)
        # trust_env=False is REQUIRED, not stylistic: with it unset httpx picks
        # up HTTP(S)_PROXY from the environment and a LAN address like
        # http://192.168.1.10:8123 gets routed to the proxy, which cannot reach
        # it. This has bitten this codebase twice before.
        #
        # timeout defaults to 8 s because the server-side tool budget is ~15 s
        # (the per-tool timeout_s is not advertised over the wire, so the
        # server's default is what applies) — a tool that blocks longer than
        # that is reported as a timeout to the LLM no matter what we do here.
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            timeout=timeout_s,
            trust_env=False,
        )
        self._cache: list[Device] = []
        self._cache_at = 0.0

    def close(self) -> None:
        self._http.close()

    # ── raw API ──────────────────────────────────────────────────────────
    def ping(self) -> bool:
        """True when the base URL + token are both good."""
        try:
            r = self._http.get("/api/")
            return r.status_code == 200
        except httpx.HTTPError as e:
            logger.warning("HA ping failed: %r", e)
            return False

    def raw_states(self) -> list[dict[str, Any]]:
        r = self._http.get("/api/states")
        r.raise_for_status()
        return r.json()

    def raw_state(self, entity_id: str) -> dict[str, Any]:
        r = self._http.get(f"/api/states/{entity_id}")
        r.raise_for_status()
        return r.json()

    def call_service(self, domain: str, service: str,
                     data: Optional[dict[str, Any]] = None) -> Any:
        r = self._http.post(f"/api/services/{domain}/{service}", json=data or {})
        r.raise_for_status()
        return r.json()

    # ── device inventory ─────────────────────────────────────────────────
    def devices(self, refresh: bool = False) -> list[Device]:
        """Controllable entities, cached briefly.

        The cache exists because a single spoken turn can hit several tools and
        each would otherwise re-fetch every entity in the house.
        """
        if not refresh and self._cache and (time.monotonic() - self._cache_at) < _CACHE_TTL_S:
            return self._cache
        out: list[Device] = []
        for e in self.raw_states():
            eid = e.get("entity_id", "")
            if eid.split(".", 1)[0] not in self.domains:
                continue
            if eid in self.exclude_entity_ids:
                continue
            state = e.get("state", "")
            if state in _DEAD_STATES:
                continue
            attrs = e.get("attributes") or {}
            # HA marks config/diagnostic entities with entity_category; those are
            # for the settings UI, not for a user to ask about out loud.
            if attrs.get("entity_category"):
                continue
            name = attrs.get("friendly_name") or eid
            out.append(Device(entity_id=eid, name=name, state=state))
        self._cache = out
        self._cache_at = time.monotonic()
        return out

    # ── resolution ───────────────────────────────────────────────────────
    def resolve(self, spoken: str, domains: Optional[tuple[str, ...]] = None) -> Device:
        """Map a spoken device reference to exactly one Device.

        Tiers, most to least confident. Each tier only wins if it produces a
        UNIQUE hit; otherwise we fall through, and an ambiguous result raises
        with candidates rather than picking one.

        ``domains`` narrows the search — e.g. set_brightness only cares about
        lights, so "客厅" can resolve even if a 客厅电视 switch also exists.
        """
        pool = self.devices()
        if domains:
            pool = [d for d in pool if d.domain in domains]
        if not pool:
            raise ResolveError("没有可控设备" if not domains else f"没有 {domains} 类型的可控设备")

        q = _normalize(spoken)
        if not q:
            raise ResolveError("设备名为空", [d.name for d in pool])

        # 1. exact entity_id (lets an LLM pass back something from list_devices)
        for d in pool:
            if spoken.strip() == d.entity_id:
                return d

        # 2. exact normalized friendly_name
        hits = [d for d in pool if _normalize(d.name) == q]
        if len(hits) == 1:
            return hits[0]

        # 3. containment either way — 客厅灯光 ⊃ 客厅灯, and 客厅 ⊂ 客厅灯
        if not hits:
            hits = [d for d in pool
                    if q in _normalize(d.name) or _normalize(d.name) in q]
        if len(hits) == 1:
            return hits[0]

        # 4. character-overlap score, for near-misses like 客厅的大灯
        if not hits:
            scored = []
            for d in pool:
                n = _normalize(d.name)
                if not n:
                    continue
                overlap = len(set(q) & set(n)) / max(len(set(n)), 1)
                if overlap >= 0.6:
                    scored.append((overlap, d))
            scored.sort(key=lambda t: t[0], reverse=True)
            # Only accept a clear winner — a tie means we genuinely don't know.
            if len(scored) == 1 or (len(scored) > 1 and scored[0][0] > scored[1][0] + 0.15):
                return scored[0][1]
            hits = [d for _, d in scored]

        if not hits:
            raise ResolveError(
                f"找不到设备 {spoken!r}", [d.name for d in pool])
        raise ResolveError(
            f"{spoken!r} 匹配到多个设备，需要说得更具体",
            [d.name for d in hits])


__all__ = ["HAClient", "Device", "ResolveError", "CONTROLLABLE_DOMAINS", "_normalize"]
