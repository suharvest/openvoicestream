"""Cloud Realtime WebSocket relay for the provider-neutral V2 endpoint."""
from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import quote

import numpy as np
from websockets.asyncio.client import connect

from server.core.realtime_provider import create_provider_adapter


def _resample_pcm16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    if source_rate == target_rate or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    output_count = int(round(len(samples) * target_rate / source_rate))
    if output_count <= 0:
        return b""
    output = np.interp(
        np.linspace(0, len(samples) - 1, output_count),
        np.arange(len(samples)),
        samples,
    )
    return np.clip(output, -32768, 32767).astype(np.int16).tobytes()


def provider_settings(name: str) -> tuple[str, list[tuple[str, str]], str]:
    if name == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for realtime provider openai")
        model = os.environ.get("OVS_REALTIME_OPENAI_MODEL", "gpt-realtime-2.1")
        url = os.environ.get(
            "OVS_REALTIME_OPENAI_URL",
            f"wss://api.openai.com/v1/realtime?model={quote(model)}",
        )
        return url, [("Authorization", f"Bearer {key}")], model
    if name == "qwen":
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("DASHSCOPE_API_KEY is required for realtime provider qwen")
        model = os.environ.get(
            "OVS_REALTIME_QWEN_MODEL", "qwen-audio-3.0-realtime-flash"
        )
        url = os.environ.get("OVS_REALTIME_QWEN_URL")
        if not url:
            raise ValueError(
                "OVS_REALTIME_QWEN_URL is required (it contains the Model Studio workspace ID)"
            )
        separator = "&" if "?" in url else "?"
        if "model=" not in url:
            url = f"{url}{separator}model={quote(model)}"
        return url, [("Authorization", f"Bearer {key}")], model
    raise ValueError(f"unsupported realtime provider: {name!r}")


async def relay_cloud_realtime(
    downstream,
    *,
    provider_name: str,
    canonical_session: dict,
    downstream_adapter,
    input_rate: int,
    create_response: bool,
    interrupt_response: bool,
) -> None:
    provider = create_provider_adapter(provider_name)
    downstream_adapter.output_sample_rate = provider.output_rate
    downstream_adapter.capabilities_override = provider.capabilities()

    try:
        url, headers, model = provider_settings(provider_name)
        downstream_adapter.provider = provider_name
        downstream_adapter.model = model
        async with connect(
            url, additional_headers=headers, max_size=None, open_timeout=10
        ) as upstream:
            created = json.loads(await asyncio.wait_for(upstream.recv(), timeout=10))
            if created.get("type") != "session.created":
                raise RuntimeError(
                    f"{provider_name}: expected session.created, got {created.get('type')!r}"
                )
            await upstream.send(json.dumps(provider.session_update(canonical_session)))
            updated = json.loads(await asyncio.wait_for(upstream.recv(), timeout=10))
            if updated.get("type") == "error":
                raise RuntimeError(str(updated.get("error") or updated))
            if updated.get("type") != "session.updated":
                raise RuntimeError(
                    f"{provider_name}: expected session.updated, got {updated.get('type')!r}"
                )
            await downstream.send_json(
                downstream_adapter.session_updated(
                    canonical_session,
                    create_response=create_response,
                    interrupt_response=interrupt_response,
                )
            )

            async def client_to_provider() -> None:
                while True:
                    message = await downstream.receive()
                    if message.get("type") == "websocket.disconnect":
                        return
                    pcm = message.get("bytes")
                    if pcm is not None:
                        pcm = _resample_pcm16(pcm, input_rate, provider.input_rate)
                        await upstream.send(json.dumps(provider.audio_append(pcm)))
                        continue
                    text = message.get("text")
                    if not text:
                        continue
                    event = json.loads(text)
                    if event.get("type") == "session.update":
                        await upstream.send(json.dumps(
                            provider.session_update(event.get("session") or {})
                        ))
                        continue
                    mapped = provider.client_event(event)
                    if mapped is None:
                        await downstream.send_json(downstream_adapter.translate({
                            "type": "error",
                            "code": "unsupported_provider_event",
                            "error": (
                                f"{event.get('type')} is not supported by "
                                f"the {provider_name} adapter"
                            ),
                            "param": "type",
                        })[0])
                        continue
                    await upstream.send(json.dumps(mapped))
                    if (
                        mapped.get("type") == "conversation.item.create"
                        and (mapped.get("item") or {}).get("type")
                        == "function_call_output"
                    ):
                        await upstream.send(json.dumps({"type": "response.create"}))

            async def provider_to_client() -> None:
                async for raw in upstream:
                    if not isinstance(raw, str):
                        continue
                    event = json.loads(raw)
                    if event.get("type") in {"session.created", "session.updated"}:
                        continue
                    output = provider.server_event(event)
                    for audio in output.audio:
                        await downstream.send_bytes(audio)
                    for normalized in output.events:
                        await downstream.send_json(normalized)

            tasks = [
                asyncio.create_task(client_to_provider(), name="realtime-cloud-up"),
                asyncio.create_task(provider_to_client(), name="realtime-cloud-down"),
            ]
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
    except Exception as exc:
        await downstream.send_json(downstream_adapter.translate({
            "type": "error",
            "code": "provider_connection_error",
            "error": str(exc),
        })[0])
        try:
            await downstream.close(code=1011)
        except Exception:
            pass
