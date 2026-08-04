"""Shared helpers for N=2 TTS stability gates.

Used by `stability_kokoro_n2.py` and `stability_matcha_n2.py`. The gate
semantics mirror the qwen3 reference gate:

  - capture N=1 baseline TTFA after warmup
  - capture pre-stress full-audio MD5 for a fixed prompt
  - run >= 30 sustained N=2 bursts
  - capture post-stress full-audio MD5
  - scan service / container logs for CUDA error signatures
  - pass only when:
      * N=2 combined TTFA p50 / N=1 TTFA p50 <= --fail-on-ratio (default 1.5)
      * pre/post MD5 byte-identical (excluding the 4-byte SR header if applicable)
      * every burst returned a complete PCM stream
      * 0 CUDA errors

The helper itself does not import TensorRT, CUDA, or backend modules — it
talks to the running service over plain HTTP. It uses only `requests`
plus stdlib so no new dependency is introduced. A `--mock` mode is
provided so the CLI surface can be exercised on machines without the
service running (Mac dev loop).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import requests  # already in pyproject (see bench/perf/client.py)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = REPO_ROOT / "bench" / "perf" / "corpus" / "tts_prompts.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "bench" / "perf" / "results"

# 4-byte sample-rate header convention (see load_2client_tts.py:30-39)
SR_HEADER_BYTES = 4

CUDA_PATTERNS = [
    r"CUDA runtime error",
    r"illegal memory access",
    r"cudaMemcpy",
    r"cudaMemsetAsync",
    r"cudaStreamSynchronize failed",
    r"execute_async_v3 returned False",
    r"CUDNN_STATUS",
    r"TensorRT.*Error Code",
]
CUDA_REGEX = re.compile("|".join(f"({p})" for p in CUDA_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompts(
    path: Path = DEFAULT_PROMPTS,
    lang: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts", [])
    if lang:
        prompts = [p for p in prompts if p.get("lang") == lang]
    if category:
        prompts = [p for p in prompts if p.get("category") == category]
    if not prompts:
        raise SystemExit(
            f"No prompts matched lang={lang} category={category} in {path}"
        )
    return prompts


def pick_prompt(prompts: list[dict[str, Any]], prompt_id: str | None) -> dict[str, Any]:
    if prompt_id:
        for p in prompts:
            if p.get("id") == prompt_id:
                return p
        raise SystemExit(f"prompt_id {prompt_id!r} not found in corpus")
    return prompts[0]


# ---------------------------------------------------------------------------
# HTTP TTS calls
# ---------------------------------------------------------------------------


@dataclass
class TtsCallResult:
    text: str
    prompt_id: str | None
    started_at: float
    ttfa_ms: float | None
    total_ms: float
    status: int
    bytes_total: int
    pcm_present: bool
    body: bytes | None
    error: str | None = None


def post_tts_stream(
    base_url: str,
    text: str,
    prompt_id: str | None,
    timeout: float,
    *,
    capture_body: bool,
    session: requests.Session | None = None,
) -> TtsCallResult:
    """POST /tts/stream and record TTFA + optional full body bytes.

    TTFA = first PCM byte past the 4-byte SR header. Matches the
    definition in load_2client_tts.py. The response is always drained to
    completion so a TTFA-only sample cannot leave uncancelled synthesis work
    occupying a worker slot and pollute the following N=1/N=2 sample.
    `capture_body=True` additionally retains the post-header PCM for MD5
    comparison.
    """
    url = base_url.rstrip("/") + "/tts/stream"
    s = session or requests
    t0 = time.perf_counter()
    started_at = time.time()
    try:
        r = s.post(url, json={"text": text}, stream=True, timeout=timeout)
    except Exception as e:
        return TtsCallResult(
            text=text, prompt_id=prompt_id, started_at=started_at,
            ttfa_ms=None, total_ms=(time.perf_counter() - t0) * 1000,
            status=-1, bytes_total=0, pcm_present=False, body=None,
            error=f"connect: {e}",
        )
    if r.status_code != 200:
        body = r.content[:512]
        r.close()
        return TtsCallResult(
            text=text, prompt_id=prompt_id, started_at=started_at,
            ttfa_ms=None, total_ms=(time.perf_counter() - t0) * 1000,
            status=r.status_code, bytes_total=len(body), pcm_present=False,
            body=None, error=f"http {r.status_code}: {body!r}",
        )
    first_audio_t: float | None = None
    total_bytes = 0
    chunks: list[bytes] = [] if capture_body else []  # only used if capture
    try:
        # Read the protocol prefix exactly. iter_content(chunk_size=4096)
        # measures the first aggregated 4 KiB, not the first PCM byte.
        header = r.raw.read(SR_HEADER_BYTES)
        total_bytes += len(header)
        first_pcm = r.raw.read(1)
        if first_pcm:
            first_audio_t = time.perf_counter()
            total_bytes += 1
            if capture_body:
                chunks.append(first_pcm)
        while True:
            chunk = r.raw.read(64 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if capture_body:
                chunks.append(chunk)
    finally:
        r.close()
    total_ms = (time.perf_counter() - t0) * 1000
    ttfa_ms = (first_audio_t - t0) * 1000 if first_audio_t else None
    body_bytes = b"".join(chunks) if capture_body else None
    pcm_present = total_bytes > SR_HEADER_BYTES
    return TtsCallResult(
        text=text, prompt_id=prompt_id, started_at=started_at,
        ttfa_ms=ttfa_ms, total_ms=total_ms,
        status=r.status_code, bytes_total=total_bytes,
        pcm_present=pcm_present, body=body_bytes,
    )


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def p50(values: list[float]) -> float:
    return percentile(values, 0.5)


def p95(values: list[float]) -> float:
    return percentile(values, 0.95)


# ---------------------------------------------------------------------------
# Log scanning
# ---------------------------------------------------------------------------


def scan_log_text(text: str) -> tuple[int, list[str]]:
    hits = []
    for line in text.splitlines():
        if CUDA_REGEX.search(line):
            hits.append(line.rstrip())
    return len(hits), hits[:50]  # cap stored excerpt


def scan_log_file(path: Path, since_iso: str | None) -> tuple[int, list[str]]:
    if not path.exists():
        return 0, [f"[log file not found: {path}]"]
    text = path.read_text(encoding="utf-8", errors="replace")
    if since_iso:
        # Naive line-prefix filter: keep lines whose first 19 chars >= since_iso[:19]
        cutoff = since_iso[:19]
        filtered = []
        for line in text.splitlines():
            prefix = line[:19]
            if len(prefix) == 19 and prefix >= cutoff:
                filtered.append(line)
        text = "\n".join(filtered) if filtered else text
    return scan_log_text(text)


def scan_docker_container(name: str, since_iso: str | None) -> tuple[int, list[str]]:
    cmd = ["docker", "logs"]
    if since_iso:
        cmd += ["--since", since_iso]
    cmd.append(name)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return 0, ["[docker CLI not available]"]
    except subprocess.TimeoutExpired:
        return 0, ["[docker logs timed out]"]
    if proc.returncode != 0:
        return 0, [f"[docker logs rc={proc.returncode}: {proc.stderr.strip()[:200]}]"]
    return scan_log_text((proc.stdout or "") + "\n" + (proc.stderr or ""))


# ---------------------------------------------------------------------------
# Backend / runtime introspection
# ---------------------------------------------------------------------------


def fetch_backend_info(base_url: str, timeout: float = 10.0) -> dict[str, Any]:
    out: dict[str, Any] = {"health": None, "metadata": None}
    try:
        r = requests.get(base_url.rstrip("/") + "/health", timeout=timeout)
        if r.status_code == 200:
            out["health"] = r.json()
    except Exception as e:
        out["health_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# Gate runner
# ---------------------------------------------------------------------------


@dataclass
class GateConfig:
    backend_label: str          # "kokoro_trt" or "matcha_trt"
    expected_backend_substr: tuple[str, ...]  # match against /health tts_backend
    fallback_env_var: str       # e.g. OVS_TTS_STREAM_MAX_WORKERS_KOKORO
    prompt_lang: str | None     # "en", "zh", or None for both
    base_url: str
    bursts: int
    warmup: int
    timeout: float
    fail_on_ratio: float
    output_dir: Path
    prompt_id: str | None
    category: str | None
    scan_log: Path | None
    container: str | None


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _ts_for_path() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_gate(cfg: GateConfig) -> dict[str, Any]:
    started_iso = _now_iso()
    prompts = load_prompts(lang=cfg.prompt_lang, category=cfg.category)
    fixed_prompt = pick_prompt(prompts, cfg.prompt_id)

    backend_info = fetch_backend_info(cfg.base_url)
    detected_backend = None
    if backend_info.get("health"):
        detected_backend = backend_info["health"].get("tts_backend")

    print(f"[{cfg.backend_label}] base_url={cfg.base_url}", flush=True)
    print(f"[{cfg.backend_label}] tts_backend={detected_backend}", flush=True)
    # Codex Week 3 BLOCKER 2: backend mismatch must HARD-FAIL the gate
    # (exit 2) — silent warning let kokoro-labelled runs proceed against a
    # qwen3 server and "pass" without exercising the kokoro code path.
    if detected_backend and cfg.expected_backend_substr:
        if not any(sub in (detected_backend or "") for sub in cfg.expected_backend_substr):
            reason = (
                f"backend_mismatch: expected one of {list(cfg.expected_backend_substr)}, "
                f"got {detected_backend!r}"
            )
            print(f"ERROR: {reason}", flush=True)
            return _build_failure_report(
                cfg, started_iso, detected_backend,
                n1_ttfas=[], n1_errors=0, reason=reason,
            )

    session = requests.Session()

    # Warmup
    for i in range(cfg.warmup):
        r = post_tts_stream(
            cfg.base_url, fixed_prompt["text"], fixed_prompt.get("id"),
            cfg.timeout, capture_body=False, session=session,
        )
        print(f"[{cfg.backend_label}] warmup {i+1}/{cfg.warmup} status={r.status} "
              f"ttfa={r.ttfa_ms}", flush=True)

    # N=1 baseline
    n1_ttfas: list[float] = []
    n1_errors = 0
    n1_runs = max(10, cfg.warmup * 3)
    for i in range(n1_runs):
        prompt = prompts[i % len(prompts)]
        r = post_tts_stream(
            cfg.base_url, prompt["text"], prompt.get("id"),
            cfg.timeout, capture_body=False, session=session,
        )
        if r.error or r.ttfa_ms is None or not r.pcm_present:
            n1_errors += 1
            print(f"[{cfg.backend_label}] N=1 #{i} ERROR: {r.error}", flush=True)
            continue
        n1_ttfas.append(r.ttfa_ms)
    n1_p50 = p50(n1_ttfas) if n1_ttfas else float("nan")
    n1_p95 = p95(n1_ttfas) if n1_ttfas else float("nan")
    print(f"[{cfg.backend_label}] N=1: p50={n1_p50:.1f}ms p95={n1_p95:.1f}ms "
          f"errors={n1_errors}", flush=True)

    # Pre-stress audio capture
    pre = post_tts_stream(
        cfg.base_url, fixed_prompt["text"], fixed_prompt.get("id"),
        cfg.timeout, capture_body=True, session=session,
    )
    if pre.error or not pre.body:
        return _build_failure_report(
            cfg, started_iso, detected_backend, n1_ttfas, n1_errors,
            reason=f"pre-stress capture failed: {pre.error}",
        )
    pre_md5 = hashlib.md5(pre.body).hexdigest()
    print(f"[{cfg.backend_label}] pre-stress MD5={pre_md5} "
          f"bytes={len(pre.body)}", flush=True)

    # N=2 sustained bursts
    burst_pairs: list[dict[str, Any]] = []
    http_errors = 0
    bursts_completed = 0
    client0_ttfas: list[float] = []
    client1_ttfas: list[float] = []
    combined_ttfas: list[float] = []

    def _one(prompt: dict[str, Any]) -> TtsCallResult:
        return post_tts_stream(
            cfg.base_url, prompt["text"], prompt.get("id"),
            cfg.timeout, capture_body=False,
        )

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="stability-n2") as ex:
        for b in range(cfg.bursts):
            p0 = prompts[(b * 2) % len(prompts)]
            p1 = prompts[(b * 2 + 1) % len(prompts)]
            fut0 = ex.submit(_one, p0)
            fut1 = ex.submit(_one, p1)
            r0 = fut0.result()
            r1 = fut1.result()
            pair_ok = True
            for r in (r0, r1):
                if r.error or r.ttfa_ms is None or not r.pcm_present:
                    http_errors += 1
                    pair_ok = False
            if pair_ok:
                bursts_completed += 1
                client0_ttfas.append(r0.ttfa_ms)  # type: ignore[arg-type]
                client1_ttfas.append(r1.ttfa_ms)  # type: ignore[arg-type]
                combined_ttfas.extend([r0.ttfa_ms, r1.ttfa_ms])  # type: ignore[list-item]
            burst_pairs.append({
                "burst": b,
                "client0": {"prompt_id": p0.get("id"), "ttfa_ms": r0.ttfa_ms,
                            "status": r0.status, "pcm_present": r0.pcm_present,
                            "error": r0.error},
                "client1": {"prompt_id": p1.get("id"), "ttfa_ms": r1.ttfa_ms,
                            "status": r1.status, "pcm_present": r1.pcm_present,
                            "error": r1.error},
            })
            if (b + 1) % 10 == 0:
                print(f"[{cfg.backend_label}] burst {b+1}/{cfg.bursts} "
                      f"ok={bursts_completed} err={http_errors}", flush=True)

    n2_c0_p50 = p50(client0_ttfas)
    n2_c1_p50 = p50(client1_ttfas)
    n2_comb_p50 = p50(combined_ttfas)
    n2_comb_p95 = p95(combined_ttfas)
    ratio = (n2_comb_p50 / n1_p50) if n1_p50 and n1_p50 == n1_p50 else float("nan")
    print(f"[{cfg.backend_label}] N=2: comb_p50={n2_comb_p50:.1f}ms "
          f"comb_p95={n2_comb_p95:.1f}ms ratio={ratio:.2f}", flush=True)

    # Post-stress audio capture
    post = post_tts_stream(
        cfg.base_url, fixed_prompt["text"], fixed_prompt.get("id"),
        cfg.timeout, capture_body=True, session=session,
    )
    if post.error or not post.body:
        return _build_failure_report(
            cfg, started_iso, detected_backend, n1_ttfas, n1_errors,
            reason=f"post-stress capture failed: {post.error}",
            extra_bursts=burst_pairs,
        )
    post_md5 = hashlib.md5(post.body).hexdigest()
    md5_stable = (pre_md5 == post_md5) and (len(pre.body) == len(post.body))
    print(f"[{cfg.backend_label}] post-stress MD5={post_md5} "
          f"bytes={len(post.body)} stable={md5_stable}", flush=True)

    # Log scan
    if cfg.container:
        cuda_count, cuda_lines = scan_docker_container(cfg.container, started_iso)
        log_source = f"docker:{cfg.container}"
    elif cfg.scan_log:
        cuda_count, cuda_lines = scan_log_file(cfg.scan_log, started_iso)
        log_source = f"file:{cfg.scan_log}"
    else:
        cuda_count, cuda_lines = 0, ["[no log source configured; scan skipped]"]
        log_source = "none"
    print(f"[{cfg.backend_label}] cuda_error_count={cuda_count} "
          f"log_source={log_source}", flush=True)

    failure_reasons: list[str] = []
    if bursts_completed < cfg.bursts:
        failure_reasons.append(
            f"only {bursts_completed}/{cfg.bursts} bursts completed cleanly"
        )
    if http_errors > 0:
        failure_reasons.append(f"http_error_count={http_errors}")
    if cuda_count > 0:
        failure_reasons.append(f"cuda_error_count={cuda_count}")
    if not md5_stable:
        failure_reasons.append("pre/post MD5 differ")
    if ratio != ratio:
        failure_reasons.append("N=1 baseline empty; cannot compute ratio")
    elif ratio > cfg.fail_on_ratio:
        failure_reasons.append(
            f"TTFA ratio {ratio:.2f} > {cfg.fail_on_ratio}"
        )
    if log_source == "none":
        failure_reasons.append(
            "no log source configured (--container or --scan-log)"
        )

    passed = not failure_reasons

    report = {
        "backend": cfg.backend_label,
        "detected_backend": detected_backend,
        "base_url": cfg.base_url,
        "started_at": started_iso,
        "ended_at": _now_iso(),
        "bursts_requested": cfg.bursts,
        "bursts_completed": bursts_completed,
        "fallback_env_var": cfg.fallback_env_var,
        "n1": {
            "ttfa_ms": n1_ttfas,
            "p50_ms": n1_p50,
            "p95_ms": n1_p95,
            "errors": n1_errors,
        },
        "n2": {
            "pairs": burst_pairs,
            "client0_p50_ms": n2_c0_p50,
            "client1_p50_ms": n2_c1_p50,
            "combined_p50_ms": n2_comb_p50,
            "combined_p95_ms": n2_comb_p95,
            "ratio_vs_n1_p50": ratio,
        },
        "md5": {
            "prompt_id": fixed_prompt.get("id"),
            "pre": pre_md5,
            "post": post_md5,
            "stable": md5_stable,
            "bytes_pre": len(pre.body),
            "bytes_post": len(post.body),
        },
        "http_error_count": http_errors,
        "cuda_error_count": cuda_count,
        "cuda_log_excerpt": cuda_lines,
        "log_source": log_source,
        "pass": passed,
        "failure_reasons": failure_reasons,
        "repro_command": _repro_command(cfg),
    }
    return report


def _repro_command(cfg: GateConfig) -> str:
    parts = [
        "python", f"bench/perf/stability_{cfg.backend_label}_n2.py",
        f"--base-url {cfg.base_url}",
        f"--bursts {cfg.bursts}",
        f"--warmup {cfg.warmup}",
        f"--timeout {int(cfg.timeout)}",
        f"--fail-on-ratio {cfg.fail_on_ratio}",
    ]
    if cfg.prompt_id:
        parts.append(f"--prompt-id {cfg.prompt_id}")
    if cfg.category:
        parts.append(f"--category {cfg.category}")
    if cfg.container:
        parts.append(f"--container {cfg.container}")
    if cfg.scan_log:
        parts.append(f"--scan-log {cfg.scan_log}")
    return " ".join(parts)


def _build_failure_report(
    cfg: GateConfig, started_iso: str, detected_backend: str | None,
    n1_ttfas: list[float], n1_errors: int, *, reason: str,
    extra_bursts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "backend": cfg.backend_label,
        "detected_backend": detected_backend,
        "base_url": cfg.base_url,
        "started_at": started_iso,
        "ended_at": _now_iso(),
        "bursts_requested": cfg.bursts,
        "bursts_completed": 0,
        "fallback_env_var": cfg.fallback_env_var,
        "n1": {
            "ttfa_ms": n1_ttfas,
            "p50_ms": p50(n1_ttfas),
            "p95_ms": p95(n1_ttfas),
            "errors": n1_errors,
        },
        "n2": {"pairs": extra_bursts or []},
        "md5": {},
        "http_error_count": -1,
        "cuda_error_count": -1,
        "log_source": "n/a",
        "pass": False,
        "failure_reasons": [reason],
        "repro_command": _repro_command(cfg),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_results(report: dict[str, Any], cfg: GateConfig) -> tuple[Path, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ts = _ts_for_path()
    base = cfg.output_dir / f"{cfg.backend_label}_n2_stability_{ts}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def render_markdown(r: dict[str, Any]) -> str:
    n1 = r.get("n1", {}) or {}
    n2 = r.get("n2", {}) or {}
    md = r.get("md5", {}) or {}
    lines = []
    lines.append(f"# {r['backend']} N=2 Stability Gate")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- backend label: `{r['backend']}`")
    lines.append(f"- detected tts_backend: `{r.get('detected_backend')}`")
    lines.append(f"- base_url: `{r['base_url']}`")
    lines.append(f"- started_at: {r['started_at']}")
    lines.append(f"- ended_at: {r['ended_at']}")
    lines.append(f"- bursts: {r.get('bursts_completed')}/{r.get('bursts_requested')}")
    lines.append(f"- pass: **{r.get('pass')}**")
    if r.get("failure_reasons"):
        lines.append("- failure reasons:")
        for reason in r["failure_reasons"]:
            lines.append(f"  - {reason}")
    lines.append("")
    lines.append("## Gate Result")
    lines.append(f"- TTFA ratio N=2/N=1: {n2.get('ratio_vs_n1_p50')}")
    lines.append(f"- HTTP error count: {r.get('http_error_count')}")
    lines.append(f"- CUDA error count: {r.get('cuda_error_count')}")
    lines.append(f"- MD5 stable: {md.get('stable')}")
    lines.append("")
    lines.append("## N=1 TTFA")
    lines.append(f"- p50: {n1.get('p50_ms')} ms")
    lines.append(f"- p95: {n1.get('p95_ms')} ms")
    lines.append(f"- errors: {n1.get('errors')}")
    lines.append("")
    lines.append("## N=2 TTFA")
    lines.append(f"- client0 p50: {n2.get('client0_p50_ms')} ms")
    lines.append(f"- client1 p50: {n2.get('client1_p50_ms')} ms")
    lines.append(f"- combined p50: {n2.get('combined_p50_ms')} ms")
    lines.append(f"- combined p95: {n2.get('combined_p95_ms')} ms")
    lines.append("")
    lines.append("## MD5 Stability")
    lines.append(f"- prompt_id: `{md.get('prompt_id')}`")
    lines.append(f"- pre: `{md.get('pre')}` ({md.get('bytes_pre')} bytes)")
    lines.append(f"- post: `{md.get('post')}` ({md.get('bytes_post')} bytes)")
    lines.append(f"- stable: {md.get('stable')}")
    lines.append("")
    lines.append("## Errors and CUDA Log Scan")
    lines.append(f"- log source: `{r.get('log_source')}`")
    excerpt = r.get("cuda_log_excerpt") or []
    if excerpt:
        lines.append("- excerpt:")
        lines.append("```")
        for line in excerpt[:20]:
            lines.append(line)
        lines.append("```")
    lines.append("")
    lines.append("## Reproduction Command")
    lines.append("```")
    lines.append(str(r.get("repro_command", "")))
    lines.append("```")
    lines.append("")
    lines.append("## Fallback")
    lines.append(
        f"If this gate fails, set `{r.get('fallback_env_var')}=1` to force "
        f"single-slot for this backend without muting other backends. "
        f"If backend-specific env is not supported by the deployed image, "
        f"fall back to `OVS_TTS_STREAM_MAX_WORKERS=1` (affects all TTS backends)."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------


def build_mock_report(cfg: GateConfig) -> dict[str, Any]:
    """Synthesize a deterministic PASS report for CLI smoke / Mac dry-runs.

    No HTTP calls, no log scanning. Used to verify the script is wired
    up correctly when no Jetson is available.
    """
    started = _now_iso()
    fake_n1 = [120.0 + i for i in range(10)]
    fake_combined = [150.0 + (i % 5) for i in range(60)]  # 30 bursts × 2 clients
    report = {
        "backend": cfg.backend_label,
        "detected_backend": f"jetson.{cfg.backend_label}",
        "base_url": cfg.base_url,
        "started_at": started,
        "ended_at": _now_iso(),
        "bursts_requested": cfg.bursts,
        "bursts_completed": cfg.bursts,
        "fallback_env_var": cfg.fallback_env_var,
        "n1": {
            "ttfa_ms": fake_n1,
            "p50_ms": p50(fake_n1),
            "p95_ms": p95(fake_n1),
            "errors": 0,
        },
        "n2": {
            "pairs": [],
            "client0_p50_ms": 150.0,
            "client1_p50_ms": 151.0,
            "combined_p50_ms": p50(fake_combined),
            "combined_p95_ms": p95(fake_combined),
            "ratio_vs_n1_p50": p50(fake_combined) / p50(fake_n1),
        },
        "md5": {
            "prompt_id": cfg.prompt_id or "mock-prompt",
            "pre": "0" * 32,
            "post": "0" * 32,
            "stable": True,
            "bytes_pre": 1024,
            "bytes_post": 1024,
        },
        "http_error_count": 0,
        "cuda_error_count": 0,
        "cuda_log_excerpt": ["[mock mode: no log scan]"],
        "log_source": "mock",
        "pass": True,
        "failure_reasons": [],
        "repro_command": _repro_command(cfg),
        "mock": True,
    }
    return report


# ---------------------------------------------------------------------------
# CLI builder
# ---------------------------------------------------------------------------


def build_parser(backend_label: str, fallback_env_var: str,
                 default_lang: str | None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"{backend_label} N=2 production stability gate "
                    f"(mirrors qwen3_trt reference gate)",
    )
    p.add_argument("--base-url", default="http://localhost:8621",
                   help="Service base URL (default: %(default)s)")
    p.add_argument("--bursts", type=int, default=30,
                   help="Number of N=2 bursts (default: %(default)s)")
    p.add_argument("--warmup", type=int, default=3,
                   help="Warmup requests before measurement (default: %(default)s)")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="Per-request timeout in seconds (default: %(default)s)")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                   help="Where to write JSON + Markdown reports")
    p.add_argument("--prompt-id", default=None,
                   help="Fixed prompt id for MD5 capture (default: first matching)")
    p.add_argument("--lang", default=default_lang,
                   help=f"Prompt language filter (default: {default_lang})")
    p.add_argument("--category", default=None,
                   help="Prompt category filter (e.g. short, long)")
    p.add_argument("--scan-log", default=None,
                   help="Path to a service log file for CUDA error scanning")
    p.add_argument("--container", default=None,
                   help="Docker container name; if set, scan via `docker logs --since`")
    p.add_argument("--fail-on-ratio", type=float, default=1.5,
                   help="Max acceptable N=2/N=1 TTFA ratio (default: %(default)s)")
    p.add_argument("--mock", action="store_true",
                   help="Generate a synthetic PASS report without hitting the service "
                        "(Mac dry-run / CLI smoke)")
    return p


def cfg_from_args(args: argparse.Namespace, backend_label: str,
                  expected_backend_substr: tuple[str, ...],
                  fallback_env_var: str) -> GateConfig:
    return GateConfig(
        backend_label=backend_label,
        expected_backend_substr=expected_backend_substr,
        fallback_env_var=fallback_env_var,
        prompt_lang=args.lang,
        base_url=args.base_url,
        bursts=args.bursts,
        warmup=args.warmup,
        timeout=args.timeout,
        fail_on_ratio=args.fail_on_ratio,
        output_dir=Path(args.output_dir),
        prompt_id=args.prompt_id,
        category=args.category,
        scan_log=Path(args.scan_log) if args.scan_log else None,
        container=args.container,
    )


def main_entry(backend_label: str, expected_backend_substr: tuple[str, ...],
               fallback_env_var: str, default_lang: str | None) -> int:
    parser = build_parser(backend_label, fallback_env_var, default_lang)
    args = parser.parse_args()
    cfg = cfg_from_args(args, backend_label, expected_backend_substr,
                        fallback_env_var)
    if args.mock:
        report = build_mock_report(cfg)
    else:
        report = run_gate(cfg)
    json_path, md_path = write_results(report, cfg)
    print(f"[{cfg.backend_label}] wrote {json_path}")
    print(f"[{cfg.backend_label}] wrote {md_path}")
    print(f"[{cfg.backend_label}] pass={report.get('pass')} "
          f"reasons={report.get('failure_reasons')}")
    return 0 if report.get("pass") else 2
