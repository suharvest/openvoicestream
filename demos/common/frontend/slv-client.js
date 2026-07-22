/* SLV browser SDK skeleton (native ES module, no build step).
 *
 * Wire formats (verified against server/main.py):
 *   /asr/stream  WS   — client sends raw int16 PCM (16 kHz mono) binary frames;
 *                       empty binary frame = end-of-utterance. Server sends JSON
 *                       {"text","is_final","is_stable",...}; final frames may
 *                       carry {speaker, speaker_conf} when ?diarize=true.
 *   /tts/stream  POST — body TTSRequest {text, speaker_id?, speed?, pitch?,
 *                       language?, voice?}; response bytes: first 4 bytes =
 *                       sample_rate (uint32 LE), then raw int16 PCM chunks.
 *
 * P0 scope: WS ASR client + streaming TTS player. Microphone capture
 * (AudioWorklet, 48k float → 16k PCM16 resample) lands in P1 — see MicCapture.
 */

/* ── ASR streaming client ────────────────────────────────────────────── */

export class ASRStreamClient {
  /**
   * @param {object} opts
   * @param {string}  [opts.baseUrl]   http(s) origin of the SLV server (default: page origin)
   * @param {string}  [opts.language]  "auto" | "zh" | "en" | ...
   * @param {number}  [opts.sampleRate]
   * @param {boolean} [opts.diarize]   ask for speaker labels on finals
   * @param {function} [opts.onPartial] (msg) partial hypothesis
   * @param {function} [opts.onFinal]   (msg) finalized utterance
   * @param {function} [opts.onEvent]   (msg) any other JSON event (vad_endpoint, summary, error)
   * @param {function} [opts.onClose]   (ev)
   */
  constructor(opts = {}) {
    this.opts = opts;
    this.ws = null;
  }

  get _wsUrl() {
    const base = this.opts.baseUrl || window.location.origin;
    const u = new URL("/asr/stream", base);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    u.searchParams.set("language", this.opts.language || "auto");
    u.searchParams.set("sample_rate", String(this.opts.sampleRate || 16000));
    if (this.opts.diarize) u.searchParams.set("diarize", "true");
    return u.toString();
  }

  connect() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this._wsUrl);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => resolve(this);
      ws.onerror = (e) => reject(e);
      ws.onclose = (ev) => this.opts.onClose && this.opts.onClose(ev);
      ws.onmessage = (ev) => {
        let msg;
        try { msg = JSON.parse(ev.data); } catch { return; }
        if (msg.is_final) this.opts.onFinal && this.opts.onFinal(msg);
        else if (typeof msg.text === "string") this.opts.onPartial && this.opts.onPartial(msg);
        else this.opts.onEvent && this.opts.onEvent(msg);
      };
      this.ws = ws;
    });
  }

  /** @param {Int16Array|ArrayBuffer} pcm 16 kHz mono int16 samples */
  sendPcm(pcm) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(pcm instanceof Int16Array ? pcm.buffer : pcm);
    }
  }

  /** Force end-of-utterance (server VAD normally does this on silence). */
  endUtterance() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(new ArrayBuffer(0));
    }
  }

  close() {
    if (this.ws) { this.ws.close(); this.ws = null; }
  }
}

/* ── TTS streaming player ────────────────────────────────────────────── */

export class TTSStreamPlayer {
  /**
   * @param {object} opts
   * @param {string} [opts.baseUrl]      http(s) origin of the SLV server
   * @param {AudioContext} [opts.audioContext]  shared context (created lazily otherwise)
   */
  constructor(opts = {}) {
    this.baseUrl = opts.baseUrl || window.location.origin;
    this.ctx = opts.audioContext || null;
    this._abort = null;
    this._sources = [];
  }

  _ensureCtx() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  }

  /**
   * Stream-synthesize and play. Resolves when playback of all received audio ends.
   * @param {object} req  {text, speaker_id?, speed?, pitch?, language?, voice?}
   * @param {object} [cb] {onTTFA(seconds), onChunk(int16Len), onDone()}
   */
  async speak(req, cb = {}) {
    this.stop();
    const ctx = this._ensureCtx();
    this._abort = new AbortController();
    const t0 = performance.now();

    const resp = await fetch(new URL("/tts/stream", this.baseUrl), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal: this._abort.signal,
    });
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try { detail = JSON.stringify(await resp.json()); } catch { /* keep */ }
      throw new Error(`TTS stream failed: ${detail}`);
    }

    const reader = resp.body.getReader();
    let header = new Uint8Array(0);   // first 4 bytes: sample_rate uint32 LE
    let sampleRate = 0;
    let leftover = new Uint8Array(0); // odd trailing byte between chunks
    let playhead = 0;                 // ctx.currentTime-based scheduling cursor
    let firstAudio = true;

    const scheduleChunk = (int16) => {
      if (!int16.length) return;
      const buf = ctx.createBuffer(1, int16.length, sampleRate);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < int16.length; i++) ch[i] = int16[i] / 32768;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      const startAt = Math.max(ctx.currentTime + 0.02, playhead);
      src.start(startAt);
      playhead = startAt + buf.duration;
      this._sources.push(src);
    };

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      let bytes = value;
      if (sampleRate === 0) {
        const merged = new Uint8Array(header.length + bytes.length);
        merged.set(header); merged.set(bytes, header.length);
        if (merged.length < 4) { header = merged; continue; }
        sampleRate = new DataView(merged.buffer).getUint32(0, true);
        bytes = merged.subarray(4);
      }
      if (leftover.length) {
        const merged = new Uint8Array(leftover.length + bytes.length);
        merged.set(leftover); merged.set(bytes, leftover.length);
        bytes = merged;
        leftover = new Uint8Array(0);
      }
      const evenLen = bytes.length - (bytes.length % 2);
      if (bytes.length % 2) leftover = bytes.slice(evenLen);
      const int16 = new Int16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + evenLen));
      if (firstAudio && int16.length) {
        firstAudio = false;
        cb.onTTFA && cb.onTTFA((performance.now() - t0) / 1000);
      }
      cb.onChunk && cb.onChunk(int16.length);
      scheduleChunk(int16);
    }

    // Resolve after the scheduled tail finishes playing.
    const remaining = Math.max(0, playhead - ctx.currentTime);
    await new Promise((r) => setTimeout(r, remaining * 1000 + 50));
    cb.onDone && cb.onDone();
  }

  /** Barge-in: abort the fetch and silence everything scheduled. */
  stop() {
    if (this._abort) { this._abort.abort(); this._abort = null; }
    for (const s of this._sources) { try { s.stop(); } catch { /* already ended */ } }
    this._sources = [];
  }
}

/* ── Microphone capture (P1) ─────────────────────────────────────────── */

export class MicCapture {
  /**
   * TODO(P1): implement AudioWorklet-based capture:
   *   1. getUserMedia({audio: {echoCancellation, noiseSuppression}})
   *   2. AudioWorkletNode pulling 128-frame float32 blocks at ctx.sampleRate (typically 48k)
   *   3. linear resample → 16 kHz, float32 → int16 (clamp, ×32767)
   *   4. batch ~20 ms frames and invoke onPcmChunk(Int16Array)
   * Also expose a live RMS level for the volume-bar UI (design principle #2).
   */
  constructor(opts = {}) {
    this.opts = opts; // {targetSampleRate=16000, onPcmChunk, onLevel}
  }

  async start() {
    throw new Error("MicCapture is not implemented yet (planned for P1)");
  }

  stop() {}
}
