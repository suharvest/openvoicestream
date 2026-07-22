/* MicCapture — browser microphone → 16 kHz mono PCM16 chunks (native ES module).
 *
 * getUserMedia → AudioContext + AudioWorklet (/common/mic-worklet.js):
 *   - resamples the native-rate float32 stream to 16 kHz int16 on the audio
 *     thread, delivered as ~40 ms Int16Array chunks (640 samples / 1280 bytes);
 *   - reports a live level (RMS + peak, 0..1) for volume-bar UI;
 *   - clean start()/stop(); start() rejects with a human-readable error that
 *     carries bilingual text in `err.messages = { zh, en }` and a stable
 *     `err.code` (permission_denied / no_microphone / mic_busy /
 *     insecure_context / no_worklet / audio_setup_failed / mic_failed).
 *
 * Usage:
 *   import { MicCapture } from "/common/mic-capture.js";
 *   const mic = new MicCapture({
 *     onPcmChunk: (int16) => ws.send(int16.buffer),
 *     onLevel: (rms, peak) => bar.style.width = `${Math.min(100, peak * 140)}%`,
 *   });
 *   await mic.start();
 *   ...
 *   mic.stop();
 */

const DEFAULT_WORKLET_URL = "/common/mic-worklet.js";

function micError(code, zh, en, cause) {
  const err = new Error(en);
  err.name = "MicCaptureError";
  err.code = code;
  err.messages = { zh, en };
  if (cause !== undefined) err.cause = cause;
  return err;
}

function translateGetUserMediaError(e) {
  const name = e && e.name;
  if (name === "NotAllowedError" || name === "PermissionDeniedError" || name === "SecurityError") {
    return micError(
      "permission_denied",
      "麦克风权限被拒绝。请点击浏览器地址栏的权限图标，允许使用麦克风后重试。",
      "Microphone access was denied. Allow the microphone in the browser's address-bar permission prompt, then retry.",
      e,
    );
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return micError(
      "no_microphone",
      "没有检测到麦克风。请接上麦克风（或检查系统输入设备）后重试。",
      "No microphone detected. Plug one in (or check the system input device) and retry.",
      e,
    );
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return micError(
      "mic_busy",
      "麦克风被其他应用占用，无法打开。请关闭占用麦克风的应用后重试。",
      "The microphone is busy in another application. Close it and retry.",
      e,
    );
  }
  return micError(
    "mic_failed",
    `无法打开麦克风：${name || e}`,
    `Could not open the microphone: ${name || e}`,
    e,
  );
}

export class MicCapture {
  /**
   * @param {object} opts
   * @param {number}   [opts.targetSampleRate=16000]
   * @param {number}   [opts.chunkMs=40]         PCM chunk size delivered to onPcmChunk
   * @param {string}   [opts.workletUrl="/common/mic-worklet.js"]
   * @param {function} [opts.onPcmChunk]  (Int16Array) 16 kHz mono PCM16 chunk
   * @param {function} [opts.onLevel]     (rms, peak) live input level, 0..1
   */
  constructor(opts = {}) {
    this.targetSampleRate = opts.targetSampleRate || 16000;
    this.chunkMs = opts.chunkMs || 40;
    this.workletUrl = opts.workletUrl || DEFAULT_WORKLET_URL;
    this.onPcmChunk = opts.onPcmChunk || null;
    this.onLevel = opts.onLevel || null;
    this.ctx = null;
    this.stream = null;
    this.source = null;
    this.node = null;
    this.sink = null;
    this.running = false;
  }

  /** Native AudioContext rate (0 until started). */
  get contextSampleRate() {
    return this.ctx ? this.ctx.sampleRate : 0;
  }

  async start() {
    if (this.running) return this;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw micError(
        "insecure_context",
        "浏览器只允许在 HTTPS 或 http://localhost 下使用麦克风。请改用 localhost 端口转发或 HTTPS 打开本页。",
        "Browsers only expose the microphone on HTTPS or http://localhost. Open this page via localhost port-forwarding or HTTPS.",
      );
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) {
      throw micError(
        "no_audiocontext",
        "此浏览器不支持 Web Audio，无法采集麦克风。请换用最新版 Chrome / Edge / Safari / Firefox。",
        "This browser lacks Web Audio support. Use a recent Chrome / Edge / Safari / Firefox.",
      );
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (e) {
      throw translateGetUserMediaError(e);
    }

    try {
      this.ctx = new AC();
      if (this.ctx.state === "suspended") await this.ctx.resume();
      if (!this.ctx.audioWorklet) {
        throw micError(
          "no_worklet",
          "此浏览器不支持 AudioWorklet，无法低延迟采集音频。请换用最新版浏览器。",
          "This browser lacks AudioWorklet support. Use a recent browser.",
        );
      }
      await this.ctx.audioWorklet.addModule(this.workletUrl);

      this.source = this.ctx.createMediaStreamSource(this.stream);
      this.node = new AudioWorkletNode(this.ctx, "mic-capture", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
        processorOptions: {
          targetSampleRate: this.targetSampleRate,
          chunkMs: this.chunkMs,
        },
      });
      this.node.port.onmessage = (ev) => {
        if (!this.running) return;
        const m = ev.data || {};
        if (m.type === "pcm") {
          if (this.onPcmChunk) this.onPcmChunk(new Int16Array(m.buffer));
        } else if (m.type === "level") {
          if (this.onLevel) this.onLevel(m.rms, m.peak);
        }
      };

      // Keep the worklet pulled by the rendering graph without audible
      // output: route through a muted gain into the destination.
      this.sink = this.ctx.createGain();
      this.sink.gain.value = 0;
      this.source.connect(this.node);
      this.node.connect(this.sink);
      this.sink.connect(this.ctx.destination);
    } catch (e) {
      this._teardown();
      if (e && e.messages) throw e; // already a MicCaptureError
      throw micError(
        "audio_setup_failed",
        `音频初始化失败：${(e && e.message) || e}`,
        `Audio setup failed: ${(e && e.message) || e}`,
        e,
      );
    }

    this.running = true;
    return this;
  }

  /** Idempotent: stops tracks, disconnects nodes, closes the context. */
  stop() {
    this.running = false;
    this._teardown();
  }

  _teardown() {
    if (this.node) {
      try { this.node.port.postMessage("stop"); } catch { /* detached */ }
      try { this.node.disconnect(); } catch { /* already */ }
    }
    for (const n of [this.source, this.sink]) {
      if (n) { try { n.disconnect(); } catch { /* already */ } }
    }
    if (this.stream) {
      for (const t of this.stream.getTracks()) {
        try { t.stop(); } catch { /* already */ }
      }
    }
    if (this.ctx) { try { this.ctx.close(); } catch { /* already */ } }
    this.node = this.source = this.sink = this.ctx = this.stream = null;
  }
}
