/* AudioWorklet processor for MicCapture (loaded via audioWorklet.addModule).
 *
 * Runs on the audio rendering thread. Receives float32 blocks at the
 * context's native rate (`sampleRate` global, typically 48000), linearly
 * resamples to the target rate (default 16000), converts to int16 and posts
 * fixed-size PCM chunks (default 40 ms = 640 samples) to the main thread as
 * transferable ArrayBuffers. Also posts throttled level messages (RMS + peak
 * measured on the native-rate signal) for the live volume bar.
 *
 * Messages posted to port:
 *   { type: "pcm",   buffer: ArrayBuffer }   // Int16Array contents, mono
 *   { type: "level", rms: number, peak: number }   // 0..1 floats
 * Messages accepted on port:
 *   "stop"  → processor returns false on next process() and unwinds
 */

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = (options && options.processorOptions) || {};
    this.targetRate = o.targetSampleRate || 16000;
    this.chunkSamples = Math.max(
      1, Math.round(((o.chunkMs || 40) * this.targetRate) / 1000)
    );
    // Post a level update roughly every 32 ms (128-frame blocks at native rate).
    this.levelIntervalBlocks =
      o.levelIntervalBlocks || Math.max(1, Math.round((0.032 * sampleRate) / 128));
    this.ratio = sampleRate / this.targetRate;

    this.chunk = new Int16Array(this.chunkSamples);
    this.chunkLen = 0;
    this.rem = new Float32Array(0); // unconsumed native-rate tail
    this.frac = 0;                  // fractional resample cursor into `rem`
    this.blockCount = 0;
    this.levelPeak = 0;
    this.levelSumSq = 0;
    this.levelSamples = 0;
    this.stopped = false;
    this.port.onmessage = (ev) => {
      if (ev.data === "stop") this.stopped = true;
    };
  }

  process(inputs) {
    if (this.stopped) return false;
    const input = inputs[0] && inputs[0][0];
    if (!input || input.length === 0) return true;

    // ── level metering (native rate, before resample) ──
    for (let i = 0; i < input.length; i++) {
      const v = input[i];
      this.levelSumSq += v * v;
      const a = v < 0 ? -v : v;
      if (a > this.levelPeak) this.levelPeak = a;
    }
    this.levelSamples += input.length;
    if (++this.blockCount >= this.levelIntervalBlocks) {
      this.port.postMessage({
        type: "level",
        rms: Math.sqrt(this.levelSumSq / this.levelSamples),
        peak: this.levelPeak,
      });
      this.blockCount = 0;
      this.levelPeak = 0;
      this.levelSumSq = 0;
      this.levelSamples = 0;
    }

    // ── linear resample with fractional-cursor continuity across blocks ──
    const merged = new Float32Array(this.rem.length + input.length);
    merged.set(this.rem);
    merged.set(input, this.rem.length);
    let pos = this.frac;
    while (pos + 1 < merged.length) {
      const i0 = pos | 0;
      const t = pos - i0;
      let s = merged[i0] * (1 - t) + merged[i0 + 1] * t;
      if (s > 1) s = 1;
      else if (s < -1) s = -1;
      this.chunk[this.chunkLen++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.chunkLen === this.chunkSamples) {
        const out = this.chunk;
        this.port.postMessage({ type: "pcm", buffer: out.buffer }, [out.buffer]);
        this.chunk = new Int16Array(this.chunkSamples);
        this.chunkLen = 0;
      }
      pos += this.ratio;
    }
    const consumed = Math.min(pos | 0, merged.length);
    this.rem = merged.slice(consumed);
    this.frac = pos - consumed;
    return true;
  }
}

registerProcessor("mic-capture", MicCaptureProcessor);
