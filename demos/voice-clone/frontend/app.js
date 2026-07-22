/* Voice Clone demo — record 10 s of your voice → enroll → speak any text with it.
 *
 * State machine (single source of truth: `state`):
 *
 *   probing ──(supports_voice_cloning=false / unreachable)──▶ gated (full-screen notice)
 *      │ ok
 *      ▼
 *    idle ──mic click──▶ recording ──(10 s auto-stop | manual stop ≥3 s)──▶ enrolling
 *      ▲                     │ manual stop <3 s → human hint, back to idle
 *      │                     ▼
 *      │◀── enroll failed (human wording + retry keeps the recording)
 *      │                 enrolling ──200──▶ ready (step 3: text + play)
 *      │
 *   ready ──play──▶ playing ──done/stop──▶ ready
 *
 * Picking a previously enrolled voice from the sidebar jumps straight to ready.
 *
 * Audio path: MicCapture (shared /common/mic-capture.js) yields 16 kHz mono
 * PCM16 chunks; on stop they are concatenated into a WAV blob (44-byte RIFF
 * header written here) and POSTed as a raw body to /api/enroll, which the demo
 * backend repacks into the multipart form SLV /tts/voices/enroll expects.
 * Playback uses the shared TTSStreamPlayer with {text, voice: <voice_id>} —
 * the backend forwards it to SLV /tts/stream (VoiceProfile `voice` selector).
 */

import { createStatusPill, createMetricCard, createModelSwitchPanel } from "/common/ui.js";
import { TTSStreamPlayer } from "/common/slv-client.js";
import { MicCapture } from "/common/mic-capture.js";

/* ── constants ────────────────────────────────────────────────────── */
const SAMPLE_RATE = 16000;
const MAX_RECORD_S = 10;   // auto-stop (server accepts 3-15 s references)
const MIN_RECORD_S = 3;    // server-side contract: "reference wav (3-15s)"
const RING_CIRCUMFERENCE = 490; // matches stroke-dasharray in index.html

/* ── i18n (default zh, shared localStorage key with the gallery) ──── */
const I18N = {
  zh: {
    back: "← 返回演示门户",
    step1: "录 10 秒你的声音",
    step2: "等待声音注册",
    step3: "用你的声音朗读",
    retry: "重试",
    privacy: "录音仅在本设备处理，不上传云端",
    recIdle: "点击麦克风，朗读几句话（3–10 秒），说什么都行",
    recRecording: "录音中… 再次点击结束（至少 3 秒），10 秒自动结束",
    recTooShort: "录音太短了：请至少录满 3 秒再结束。",
    enrolling: "正在注册你的声音（设备本地分析音色）…",
    enrollDoneStatus: "声音注册成功 ✓ 现在输入文本，用你的声音朗读",
    errEnrollFailed: "注册失败：语音服务处理这段录音时出错了。可以直接重试，或重新录一段更清晰的声音。",
    errEnrollUnsupported: "这台设备不支持在线注册声音（需在 GPU 主机上完成 enroll），请联系工作人员预先注册。",
    errEnrollUnreachable: "注册失败：语音服务暂时不可达。请确认设备电源与网络后重试。",
    errVoices: "拿不到已注册声音列表：语音服务暂时不可达。",
    errTts: "合成失败：语音服务没有响应。请确认设备在线后重试。",
    errTtsHttp: (s) => `合成失败（服务返回 ${s}）。请稍后重试。`,
    gateUnsupportedTitle: "此设备暂不支持声音克隆",
    gateUnsupportedMsg: "当前 TTS 引擎没有声音克隆能力。请在门户切换到支持克隆的模型（如 SparkTTS），或换用支持的设备。",
    gateNotReadyTitle: "语音引擎启动中",
    gateNotReadyMsg: "TTS 引擎尚未就绪，请稍候片刻再试。",
    gateDownTitle: "语音服务不可达",
    gateDownMsg: "连不上本机语音服务。请确认设备电源与网络后重试。",
    switchTitle: "模型切换",
    voicesLabel: "已注册声音",
    voicesNone: "还没有注册的声音",
    voicesLoaded: (n) => `${n} 个已注册声音`,
    voicePlaceholder: "── 选择一个历史声音 ──",
    rerecord: "🎙 重新录一段",
    placeholder: "输入任意文本，用你刚注册的声音朗读…",
    ttfaLabel: "TTFA 首包延迟",
    durLabel: "已播放",
    charsLabel: "生成字符数",
    statusIdle: "就绪 — 点击 ▶ 用你的声音开始朗读",
    statusSynth: "合成中，等待首块音频…",
    statusPlaying: "流式播放中（你的声音，本机实时合成）",
    statusDone: "播放完成 ✓",
    statusStopped: "已停止",
    engineLoading: "连接语音服务…",
    engineDown: "语音服务离线",
    engineReady: "支持声音克隆",
    footer: "seeed-local-voice · 录音与声音克隆完全在本机边缘设备完成，无云端",
    aboutClone: "声音克隆＝用几秒参考音频提取你的音色特征（VoiceProfile），随后任意文本都能用这个音色实时合成。全过程在本设备完成：录音不上传云端，这正是边缘本地语音的核心卖点。",
    countdown: (s) => `${s.toFixed(1)} s`,
  },
  en: {
    back: "← Back to gallery",
    step1: "Record 10 s of your voice",
    step2: "Enrolling your voice",
    step3: "Speak with your voice",
    retry: "Retry",
    privacy: "Audio is processed on this device only — never uploaded to any cloud",
    recIdle: "Tap the mic and read a few sentences (3–10 s) — anything works",
    recRecording: "Recording… tap again to finish (min 3 s), auto-stops at 10 s",
    recTooShort: "Too short: please record at least 3 seconds.",
    enrolling: "Enrolling your voice (on-device timbre analysis)…",
    enrollDoneStatus: "Voice enrolled ✓ Now type some text and hear it in your voice",
    errEnrollFailed: "Enrollment failed: the voice server couldn't process this recording. Retry, or record a cleaner take.",
    errEnrollUnsupported: "This device can't enroll voices in-process (needs a GPU host). Ask staff to pre-enroll one.",
    errEnrollUnreachable: "Enrollment failed: the voice server is unreachable. Check the device power/network and retry.",
    errVoices: "Couldn't load enrolled voices: the voice server is unreachable.",
    errTts: "Synthesis failed: the voice server didn't respond. Check the device is online and retry.",
    errTtsHttp: (s) => `Synthesis failed (server returned ${s}). Try again.`,
    gateUnsupportedTitle: "Voice cloning isn't available on this device",
    gateUnsupportedMsg: "The current TTS engine has no voice-clone capability. Switch to a clone-capable model (e.g. SparkTTS) in the gallery, or use a supported device.",
    gateNotReadyTitle: "Voice engine is starting",
    gateNotReadyMsg: "The TTS engine isn't ready yet. Give it a moment and retry.",
    gateDownTitle: "Voice server unreachable",
    gateDownMsg: "Can't reach the on-device voice server. Check the device power/network and retry.",
    switchTitle: "Switch model",
    voicesLabel: "Enrolled voices",
    voicesNone: "No enrolled voices yet",
    voicesLoaded: (n) => `${n} enrolled voice${n === 1 ? "" : "s"}`,
    voicePlaceholder: "── pick a previous voice ──",
    rerecord: "🎙 Record a new voice",
    placeholder: "Type anything — it will be read in the voice you just enrolled…",
    ttfaLabel: "TTFA · time to first audio",
    durLabel: "Played",
    charsLabel: "Characters",
    statusIdle: "Ready — press ▶ to speak in your voice",
    statusSynth: "Synthesizing, waiting for first audio…",
    statusPlaying: "Streaming playback (your voice, synthesized on-device)",
    statusDone: "Done ✓",
    statusStopped: "Stopped",
    engineLoading: "Connecting to voice server…",
    engineDown: "Voice server offline",
    engineReady: "Voice cloning ready",
    footer: "seeed-local-voice · recording and cloning happen entirely on this edge device, no cloud",
    aboutClone: "Voice cloning extracts your timbre (a VoiceProfile) from a few seconds of reference audio; any text can then be synthesized in that voice in real time. Everything runs on this device — your recording never leaves it. That's the whole point of edge-local voice.",
    countdown: (s) => `${s.toFixed(1)} s`,
  },
};
let lang = localStorage.getItem("slv-demo-lang") || "zh";
const t = (k) => I18N[lang][k];

/* Sample sentences for step 3 */
const SAMPLES = [
  { tag: { zh: "中文", en: "ZH" },
    text: "大家好，这是我刚刚在这台边缘设备上克隆出来的声音，整个过程没有经过任何云端。" },
  { tag: { zh: "英文", en: "EN" },
    text: "Hi! This is my cloned voice, generated entirely on this edge device in real time." },
  { tag: { zh: "混合", en: "Mixed" },
    text: "只用了 10 秒参考音频，本地 TTS 引擎就能用我的音色 streaming 朗读任意文本。" },
  { tag: { zh: "短句", en: "Short" },
    text: "My voice. My device. No cloud." },
];

/* ── DOM refs ─────────────────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);
const micBtn = $("mic-btn");
const iconMic = $("icon-mic");
const iconRecStop = $("icon-rec-stop");
const ringProg = $("ring-prog");
const recCount = $("rec-count");
const recStatus = $("rec-status");
const levelBar = $("level-bar");
const enrollPanel = $("enroll-panel");
const recordPanel = $("record-panel");
const synthPanel = $("synth-panel");
const playPanel = $("play-panel");
const metricsBox = $("metrics");
const textArea = $("text");
const playBtn = $("play-btn");
const iconPlay = $("icon-play");
const iconStop = $("icon-stop");
const waveEl = $("wave");
const playStatusEl = $("play-status");
const voiceSelect = $("voice-select");

/* waveform bars */
for (let i = 0; i < 28; i++) {
  const bar = document.createElement("span");
  bar.style.setProperty("--i", i);
  waveEl.appendChild(bar);
}

/* metric cards: big TTFA (ms) + small duration/chars */
const ttfaCard = createMetricCard(metricsBox, { label: "", unit: "ms", digits: 0 });
const durCard = createMetricCard(metricsBox, { label: "", unit: "s", digits: 1 });
const charsCard = createMetricCard(metricsBox, { label: "", unit: "", digits: 0 });
durCard.el.classList.add("small");
charsCard.el.classList.add("small");

const enginePill = createStatusPill($("engine-pill"), { state: "busy", text: "…" });

/* ── state ────────────────────────────────────────────────────────── */
let state = "probing"; // probing | gated | idle | recording | enrolling | ready | playing
let capabilities = null;   // /api/capabilities payload
let voices = [];           // /api/voices list
let activeVoiceId = null;  // enrolled voice used for synthesis
let lastError = null;      // {retry: fn}

let mic = null;
let pcmChunks = [];        // Int16Array[]
let pcmSamples = 0;
let recStartedAt = 0;
let recTimer = null;       // rAF handle
let lastWav = null;        // Blob — kept so an enroll retry doesn't re-record

const player = new TTSStreamPlayer(); // page origin → backend proxies /tts/stream
let playing = false;
let durTimer = null;
let firstAudioAt = null;

/* ── i18n rendering ───────────────────────────────────────────────── */
function applyStaticI18n() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  for (const node of document.querySelectorAll("[data-i18n]")) {
    node.textContent = t(node.dataset.i18n);
  }
  $("lang-btn").textContent = lang === "zh" ? "English" : "中文";
  textArea.placeholder = t("placeholder");
  ttfaCard.setLabel(t("ttfaLabel"));
  durCard.setLabel(t("durLabel"));
  charsCard.setLabel(t("charsLabel"));
  renderChips();
  renderEnginePill();
  renderVoices();
  renderStateUI();
}

function renderChips() {
  const box = $("chips");
  box.innerHTML = "";
  for (const s of SAMPLES) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = s.tag[lang];
    chip.append(tag, document.createTextNode(s.text));
    chip.title = s.text;
    chip.addEventListener("click", () => {
      textArea.value = s.text;
      renderStateUI();
    });
    box.appendChild(chip);
  }
}

/* ── three-step guide + panel visibility (single render pass) ─────── */
function renderStateUI() {
  const setStep = (el, cls) => {
    el.classList.remove("active", "done");
    if (cls) el.classList.add(cls);
  };
  const enrolled = activeVoiceId !== null;
  setStep($("step-1"),
    enrolled ? "done" : (state === "recording" || state === "idle" ? "active" : ""));
  setStep($("step-2"), enrolled ? "done" : (state === "enrolling" ? "active" : ""));
  setStep($("step-3"), playing ? "done" : (enrolled ? "active" : ""));

  recordPanel.classList.toggle("hidden", enrolled || state === "enrolling");
  enrollPanel.classList.toggle("visible", state === "enrolling");
  synthPanel.classList.toggle("hidden", !enrolled);
  playPanel.classList.toggle("hidden", !enrolled);
  metricsBox.classList.toggle("hidden", !enrolled);

  micBtn.disabled = state === "probing" || state === "gated" || state === "enrolling";
  playBtn.disabled = !(enrolled && (textArea.value.trim().length > 0 || playing));

  if (state === "recording") recStatus.textContent = t("recRecording");
  else if (state === "idle" && !recStatus.dataset.sticky) recStatus.textContent = t("recIdle");
  recStatus.classList.toggle("live", state === "recording");
  $("enroll-status").textContent = t("enrolling");
}
textArea.addEventListener("input", renderStateUI);

/* ── engine pill / capability gate ────────────────────────────────── */
function renderEnginePill() {
  if (state === "probing") enginePill.set("busy", t("engineLoading"));
  else if (state === "gated") enginePill.set("err", t("engineDown"));
  else enginePill.set("ok", (capabilities && capabilities.model_id) || t("engineReady"));
}

function showGate(titleKey, msgKey, detail) {
  state = "gated";
  $("gate-title").textContent = t(titleKey);
  $("gate-msg").textContent = t(msgKey);
  $("gate-detail").textContent = detail || "";
  $("gate").classList.add("visible");
  renderEnginePill();
  renderStateUI();
}
function hideGate() {
  $("gate").classList.remove("visible");
}

async function probeCapabilities() {
  state = "probing";
  hideGate();
  renderEnginePill();
  let body = null;
  let httpStatus = 0;
  try {
    const resp = await fetch("/api/capabilities");
    httpStatus = resp.status;
    body = await resp.json();
  } catch (e) {
    showGate("gateDownTitle", "gateDownMsg", String(e.message || e));
    return;
  }
  capabilities = body;
  if (!body || body.reachable === false || httpStatus === 502) {
    showGate("gateDownTitle", "gateDownMsg", body && (body.message || body.error));
    return;
  }
  if (body.reason === "tts_not_ready") {
    showGate("gateNotReadyTitle", "gateNotReadyMsg", body.detail);
    return;
  }
  if (!body.supports_voice_cloning) {
    showGate("gateUnsupportedTitle", "gateUnsupportedMsg",
             body.backend ? `backend: ${body.backend}` : "");
    return;
  }
  state = "idle";
  renderEnginePill();
  renderStateUI();
  loadVoices();
}
$("gate-retry").addEventListener("click", probeCapabilities);

/* ── enrolled voices (sidebar history) ────────────────────────────── */
function renderVoices() {
  voiceSelect.innerHTML = "";
  const ph = document.createElement("option");
  ph.value = "";
  ph.textContent = t("voicePlaceholder");
  voiceSelect.appendChild(ph);
  for (const v of voices) {
    const opt = document.createElement("option");
    opt.value = v.voice_id;
    opt.textContent = v.voice_id;
    voiceSelect.appendChild(opt);
  }
  voiceSelect.value = activeVoiceId && voices.some((v) => v.voice_id === activeVoiceId)
    ? activeVoiceId : "";
  $("voices-hint").textContent =
    voices.length ? t("voicesLoaded")(voices.length) : t("voicesNone");
}

async function loadVoices() {
  try {
    const resp = await fetch("/api/voices");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const body = await resp.json();
    voices = body.voices || [];
  } catch (e) {
    voices = [];
    $("voices-hint").textContent = `${t("errVoices")} (${e.message || e})`;
  }
  renderVoices();
}

voiceSelect.addEventListener("change", () => {
  const vid = voiceSelect.value;
  if (!vid) return;
  activeVoiceId = vid;      // jump straight to step 3 with a historical voice
  state = "ready";
  hideError();
  renderStateUI();
});

$("rerecord-btn").addEventListener("click", () => {
  if (playing) player.stop();
  activeVoiceId = null;
  lastWav = null;
  voiceSelect.value = "";
  state = "idle";
  delete recStatus.dataset.sticky;
  hideError();
  renderStateUI();
});

/* ── error box (human wording + retry, never raw JSON) ────────────── */
function showError(msg, detail, retry) {
  lastError = { retry };
  $("error-msg").textContent = msg;
  $("error-detail").textContent = detail || "";
  $("error-box").classList.add("visible");
}
function hideError() {
  lastError = null;
  $("error-box").classList.remove("visible");
}
$("error-retry").addEventListener("click", () => {
  const retry = lastError && lastError.retry;
  hideError();
  if (retry) retry();
});

/* ── recording (step 1) ───────────────────────────────────────────── */
function recordedSeconds() {
  return pcmSamples / SAMPLE_RATE;
}

function setRecordingUI(on) {
  micBtn.classList.toggle("recording", on);
  iconMic.style.display = on ? "none" : "";
  iconRecStop.style.display = on ? "" : "none";
  micBtn.setAttribute("aria-label", on ? "stop recording" : "record");
  if (!on) {
    ringProg.style.strokeDashoffset = String(RING_CIRCUMFERENCE);
    levelBar.style.width = "0%";
    recCount.textContent = "";
  }
}

function tickRecording() {
  if (state !== "recording") return;
  const s = Math.min(MAX_RECORD_S, (performance.now() - recStartedAt) / 1000);
  recCount.textContent = t("countdown")(MAX_RECORD_S - s);
  ringProg.style.strokeDashoffset =
    String(RING_CIRCUMFERENCE * (1 - s / MAX_RECORD_S));
  if (s >= MAX_RECORD_S) {
    stopRecording(true);
    return;
  }
  recTimer = requestAnimationFrame(tickRecording);
}

async function startRecording() {
  hideError();
  delete recStatus.dataset.sticky;
  pcmChunks = [];
  pcmSamples = 0;
  mic = new MicCapture({
    onPcmChunk: (int16) => {
      if (state !== "recording") return;
      pcmChunks.push(int16);
      pcmSamples += int16.length;
    },
    onLevel: (_rms, peak) => {
      if (state !== "recording") return;
      levelBar.style.width = `${Math.min(100, peak * 140).toFixed(1)}%`;
    },
  });
  try {
    await mic.start();
  } catch (e) {
    mic = null;
    // MicCaptureError carries bilingual human wording
    const msg = (e.messages && e.messages[lang]) || String(e.message || e);
    showError(msg, e.code || "", startRecording);
    return;
  }
  state = "recording";
  recStartedAt = performance.now();
  setRecordingUI(true);
  renderStateUI();
  recTimer = requestAnimationFrame(tickRecording);
}

function stopRecording(auto) {
  if (recTimer) { cancelAnimationFrame(recTimer); recTimer = null; }
  const seconds = recordedSeconds();
  if (!auto && seconds < MIN_RECORD_S) {
    // Too short — discard, human hint, back to idle (no half-baked enrolls).
    if (mic) { mic.stop(); mic = null; }
    state = "idle";
    setRecordingUI(false);
    recStatus.dataset.sticky = "1"; // survives renderStateUI's idle overwrite
    renderStateUI();
    recStatus.textContent = t("recTooShort");
    return;
  }
  if (mic) { mic.stop(); mic = null; }
  setRecordingUI(false);
  lastWav = pcmToWav(pcmChunks, pcmSamples, SAMPLE_RATE);
  pcmChunks = [];
  pcmSamples = 0;
  enroll();
}

micBtn.addEventListener("click", () => {
  if (state === "recording") stopRecording(false);
  else if (state === "idle") startRecording();
});

/* PCM16 chunks → WAV blob (44-byte RIFF header, mono 16-bit) */
function pcmToWav(chunks, totalSamples, sampleRate) {
  const dataLen = totalSamples * 2;
  const buf = new ArrayBuffer(44 + dataLen);
  const dv = new DataView(buf);
  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  dv.setUint32(4, 36 + dataLen, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  dv.setUint32(16, 16, true);          // fmt chunk size
  dv.setUint16(20, 1, true);           // PCM
  dv.setUint16(22, 1, true);           // mono
  dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * 2, true); // byte rate
  dv.setUint16(32, 2, true);           // block align
  dv.setUint16(34, 16, true);          // bits per sample
  writeStr(36, "data");
  dv.setUint32(40, dataLen, true);
  let off = 44;
  for (const c of chunks) {
    for (let i = 0; i < c.length; i++, off += 2) dv.setInt16(off, c[i], true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

/* ── enrollment (step 2) ──────────────────────────────────────────── */
function newVoiceId() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `web-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-` +
         `${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

async function enroll() {
  if (!lastWav) return;
  hideError();
  state = "enrolling";
  renderStateUI();

  const vid = newVoiceId();
  let resp = null;
  let body = {};
  try {
    resp = await fetch(`/api/enroll?voice_id=${encodeURIComponent(vid)}`, {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: lastWav,
    });
    body = await resp.json().catch(() => ({}));
  } catch (e) {
    state = "idle";
    renderStateUI();
    showError(t("errEnrollUnreachable"), String(e.message || e), enroll);
    return;
  }

  if (!resp.ok) {
    state = "idle";
    renderStateUI();
    if (resp.status === 501) {
      // Jetson in-process enroll unavailable — server points at host enrollment.
      showError(t("errEnrollUnsupported"), body.error || "", null);
    } else if (resp.status === 502) {
      showError(t("errEnrollUnreachable"), body.message || "", enroll);
    } else {
      // Retry reuses lastWav — the visitor doesn't have to record again.
      showError(t("errEnrollFailed"), body.error || `HTTP ${resp.status}`, enroll);
    }
    return;
  }

  activeVoiceId = body.voice_id || vid;
  state = "ready";
  renderStateUI();
  setStatus("statusIdle");
  recStatus.textContent = "";
  $("voices-hint").textContent = t("enrollDoneStatus");
  loadVoices();
}

/* ── playback (step 3) ────────────────────────────────────────────── */
function setStatus(key, live = false) {
  playStatusEl.textContent = t(key);
  playStatusEl.classList.toggle("live", live);
}

function setPlayingUI(on) {
  playing = on;
  playBtn.classList.toggle("playing", on);
  waveEl.classList.toggle("playing", on);
  iconPlay.style.display = on ? "none" : "";
  iconStop.style.display = on ? "" : "none";
  playBtn.setAttribute("aria-label", on ? "stop" : "play");
  renderStateUI();
}

function startDurTimer() {
  firstAudioAt = performance.now();
  durTimer = setInterval(() => {
    durCard.set((performance.now() - firstAudioAt) / 1000);
  }, 100);
}
function stopDurTimer() {
  if (durTimer) { clearInterval(durTimer); durTimer = null; }
  if (firstAudioAt != null) {
    durCard.set((performance.now() - firstAudioAt) / 1000);
  }
  firstAudioAt = null;
}

async function play() {
  const text = textArea.value.trim();
  if (!text || !activeVoiceId) return;
  hideError();
  ttfaCard.reset();
  durCard.reset();
  charsCard.set(text.length);
  setPlayingUI(true);
  setStatus("statusSynth", true);

  let stopped = false;
  try {
    /* TTSStreamPlayer POSTs {origin}/tts/stream — our backend forwards it to
       SLV /tts/stream with the `voice` (VoiceProfile) selector intact. */
    await player.speak({ text, voice: activeVoiceId }, {
      onTTFA: (seconds) => {
        ttfaCard.set(seconds * 1000);
        setStatus("statusPlaying", true);
        startDurTimer();
      },
    });
  } catch (e) {
    if (e.name === "AbortError") {
      stopped = true; // user pressed stop — not an error
    } else {
      const httpMatch = /HTTP (\d+)|"status":\s*(\d+)/.exec(String(e.message || ""));
      const msg = /failed to fetch|networkerror|load failed/i.test(String(e.message || e))
        ? t("errTts")
        : (httpMatch ? t("errTtsHttp")(httpMatch[1] || httpMatch[2]) : t("errTts"));
      showError(msg, String(e.message || e), play);
    }
  } finally {
    stopDurTimer();
    setPlayingUI(false);
    setStatus(lastError ? "statusIdle" : (stopped ? "statusStopped" : "statusDone"));
  }
}

playBtn.addEventListener("click", () => {
  if (playing) {
    player.stop(); // aborts the fetch; play()'s finally handles UI teardown
  } else {
    play();
  }
});

/* ── boot ─────────────────────────────────────────────────────────── */
$("back-link").href = `//${window.location.hostname}:8700/`;
$("lang-btn").addEventListener("click", () => {
  lang = lang === "zh" ? "en" : "zh";
  localStorage.setItem("slv-demo-lang", lang);
  applyStaticI18n();
});

applyStaticI18n();
probeCapabilities();

// Inline model switch — voice clone rides the TTS engine, so pin to ["tts"].
createModelSwitchPanel($("switch-panel"), { kinds: ["tts"] });
