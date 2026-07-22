/* SLV demo gallery — shared UI components (native ES module, no build step).
 *
 * Exports:
 *   createStatusPill(container, opts)      → { set(state, text), el }
 *   createMetricCard(container, opts)      → { set(value), reset(), el }
 *   createModelSwitchPanel(container, opts)→ { refresh(), el }
 *
 * All components render into `container` (an Element) and are dependency-free.
 */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function firstSentence(text, maxChars) {
  const cut = text.search(/[。．.!?！？]\s|[。！？](?=\S)/);
  let s = cut >= 0 ? text.slice(0, cut + 1) : text;
  if (s.length > maxChars) s = s.slice(0, maxChars - 1) + "…";
  return s;
}

/* ── status pill ─────────────────────────────────────────────────────────
 * states: "ok" | "warn" | "err" | "busy" | "idle" */
export function createStatusPill(container, { state = "idle", text = "" } = {}) {
  const pill = el("span", "pill", text);
  if (state !== "idle") pill.classList.add(state);
  container.appendChild(pill);
  return {
    el: pill,
    set(newState, newText) {
      pill.classList.remove("ok", "warn", "err", "busy");
      if (newState && newState !== "idle") pill.classList.add(newState);
      if (newText != null) pill.textContent = newText;
    },
  };
}

/* ── big-number metric card ──────────────────────────────────────────────
 * e.g. TTFA `0.42s` updating live. set(0.42) animates a pop pulse. */
export function createMetricCard(container, { label = "", unit = "s", digits = 2, idleText = "—" } = {}) {
  const card = el("div", "metric-card");
  const value = el("div", "metric-value idle", idleText);
  const labelEl = el("div", "metric-label", label);
  card.append(value, labelEl);
  container.appendChild(card);

  return {
    el: card,
    set(num) {
      value.classList.remove("idle");
      value.textContent = Number(num).toFixed(digits);
      const u = el("span", "metric-unit", unit);
      value.appendChild(u);
      card.classList.remove("pulse");
      void card.offsetWidth; // restart animation
      card.classList.add("pulse");
    },
    reset() {
      value.classList.add("idle");
      value.textContent = idleText;
    },
    setLabel(text) { labelEl.textContent = text; },
  };
}

/* ── model switch panel ──────────────────────────────────────────────────
 * Shared by the gallery portal and every demo page header.
 *
 * opts:
 *   api            base path of the demo backend API (default "/api")
 *   kinds          array of switchable kinds, e.g. ["asr"] / ["tts"] /
 *                  ["asr","tts"] (default both). A single kind hides the
 *                  ASR/TTS toggle row and pins the panel to that kind — used
 *                  by demos that only exercise one side (asr-caption → ["asr"],
 *                  tts-playground → ["tts"]). The gallery passes both (tabs).
 *   strings        i18n overrides, see DEFAULT_STRINGS below
 *   pollMs         status poll interval during a switch (default 600)
 *   onSwitched(kind, result)  optional callback after a settled switch
 *
 * Expects the demo backend to expose:
 *   GET  {api}/profiles  → { profiles: [{name, description, asr_backend, tts_backend}] }
 *   GET  {api}/status    → { slv: { backend_status: { tts: {state, profile_name, backend_name}, asr: {...} } } }
 *   POST {api}/switch    → passthrough of SLV /admin/backend/reload
 */
const DEFAULT_STRINGS = {
  kindAsr: "ASR 识别",
  kindTts: "TTS 合成",
  switchBtn: "切换模型",
  switching: "切换中…",
  draining: "正在收尾进行中的会话（drain）…",
  reloading: "正在加载新模型（RELOADING）…",
  done: "切换成功",
  rolledBack: "切换失败，已自动回滚到原模型",
  failed: "切换失败",
  unreachable: "语音服务不可达",
  current: "当前",
  noProfiles: "没有可用的模型组合",
  noLoadable: "该设备无可切换的 {kind} 模型",
  noHotReload: "该设备的引擎不支持运行时切换（更换模型需重启容器）",
};

export function createModelSwitchPanel(container, opts = {}) {
  const api = (opts.api || "/api").replace(/\/$/, "");
  const S = { ...DEFAULT_STRINGS, ...(opts.strings || {}) };
  const pollMs = opts.pollMs || 600;
  // Which kinds this demo exposes. A single kind pins the panel to it and
  // hides the toggle row; both keeps the ASR/TTS tabs (gallery default).
  const kinds = Array.isArray(opts.kinds) && opts.kinds.length
    ? opts.kinds.filter((k) => k === "tts" || k === "asr")
    : ["tts", "asr"];
  const singleKind = kinds.length === 1;
  // Hard device trait per kind (e.g. RK NPU engines): once SLV answers
  // hot_reload_not_supported, stop offering the switch for that kind.
  const noReload = { tts: false, asr: false };

  const root = el("div", "switch-panel");
  const kindRow = el("div", "switch-kind");
  const btnTts = el("button", "active", S.kindTts);
  const btnAsr = el("button", "", S.kindAsr);
  kindRow.append(btnTts, btnAsr);
  // Single-kind demos never switch kinds — drop the toggle row entirely.
  if (singleKind) kindRow.style.display = "none";

  const currentLine = el("div", "muted switch-current", "");
  const select = document.createElement("select");
  const desc = el("div", "switch-desc", "");
  const switchBtn = el("button", "btn-primary", S.switchBtn);
  const progress = el("div", "switch-progress");
  const bar = el("div", "bar");
  const progressText = el("span", "", "");
  progress.append(bar, progressText);
  const result = el("div", "switch-result", "");

  root.append(kindRow, currentLine, select, desc, switchBtn, progress, result);
  container.appendChild(root);

  // Default to TTS when available (preserves the gallery's initial tab),
  // otherwise the sole pinned kind.
  let kind = kinds.includes("tts") ? "tts" : kinds[0];
  let profiles = [];
  let loadableFiltered = false;
  let pollTimer = null;

  function setKind(newKind) {
    kind = newKind;
    btnTts.classList.toggle("active", kind === "tts");
    btnAsr.classList.toggle("active", kind === "asr");
    result.className = noReload[kind] ? "switch-result warn" : "switch-result";
    result.textContent = noReload[kind] ? S.noHotReload : "";
    renderCurrent();
    // Each kind has its own loadable set — re-fetch scoped to this kind so the
    // dropdown only lists profiles the device can actually load for it.
    refresh();
  }
  btnTts.addEventListener("click", () => setKind("tts"));
  btnAsr.addEventListener("click", () => setKind("asr"));

  select.addEventListener("change", () => {
    const p = profiles.find((x) => x.name === select.value);
    // Profile descriptions are engineering notes — surface only the first
    // sentence to demo visitors, never the full internals dump.
    desc.textContent = p ? firstSentence(p.description || "", 160) : "";
  });

  let lastStatus = null;
  function renderCurrent() {
    const entry = lastStatus?.slv?.backend_status?.[kind];
    if (!entry) { currentLine.textContent = ""; return; }
    // Prefer the friendly model name (the deduped list marks the loaded one);
    // fall back to the raw backend name when the list hasn't loaded yet.
    const cur = profiles.find((p) => p.current);
    const model = cur?.label || entry.backend_name || "?";
    currentLine.textContent = `${S.current}: ${model} (${entry.state || "?"})`;
  }

  async function refresh() {
    try {
      const [pRes, sRes] = await Promise.all([
        fetch(`${api}/profiles?kind=${encodeURIComponent(kind)}`).then((r) => r.json()),
        fetch(`${api}/status`).then((r) => r.json()),
      ]);
      profiles = pRes.profiles || [];
      loadableFiltered = !!pRes.loadable_filtered;
      lastStatus = sRes;
      select.innerHTML = "";
      if (!profiles.length) {
        // When the SLV answered the loadable pre-flight (loadable_filtered) and
        // still nothing survived, this device genuinely can't switch this kind.
        const msg = loadableFiltered
          ? S.noLoadable.replace("{kind}", kind.toUpperCase())
          : S.noProfiles;
        select.appendChild(el("option", "", msg));
        switchBtn.disabled = true;
      } else {
        for (const p of profiles) {
          // Show the MODEL name (Qwen3-ASR / Matcha / …), not the bundle stem;
          // value stays the representative profile the SLV reloads.
          const opt = el("option", "", p.label || p.name);
          opt.value = p.name;
          if (p.current) opt.selected = true;
          select.appendChild(opt);
        }
        switchBtn.disabled = noReload[kind];
        if (noReload[kind]) {
          result.className = "switch-result warn";
          result.textContent = S.noHotReload;
        }
        select.dispatchEvent(new Event("change"));
      }
      renderCurrent();
    } catch (e) {
      result.className = "switch-result err";
      result.textContent = `${S.unreachable}: ${e.message || e}`;
    }
  }

  async function pollUntilSettled() {
    // Poll {api}/status until the manager leaves DRAINING/RELOADING.
    for (;;) {
      await new Promise((r) => { pollTimer = setTimeout(r, pollMs); });
      let state = null;
      try {
        const s = await fetch(`${api}/status`).then((r) => r.json());
        lastStatus = s;
        state = s?.slv?.backend_status?.[kind]?.state || null;
      } catch { /* transient — keep polling */ }
      if (state === "draining") progressText.textContent = S.draining;
      else if (state === "reloading") progressText.textContent = S.reloading;
      else if (state === "ready" || state === "failed" || state === null) return state;
      renderCurrent();
    }
  }

  switchBtn.addEventListener("click", async () => {
    const profile = select.value;
    if (!profile) return;
    switchBtn.disabled = true;
    result.textContent = "";
    result.className = "switch-result";
    progress.classList.add("visible");
    progressText.textContent = S.switching;

    const poller = pollUntilSettled();
    let outcome;
    try {
      const resp = await fetch(`${api}/switch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind, profile }),
      });
      outcome = { httpOk: resp.ok, status: resp.status, body: await resp.json().catch(() => ({})) };
    } catch (e) {
      outcome = { httpOk: false, status: 0, body: { error: String(e) } };
    }
    clearTimeout(pollTimer);
    await Promise.race([poller, Promise.resolve()]); // stop the poll loop next tick
    progress.classList.remove("visible");
    switchBtn.disabled = false;

    const st = outcome.body?.status;
    if (outcome.httpOk && st === "reloaded") {
      result.className = "switch-result ok";
      result.textContent = `${S.done}: ${profile}`;
    } else if (st === "rolled_back") {
      result.className = "switch-result warn";
      result.textContent = S.rolledBack;
    } else if (outcome.body?.detail?.error === "hot_reload_not_supported") {
      // Backend (e.g. RK NPU engines) cannot swap at runtime — a hard
      // device trait, not a transient failure. Say so and stop inviting
      // retries for this kind.
      result.className = "switch-result warn";
      result.textContent = S.noHotReload;
      noReload[kind] = true;
    } else {
      result.className = "switch-result err";
      const detail = outcome.body?.detail?.error || outcome.body?.error || `HTTP ${outcome.status}`;
      result.textContent = `${S.failed}: ${detail}`;
    }
    await refresh();
    if (opts.onSwitched) opts.onSwitched(kind, outcome);
  });

  refresh();
  return { el: root, refresh, setStrings(next) { Object.assign(S, next); switchBtn.textContent = S.switchBtn; btnTts.textContent = S.kindTts; btnAsr.textContent = S.kindAsr; } };
}
