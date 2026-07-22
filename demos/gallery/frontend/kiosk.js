/* SLV demo gallery — kiosk attract loop (native ES module, no build step).
 *
 * State machine (only ever armed when the backend reports kiosk:true):
 *
 *   DISABLED ──setEnabled(true)──▶ WATCHING ──60 s idle──▶ ATTRACT
 *      ▲                             │  ▲                     │
 *      └────setEnabled(false)────────┘  └──any touch/click────┘
 *
 *   WATCHING  idle timer armed; any pointer/key/touch/wheel activity resets it.
 *   ATTRACT   fullscreen carousel over the portal, one slide every 8 s;
 *             any interaction tears the overlay down and returns to WATCHING.
 *
 * Timing knobs (frontend-only; the backend only carries the kiosk flag):
 *   - idle threshold: 60 s default, override with URL param ?kiosk_idle_s=NN
 *   - slide interval:  8 s default, override with URL param ?kiosk_slide_s=NN
 *
 * Zero dependencies; transitions are pure CSS (see "kiosk attract overlay"
 * section in /common/ui.css).
 */

const DEFAULT_IDLE_S = 60;
const DEFAULT_SLIDE_S = 8;

const ACTIVITY_EVENTS = ["pointerdown", "mousemove", "keydown", "touchstart", "wheel"];
const EXIT_EVENTS = ["pointerdown", "keydown", "touchstart"];

function paramSeconds(name, fallbackS) {
  const raw = new URLSearchParams(window.location.search).get(name);
  const n = Number(raw);
  return raw != null && Number.isFinite(n) && n > 0 ? n : fallbackS;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/* opts:
 *   getSlides()  → [{ headline, sub?, metric?, metricLabel? }]  (re-read on
 *                  every attract entry so language/catalog changes apply)
 *   getHint()    → string  ("touch anywhere to start" in the current language)
 *   idleMs / slideMs  optional overrides (tests); URL params win over defaults.
 */
export function createAttractLoop(opts = {}) {
  const idleMs = opts.idleMs ?? paramSeconds("kiosk_idle_s", DEFAULT_IDLE_S) * 1000;
  const slideMs = opts.slideMs ?? paramSeconds("kiosk_slide_s", DEFAULT_SLIDE_S) * 1000;
  const getSlides = opts.getSlides || (() => []);
  const getHint = opts.getHint || (() => "");

  let enabled = false;        // DISABLED vs WATCHING/ATTRACT
  let idleTimer = null;
  let slideTimer = null;
  let overlay = null;         // non-null ⇔ state === ATTRACT
  let slideEls = [];
  let slideIdx = 0;

  /* ── WATCHING: idle detection ────────────────────────────────────────── */

  function armIdleTimer() {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(enterAttract, idleMs);
  }

  function onActivity() {
    if (!enabled || overlay) return; // ATTRACT exits via its own listener
    armIdleTimer();
  }

  /* ── ATTRACT: fullscreen carousel ────────────────────────────────────── */

  function buildOverlay() {
    const slides = getSlides();
    if (!slides.length) return null;

    const root = el("div", "attract-overlay");
    root.setAttribute("role", "button");
    root.setAttribute("aria-label", getHint());
    slideEls = slides.map((s) => {
      const slide = el("div", "attract-slide");
      if (s.metric) {
        const metric = el("div", "attract-metric", s.metric);
        const label = el("div", "attract-metric-label", s.metricLabel || "");
        slide.append(metric, label);
      }
      if (s.headline) slide.append(el("div", "attract-headline", s.headline));
      if (s.sub) slide.append(el("div", "attract-sub", s.sub));
      root.append(slide);
      return slide;
    });
    root.append(el("div", "attract-hint", getHint()));
    return root;
  }

  function showSlide(i) {
    slideIdx = ((i % slideEls.length) + slideEls.length) % slideEls.length;
    slideEls.forEach((s, k) => s.classList.toggle("active", k === slideIdx));
  }

  function enterAttract() {
    if (!enabled || overlay) return;
    overlay = buildOverlay();
    if (!overlay) { armIdleTimer(); return; } // nothing to show yet — keep watching
    document.body.appendChild(overlay);
    document.body.classList.add("attract-active");
    showSlide(0);
    slideTimer = setInterval(() => showSlide(slideIdx + 1), slideMs);
    for (const ev of EXIT_EVENTS) overlay.addEventListener(ev, exitAttract);
  }

  function exitAttract() {
    if (!overlay) return;
    clearInterval(slideTimer);
    slideTimer = null;
    overlay.remove();
    overlay = null;
    slideEls = [];
    document.body.classList.remove("attract-active");
    if (enabled) armIdleTimer(); // back to WATCHING
  }

  /* ── public API ──────────────────────────────────────────────────────── */

  for (const ev of ACTIVITY_EVENTS) {
    window.addEventListener(ev, onActivity, { passive: true });
  }

  return {
    /* Called after every /api/catalog refresh with the backend kiosk flag.
     * Non-kiosk deployments therefore never arm a timer — behavior identical
     * to before this module existed. */
    setEnabled(next) {
      next = !!next;
      if (next === enabled) return;
      enabled = next;
      if (enabled) {
        armIdleTimer();
      } else {
        clearTimeout(idleTimer);
        idleTimer = null;
        exitAttract();
      }
    },
    get state() {
      if (!enabled) return "disabled";
      return overlay ? "attract" : "watching";
    },
  };
}
