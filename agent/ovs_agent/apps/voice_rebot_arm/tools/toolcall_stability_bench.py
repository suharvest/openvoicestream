"""Tool-call stability bench for the reBot voice-arm demo.

WHY THIS EXISTS
---------------
Concern: does the small LLM (Qwen3-4B-AWQ on edge-llm) lose tool-calling
accuracy after many calls in a demo session, and should we cap conversation
history?

Architectural finding (voxedge/engine/conversation.py:910/929/978): in
server-loop mode the LLM turn is built FRESH every utterance —
``[{system}, {user}]`` — with NO persistent cross-turn history member. So the
"history accumulates → model mimics history → stops emitting tool_calls"
failure (KNOWN_ISSUES ISSUE-001) is structurally impossible in production, and
a history cap is moot there. What remains is the model's per-command
tool-selection reliability, which is independent of turn count.

This bench measures exactly that. It replays the production server-loop request
shape (the real system prompt + the 8 advertised tool schemas) against an
edge-llm endpoint for a matrix of [command × paraphrase] × N repeats, and:
  * scores tool-selection accuracy (right tool name) + arg correctness,
  * checks determinism (same input → same tool across repeats),
  * runs a long SEQUENTIAL mixed sequence to expose any server-side KV/prefix
    drift (if accuracy were turn-dependent, it would show here),
  * lists every failure (wrong tool / no tool / wrong args) so the presenter
    knows which phrasings are fragile.

It is model-swappable (``--base-url`` / ``--model``) so the SAME matrix can be
run against Qwen3-4B-AWQ (current), Qwen3.5-4B-GDN, and the GDN-MTP variant to
pick the most demo-stable model.

Run inside a container on the voice-arm network (reaches edge-llm:8000), e.g.:
  docker exec voice-rebot-arm python /home/seeed/toolcall_stability_bench.py \
    --base-url http://edge-llm:8000/v1 --model Qwen/Qwen3-4B-AWQ --repeats 5
Stdlib only (urllib) so it runs in any python container.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import defaultdict


# ── production system prompt (voice_rebot_arm/config.yaml, v5 2026-06-12) ──
SYSTEM_PROMPT = (
    'You are a voice controller for a reBot B601-DM robotic arm.\n'
    'The user speaks commands aloud and you reply via TTS.\n'
    '\n'
    'Behaviour:\n'
    '- When the user clearly requests a motion, call the matching tool\n'
    '  DIRECTLY — do NOT write any text before or alongside the call. The\n'
    '  system already plays a spoken acknowledgement the moment the tool\n'
    '  starts, so any text you add would be spoken ON TOP of it ("好的"\n'
    '  twice). Examples:\n'
    '    User: "挥手"       → call wave()\n'
    '    User: "回到原位"   → call go_home()\n'
    '    User: "回家"       → call go_home()\n'
    '       (the arm returning to its home pose — a command, not chit-chat)\n'
    '    User: "张开夹爪"   → call open_gripper()\n'
    '    User: "灰手"       → call wave()\n'
    "       (ASR misheard '挥手'; clearly a command, so recover it)\n"
    '    User: "把盒子放回去" → call put_down()\n'
    '       (putting DOWN the held object — even though an object is named,\n'
    '        this is put_down, NOT a new grasp)\n'
    '    User: "放回原位"     → call put_down()\n'
    '    User: "放回去"       → call put_down()\n'
    "       ('放' = release the HELD object; '原位' / '去' only say WHERE it\n"
    "        goes. The verb '放' decides put_down — do NOT pick go_home just\n"
    "        because '原位' looks like '回到原位'. go_home moves the EMPTY\n"
    '        arm home; put_down releases what the arm is holding.)\n'
    '  AFTER a motion tool returns, do NOT speak either — reply with an\n'
    '  EMPTY message (no words at all). The system plays a completion tone\n'
    '  when the motion finishes; any text you add only delays the next\n'
    '  command. You only speak words when NO tool is called (questions,\n'
    '  chit-chat, refusals).\n'
    "- To pick the right tool, match the user's INTENT against the trigger\n"
    "  words in each tool's description. Trigger examples (not exhaustive):\n"
    "    '挥手' / '挥挥手' / '打招呼' → wave\n"
    "    '回到原位' / '回家' / '归位' / '复位' / 'home' / 'reset' → go_home\n"
    "       (ONLY when the arm just moves home and nothing is placed — if the\n"
    "        user says '放' something, it is put_down, NOT go_home)\n"
    "    '张开夹爪' / '松开' / '打开夹爪' → open_gripper\n"
    "    '闭合夹爪' / '夹紧' / '合上夹爪' → close_gripper\n"
    "    '指向' / '指一下' → point_at\n"
    "    '抓/拿起/夹住 + 物体名' → grasp_object\n"
    "    '找/搜索 + 物体名' → search_object\n"
    "    '放下' / '放回去' / '放回原位' / '放好' / '放下来' → put_down\n"
    '  Never substitute a semantically-similar tool for a different action\n'
    "  the user named — if the user names an action no tool supports ('点头',\n"
    "  '转圈', '跳舞', 'nod', 'spin'), call NO tool and reply that you don't\n"
    '  have that action. Example:\n'
    '    User: "点头"       → (no tool) "我不会点头这个动作。"\n'
    "- If the user names an OBJECT to grab/pick/hold ('夹住这个盒子',\n"
    "  '把箱子拿起来', 'grab the box'), that is grasp_object — the\n"
    '  camera-guided pick. close_gripper is ONLY for closing the empty\n'
    '  gripper when NO object is named. EXCEPTION: putting an object DOWN\n'
    "  ('把盒子放回去', '放下盒子', 'put the box back/down') is put_down\n"
    '  even when the object is named — the verb decides, not the noun.\n'
    "- Commands often come wrapped in politeness or filler ('那个，麻烦你\n"
    "  挥手好吗', '请', '帮我', '一下', '好吗'). Strip the wrapping — the\n"
    '  command inside still counts and the tool MUST be called.\n'
    '- The text comes from speech recognition and may contain NEAR-HOMOPHONE\n'
    "  mishearings of a trigger (e.g. '灰手'/'挥首' for '挥手', '必合夹爪'\n"
    "  for '闭合夹爪', '加紧' for '夹紧'). If the text is clearly a COMMAND\n"
    '  and sounds like a trigger, treat it as that trigger and call that\n'
    '  tool.\n'
    '- NOT every sentence containing a trigger word is a command. Questions\n'
    '  or talk ABOUT a command — asking how to say it in another language\n'
    "  ('挥手用英语怎么说'), what it means, how it is spelled — are NOT\n"
    '  requests to move. Answer in words and call NO tool.\n'
    '- If the text is neither a clear command nor a close mishearing of one\n'
    '  (chit-chat, unrelated questions), DO NOT call any tool — just reply\n'
    '  normally. If it sounds like a garbled command you cannot map to\n'
    '  exactly ONE trigger, reply "没听清，请再说一次。" / "Sorry, I didn\'t\n'
    '  catch that."\n'
    '\n'
    '/no_think'
)


def _fn(name: str, desc: str, params: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": params or {"type": "object", "properties": {}},
        },
    }


# ── the 8 advertised tools (actions.yaml descriptions + grasp catalog +
#    builtins), reconstructed to match what the agent advertises ──────────
_GRASP_CATALOG = ["box", "cardboard box", "carton", "package"]
_GRASP_DESC = (
    "Pick up / grasp an object using the camera-guided arm when the user asks "
    "to grab/pick something up ('抓','拿起','夹起','抓取','grab','pick up'). "
    "object_name MUST be exactly one of these catalog labels: ["
    + ", ".join(repr(c) for c in _GRASP_CATALOG)
    + "]. Map the user's spoken object to the closest catalog label and pass "
    "that English label verbatim (e.g. user says '抓盒子'/'把箱子拿起来' -> "
    "object_name='box'). Do NOT pass the user's Chinese words; the detector "
    "only knows the catalog labels above."
)

TOOLS = [
    _fn("go_home", 'Return the arm to its home / ready position. Triggers: "回到原位", "回家", "归位", "go home", "home", "reset position".'),
    _fn("open_gripper", 'Open the gripper / release. Triggers: "张开夹爪", "松开", "打开夹爪", "把夹爪打开", "夹爪打开", "open gripper", "release", "let go".'),
    _fn("close_gripper", 'Close the gripper / grasp. Triggers: "闭合夹爪", "夹紧", "抓住", "合上夹爪", "关闭夹爪", "close gripper", "grasp", "grab".'),
    _fn("wave", 'Wave hello by swinging the arm side to side. Triggers: "挥手", "挥一下手", "挥挥手", "打招呼", "打个招呼", "wave", "say hi".'),
    _fn("point_at", 'Point forward at an object. Triggers: "指向", "指一下", "point at", "point", "show me". Do not use for head nodding: "点头" is unsupported.'),
    _fn("grasp_object", _GRASP_DESC, {
        "type": "object",
        "properties": {"object_name": {"type": "string", "description": "Catalog label of the object to grasp."}},
        "required": ["object_name"],
    }),
    _fn("search_object", "Search for / locate an object by sweeping the camera around when the user asks to FIND or LOOK FOR something but not (yet) pick it up ('找一下','找找','搜索','看看有没有','find','look for','search for'). The arm scans several viewpoints, stops when it sees the object and points at it WITHOUT grasping. object_name MUST be exactly one of these catalog labels: ['box', 'cardboard box', 'carton', 'package'].", {
        "type": "object",
        "properties": {"object_name": {"type": "string", "description": "Catalog label of the object to find."}},
        "required": ["object_name"],
    }),
    _fn("put_down", "Put down / place / release the object currently held by the gripper. The arm sets it back down at the spot it was picked up from (so the camera can find it again), opens the gripper, and returns home. Use this whenever the user wants the held object put down or returned, EVEN IF they name the object — including '放回原位' (the '放' verb means release the held object; '原位' only says where, do NOT confuse with go_home). Triggers: '放下', '放下来', '放回去', '放回原位', '放好', '把盒子放回去', '放下盒子', '放回原处', '放到桌上', '把它放下', 'put it down', 'put down', 'put the box back', 'place it', 'set it down', 'drop it', 'release it'."),
    _fn("time_now", "Return the current local time as ISO 8601."),
    _fn("set_mode", "Switch the agent to a different mode.", {
        "type": "object",
        "properties": {"mode_name": {"type": "string"}},
        "required": ["mode_name"],
    }),
]


# ── test matrix: (utterance, expected_tool_or_None, expected_arg_substr) ─
# expected_tool=None means "no tool should fire" (chit-chat / out-of-scope).
MATRIX = [
    # wave
    ("挥手", "wave", None),
    ("挥挥手", "wave", None),
    ("挥一下手", "wave", None),
    ("跟大家打个招呼", "wave", None),
    # go_home
    ("回到原位", "go_home", None),
    ("回家", "go_home", None),
    ("归位", "go_home", None),
    ("复位", "go_home", None),
    # open_gripper
    ("张开夹爪", "open_gripper", None),
    ("打开夹爪", "open_gripper", None),
    ("把夹爪松开", "open_gripper", None),
    # close_gripper
    ("闭合夹爪", "close_gripper", None),
    ("夹紧", "close_gripper", None),
    ("合上夹爪", "close_gripper", None),
    # point_at
    ("指向那个物体", "point_at", None),
    ("指一下", "point_at", None),
    # grasp_object (vision grasp) — note the 抓住/抓 trigger collision with close_gripper
    ("抓盒子", "grasp_object", "box"),
    # put_down — the held-object return path; MUST win over grasp_object even
    # when the object is named (real-machine miss 2026-06-12: '把盒子放回去'
    # fired NO tool under v5).
    ("放下盒子", "put_down", None),
    ("把盒子放回去", "put_down", None),
    ("放回去", "put_down", None),
    ("put the box back", "put_down", None),
    # '放回原位' DISAMBIGUATION: '放' (place the held object) must win over
    # '原位' resembling go_home's '回到原位' — real-machine miss 2026-06-13,
    # "放回原位" fired go_home (arm went home still holding the box).
    ("放回原位", "put_down", None),
    ("把盒子放回原位", "put_down", None),
    ("放好", "put_down", None),
    # search_object
    ("找一下盒子", "search_object", "box"),
    ("看看有没有盒子", "search_object", "box"),
    ("把盒子抓起来", "grasp_object", "box"),
    ("夹起盒子", "grasp_object", "box"),
    ("抓取盒子", "grasp_object", "box"),
    ("把那个箱子拿起来", "grasp_object", "box"),
    # out-of-scope / should NOT fire a motion tool
    ("你好", None, None),
    ("今天天气怎么样", None, None),
]


# ── HARD matrix: demo-realistic robustness. Each entry is
#    (utterance, expected_tool_or_None, expected_arg_substr, category).
# expected_tool=None → no motion tool should fire (chit-chat / unsupported).
# For ASR-homophone / truncation rows the "expected" is the demo-desired
# recovery; failures are later split into DANGEROUS (a different motion fired)
# vs SAFE (no tool → presenter just repeats).
MOTION_TOOLS = {"wave", "go_home", "open_gripper", "close_gripper", "point_at", "grasp_object"}
HARD_MATRIX = [
    # colloquial / indirect phrasings that still embed the trigger word
    ("帮我挥个手", "wave", None, "colloquial"),
    ("挥一挥手", "wave", None, "colloquial"),
    ("胳膊回到原位", "go_home", None, "colloquial"),
    ("请把夹爪张开", "open_gripper", None, "colloquial"),
    ("把夹爪闭合一下", "close_gripper", None, "colloquial"),
    ("帮我把盒子拿起来", "grasp_object", "box", "colloquial"),
    # polite / distractor wrappers around the literal trigger
    ("那个，麻烦你挥手好吗", "wave", None, "wrapper"),
    ("现在请回到原位吧", "go_home", None, "wrapper"),
    ("嗯…你帮我把夹爪松开", "open_gripper", None, "wrapper"),
    # ASR homophone / near-homophone errors (what STT may actually emit)
    ("灰手", "wave", None, "asr_homophone"),
    ("挥首", "wave", None, "asr_homophone"),
    ("回到原味", "go_home", None, "asr_homophone"),
    ("张开夹抓", "open_gripper", None, "asr_homophone"),
    ("必合夹爪", "close_gripper", None, "asr_homophone"),
    ("加紧", "close_gripper", None, "asr_homophone"),
    ("抓河子", "grasp_object", "box", "asr_homophone"),
    # truncated / partial (ASR cut the tail)
    ("张开", "open_gripper", None, "truncated"),
    ("回原位", "go_home", None, "truncated"),
    ("挥", "wave", None, "truncated"),
    # English / mixed — on-site may be spoken in English
    ("wave", "wave", None, "english"),
    ("go home please", "go_home", None, "english"),
    ("grab the box", "grasp_object", "box", "english"),
    ("wave your hand", "wave", None, "english"),
    ("open the gripper", "open_gripper", None, "english"),
    ("close the gripper", "close_gripper", None, "english"),
    ("pick up the box", "grasp_object", "box", "english"),
    ("point at it", "point_at", None, "english"),
    ("reset to home position", "go_home", None, "english"),
    # intent collisions (抓住 is a close_gripper trigger, but a box is named)
    ("抓住盒子", "grasp_object", "box", "collision"),
    ("夹住这个盒子", "grasp_object", "box", "collision"),
    # traps — must NOT fire a motion tool
    ("点头", None, None, "trap_unsupported"),
    ("转个圈", None, None, "trap_unsupported"),
    ("你叫什么名字", None, None, "trap_chitchat"),
    ("给我讲个笑话", None, None, "trap_chitchat"),
    ("挥手用英语怎么说", None, None, "trap_meta"),
]


# ── trigger-guard simulation (faithful copy of app_base._server_tool_
#    trigger_guard_error + _extract/_normalize_tool_trigger_phrases). The guard
#    blocks a tool call whose user text contains none of that tool's declared
#    trigger phrases. Tools with no "Triggers:" in their description (grasp_object,
#    builtins) yield no phrases → never blocked (the scoped exemption). ──────
import re as _re
import unicodedata as _ud

GUARD_EXEMPT = {"grasp_object", "time_now", "set_mode"}


def _norm(text: str) -> str:
    return "".join(ch.lower() for ch in (text or "") if _ud.category(ch)[0] in {"L", "N"})


def _phrases(desc: str) -> list[str]:
    m = _re.search(r"(?:Triggers?|Trigger words?)\s*:\s*([^.。]*)", desc or "", _re.I | _re.S)
    if not m:
        return []
    return [p.strip() for p in _re.findall(r"""["']([^"']+)["']""", m.group(1)) if p.strip()]


_TOOL_DESC = {t["function"]["name"]: t["function"]["description"] for t in TOOLS}


def guard_blocks(user_text: str, tool: str | None) -> bool:
    """True if the scoped trigger guard would BLOCK this tool call."""
    if tool is None or tool in GUARD_EXEMPT:
        return False
    ph = _phrases(_TOOL_DESC.get(tool, ""))
    if not ph:
        return False
    nt = _norm(user_text)
    return not any(_norm(p) in nt for p in ph if _norm(p))


def call_llm(base_url: str, model: str, text: str, timeout: float = 30.0) -> dict:
    """One production-shaped chat/completions call; return parsed result."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.0,
        "stream": False,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode())
    dt = time.time() - t0
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    tool = None
    arg = None
    if tcs:
        tool = tcs[0]["function"]["name"]
        try:
            arg = json.dumps(json.loads(tcs[0]["function"].get("arguments") or "{}"), ensure_ascii=False)
        except Exception:
            arg = tcs[0]["function"].get("arguments")
    return {"tool": tool, "arg": arg, "text": (msg.get("content") or "")[:60], "latency_s": round(dt, 2)}


def score(exp_tool, exp_arg, got_tool, got_arg) -> bool:
    if exp_tool is None:
        return got_tool is None
    if got_tool != exp_tool:
        return False
    if exp_arg is not None:
        return exp_arg in (got_arg or "")
    return True


def run_hard(args) -> int:
    """Demo-realistic robustness matrix with failure classified by demo impact."""
    print(f"=== HARD robustness bench :: model={args.model} base={args.base_url} repeats={args.repeats} ===\n")
    by_cat = defaultdict(lambda: [0, 0])
    dangerous = []   # a motion command/trap → a DIFFERENT motion tool fired (arm moves wrong)
    safe_miss = []   # a motion command → no tool (arm doesn't move; presenter repeats)
    spurious = []    # a trap (no-tool expected) → a motion tool fired (arm moves unexpectedly)
    nondet = []
    ok_total = tot = 0
    for (text, exp_tool, exp_arg, cat) in HARD_MATRIX:
        seen_tools = set()
        row_ok = 0
        last = None
        for _ in range(args.repeats):
            try:
                r = call_llm(args.base_url, args.model, text)
            except Exception as e:
                r = {"tool": "ERROR:" + type(e).__name__, "arg": str(e)[:60], "text": "", "latency_s": 0}
            last = r
            seen_tools.add(r["tool"])
            tot += 1
            if score(exp_tool, exp_arg, r["tool"], r["arg"]):
                row_ok += 1
                ok_total += 1
        by_cat[cat][0] += row_ok
        by_cat[cat][1] += args.repeats
        if len(seen_tools) > 1:
            nondet.append((text, exp_tool, sorted(map(str, seen_tools))))
        # classify the row by its modal (last) outcome for demo-impact triage
        got = last["tool"] if last else None
        if exp_tool is None:  # trap
            if got in MOTION_TOOLS:
                spurious.append((cat, text, got))
        else:  # motion command expected
            if got != exp_tool:
                if got in MOTION_TOOLS:
                    dangerous.append((cat, text, exp_tool, got))
                else:  # None / chit-chat reply / error
                    safe_miss.append((cat, text, exp_tool, got))

    print("── accuracy by category ──")
    for c in sorted(by_cat):
        p, t = by_cat[c]
        print(f"  {c:18s} {p}/{t}  ({100*p//max(t,1)}%)")
    print(f"\nOVERALL HARD: {ok_total}/{tot} ({100*ok_total//max(tot,1)}%)")

    print("\n── DEMO-IMPACT TRIAGE ──")
    print(f"  ⛔ DANGEROUS (command → WRONG motion fired, arm moves wrong): {len(dangerous)}")
    for cat, text, exp, got in dangerous:
        print(f"       [{cat}] {text!r}: expected {exp} → fired {got}")
    print(f"  ⚠️  SPURIOUS (trap → motion fired, arm moves unexpectedly): {len(spurious)}")
    for cat, text, got in spurious:
        print(f"       [{cat}] {text!r}: fired {got} (should be no-tool)")
    print(f"  ✅ SAFE-MISS (command → no tool, arm still, presenter repeats): {len(safe_miss)}")
    for cat, text, exp, got in safe_miss:
        print(f"       [{cat}] {text!r}: expected {exp} → {got}")
    if nondet:
        print(f"\n  non-deterministic rows ({len(nondet)}):")
        for text, exp, seen in nondet:
            print(f"     {text!r} (exp {exp}): {seen}")
    else:
        print("\n  ✓ deterministic across all repeats")

    if args.guard:
        # Re-derive the per-row modal tool once more (cheap: 1 call/row) and
        # apply the scoped trigger guard to show the NET demo outcome.
        print("\n══ SCOPED TRIGGER-GUARD net effect (5 fixed-trigger motions guarded; "
              "grasp_object + builtins exempt) ══")
        g_correct = g_blocked_valid = g_saved_dangerous = g_saved_spurious = 0
        g_still_dangerous = g_still_spurious = 0
        blocked_valid_rows = []
        for (text, exp_tool, exp_arg, cat) in HARD_MATRIX:
            try:
                r = call_llm(args.base_url, args.model, text)
            except Exception:
                r = {"tool": None, "arg": None}
            got = r["tool"]
            blocked = guard_blocks(text, got)
            if exp_tool is None:  # trap
                if got in MOTION_TOOLS:
                    if blocked:
                        g_saved_spurious += 1
                    else:
                        g_still_spurious += 1
            else:  # command
                correct = score(exp_tool, exp_arg, got, r["arg"])
                if correct:
                    if blocked:
                        g_blocked_valid += 1
                        blocked_valid_rows.append((cat, text, exp_tool))
                    else:
                        g_correct += 1
                else:  # wrong tool
                    if got in MOTION_TOOLS and not blocked:
                        g_still_dangerous += 1
                    elif got in MOTION_TOOLS and blocked:
                        g_saved_dangerous += 1
        print(f"  ✅ valid command, fires correctly (guard passes): {g_correct}")
        print(f"  🛡️  WAS dangerous wrong-motion → now BLOCKED (safe no-op): {g_saved_dangerous}")
        print(f"  🛡️  WAS spurious trap-motion → now BLOCKED (safe no-op): {g_saved_spurious}")
        print(f"  ⛔ still-dangerous (wrong motion still fires): {g_still_dangerous}")
        print(f"  ⚠️  still-spurious (trap motion still fires): {g_still_spurious}")
        print(f"  🚧 valid command BLOCKED by guard (arm won't move — must use exact trigger): {g_blocked_valid}")
        for cat, text, exp in blocked_valid_rows:
            print(f"       [{cat}] {text!r} (wanted {exp}) — paraphrase lacks the literal trigger")
        print("\n  NET: guard converts dangerous/spurious motion into a safe no-op, at the cost of")
        print("  blocking paraphrases that omit the literal trigger. For a SCRIPTED demo using the")
        print("  exact supported commands, blocked-valid count above is the only trade.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://edge-llm:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-AWQ")
    ap.add_argument("--repeats", type=int, default=5, help="repeats per utterance (determinism)")
    ap.add_argument("--sequence", type=int, default=60, help="length of the long mixed-sequential drift run")
    ap.add_argument("--hard", action="store_true", help="run the demo-realistic HARD robustness matrix instead")
    ap.add_argument("--guard", action="store_true", help="also report the scoped trigger-guard net effect (hard mode)")
    ap.add_argument(
        "--system-prompt-file",
        help="override the baked-in production system prompt (A/B prompt "
        "engineering: validate a candidate prompt here BEFORE editing the "
        "app config)",
    )
    args = ap.parse_args()

    if args.system_prompt_file:
        global SYSTEM_PROMPT
        with open(args.system_prompt_file, encoding="utf-8") as fh:
            SYSTEM_PROMPT = fh.read().strip()
        print(f"[system prompt overridden from {args.system_prompt_file}]")

    if args.hard:
        return run_hard(args)

    print(f"=== tool-call stability bench :: model={args.model} base={args.base_url} ===")
    print(f"system_prompt_len={len(SYSTEM_PROMPT)} tools={len(TOOLS)} matrix={len(MATRIX)} repeats={args.repeats}\n")

    # Phase 1: accuracy matrix + determinism (repeat each utterance N times)
    per_tool = defaultdict(lambda: [0, 0])  # expected_tool -> [pass, total]
    failures = []
    nondet = []
    overall_pass = overall_total = 0
    lat = []
    for (text, exp_tool, exp_arg) in MATRIX:
        got = []
        for _ in range(args.repeats):
            try:
                r = call_llm(args.base_url, args.model, text)
            except Exception as e:
                r = {"tool": "ERROR:" + type(e).__name__, "arg": str(e)[:80], "text": "", "latency_s": 0}
            got.append(r)
            lat.append(r["latency_s"])
            ok = score(exp_tool, exp_arg, r["tool"], r["arg"])
            key = exp_tool or "(no-tool)"
            per_tool[key][1] += 1
            overall_total += 1
            if ok:
                per_tool[key][0] += 1
                overall_pass += 1
            else:
                failures.append((text, exp_tool, exp_arg, r["tool"], r["arg"]))
        tools_seen = {g["tool"] for g in got}
        if len(tools_seen) > 1:
            nondet.append((text, exp_tool, sorted(map(str, tools_seen))))

    print("── per-command accuracy ──")
    for k in sorted(per_tool):
        p, t = per_tool[k]
        print(f"  {k:14s} {p}/{t}  ({100*p//max(t,1)}%)")
    print(f"\nOVERALL: {overall_pass}/{overall_total} ({100*overall_pass//max(overall_total,1)}%)")
    if lat:
        s = sorted(lat)
        print(f"latency_s: p50={s[len(s)//2]:.2f} p90={s[int(len(s)*0.9)]:.2f} max={s[-1]:.2f}")

    if nondet:
        print(f"\n── NON-DETERMINISTIC inputs ({len(nondet)}) — same text, different tool across repeats ──")
        for text, exp, seen in nondet:
            print(f"  {text!r} (exp {exp}): saw {seen}")
    else:
        print("\n✓ deterministic: every utterance produced the same tool across all repeats")

    if failures:
        print(f"\n── FAILURES ({len(failures)}) ──")
        seen = set()
        for text, exp_tool, exp_arg, got_tool, got_arg in failures:
            sig = (text, got_tool, got_arg)
            if sig in seen:
                continue
            seen.add(sig)
            print(f"  {text!r}: expected {exp_tool}({exp_arg}) → got {got_tool}({got_arg})")

    # Phase 2: long sequential mixed run — if accuracy degraded with turn
    # count (it shouldn't, since production is stateless), it surfaces here.
    print(f"\n── long sequential run ({args.sequence} calls, mixed order) — drift check ──")
    seq = [(MATRIX[i % len(MATRIX)]) for i in range(args.sequence)]
    win = []  # rolling correctness
    first_half = second_half = 0
    half = args.sequence // 2
    for idx, (text, exp_tool, exp_arg) in enumerate(seq):
        try:
            r = call_llm(args.base_url, args.model, text)
            ok = score(exp_tool, exp_arg, r["tool"], r["arg"])
        except Exception:
            ok = False
        win.append(ok)
        if idx < half:
            first_half += int(ok)
        else:
            second_half += int(ok)
    print(f"  first-half  {first_half}/{half} ({100*first_half//max(half,1)}%)")
    print(f"  second-half {second_half}/{args.sequence-half} ({100*second_half//max(args.sequence-half,1)}%)")
    print("  → if these two halves match, accuracy is position/turn-INDEPENDENT (no degradation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
