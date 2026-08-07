#!/usr/bin/env python3
"""OVS 全流程回归基线：TTS 数字规范化 / TTS→ASR 闭环 / 连续 3 轮无槽位泄漏。

    python3 scripts/regress_pipeline.py [host:port] [容器名]
    # 默认 127.0.0.1:8621；给了容器名才会跑第 3 项（插件软链接 dlopen）

换镜像后必跑。三项都 PASS 才算「功能不衰退」。

判据说明：
- TTS 用**字节长度**而非 md5 比对。matcha_trt 是非确定性的，同一文本两次合成
  md5 就不同，但长度稳定 —— 长度相等即音素序列相同，正是要验的归一化结果。
- ASR 用 TTS 自己合成的音频回灌，无需外部语料，也顺带覆盖了 TTS 输出可用性。
- 连续 3 轮跑满是槽位泄漏的探针：泄漏时第 2/3 轮会挂起或返回空。
- /asr/stream 的协议：二进制帧 = 音频，**空二进制帧 = finalize**。没有
  begin/end 事件，发 JSON 只能是 {"command": "reset"|"end_utterance"}。
"""
import sys, json, wave, io, urllib.request, asyncio, audioop
HOSTPORT = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8621"
CONTAINER = sys.argv[2] if len(sys.argv) > 2 else None
BASE = f"http://{HOSTPORT}"
WS = f"ws://{HOSTPORT}"
ok = True
def fail(m): 
    global ok; ok = False; print("   FAIL", m)

def tts(text):
    req = urllib.request.Request(BASE + "/tts", data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def wav_to_pcm16k(raw):
    w = wave.open(io.BytesIO(raw)); sr = w.getframerate(); ch = w.getnchannels()
    pcm = w.readframes(w.getnframes()); sw = w.getsampwidth()
    if ch == 2: pcm = audioop.tomono(pcm, sw, 1, 1)
    if sr != 16000: pcm, _ = audioop.ratecv(pcm, sw, 1, sr, 16000, None)
    return pcm

# ── 1. TTS 数字规范化（长度判据；TTS 非确定性，md5 不可用）──────
print("== 1. TTS 数字规范化 ==")
try:
    a, b = tts("库存945个"), tts("库存九百四十五个")
    print(f"   阿拉伯 {len(a)}B / 中文 {len(b)}B")
    if len(a) == len(b): print("   PASS 时长一致 → 归一化生效")
    else: fail(f"长度不等，数字未被正确读出")
    c = tts("0002")
    print(f"   '0002' -> {len(c)}B", "PASS" if len(c) > 4000 else "FAIL 疑似静音")
    if len(c) <= 4000: fail("纯数字静音")
except Exception as e: fail(e)

# ── 2+3. TTS→ASR 闭环 × 3 轮（vad=none 完整分段 + 槽位不泄漏）──
print("== 2. TTS→ASR 闭环 × 3 轮 (vad=none) ==")
SENT = "帮我查一下M6螺母的库存"
KEYS = ["帮我查", "螺母", "库存"]
try:
    import websockets
except ImportError:
    fail("缺 websockets"); websockets = None

async def rt(pcm):
    url = f"{WS}/asr/stream?vad=none&sample_rate=16000"
    out = ""
    async with websockets.connect(url, open_timeout=30, max_size=None) as ws:
        step = 16000 * 2 // 10   # 100ms
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i:i+step]); await asyncio.sleep(0.005)
        await ws.send(b"")                       # 空二进制帧 = finalize
        while True:
            m = await asyncio.wait_for(ws.recv(), timeout=60)
            if isinstance(m, bytes): continue
            d = json.loads(m)
            if d.get("text"): out = d["text"]
            if d.get("is_final"): break
    return out

if websockets:
    try:
        pcm = wav_to_pcm16k(tts(SENT))
        print(f"   合成 {len(pcm)/32000:.2f}s @16k")
        for i in range(1, 4):
            t = asyncio.run(rt(pcm))
            miss = [k for k in KEYS if k not in t]
            print(f"   轮{i}: {t!r}  {'PASS' if not miss else 'FAIL 缺 '+str(miss)}")
            if miss: fail(f"轮{i} 识别不全")
    except Exception as e: fail(e)

# ── 3. 插件软链接可加载（需容器名；HTTP 回归覆盖不到）─────────
# 上面的 ASR 走的是 /opt/edgellm-v091/ 下的插件，与 /opt/edgellm{,-bin}/ 那两条
# 指向 /opt/jv-workers/ 的软链接无关 —— 后者由 SparkTTS / CustomVoice / 旧
# multilang profile 使用。跑过 HTTP 回归不等于验过这些链接，必须单独 dlopen。
if CONTAINER:
    print(f"== 3. 插件软链接 dlopen ({CONTAINER}) ==")
    probe = (
        "import ctypes,os,sys\n"
        "ps=['/opt/edgellm/libNvInfer_edgellm_plugin.so',"
        "'/opt/edgellm/libNvInfer_edgellm_plugin_asr.so',"
        "'/opt/edgellm-bin/libNvInfer_edgellm_plugin.so',"
        "'/opt/edgellm-bin/libNvInfer_edgellm_plugin_asr.so']\n"
        "bad=0\n"
        "for p in ps:\n"
        "    k='symlink' if os.path.islink(p) else 'regular'\n"
        # 每个插件单开一次进程会太慢；同进程加载多个插件会在退出时 double free
        # （符号冲突，实体文件同样复现，与链接无关），故只验证能否打开。
        "    try: ctypes.CDLL(p); print('   OK  %s [%s]'%(p,k))\n"
        "    except Exception as e: bad=1; print('   FAIL %s [%s] %s'%(p,k,e))\n"
        "    if os.path.islink(p) and not os.path.exists(p):\n"
        "        bad=1; print('   FAIL 悬空链接 '+p)\n"
        "sys.exit(bad)\n"
    )
    import subprocess
    r = subprocess.run(["docker", "exec", CONTAINER, "python3", "-c", probe],
                       capture_output=True, text=True)
    print(r.stdout.rstrip() or r.stderr.rstrip()[:300])
    if r.returncode != 0: fail("插件链接不可加载")

h = json.loads(urllib.request.urlopen(BASE + "/health", timeout=10).read())
print(f"== 健康: asr={h.get('asr')} tts={h.get('tts')} ==")
print("\n==== 总体:", "PASS" if ok else "FAIL", "====")
sys.exit(0 if ok else 1)
