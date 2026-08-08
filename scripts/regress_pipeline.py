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

def rms(raw):
    """WAV 的均方根振幅。判静音要看能量，不能看字节数 —— 一段全零的 WAV
    照样有几万字节（codex review 2026-08-08 指出的假阳性）。"""
    w = wave.open(io.BytesIO(raw))
    pcm = w.readframes(w.getnframes())
    if not pcm:
        return 0.0
    return audioop.rms(pcm, w.getsampwidth())


# ── 1. TTS 数字规范化 ────────────────────────────────────────────
# 判据用长度而非 md5：matcha_trt 非确定性，同一文本两次合成 md5 就不同，
# 长度才稳定。但长度相等只是必要条件 —— 真正证明「数字被念出来」要靠把音频
# 回灌 ASR 看转写内容，所以下面加了语义校验。
print("== 1. TTS 数字规范化 ==")
try:
    a, b = tts("库存945个"), tts("库存九百四十五个")
    print(f"   阿拉伯 {len(a)}B / 中文 {len(b)}B")
    if len(a) == len(b):
        print("   PASS 时长一致")
    else:
        fail("长度不等，数字未被正确读出")
    c = tts("0002")
    energy = rms(c)
    ok_energy = energy > 200          # 全零 WAV 的 rms 恰为 0
    print(f"   '0002' -> {len(c)}B rms={energy}  "
          f"{'PASS 有实际音频' if ok_energy else 'FAIL 静音（字节数不足以证明非静音）'}")
    if not ok_energy:
        fail("纯数字静音")
except Exception as e:
    fail(e)

# ── 2+3. TTS→ASR 闭环 × 3 轮（vad=none 完整分段 + 槽位不泄漏）──
print("== 2. TTS→ASR 闭环 × 3 轮 (vad=none) ==")
SENT = "帮我查一下M6螺母的库存"
KEYS = ["帮我查", "螺母", "库存"]
try:
    import websockets
except ImportError:
    fail("缺 websockets"); websockets = None

async def rt(pcm):
    """返回 (final 文本, 错误)。

    只取 **final 帧**的文本，不拿 partial 兜底：早先的写法是「有 text 就记下来」，
    于是服务端返回错误 final 时，上一帧残留的 partial 会让断言照样通过
    （codex review 2026-08-08 指出的假阳性）。
    """
    url = f"{WS}/asr/stream?vad=none&sample_rate=16000"
    final_text, err = None, None
    async with websockets.connect(url, open_timeout=30, max_size=None) as ws:
        step = 16000 * 2 // 10   # 100ms
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i:i+step]); await asyncio.sleep(0.005)
        await ws.send(b"")                       # 空二进制帧 = finalize
        while True:
            m = await asyncio.wait_for(ws.recv(), timeout=60)
            if isinstance(m, bytes): continue
            d = json.loads(m)
            if d.get("error"):
                err = d["error"]; break
            if d.get("is_final"):
                final_text = d.get("text") or ""
                break
    return final_text, err

if websockets:
    try:
        wav = tts(SENT)
        pcm = wav_to_pcm16k(wav)
        print(f"   合成 {len(pcm)/32000:.2f}s @16k rms={rms(wav)}")
        for i in range(1, 4):
            t, err = asyncio.run(rt(pcm))
            if err:
                print(f"   轮{i}: 服务端错误 {err!r}"); fail(f"轮{i} 服务端报错"); continue
            if t is None:
                print(f"   轮{i}: 未收到 final 帧"); fail(f"轮{i} 无 final"); continue
            miss = [k for k in KEYS if k not in t]
            print(f"   轮{i}: {t!r}  {'PASS' if not miss else 'FAIL 缺 '+str(miss)}")
            if miss: fail(f"轮{i} 识别不全")
        # 语义校验数字归一化：把「库存945个」的音频回灌 ASR，转写里必须出现
        # 中文数字读法。长度相等只说明时长一样，说不出「数字确实被念了」。
        t945, err945 = asyncio.run(rt(wav_to_pcm16k(tts("库存945个"))))
        spoken = any(k in (t945 or "") for k in ("九百四十五", "945"))
        print(f"   数字语义: {t945!r}  {'PASS' if spoken else 'FAIL 未念出数字'}")
        if not spoken: fail("阿拉伯数字未被念出")
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
        # 这四条路径按设计必须是软链接：换回实体文件意味着去重失效、镜像凭空
        # 胖 141MB，而只验 dlopen 是发现不了的。
        "    if not os.path.islink(p):\n"
        "        bad=1; print('   FAIL 应为软链接却是实体文件 '+p); continue\n"
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

# 健康状态必须参与判定：只打印的话，后端挂了而前面靠缓存/残留侥幸通过时，
# 脚本仍会报 PASS（codex review 2026-08-08）。
h = json.loads(urllib.request.urlopen(BASE + "/health", timeout=10).read())
healthy = bool(h.get("asr")) and bool(h.get("tts"))
print(f"== 健康: asr={h.get('asr')} tts={h.get('tts')} "
      f"{'PASS' if healthy else 'FAIL 后端未就绪'} ==")
if not healthy:
    fail("健康检查未通过")
print("\n==== 总体:", "PASS" if ok else "FAIL", "====")
sys.exit(0 if ok else 1)
