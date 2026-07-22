# voice_rebot_arm — reBot B601-DM 运维 / 验收 Runbook

语音控制 reBot B601-DM 机械臂的 app。复用我们的语音栈（wake / ASR / LLM 工具 / TTS）
+ Actuator ABC 框架；Phase A = 笛卡尔预设动作，Phase B = 去 torch 视觉抓取。

设备：**seeed-orin-nx**（Jetson Orin NX 16G, JetPack 6.2.1, CUDA 12.6, Python 3.10）。
访问：直接 **SSH**（`ssh seeed@seeed-orin-nx`；docker / 串口 / dpkg 需 `sudo`）。
机械臂在 **`/dev/ttyACM1`**（Damiao CAN via DM-serial）。**`/dev/ttyACM0` 是 SO-ARM，勿碰。**

---

## 0. 当前状态（2026-06-03）

- **Phase A — 部署 LIVE 已验**：容器 `voice-rebot-arm` 运行中（server-loop，8 工具 advertise 给 SLV）。
  actuator 连 ttyACM1 读到真实位姿、唤醒词 listening、go_home 通电动作 PASS。
  **未验**：真人对麦说命令（硬件麦无法软件注入，需到场，见 §2）。
- **Phase B — 感知半条链真机验通（dry-run）**：Orbbec 采帧 → YOLOE-seg ONNX 检测
  → 短轴抓取位姿（相机系）全跑通。**未做**：手眼标定 + 监督式真抓取（见 §3）。

---

## 1. 镜像 / 部署速查

- 镜像：`voice-rebot-arm:dev`（设备本地构建；`agent/Dockerfile.rebot-arm`，context=repo root）。
- 重新构建（设备上，前台）：
  ```
  # 在 device：把最新 agent/ 放到 /home/seeed/vra_build/agent/ 后
  docker build -f /home/seeed/vra_build/agent/Dockerfile.rebot-arm -t voice-rebot-arm:dev /home/seeed/vra_build
  ```
- **B601-DM 与 SO-ARM 二选一**（共用唯一 reSpeaker 麦 + 16G 内存，且内存余量极小）：
  ```
  # 上 B601-DM（停 SO-ARM）
  docker stop voice-arm && docker start voice-rebot-arm
  # 切回 SO-ARM
  docker rm -f voice-rebot-arm && docker start voice-arm   # 或 docker stop voice-rebot-arm
  ```
  `seeed-voice`(ASR/TTS) / `edge-llm`(LLM) 常驻共享，**绝不 stop/down**。
- 首次/换 image 起容器（完整 docker run，已含全部 env）：见 §5。
- 日志：设备上 `sudo docker logs --tail 80 voice-rebot-arm`（远程一行：`ssh seeed@seeed-orin-nx sudo docker logs --tail 80 voice-rebot-arm`）。

---

## 2. Phase A 验收（语音 → 预设动作）— 需到场

1. 确认容器在跑且 SO-ARM 已停：`docker ps`（voice-rebot-arm Up、voice-arm 不在）。
2. **桌面清空、人在急停旁。**
3. 对 reSpeaker 麦说：**“Hey Jarvis”**（唤醒）→ 应答后 → 命令：
   - **“回到原位”** → `go_home`
   - **“挥手”** → `wave`（左右摆几下）
   - **“指一下”** → `point_at`
   - **“张开夹爪”** → `open_gripper`（开 6cm）
   - **“夹紧”/“抓住”** → `close_gripper`（grasp 0.2 N·m）
4. 不灵排查：`docker logs --tail 80 voice-rebot-arm`（看 wake 触发 / ASR 文本 / SLV tool_call / actuator 执行）。
5. 也可绕过语音直接测动作（HTTP，仍会动臂，需盯着）：
   `curl -X POST http://localhost:8775/actions/go_home/test`

动作坐标已在真机 IK 标定（`actions.yaml`，安全盒 x∈[0.20,0.34] y∈[±0.14] z∈[0.12,0.34]）。

---

## 3. Phase B 完成步骤（视觉抓取）— 含一次性手眼标定（需到场）

感知链已真机验通（见 §0）。到“完整抓取”还差以下，全是纯工程、无新未知：

### 3.1 grasp 词表 ONNX —— ✅ 已完成（2026-06-04）
已导出 7 类词表（与 reBot demo config 一致，顺序即 `names`）：
`["yellow banana","water bottle","light blue coffee cup","cup","green object","red object","tool"]`
- 设备上：`/home/seeed/perception_dryrun/yoloe-26s-seg-grasp7.onnx`（md5 `71d303c3...`）
- **TRT engine 已预编译缓存**：`/home/seeed/perception_dryrun/trt_cache/`（45MB .engine + timing cache，sm87）。
  编译是在停 edge-llm 腾出内存的窗口做的（冷编译 ~218s）；**缓存后即使内存紧（edge-llm 在跑）
  也能直接加载推理，不再触发编译 OOM**。dry-run 脚本 providers 已带 TRT cache 选项。
- 注意：一次性脚本里 TRT "warm" 跑 ~3.2s 是进程冷启（session 初始化+engine 反序列化）开销；
  **常驻进程（grasp_service）里 session 只建一次，稳态单帧推理是几十 ms 级**，远快于 CPU 1.18s。
  换词表需重导 ONNX + 重建 engine（再开一次腾内存窗口）。
- **ONNX 已发布**：两个 ONNX（`test_yolo_onnx.py` 的 person/bus fixture + 上面的 grasp7 词表）
  已上传至 `https://huggingface.co/harvestsu/yoloe-26s-seg-onnx`，设备 / 复现可从此拉取。

### 3.2 手眼标定（eye-in-hand）→ `hand_eye.npz` ★硬门槛，必须到场
当前**没有** `config/calibration/orbbec_gemini2/hand_eye.npz`，没它相机系位姿转不到机械臂基座系 → **无法抓**。
- 准备：打印 ArUco 板（reBot 仓 `aruco100x100.pdf`），固定在工作台。
- 用 reBot-DevArm-Grasp 仓的 `scripts/collect_handeye_eih.py` + `calibration/hand_eye.py`（TSAI）：
  机械臂带相机走 N 个姿态，每姿态记 (TCP 位姿, 板的相机系位姿) → 解 eye-in-hand → 存 `hand_eye.npz`。
- 这一步**机械臂会动 + 需人监督 + 需标定板**，无法远程自动化。
- 产物放：`config/calibration/orbbec_gemini2/hand_eye.npz`（`utils/camera_utils.load_hand_eye` 读这里）。

### 3.3 把 perception 接入 grasp_service / 容器
- 容器镜像已含 cv2/onnxruntime/scipy；**缺 `pyorbbecsdk2` + ONNX engine + hand_eye.npz**，需加进镜像或挂载。
- 相机 SDK：`pip install pyorbbecsdk2`（cp310/aarch64 wheel，无需编译；**导入名是 `pyorbbecsdk`**）。
- 把 grasp 词表 ONNX + hand_eye.npz 挂进 `/opt/seeed/voice_rebot_arm/config/` 或烤进镜像。
- `grasp_object(object_name)` 工具已注册（Phase B），走 `grasp_service.run_grasp_once`。

### 3.4 监督式执行抓取（机械臂伸手，需人在场盯急停）
语音：“Hey Jarvis” → “抓那个杯子” → 相机检测 → 相机系 grasp pose → hand_eye 变换到基座
→ move_to(pregrasp→grasp) → grasp(force) → lift。barge-in/“停” 会 cancel 并安全张爪。

### 3.5 GPU 推理（可选，提速）
- onnxruntime-gpu 暴露 [TensorRT, CUDA, CPU] EP。**TRT 建 engine 内存峰值极高**——生产栈在跑时会 OOM；
  已在停 edge-llm 的窗口建好并缓存（见 §3.1），缓存后内存紧也能直接加载推理。
- **TensorRT/CUDA 用的是宿主机 JetPack 的库**：pip 的 `onnxruntime-gpu` 只带 ORT 本体，TRT EP 动态链接
  host `/lib/aarch64-linux-gnu/libnvinfer.so.10`（TensorRT 10.3）+ host CUDA 12.6。两个推论：
  ① engine cache **绑本机**（sm87 + TRT 版本），换设备/升 JetPack 要重建；
  ② **容器内要 GPU** 必须注入宿主 CUDA/TRT：`--runtime nvidia` + 宿主库挂载（参考 seeed-voice 的 GPU 容器做法），
    否则容器里只有 CPU EP（1.18s/帧，单次抓取也够用）。
- dry-run 脚本 `tools/perception_dryrun.py` 有 `OVS_ORT_PROVIDERS=cpu|cuda` 开关。

---

## 4. 已知坑（全部踩过，勿重蹈）

**镜像构建**
- base 必须 `python:3.10-slim`（Pinocchio `pin` 4.0.0 aarch64 wheel 只在 cp310 验过；3.11 无保证）。`requires-python>=3.10`。
- 缺 `build-essential` + `portaudio19-dev` → pyaudio 源码编译失败（无 aarch64 wheel）。
- `motorbridge` 不 pin 会版本漂（见过 0.2.2 / 0.4.2）→ pin `==0.2.8`（真机验证版本）。
- 唤醒词模型不在镜像、运行时也不下（无 entrypoint）→ Dockerfile `openwakeword.utils.download_models()` 构建时烧入。

**运行 config**
- SDK 默认 `channel=/dev/ttyACM0`（SO-ARM 口！）→ 设 `REBOT_CHANNEL=auto` 自动检测 B601-DM（扫描 `/dev/serial/by-id/`），或传 realpath / by-id 路径（`normalize_channel` 会 realpath 解析）。
- `grasp_force`/`open_distance_m` 等空串（`${VAR:-}` 未设）→ 旧代码 `float('')` 崩 → ArmPlugin 禁用。已修：空串当未设。
- `MIC_INDEX=auto` 或 `MIC_INDEX=reSpeaker` 均可（`resolve_input_index` 在 app.py 中已接线）。
- 缺 `OVS_AGENT_SERVER_LOOP=1` → 跑 legacy client-loop（skip advertise）。生产用 server-loop（SLV 跑 LLM+工具循环）。已设为镜像默认。
- 夹爪“只动夹爪”的帧要**省略位姿字段**（带 x/y/z 会先 move_to 让臂乱动）。

**Phase B 设备**
- Orbbec USB 节点默认 root-only → `sudo chmod a+rw /dev/bus/usb/<bus>/<dev>`（单节点，非破坏）后 SDK 才能开。
- `pyorbbecsdk`(v1) 无 cp310/aarch64 wheel；用 `pyorbbecsdk2`（pip 名），导入名 `pyorbbecsdk`。

**远程操作（SSH）**
- 设备操作直接 `ssh seeed@seeed-orin-nx`，docker/串口/系统操作需 `sudo`。
- 带空格的 env 值（如 `WAKEWORD_MODEL="hey jarvis"`）经多层 shell 转发（ssh 一行式、脚本套脚本）容易被拆——
  要么登进设备再执行，要么仔细保引号（`ssh host 'docker run ... -e "WAKEWORD_MODEL=hey jarvis" ...'`）。

---

## 5. 完整 docker run（参考）

```bash
docker run -d --name voice-rebot-arm --restart no --runtime nvidia \
  --network voice-arm_default --user 0 \
  --device /dev/snd --device /dev/ttyACM1 \
  -v /run/user:/run/user:ro -v /home/seeed/.config/pulse:/host-pulse-config:ro \
  -v /opt/seeed/voice_rebot_arm/config:/opt/seeed/voice_rebot_arm/config \
  -e CONFIG_DIR=/opt/seeed/voice_rebot_arm/config \
  -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
  -e VOICE_SERVICE_HOST=seeed-voice -e VOICE_SERVICE_PORT=8000 -e VOICE_SERVICE_URL=http://seeed-voice:8000 \
  -e LLM_SERVICE_HOST=edge-llm -e LLM_SERVICE_PORT=8000 -e LLM_SERVICE_URL=http://edge-llm:8000 \
  -e MIC_INDEX=reSpeaker -e MIC_CHANNEL_SELECT=0 -e WAKEWORD_MODEL="hey jarvis" \
  -e REBOT_CHANNEL=/dev/ttyACM1 -e REBOT_REPO_ROOT=/opt/rebot \
  -e HF_ENDPOINT=https://hf-mirror.com -e OBSERVATION_PORT=8775 \
  -e OVS_AGENT_SERVER_LOOP=1 \
  -p 8775:8775 voice-rebot-arm:dev
```
（镜像已把大部分 env 设为默认；上面显式列全便于排查。）

---

## 6. 源码地图

| 路径 | 作用 |
|---|---|
| `rebot_arm.py` | vendored RebotArm 封装（SDK 延迟 import、channel 覆盖、夹爪力控） |
| `rebot_actuator.py` | `RebotArmActuator(Actuator)`：frame→笛卡尔 waypoint，夹爪带符号幅度 |
| `actions.yaml` | 5 个动作（真机 IK 标定的笛卡尔坐标） |
| `config.yaml` | app 配置（${VAR} 由 ovs-agent 自身 env 替换） |
| `app.py` | `VoiceRebotArmApp`：注册 ArmPlugin + GraspPlugin |
| `grasp_plugin.py` / `grasp_service.py` | Phase B：`grasp_object` 工具 + 抓取流水线（cancel 感知） |
| `perception/yolo_onnx.py` | 去 torch YOLOE-seg（onnxruntime + numpy/cv2 后处理） |
| `perception/ordinary_grasp.py` / `transforms.py` / `camera/` | 短轴抓取 / 坐标变换 / 相机驱动（vendored 去 torch） |
| `tools/perception_dryrun.py` | Phase B 感知 dry-run 参考脚本（不动臂；路径为 seeed-orin-nx 设置） |

---

## 7. 多目标抓取（瓶子 / 杯子 / 水果）—— 2026-07-07 内部发布

Phase B 原本只抓盒子。本节记录扩展到 **水杯（✅ 已验证）/ 直立水瓶（✅ 已验证）/ 水果（代码就绪，未实测）** 的步骤与原理。盒子回归测试通过（box ✅ + cup ✅ + water bottle ✅ 同一配置下全部可抓）。

### 7.1 一次性：导出多类别检测 ONNX（必须）

盒子 demo 的 `yoloe-26s-seg-box.onnx` 只烧了 box 类，对瓶/杯 `num_detections=0`。用开放词表底模重导（宿主机需 ultralytics，≈30s CPU）：

```bash
cd /home/harvest && python3 openvoicestream/agent/ovs_agent/apps/voice_rebot_arm/tools/export_yoloe_seg_model.py \
  --weights yoloe-26s-seg.pt \
  --classes box "cardboard box" carton package "small cardboard box" "brown box" "yellow banana" orange "water bottle" cup \
  --out yoloe-26s-seg-multi.onnx
sudo cp yoloe-26s-seg-multi.onnx /opt/rebot-models/
```

类别顺序必须与 `config.yaml` 的 `yolo_classes` 一致。校验输出 shape：`output0 [1,300,38]` + `output1 [1,32,160,160]`。

### 7.2 生效的默认值（已烧进 config.yaml，env 可覆盖）

| 配置 | 值 | 原因（真机 2026-07-07） |
|---|---|---|
| `yolo_model_path` | `yoloe-26s-seg-multi.onnx` | 多类别检测 |
| `conf` | 0.06 | 瓶子只有 0.11-0.19 置信度，0.10 地板导致间歇性"找不到" |
| `grasp_force_by_class."water bottle"` | 0.8 | 0.5 固定力仍打滑 |
| `grasp_force_fixed_classes` | + `"water bottle"` | 硬塑料瓶不压缩 → 自适应 ramp 停在 ~0.2 = 抓不住（与盒子同一个坑） |
| `insertion_depth_m` | 0.040 | 0.025 只有指尖碰到曲面；盒子在 0.040 下回归通过 |

### 7.3 感知侧改动（perception/ordinary_grasp.py）

1. **直立圆柱逃生通道**：tall-box 守卫（`_major_is_vertical` / FORCE-SIDE `_tall_box`）会把"直立的高物体"一律当盒子送去 side_face（真机现象：0.15m 瓶子在桌面高度 z≈0.02 抓空）。新增前置分支：`elong≥4.0 + extent_minor≤0.07 + top is None`（盒子有 RANSAC 顶面、瓶子没有 → 这一项区分盒/瓶）→ 直接走 `_descriptor_grasp`。
2. **水平接近**：cylinder/elongated 路线把相机射线 approach 压平到水平面（俯仰 0.573rad → 0.000，眼在手相机 ~33° 下视角不再"从上往下扎"）。
3. **钉死抓取高度**：质心高度逐帧漂移（同一瓶子 0.098↔0.057）→ 改为 base + 0.55×extent_major（0.40 抓到瓶底收窄段，太低）。
4. **recenter 0.5→0.7**：夹指越过圆柱轴线合拢，不再切线蹭壁。
5. **SHAPEDBG 日志**：每次形状描述子构建打一行 INFO（elong/spine/extents/table_proj），现场排障主信号。参考值：直立 0.15m 果汁瓶 elong 7-9、spine 0.02-0.05。

### 7.4 已知边界

- **透明瓶子抓不了**：Gemini2 深度对透明塑料+水近乎全盲（深度图整瓶黑洞），点云被桌面污染。换不透明物体，或等 RGB/轮廓方案。**演示请用不透明瓶。**
- **round 路线（橙子/苹果）未实测**：代码与 0.7 recenter 就绪，catalog 有 "orange"，但真机没跑过 —— 上水果前先看 SHAPEDBG。
- 瓶子置信度依旧偏低（0.11-0.19），偶发 `num_detections=0` → 重说指令即可；`pregrasp IK failed` 偶发（同盒子时代的 scan-pose IK 抖动），重试可过。
- 语音入口不变："Hey Jarvis, grab the water bottle / cup / box"。
