# Isaac Sim 抓取仿真 — 可复现 Runbook（换设备直接续）

**目的**:在任意 Linux + RTX GPU 机器上（vast.ai / 自有 / 其他云）从零拉起 Isaac Sim 4.5 无头抓取仿真环境。**一条命令 provision**（见 §3），或按下面手动步骤。

**为什么必须 Linux**:Isaac 无头渲染在 Linux 用 EGL/Vulkan 离屏，无会话依赖。**Windows 不行**——GPU 帧 present 需真交互 console 会话，session 0/RDP 都崩（`app.update()`/`World.reset()`/`app.close()` access violation），远程自动登录也点不亮。详见 memory `isaac_grasp_sim_buildout_2026_06_14`。别再碰 Windows。

---

## 0. 硬件/镜像要求
- GPU: **带 RT core 的 RTX**（RTX 3090/4090/A5000/A6000/L4/L40）。**避开 A100/H100**（无 RT core，渲染器不友好）。VRAM ≥ 16GB（24GB 舒适）。
- 磁盘 ≥ 60GB（Isaac 镜像 22GB + 资产 + 缓存）。CUDA driver ≥ 535（实测 570.190 OK）。
- 镜像:`nvcr.io/nvidia/isaac-sim:4.5.0`（**匿名可拉**，无需 NGC key；22GB，慢主机要 ~20min）。

## 1. vast.ai 租实例
账户:`suharvest@gmail.com`，`vastai` CLI 已认证（key 文件 `~/.config/vastai/vast_api_key`），SSH pubkey `my-macbook`(ed25519) 已注册到 vast。
```bash
# 搜（RT core / 单卡 / 大盘 / 可靠 / 快网，按价排序）
vastai search offers 'num_gpus=1 gpu_name in [RTX_4090,RTX_3090,RTX_A5000,RTX_A6000] disk_space>=80 reliability>0.97 cuda_vers>=12.0 rentable=true inet_down>=200' -o 'dph+' --limit 12
# 租（取上面某行的 ID）
vastai create instance <OFFER_ID> --image nvcr.io/nvidia/isaac-sim:4.5.0 --disk 80 --ssh --onstart-cmd 'tail -f /dev/null'
# 看状态 + SSH 地址（等 actual_status: loading→running，22GB 镜像拉完才 SSH-able）
vastai show instances-v1
vastai ssh-url <CONTRACT_ID>          # ssh://root@sshN.vast.ai:PORT
# 计费 ~$0.17/hr。用完务必 destroy：vastai destroy instance <CONTRACT_ID>
```
连接（key 已注册）:`ssh -p <PORT> -o StrictHostKeyChecking=accept-new root@<HOST>`

> 自有 Linux+RTX 机器:跳过租用，直接 `docker run --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y nvcr.io/nvidia/isaac-sim:4.5.0 ...`，或裸机 `pip install isaacsim[all,extscache]==4.5.0`（Ubuntu22.04/py3.10）。

## 2. 一键 provision（推荐）
在本 repo 根目录:
```bash
bash sim/provision_isaac.sh <SSH_HOST> <SSH_PORT>
```
它会:打包 sim 资产+流水线 → scp 到 /root/ → 解包 → 装依赖（`numpy<2` pin opencv）→ 跑无头冒烟自检。绿了就能跑桥接。

## 3. 手动步骤（provision 脚本做的事）
### 3a. 传资产（从本 repo 根目录）
```bash
tar czf /tmp/rebot_sim_bundle.tar.gz \
  sim/rebot_b601dm_urdf sim/calib docs/sim/isaac_bridge_spec.md \
  agent/ovs_agent/apps/voice_rebot_arm/perception \
  agent/ovs_agent/apps/voice_rebot_arm/tools/synthetic_grasp_harness.py \
  agent/ovs_agent/apps/voice_rebot_arm/tools/artifacts/ik_envelope_b601dm.csv
scp -P <PORT> -o StrictHostKeyChecking=accept-new /tmp/rebot_sim_bundle.tar.gz root@<HOST>:/root/
ssh -p <PORT> root@<HOST> "cd /root && tar xzf rebot_sim_bundle.tar.gz"
```
落地:`/root/sim/rebot_b601dm_urdf/`（fixend+gripper URDF+meshes）、`/root/sim/calib/`（hand_eye.npz / intrinsics.npz）、`/root/agent/ovs_agent/apps/voice_rebot_arm/`（流水线）、`/root/docs/sim/isaac_bridge_spec.md`。

### 3b. 装依赖（⚠️ numpy 必须 <2）
```bash
ssh -p <PORT> root@<HOST> "/isaac-sim/python.sh -m pip install pin opencv-python-headless && /isaac-sim/python.sh -m pip install 'numpy<2'"
```
**坑**:pin/opencv 会拉 numpy 2.x，破 Isaac ABI（`_ARRAY_API not found`/`numpy.core.multiarray failed to import`）→ 装完务必把 numpy 钉回 <2。验:`import numpy,cv2,pinocchio,isaacsim` 全过（numpy 1.26.4 / cv2 4.9.0 / pinocchio 4.0.0）。

### 3c. 无头冒烟自检
`scp` 上 `sim/linux_smoke.py`（仓里有），跑:
```bash
ssh -p <PORT> root@<HOST> "cd /isaac-sim && ./python.sh /root/linux_smoke.py 2>&1 | grep -iE 'SIMAPP_OK|RENDER_60_OK|WORLD_RESET_OK|WORLD_STEP_RENDER_30_OK'"
```
全绿（4 个 OK）= 平台通。首次跑要建 shader/asset 缓存，~1-3min。

## 4. Isaac python 使用铁律（踩过的坑）
- 入口 `/isaac-sim/python.sh <script.py>`，**用脚本文件不要 `-c`**。
- **EULA 必须进程内设**:脚本顶部 `import os; os.environ['OMNI_KIT_ACCEPT_EULA']='YES'`，**在 `from isaacsim import SimulationApp` 之前**（shell `export`/`set` 经 ssh 不可靠）。
- 无头:`SimulationApp({'headless':True})`。
- **不要 pip 装任何把 numpy 升到 2.x 的东西**（破 ABI）。要装先 `pin 'numpy<2'`。
- 流水线相对导入:`sys.path` 加 `/root/agent/ovs_agent/apps`，import 作 `voice_rebot_arm.perception.ordinary_grasp`。

## 5. 桥接代码 + 跑抓取仿真
桥接在实例 `/root/sim_bridge/`（按 `docs/sim/isaac_bridge_spec.md` 实现）:
- `isaac_scene.py` — URDF→USD(gripper)、地面/桌/盒、base 固定 world
- `isaac_camera.py` — eye-in-hand 相机(真 K + `T_cam2base=tcp_pose@T_result`)
- `gt_segmenter.py` — Isaac 真值实例分割 → `YoloResult`(绕开 YOLO 域差)
- `isaac_arm.py` — pinocchio IK on fixend URDF(frame `end_link`)→驱关节
- `run_grasp_sim.py` — 抓取网格,出 CSV(method/width/reachable/LIFTED/HELD/SLIPPED/KNOCKED)

阶段:P0 URDF导入+夹爪开合 → P1 相机+GT分割+`estimate_grasps` → P2 IK move → P3 单盒抓+夹持验证 → P4 扫描。
**已验**:P0 done（8-DOF,夹爪 OPEN/CLOSE OK），P1 相机内参对齐真机 Orbbec。

> GT 分割坑:盒子 prim 要打 semantic label + 挂 instance/semantic segmentation annotator + 渲染几帧后再读 annotator,否则 mask=None。

## 6. 资产/约定速查
- URDF frame:`end_link`(法兰,SDK FK 报这个);grasp TCP 在 end_link 外 +X 0.128m;夹爪 `gripper_base_joint` 原点=identity(装配解析确认对)。
- 夹爪:2 prismatic 指,各 0→0.0425m=总 0.085m 跨距,effort 91N(=1.5N·m/小齿轮 r0.0164m)。
- 真机相机:Orbbec Gemini2,K fx691.65/fy691.60/cx639.18/cy359.49 @1280×720;hand_eye `T_result`(TCP→camera,相机在 TCP 后68mm/上54mm)。
- TCP 约定:tool +X=approach,+Y=jaw 开合,+Z RH。base +X=reach,+Z=up。
- IK 包络:`ik_envelope_b601dm.csv`(甜区 pitch0.225-0.9)。

## 7. Tier A（Mac 侧,与本仿真互补,不需 GPU）
`agent/ovs_agent/apps/voice_rebot_arm/tools/{synthetic_grasp_harness,grasp_sweep}.py` + `agent/tests/test_*`:纯 numpy 合成深度,验几何/IK(扁盒 z<0.08 不可达等)。`uv run pytest` 即跑。Isaac(本 runbook)验接触物理(夹持/滑脱/碰倒)——Tier A 验不了的那半。

## 8. 桥接踩坑（2026-06-14 调试沉淀）
- **🔥 缓存 USD 必坑**:`run_held.py`/`run_grasp_sim.py` 加载 `sim_bridge/out/rebot_gripper.usd`（URDF 导入的缓存）。**改了 URDF 必须重跑 `p0_import_urdf.py` 重新生成 USD**，否则改动不进仿真（我在这上面跑了一堆假结果）。改 URDF → `python.sh p0_import_urdf.py` → 再跑抓取。
- **夹爪碰撞=box 不是 mesh**:动态抓取手指 PhysX 要凸基元；平垫 blade 用 box（尺寸/位置对齐真实 CAD：手指 +X 0.0955-0.1605m，接触 0.128m）。原网格要凸分解，过度工程。视觉可挂原 mesh。
- **pad_center 要加偏移**:手指 link 原点在夹爪根部，真实接触 pad 在 link +X 0.128m 处（=tool_offset）。`pad_center()` 必须 `link_origin + 0.128*link_X_axis`，否则偏一个手指长度。
- **tool_offset_x=0.128**（真实 CAD 法兰→接触），非早期 0.045。
