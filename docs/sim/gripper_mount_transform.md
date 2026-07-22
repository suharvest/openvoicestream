# reBot B601-DM — Gripper Mount Transform / TCP Offset (SDK audit)

Read-only retrieval from the production arm `seeed-orin-nx`, container
`voice-rebot-arm` (image `voice-rebot-arm:v0.8.0-vision-20260613i`), SDK at
`/opt/rebot/reBotArm_control_py/`. Date: 2026-06-14.

## TL;DR

- **No gripper-inclusive URDF exists in the SDK.** Both URDF description dirs are
  byte-identical (`md5 4382189c4f7c8947599f97c6a4c9f5bb`) and contain only the
  6-link arm + a fixed `end_link` flange. No finger links, no prismatic/gripper
  joint, no finger meshes anywhere in `/opt/rebot`.
- **`get_tcp_pose()` does not exist in the SDK** (that is OUR app term). The SDK
  FK returns the bare **`end_link` flange** pose. No tool offset is added.
- The only authoritative transform the SDK gives for the end effector is the
  **flange frame** `end_joint`: `link6 → end_link`, `xyz = (0, 0, 0.15539)`,
  `rpy = (0, -1.5708, 3.1415)`.
- The gripper is a single **damiao 4310 rotary motor** (`motor_id 0x07`). The SDK
  exposes it purely in **radians** (MIT / POS_VEL / VEL). There is **no
  stroke→metre mapping in the SDK** — the `pos → 0..0.09 m` map lives only in OUR
  `rebot_arm.py:515`.
- ⇒ Our hand-built `reBot-DevArm_gripper.urdf` `gripper_base_joint` origin
  **must come from finger/mount CAD geometry** — the SDK provides nothing to set
  it. The grasp-point offset is purely geometric (finger reach from the flange).

---

## 1. Gripper-inclusive URDF — NONE

`find` over the whole SDK returns exactly one URDF (the fixed-end / no-gripper
variant). There are two description directories with confusingly swapped names,
but they hold the **same** URDF (byte-identical) and the **same** 8 arm meshes
(no finger STLs):

```
===FIND===
/opt/rebot/reBotArm_control_py/urdf/reBot-DevArm_fixend_description/urdf/reBot-DevArm_fixend.urdf
===XACRO===
(none)
===LSURDF===
/opt/rebot/reBotArm_control_py/urdf/:
reBot-DevArm_description_fixend
reBot-DevArm_fixend_description

.../reBot-DevArm_fixend_description/meshes:
base_link.STL  end_link.STL  link1.STL link2.STL link3.STL link4.STL link5.STL link6.STL
.../reBot-DevArm_fixend_description/urdf:
reBot-DevArm_fixend.csv  reBot-DevArm_fixend.urdf

.../reBot-DevArm_description_fixend/meshes:   (identical set)
base_link.STL  end_link.STL  link1..link6.STL
.../reBot-DevArm_description_fixend/urdf:
reBot-DevArm_fixend.csv  reBot-DevArm_fixend.urdf
```

Both URDFs are identical:

```
===MD5-URDF===
4382189c4f7c8947599f97c6a4c9f5bb  .../reBot-DevArm_fixend_description/urdf/reBot-DevArm_fixend.urdf
4382189c4f7c8947599f97c6a4c9f5bb  .../reBot-DevArm_description_fixend/urdf/reBot-DevArm_fixend.urdf
```

No finger/gripper meshes anywhere:

```
===ALL-STL  (grep grip|finger|jaw|hand|claw)===
(empty)
```

**Nothing was copied to the Mac — there is no gripper geometry to copy.** Our
already-retrieved `reBot-DevArm_fixend.urdf` is the complete arm-side geometry
the SDK ships.

## 2. TCP / tool-tip offset — `get_tcp_pose()` returns the bare flange

Grep for `get_tcp_pose` / `tcp_pose` across the SDK returns **zero hits** — the
term is ours. The SDK FK default frame is `end_link` (the flange), and no tool
offset is layered on:

```
forward_kinematics.py:16:    get_end_effector_frame_id,
forward_kinematics.py:20:_DEFAULT_FRAME = "end_link"
forward_kinematics.py:75:        frame_name: 要查询的帧（默认: ``end_link``）。
robot_model.py:98:def get_end_effector_frame_id(model: pin.Model) -> int:
robot_model.py:99:    """返回末端操作帧 ``end_link`` 的索引。"""
robot_model.py:100:    return model.getFrameId("end_link")
```

The authoritative flange transform (the deepest frame the URDF defines), from
`reBot-DevArm_fixend.urdf:435-444`:

```xml
<joint name="end_joint" type="fixed">
  <origin xyz="0 0 0.15539" rpy="0 -1.5708 3.1415" />
  <parent link="link6" />
  <child  link="end_link" />
  <axis   xyz="0 0 0" />
</joint>
```

**Authoritative end-effector frame = `end_link`** at
`link6 → end_link: xyz=(0, 0, 0.15539) m, rpy=(0, -π/2, π)`.
There is **no** flange→TCP tool offset in the SDK. Any grasp-point offset beyond
`end_link` is purely geometric (finger reach), defined by us, not the SDK.

## 3. Gripper kinematics / stroke→joint mapping — NONE in SDK

`config/gripper.yaml` (full file) — a single damiao 4310 motor, no geometry:

```yaml
channel: /dev/ttyACM0
gripper:
  - name: gripper
    motor_id: 0x07
    feedback_id: 0x17
    model: "4310"
    vendor: "damiao"
    MIT:     { kp: 8.0, kd: 1.0 }
    POS_VEL: { vel_kp: 0.0008, vel_ki: 0.002, pos_kp: 50.0, pos_ki: 1.0, vlim: 3.0 }
```

`actuator/gripper.py` exposes the motor purely in **radians** — `mit(pos=…rad)`,
`pos_vel(pos=…rad)`, `get_state()` returns `pos[rad] / vel[rad/s] / torq[Nm]`
(see `example/gripper_test.py`: `pos=%+.4f rad`). There is **no** finger
stroke→metre table, no `0.09`, no width/open/close constants in the SDK.

⇒ The `pos / -5.0 rad → 0..0.09 m` linear map referenced at our
`rebot_arm.py:515` is **our own calibration**, not sourced from the SDK. Treat it
as ours to own/verify.

## 4. Implication for our hand-built `reBot-DevArm_gripper.urdf`

- `gripper_base_joint` origin (currently identity placeholder): **the SDK gives
  no value for this.** It must come from the **finger/gripper mount CAD geometry**
  (the physical mount plate fixed to `end_link`). The SDK ships zero finger
  geometry, so nothing here can be lifted from it.
- The correct anchor frame to attach our gripper sub-tree to is the SDK's
  **`end_link`** (flange), whose pose relative to `link6` is the
  `xyz=(0,0,0.15539), rpy=(0,-π/2,π)` above. Our `gripper_base_joint` should be a
  fixed joint `end_link → gripper_base` with an origin measured from CAD/mech
  drawing of the B601-DM gripper mount — NOT identity.
- Our `-5.0 rad → 0.09 m` finger map and grasp force config remain ours to
  maintain; the SDK only provides raw radian motor control + PID gains.

## Evidence — provenance
- Device: `seeed-orin-nx` (production, read-only). Container: `voice-rebot-arm`.
- All commands were `docker exec voice-rebot-arm sh -c "<read-only>"` via
  `fleet exec`. No mutations, no `--sudo`, no git, no files written to the device
  beyond none (no `/tmp` stage needed — nothing to copy out).
- Raw outputs for find / ls -R / md5sum / grep / cat are reproduced inline above.
