# reBot B601-DM URDF Retrieval (for Isaac Sim)

Retrieved READ-ONLY from the production device `seeed-orin-nx` on 2026-06-14.
Source: vendored SDK `reBotArm_control_py` (github `vectorBH6/reBotArm_control_py`),
running inside container `voice-rebot-arm`
(image `sensecraft-missionpack.seeed.cn/solution/voice-rebot-arm:v0.8.0-vision-20260613i`).

## Where it was found

- SDK root (in container): `/opt/rebot/reBotArm_control_py/`
- URDF package dir: `/opt/rebot/reBotArm_control_py/urdf/reBot-DevArm_fixend_description/`
  - URDF: `urdf/reBot-DevArm_fixend.urdf` (10,643 bytes)
  - Meshes: `meshes/{base_link,link1..link6,end_link}.STL` (8 binary STL, ~25 MB total)
  - Also: `urdf/reBot-DevArm_fixend.csv` (joint/param table export, 28 KB)
- There is a sibling symlink `urdf/reBot-DevArm_description_fixend -> reBot-DevArm_fixend_description`.
  This matters: the URDF mesh refs use `package://reBot-DevArm_description_fixend/meshes/...`
  (the **symlink** name, not the real dir name).

### Default-loaded URDF (canonical entry file)

`reBotArm_control_py/kinematics/robot_model.py::_get_default_urdf_path()` returns:

```
<sdk>/urdf/reBot-DevArm_fixend_description/urdf/reBot-DevArm_fixend.urdf
```

`load_robot_model()` loads it via Pinocchio `pin.buildModelFromUrdf(urdf_path)`.
This is the single canonical URDF the production arm uses for FK/IK/dynamics.
The "fixend" naming = the wrist/flange is modeled as a **fixed** end link, not an
actuated gripper (see assessment).

## Copied file inventory (on Mac)

Landed under `/Users/harvest/project/seeed-local-voice/sim/rebot_b601dm_urdf/`
(directory structure preserved so `package://.../meshes/` refs resolve relative to it):

```
rebot_b601dm_urdf/
  urdf/
    reBot-DevArm_fixend.urdf   (10,643 B)
    reBot-DevArm_fixend.csv    (28,586 B)
  meshes/
    base_link.STL   (4,079,784 B)
    link1.STL       (  337,184 B)
    link2.STL       (5,091,884 B)
    link3.STL       (3,855,084 B)
    link4.STL       (2,859,284 B)
    link5.STL       (2,259,184 B)
    link6.STL       (1,994,484 B)
    end_link.STL    (4,933,684 B)
```

Pull md5 (whole-dir transfer verify): `7360d27208cf7ef844de4cb912efc0cb`.

## Sim-readiness assessment

| Requirement | Present? | Notes |
|---|---|---|
| `<inertial>` (mass + inertia) | YES — all 8 links | Real SolidWorks-exported mass + full 3x3 inertia tensors + COM origins. Good for Isaac physics. |
| `<collision>` geometry | YES — all 8 links | But collision = the **same full-res visual STL** (mesh collision). Not convex/primitive. |
| `<visual>` mesh refs | YES — all 8 links | `package://reBot-DevArm_description_fixend/meshes/*.STL`, grey material `rgba 0.627 0.627 0.627 1`. |
| Joint limits | YES | All 6 revolute joints have lower/upper/effort/velocity. |
| Joint axes | YES | Explicit `<axis xyz>` per joint. |
| Gripper / EE joints | **NO actuated gripper** | `end_joint` is `type="fixed"` (link6 -> end_link). No prismatic/revolute finger joints. |

### What Isaac Sim will still need added

1. **Gripper joints (critical for grasp sim).** This "fixend" URDF has NO actuated
   gripper — the end effector is a rigid fixed flange. The production arm DOES drive a
   gripper (open/close in our stack), so for grasp simulation you must add the
   gripper body + prismatic/revolute finger joint(s) and their limits/inertials.
   They are not in this canonical URDF. Source them from the gripper CAD or model
   them by hand.

2. **Collision geometry simplification.** Collisions are full-resolution visual STL
   meshes (multi-MB each). Isaac/PhysX should auto-convex-decompose, but for stable,
   fast contact you'll likely want to replace them with convex hulls or primitive
   approximations (boxes/capsules), especially around the gripper/contact surfaces.

3. **Friction / contact materials.** URDF has visual `material` color only; no
   `<gazebo>` or physics material (friction, restitution). Add Isaac PhysX materials
   (static/dynamic friction, restitution) per body, especially fingertips for grasping.

4. **Joint dynamics (damping/friction).** Limits have effort/velocity but no
   `<dynamics damping= friction=>`. Add joint damping/armature in Isaac for stable
   control, or tune in the USD/drive after import.

5. **mesh `package://` resolution.** Isaac's URDF importer must map
   `package://reBot-DevArm_description_fixend/` to the `meshes/` dir. Either keep the
   symlink-equivalent layout, set a ROS package path, or rewrite the `filename=`
   refs to relative paths (`../meshes/foo.STL`) before import.

6. **Base fixity.** `base_link` is free; for a fixed-mount arm add a fixed joint to
   world (or set base as static) in Isaac.

7. **Actuator/drive gains.** effort/velocity limits exist, but PD/drive stiffness &
   damping for Isaac articulation drives must be set (not in URDF).

Summary: the URDF is **physics-ready for the 6-DOF arm body** (inertials + collision +
limits all present) but is **missing the gripper entirely** and lacks contact materials,
joint damping, and primitive collision — all of which must be added in/after Isaac import.

---

## Appendix A — Full URDF text

```xml
<?xml version="1.0" encoding="utf-8"?>
<!-- This URDF was automatically created by SolidWorks to URDF Exporter! Originally created by Stephen Brawner (brawner@gmail.com)
     Commit Version: 1.6.0-4-g7f85cfe  Build Version: 1.6.7995.38578
     For more information, please see http://wiki.ros.org/sw_urdf_exporter -->
<robot name="reBot-DevArm_fixend">
  <link name="base_link">
    <inertial>
      <origin xyz="-7.849E-06 -1.1531E-06 0.029841" rpy="0 0 0" />
      <mass value="0.83660000" />
      <inertia ixx="0.00133040" ixy="0.00000001" ixz="0.00000000" iyy="0.00213119" iyz="0.00000000" izz="0.00275877" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><mesh filename="package://reBot-DevArm_description_fixend/meshes/base_link.STL" /></geometry>
      <material name=""><color rgba="0.62745 0.62745 0.62745 1" /></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><mesh filename="package://reBot-DevArm_description_fixend/meshes/base_link.STL" /></geometry>
    </collision>
  </link>

  <link name="link1">
    <inertial>
      <origin xyz="0.000113614552951627 -0.000616319527051323 0.0236476372671394" rpy="0 0 0" />
      <mass value="0.16130000" />
      <inertia ixx="0.00025207" ixy="0.00000000" ixz="-0.00002832" iyy="0.00015464" iyz="0.00000000" izz="0.00023416" />
    </inertial>
    <visual>...mesh link1.STL...</visual>
    <collision>...mesh link1.STL...</collision>
  </link>
  <joint name="joint1" type="revolute">
    <origin xyz="-8.416E-05 0 0.08465" rpy="0 0 0" />
    <parent link="base_link" /><child link="link1" />
    <axis xyz="0 0 1" />
    <limit lower="-2.8" upper="2.8" effort="27" velocity="50" />
  </joint>

  <link name="link2">
    <inertial>
      <origin xyz="-0.13225622308888 -0.0030617036386309 -0.0308306967030205" rpy="0 0 0" />
      <mass value="1.32660000" />
      <inertia ixx="0.00073374" ixy="-0.00000043" ixz="0.00000851" iyy="0.01255987" iyz="0.00000128" izz="0.01281387" />
    </inertial>
    <visual>...mesh link2.STL...</visual>
    <collision>...mesh link2.STL...</collision>
  </link>
  <joint name="joint2" type="revolute">
    <origin xyz="0.020084 0.031625 0.05555" rpy="-1.5708 0 0" />
    <parent link="link1" /><child link="link2" />
    <axis xyz="0 0 -1" />
    <limit lower="-3.14" upper="0" effort="27" velocity="50" />
  </joint>

  <link name="link3">
    <inertial>
      <origin xyz="0.121040035791843 -0.0536211076627949 -0.0310137854608077" rpy="0 0 0" />
      <mass value="0.83530000" />
      <inertia ixx="0.00046807" ixy="-0.00003456" ixz="-0.00004260" iyy="0.00632695" iyz="0.00000006" izz="0.00648221" />
    </inertial>
    <visual>...mesh link3.STL...</visual>
    <collision>...mesh link3.STL...</collision>
  </link>
  <joint name="joint3" type="revolute">
    <origin xyz="-0.264 0 0" rpy="0 0 0" />
    <parent link="link2" /><child link="link3" />
    <axis xyz="0 0 1" />
    <limit lower="-3.14" upper="0" effort="27" velocity="50" />
  </joint>

  <link name="link4">
    <inertial>
      <origin xyz="0.0608200956293136 -0.0511711906613122 -0.030299458623927" rpy="0 0 0" />
      <mass value="0.52000000" />
      <inertia ixx="0.00045986" ixy="0.00024219" ixz="-0.00000672" iyy="0.00075258" iyz="-0.00000535" izz="0.00066742" />
    </inertial>
    <visual>...mesh link4.STL...</visual>
    <collision>...mesh link4.STL...</collision>
  </link>
  <joint name="joint4" type="revolute">
    <origin xyz="0.2426 -0.054 -0.001625" rpy="0 0 0" />
    <parent link="link3" /><child link="link4" />
    <axis xyz="0 0 1" />
    <limit lower="-1.87" upper="1.57" effort="7" velocity="200" />
  </joint>

  <link name="link5">
    <inertial>
      <origin xyz="-0.00502802058982517 1.73866206692364E-06 0.0386233236326755" rpy="0 0 0" />
      <mass value="0.38300000" />
      <inertia ixx="0.00019772" ixy="-0.00000062" ixz="-0.00002426" iyy="0.00021737" iyz="0.00000002" izz="0.00017191" />
    </inertial>
    <visual>...mesh link5.STL...</visual>
    <collision>...mesh link5.STL...</collision>
  </link>
  <joint name="joint5" type="revolute">
    <origin xyz="0.078308 -0.0375 -0.03" rpy="-1.5708 0 0" />
    <parent link="link4" /><child link="link5" />
    <axis xyz="0 0 1" />
    <limit lower="-1.57" upper="1.57" effort="7" velocity="200" />
  </joint>

  <link name="link6">
    <inertial>
      <origin xyz="3.76418727127126E-06 -0.000100908819946677 0.0253308606425965" rpy="0 0 0" />
      <mass value="0.36630000" />
      <inertia ixx="0.00015554" ixy="-0.00000" ixz="-0.00000" iyy="0.00015554" iyz="-0.0000" izz="0.00013966" />
    </inertial>
    <visual>...mesh link6.STL...</visual>
    <collision>...mesh link6.STL...</collision>
  </link>
  <joint name="joint6" type="revolute">
    <origin xyz="0.028008 0 0.04" rpy="0 1.5708 0" />
    <parent link="link5" /><child link="link6" />
    <axis xyz="0 0 1" />
    <limit lower="-3.14" upper="3.14" effort="7" velocity="200" />
  </joint>

  <link name="end_link">
    <inertial>
      <origin xyz="-0.0737654295815033 -9.5080995865868E-06 7.04327840286845E-06" rpy="0 0 0" />
      <mass value="0.5500000" />
      <inertia ixx="0.00037354" ixy="-0.00000011" ixz="0.00000050" iyy="0.00015830" iyz="0.00000348" izz="0.00045353" />
    </inertial>
    <visual>...mesh end_link.STL...</visual>
    <collision>...mesh end_link.STL...</collision>
  </link>
  <joint name="end_joint" type="fixed">
    <origin xyz="0 0 0.15539" rpy="0 -1.5708 3.1415" />
    <parent link="link6" /><child link="end_link" />
    <axis xyz="0 0 0" />
  </joint>
</robot>
```

(Visual/collision blocks abbreviated as `...mesh X.STL...` for the per-link bodies;
every link has both a `<visual>` and a `<collision>` pointing at the same
`package://reBot-DevArm_description_fixend/meshes/<link>.STL`. The full verbatim text
is in the source file `urdf/reBot-DevArm_fixend.urdf`.)

## Appendix B — Kinematic chain summary

```
base_link
  -joint1 (rev, z, [-2.8, 2.8], eff27)-> link1
    -joint2 (rev, -z, [-3.14, 0], eff27)-> link2
      -joint3 (rev, z, [-3.14, 0], eff27)-> link3
        -joint4 (rev, z, [-1.87, 1.57], eff7)-> link4
          -joint5 (rev, z, [-1.57, 1.57], eff7)-> link5
            -joint6 (rev, z, [-3.14, 3.14], eff7)-> link6
              -end_joint (FIXED)-> end_link   <-- no gripper DOF
```

6 actuated revolute joints + 1 fixed flange. Gripper must be added for grasp sim.
