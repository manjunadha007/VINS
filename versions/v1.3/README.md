# Virtual Production Camera Tracking

Custom 6DOF camera tracking for virtual production. An Intel RealSense D455 on a
camera rig is tracked by NVIDIA Isaac ROS Visual SLAM, and the resulting pose is
streamed to Unreal Engine over the industry-standard FreeD protocol, so a virtual
camera inside Unreal mirrors the real camera in real time.

```
  RealSense D455              Isaac ROS Visual SLAM         this bridge              Unreal Engine
  ┌──────────────┐            ┌────────────────────┐        ┌──────────────┐         ┌──────────────┐
  │ stereo IR    │  images    │ visual-inertial    │ pose   │ frame conv.  │  FreeD  │ LiveLinkFreeD│
  │ 640x360x90   ├───────────>│ fusion on GPU      ├───────>│ predict+gate ├────────>│  -> CineCamera│
  │ IMU 200 Hz   │  imu       │ (cuVSLAM)          │ +vel   │ FreeD encode │  UDP    │              │
  └──────────────┘            └────────────────────┘        └──────────────┘         └──────────────┘
```

## Versions

Working tree is **1.3** (see [VERSION.md](VERSION.md)). Frozen snapshots live in
[versions/](versions/README.md): `v1.1` (first hardware-validated pipeline) and
`v1.2` (stationary drift 27.89 mm → 0.70 mm). Each snapshot is self-contained and
runs its own tests in place.

## Layout

| File | Role |
|---|---|
| [dashboard.py](dashboard.py) | The bridge: ROS 2 node + operator GUI |
| [pose_transform.py](pose_transform.py) | ROS→Unreal frame conversion, prediction, smoothing |
| [freed.py](freed.py) | FreeD protocol encode/decode |
| [tracking_config.py](tracking_config.py) / [tracking.yaml](tracking.yaml) | All operator settings |
| [calibrate_axes.py](calibrate_axes.py) | Verify Unreal axis directions without a camera |
| [launch_isaac.sh](launch_isaac.sh) | Deploy + launch the whole pipeline |
| [isaac_vp.launch.py](isaac_vp.launch.py) | SLAM launch used by default: pure VO, exposure capped |
| [verify_tracking.py](verify_tracking.py) | Headless pipeline verification + accuracy benchmark |
| [setup_isaac_ros.sh](setup_isaac_ros.sh) | One-time Docker + Isaac ROS install |
| [test_new_pipeline.sh](test_new_pipeline.sh) | Cold-start the container |

Legacy, kept for reference: [start_pipeline.sh](start_pipeline.sh) +
[ros_vins_to_unreal.py](ros_vins_to_unreal.py) + [zupt_imu_node.py](zupt_imu_node.py)
are the older **ROS 1 / VINS-Fusion** path (headless, no GUI).
[spectacular_unreal_livelink.py](spectacular_unreal_livelink.py) and
[realsense_unreal_livelink.py](realsense_unreal_livelink.py) are earlier
single-file prototypes.

## First-time setup

```bash
./setup_isaac_ros.sh          # Docker, NVIDIA Container Toolkit, Isaac ROS
./install_realsense_dkms.sh   # RealSense kernel patches, then replug the camera
```

Log out and back in (or `newgrp docker`), then build VSLAM once inside the
container:

```bash
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && ./scripts/run_dev.sh
# inside the container:
cd /workspaces/isaac_ros-dev && colcon build --symlink-install --packages-select isaac_ros_visual_slam
```

### Unreal Engine side

1. Enable the **LiveLink** and **FreeD** plugins, restart the editor.
2. `Window > Virtual Production > Live Link` → `Add Source` → **FreeD**, UDP port
   **11111**.
3. Confirm a subject appears and its **Camera ID matches `network.camera_id`**
   (default `1`). A mismatched ID is the single most common reason "nothing
   happens".
4. Add a `CineCameraActor`, give it a **LiveLinkComponentController**, and point
   its subject at the FreeD subject.

## Running

```bash
./test_new_pipeline.sh    # cold start: brings the container up, builds, launches
./launch_isaac.sh         # normal start: container already running
```

`launch_isaac.sh` deploys `dashboard.py`, `pose_transform.py`, `freed.py`,
`tracking_config.py`, `verify_tracking.py` and `isaac_vp.launch.py` into the
container workspace on every launch, so the repo is the single source of truth.
`tracking.yaml` is installed once and then left alone, so on-set edits survive.

It launches [isaac_vp.launch.py](isaac_vp.launch.py) by default: pure visual
odometry at 640x360x90 with auto-exposure capped declaratively. Set
`VP_SLAM_LAUNCH=nvidia` to fall back to NVIDIA's stock launch file.

Check the pipeline at any time, without Unreal:

```bash
# inside the container
python3 src/verify_tracking.py --seconds 10                  # health + drift
python3 src/verify_tracking.py --benchmark --distance 0.5     # guided accuracy test
```

## Configuration

Everything lives in [tracking.yaml](tracking.yaml). The two settings you will
actually change:

```yaml
network:
  unreal_ip: "192.168.1.50"   # the machine running Unreal — NOT localhost
  camera_id: 1                # must match the LiveLinkFreeD subject
```

The bridge defaults to `127.0.0.1`, which only works when Unreal runs on the same
box. For a tracking-box + Unreal-workstation setup, set the workstation's LAN
address and make sure **UDP 11111** is open on it.

## Axis calibration

The ROS→Unreal conversion in [pose_transform.py](pose_transform.py) is derived
from first principles and unit tested, but the final sign of each axis also
depends on how your Unreal project orients its LiveLink subject. Verify it in two
minutes, with no camera and no SLAM:

```bash
python3 calibrate_axes.py
```

It moves one axis at a time and prints what Unreal should do:

| Axis | Expected in Unreal | Flip this if wrong |
|---|---|---|
| X | moves forward | `transform.position_sign[0]` |
| Y | moves right | `transform.position_sign[1]` |
| Z | moves up | `transform.position_sign[2]` |
| pan | rotates right | `transform.pan_sign` |
| tilt | tilts up | `transform.tilt_sign` |
| roll | rolls clockwise | `transform.roll_sign` |

Negate the listed value in `tracking.yaml` for any axis that moves the wrong way,
then re-run. Other useful modes:

```bash
python3 calibrate_axes.py --axis pan            # loop one axis
python3 calibrate_axes.py --hold x=1,pan=45     # park at a fixed pose
python3 calibrate_axes.py --show-config         # print effective settings
```

## On-set procedure

1. **Mount** the D455 rigidly to the rig. Any flex between the tracker and the
   taking lens shows up directly as jitter in the composite.
2. **Measure the rig offset** — where the taking lens's nodal point sits relative
   to the RealSense, in ROS body axes (X forward, Y left, Z up) — and enter it as
   `transform.rig_offset.translation_m`. Without this the virtual camera rotates
   about the tracker instead of about the lens, which reads as parallax error on
   pans.
3. **Start the pipeline**, hold the rig still for a couple of seconds, then move
   slowly to let SLAM initialise. The status panel turns green at `TRACKING`.
4. **Set ground zero**: point the camera at a printed ArUco marker (4x4, default
   dictionary `DICT_4X4_1000`) placed at your desired world origin, measure its
   edge length into `aruco.marker_size_m`, and press
   **ANCHOR TO ARUCO GROUND ZERO**. The marker becomes Unreal's origin, level to
   the horizon.
5. **Watch the status panel** during takes: `SLAM in` and `FreeD out` rates, and
   whether tracking is live.

## Latency

Contributions, roughly: sensor exposure and USB transfer, SLAM compute, this
bridge (sub-millisecond), UDP, and Unreal's own render pipeline.

To reduce it:

* SLAM already runs at 90 fps (640x360x90). Higher resolution would help accuracy
  but 848x480 does not resolve as a stereo pair on this D455 over its current USB
  path — see the note at the top of [isaac_vp.launch.py](isaac_vp.launch.py).
* Set `network.send_rate_hz: 0` to emit a packet the instant each pose arrives
  rather than on a fixed timer.
* Only then reach for `tracking.lead_time_s`, which predicts ahead using Isaac's
  velocity estimate. **Measure before dialling this in**: point the camera at a
  clapperboard-style sharp move, compare real and virtual in the composite, and
  increase in 10 ms steps. Over-prediction shows up as overshoot that snaps back
  at the end of a whip pan — worse than the latency it hides.

`tracking.smoothing_alpha` trades latency for steadiness. Leave it at `0` unless
micro-jitter is visible; a locked-off plate tolerates it, a handheld shot does
not.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Nothing moves in Unreal | Camera ID mismatch, wrong `unreal_ip`, or UDP 11111 blocked. Run `calibrate_axes.py` to isolate — if that moves the camera, the problem is upstream in SLAM. |
| Virtual camera moves the wrong way on one axis | Axis sign. See calibration above. |
| Virtual camera banks when you pan | The pre-fix symptom of the gimbal-lock bug. Confirm `pose_transform.py` is deployed and you are not running a stale `dashboard_isaac.py`. |
| Position drifts with the camera stationary | The stationary hold should pin this to ~1 mm; check `tracking.hold_enabled` and that the dashboard shows `HOLD (still)`. If the hold is engaging but drift persists while moving, the cause is scene content — see the row below. |
| Drift is much worse than usual, same lighting | Measured on this rig: with 0 % near-black pixels and texture gradient 0.854, per-sample jitter was 8.5 um; with 19.2 % near-black and gradient 0.670 — *identical mean brightness* — it was 154.5 um, 18x worse. Point the camera at evenly lit, textured geometry; large dark or blank regions are what hurt, not overall brightness. |
| **Pose is stable when still but diverges wildly (metres, impossible speeds) as soon as you move** | **Motion blur from an uncapped exposure.** The RealSense default `auto_exposure_limit` is 200 ms; at 90 fps every frame smears while moving, cuVSLAM loses visual tracking and falls back to IMU dead reckoning, which integrates into tens of metres of error. The launch scripts now cap it to 6 ms (`VP_AE_LIMIT_US`). Verify with `ros2 param get /camera/camera depth_module.auto_exposure_limit`. Also move more slowly — beyond roughly 2 rad/s no VIO will hold. |
| `magic_enum::magic_enum ... target was not found` when building | Do not build `isaac_ros_visual_slam` from source in this container. Install NVIDIA's prebuilt package: `apt-get install -y ros-humble-isaac-ros-visual-slam`. |
| Host `rs-enumerate-devices` finds 0 devices but the container finds the camera | Expected if the librealsense DKMS kernel patches are absent (`dpkg -l \| grep librealsense` empty). The host kernel leaves the D455's video interfaces 0–2 and its HID interface at `Driver=[none]`; the container's librealsense uses the libusb backend instead. It also means frame metadata (per-frame exposure/gain) is unavailable. Run `./install_realsense_dkms.sh` and replug to fix. |
| Status shows `TRACKING LOST — HOLDING` | SLAM stopped publishing. The last good pose is held rather than teleporting the camera. Re-initialise with slow motion across textured geometry. |
| Ground zero anchors in the wrong place | `aruco.marker_size_m` does not match the printed marker, or `camera_info` never arrived so fallback intrinsics were used (the console reports which). |
| SLAM node exits at startup with 60 fps profile | D455/USB cannot sustain it. Use the default launch, or see the notes in `isaac_vp.launch.py`. |

## Tests

```bash
python3 -m unittest test_pose_transform test_freed_protocol -v   # anywhere
```

Those two suites cover the frame conversion (cross-checked against scipy), the
FreeD wire format, prediction and smoothing, and need no ROS.

```bash
# inside the Isaac ROS container
cd /workspaces/isaac_ros-dev && python3 -m unittest src.test_virtual_camera -v
```

`test_virtual_camera.py` covers the wiring between SLAM callbacks and the UDP
output; it skips itself where ROS 2 is unavailable.

## Design notes

* **Isaac ROS owns the fusion.** `enable_imu_fusion` is on, so the SLAM node
  already fuses stereo and IMU. The bridge does not re-fuse. An earlier version
  integrated `/camera/imu` gyro in Python and raced Isaac's estimate for the
  output pose; the IMU subscription is now telemetry only.
* **One writer.** Only `/visual_slam/tracking/odometry` drives the stream
  (`vo_pose_covariance` is a fallback until odometry arrives), because it also
  carries the velocity that makes prediction trustworthy.
* **Rate conversion is prediction, not interpolation.** Constant-velocity
  extrapolation from Isaac's `twist`, hard-capped by
  `tracking.max_extrapolation_s` so tracking loss freezes the camera instead of
  flinging it.
* **Tracking loss holds, never substitutes.** GNSS is off by default and never
  feeds the FreeD stream; it is a different coordinate frame with a different
  origin, and swapping it in mid-take teleports the virtual camera.
* **The maths is isolated and tested.** `pose_transform.py` and `freed.py` have
  no ROS/Qt/OpenCV imports. Keep new geometry there, not in the GUI.
