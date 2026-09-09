# Virtual Production Camera Tracking


The dashboard uses one **Operate → Auto lineup** checkbox for the flat
**DICT_4X4_50, 10 cm ArUco marker**. Place the locator at the physical marker's
centre, set its heading, and enable the checkbox. It turns green only when the
camera is locked with fresh scan fusion. Marker settings and detailed diagnostics
are in **Setup**. Follow [LINEUP_LITE.md](LINEUP_LITE.md) for placement, indicator
states and recovery. The 3D camera and FreeD follow the same supervised pose.

See [PROJECT_ASSESSMENT.md](PROJECT_ASSESSMENT.md) for the project inventory,
fallback behavior and remaining calibration work, and [RTK_SETUP.md](RTK_SETUP.md)
for a beginner guide to your ZED-F9P. Run `python3 tracking_preflight.py` to check
the configured scan and available dependencies without starting hardware.

The project configuration enables supervised output; older deployed
configurations retain their existing behavior until their `fusion` settings are
updated. Existing files in the ROS workspace are preserved by deployment.

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

The original release baseline is **1.3** (see [VERSION.md](VERSION.md)). Frozen snapshots live in
[versions/](versions/README.md): `v1.1` (first hardware-validated pipeline) and
`v1.2` (stationary drift 27.89 mm → 0.70 mm). Each snapshot is self-contained and
runs its own tests in place.

## Layout

| File | Role |
|---|---|
| [dashboard.py](dashboard.py) | The bridge: ROS 2 node + operator GUI |
| [pose_transform.py](pose_transform.py) | ROS→Unreal frame conversion, prediction, smoothing, SE(3) maths |
| [lidar_map_matcher.py](lidar_map_matcher.py) | Absolute localisation against the LiDAR scan |
| [image_alignment.py](image_alignment.py) | Camera-image lineup against the scan (3DE-style) |
| [freed.py](freed.py) | FreeD protocol encode/decode |
| [tracking_config.py](tracking_config.py) / [tracking.yaml](tracking.yaml) | All operator settings |
| [calibrate_axes.py](calibrate_axes.py) | Verify Unreal axis directions without a camera |
| [launch_isaac.sh](launch_isaac.sh) | Deploy + launch the whole pipeline |
| [isaac_vp.launch.py](isaac_vp.launch.py) | SLAM launch used by default: pure VO, exposure capped |
| [verify_tracking.py](verify_tracking.py) | Headless pipeline verification + accuracy benchmark |
| [setup_isaac_ros.sh](setup_isaac_ros.sh) | One-time Docker + Isaac ROS install |
| [test_new_pipeline.sh](test_new_pipeline.sh) | Cold-start the container |
| [bench_fusion.py](bench_fusion.py) | Map-fusion accuracy/timing benchmark against a scan |
| [smoke_locator_lock.py](smoke_locator_lock.py) | End-to-end check of the locator lock-on path |

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

It launches [isaac_vp.launch.py](isaac_vp.launch.py) by default: visual-inertial odometry without loop closure at 848x480x90 with auto-exposure capped declaratively. Set
`VP_SLAM_LAUNCH=nvidia` to fall back to NVIDIA's stock launch file.

Check the pipeline at any time, without Unreal:

```bash
# inside the container
python3 src/verify_tracking.py --seconds 10                  # health + drift
python3 src/verify_tracking.py --benchmark --distance 0.5     # guided accuracy test
```

## 3D viewer channel box

Select **Camera**, **Point Cloud**, **Trail**, **World Origin**, or **Locator**
in the outliner. The **Channel Box** below it shows **Position X/Y/Z** in metres
and **Rotation X/Y/Z** (roll/pitch/yaw) in degrees. Type a value and press
**Enter** or **Tab**, or use the arrow buttons. Fields follow gizmo edits and
external pose changes; live updates preserve a value while you are typing it.
Drag the bottom edge to resize the panel, or scroll the channels in a short view.

- **Camera:** live position and rotation readouts. The dashboard prevents manual
  edits from replacing the physical camera's tracked pose.
- **Point Cloud:** position is the scan centre in the displayed world; rotations
  turn around that centre and preserve the import scale. Locked scans show
  disabled channels.
- **Locator:** edits the known physical marker's centre and orientation.
- **Trail:** offsets and rotates the displayed path in all three views. Recorded
  samples remain intact, and new samples use the same display offset.
- **World Origin:** moves and rotates the reference grid and axes in the 3D
  view. This display transform leaves the tracking coordinate system intact.

**File → Save** includes the trail and reference-grid display transforms along
with the scan alignment and locator pose.

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

## Headless UDP anchoring (marker → Unreal anchor Actor)

For work against a LiDAR-scanned set, the tracking output should come out in the
*scan's* coordinate frame rather than one determined by wherever the operator was
standing. That needs the marker to map onto a known point in the scan:

1. Tape the printed ArUco marker to a known spot in the physical set.
2. In Unreal, place an anchor Actor at that same spot in the imported scan.
3. Have the plugin send the Actor's transform, in Unreal world units:

   ```json
   {"cmd": "anchor_lidar",
    "marker_map_position": [250.0, -400.0, 120.0],
    "marker_map_rotation": [30.0, 0.0, 0.0]}
   ```

   to UDP **11114** (`network.command_port`). Position is centimetres, rotation is
   (pan, tilt, roll) in degrees — straight off the Actor. The reply confirms the
   converted target in ROS metres.
4. Point the camera at the marker. On the next detection the marker is mapped onto
   that transform and the stream switches into the scan's frame. No further
   message is needed: the next FreeD packet already carries the new pose.

Other commands on the same port: `anchor_aruco` (marker becomes the origin),
`reset_pose`, `set_offset`, `ping`.

The anchor **refuses** if depth disagrees with the PnP distance by more than
`aruco.depth_tolerance_m`, and stays armed for a retry. Four coplanar corners are
pose-ambiguous, so this is what stops a mirrored solve from taking the whole
virtual set with it. Launch with `VP_ENABLE_DEPTH=1` to get the check; without
depth the solve still runs, reported as `(no depth)`.

## Absolute localisation against the scan

Visual odometry is smooth but drifts, and starts at an arbitrary origin. The
LiDAR scan is the real world, surveyed once, and does not drift. With
`lidar.enabled` on, the D455's depth stream is matched against the scan a few
times a second and the result gives the camera's pose **in the set** — position
and rotation — corrected continuously.

What is estimated is a *transform*, not a position:

```
T_map_cam(t)  =  T_slam_to_map  @  T_slam_cam(t)
                 └ slow, fused ┘   └ fast, from VIO at 90 Hz ┘
```

`T_slam_to_map` is the accumulated drift expressed as a rigid transform. It
changes by millimetres per second, so it can be filtered hard without adding any
latency to the actual camera movement — all the fast motion still comes from
cuVSLAM. The fused transform drives ground zero, so the FreeD stream, the
on-screen camera dot and the status readout cannot disagree.

### Marker placement and scan refinement

The dashboard locator represents the **physical marker**, including its height
and printed heading. **Auto lineup** derives the camera pose from a fresh marker
observation, then refines it against nearby scan geometry. The camera remains at
its measured distance from the marker. A rejected match preserves the current
alignment and reports its reason beside the checkbox.

### The hint also says *where to search*

`lidar.hint_radius_m` / `hint_radius_deg` (default ±0.5 m / ±20°) bound how far
from the hint the answer may be. A match outside that is not a refinement of
your hint — it is a different part of the room that happened to look similar —
and it is refused, with the panel telling you so.

This matters more than it sounds. Gated loosely, a close-up of a table corner in
a cluttered room locked onto geometry **1.1 m and 42° away**, reported success,
and left continuous fusion permanently asking for a 409 mm correction the
runtime gate would not let it make.

Candidates are ranked on how much of the view lands on the scan, plus a light
prior favouring the hint. Deliberately not the *median* distance: from an
ambiguous viewpoint a solution 233 mm out of place had the better median (29 mm
against 38 mm) because it aligned some structure very tightly, and gave itself
away only in the tail — 86 % of points within 100 mm against 94 %.

Measured on `Room01_Scaled_UE01.obj`, hint deliberately wrong by the amount
shown, camera looking across a cluttered room:

| hint placed | solved to | on scan | time |
|---|---|---|---|
| 8 cm / 2° | 9 mm / 0.08° | 100 % | 0.4 s |
| 30 cm / 8° | 2 mm / 0.26° | 100 % | 0.2 s |
| 45 cm / 15° | 3 mm / 0.10° | 100 % | 0.7 s |
| 1.1 m / 42° | refused — outside the search region | — | 0.5 s |

Heading is what a hand-placed hint gets most wrong and what ICP recovers least
well on its own, so several yaw offsets are screened on the coarse pyramid and a
shortlist refined at full resolution. The swing is about the **displayed**
vertical — the axis the locator's own heading ring turns about — which for a
camera tilted down at a table is nowhere near the camera's own up axis.

### Scan units — read this first

**A scan exported from Unreal is in centimetres and everything here works in
metres.** Left unreconciled, ICP is matching a metre-scale depth cloud against a
centimetre-scale map and can never converge — and the camera reads tens of
metres out, because the locator lives in that same oversized frame.

This is now handled for you in two places: `lidar.map_scale` is applied to the
viewport's **S** box at startup, and importing a scan whose extent is
implausible sets it automatically, saying which reading it chose:

```
[LiDAR] scan is 309 units across — reading that as centimetres (3.1 m);
        alignment scale set to 0.01. Override with lidar.map_scale in tracking.yaml.
```

The **S** box is authoritative once you touch it — auto-detection never
overrides a value you set. Check the load line either way:

```
[LiDAR] Map ready: 30512 points, 3.1 x 2.3 x 2.9 m (scale 0.01, voxel 0.05 m, ...)
```

If that line says your room is 309 m across, the scale is the problem.

### Scan overlay and the outliner

**Setup → Show scan overlay** draws the scan edges over the camera view and
is enabled after a successful lineup. Check the projected edges against the
physical room. The live Camera channels show full position and rotation; the
camera dot is driven by tracking. Select Locator, Point Cloud, Trail or World
Origin to edit their transforms as described in the Channel Box section above.
A stored scan alignment stays locked. The gizmo toggle controls handle visibility
and picking.

The older survey-point and manual image-alignment solvers remain in the source
for development; their controls are no longer shown in the dashboard.

### Does the picture line up? (3DE-style lineup)

Depth-vs-scan ICP answers "is the camera where we think it is" using geometry it
can measure to ~6 m. [image_alignment.py](image_alignment.py) answers the
question a matchmove artist asks instead: **project the scan and see whether its
edges land on the picture.** It ray-casts the scan from the current pose, takes
the silhouettes, and measures how far each one sits from the nearest edge in the
live IR frame:

```
Image lineup: 71% of the scan's edges on the picture (2.2px)
```

It is deliberately a **read-only check** plus an AR overlay. Letting it drive the
pose is `lidar.image_align_refine`, and it is off — see the honest limits below.

Why edges rather than matching the picture to the scan's texture: the scan
carries photogrammetry colour and the live frame is the D455's infrared imager
with its projector off. Those are different modalities under different lighting,
so comparing intensities compares the lighting. A wall/floor junction is an edge
in both.

**Measured limits.** The score is a real measurement — it peaks at the true pose
in all three translation axes and in yaw, which is what makes it worth showing.
The *optimiser* is not yet trustworthy enough to move the pose: against
`Room01_Scaled_UE01.obj` its cost minimum sits **30–60 mm away from the true
camera pose**, while depth ICP is already solving to ~3 mm, so enabling
refinement would make the answer worse. Two things would likely close that, and
both need real footage rather than synthetic renders to tune against: a cleaner
(decimated or smoothed) scan, and the albedo edges a real IR frame has and a
shaded render does not.

### Reading the status panel

```
── Map Fusion (locator hint) ──
Lock: 100% on scan, 17mm median miss   solved 252mm / 3.0° off hint
Search region: ±0.50m / ±20°
Live: 100% on scan   drift 0mm / 0.01°   applied 0.0mm
Axes constrained: 3/3   Inliers: 1808   ICP: 108ms
── Depth vs Scan (12ms) ──
Agreement: 87.3%  Mean: 41mm  (500 rays)
```

`Lock` is the quality of the placement itself and does not change during a take;
`Live` is the running check that the view still matches. They are kept apart
because a *rejected* live fix reports zero by construction, and it used to
overwrite the lock figures — so a good lock could read `On scan: 0%`.

**`On scan`** is the number that answers "does the camera view match the LiDAR":
the fraction of the depth image landing within `verify_threshold_m` of the
scanned surface. It is deliberately *not* ICP's own fitness, which at a wide
search radius reads near 1.0 almost wherever the cloud is put — measured on this
scan, a solution 530 mm out of place still scored 0.75.

`Solved off hint` shows how far the answer moved away from the locator.
`Live drift` is the drift being removed during the take. **`Axes constrained`**
is the one worth watching: ICP against a single flat wall is free to slide along it, so the
observability of each axis is measured from the spread of matched surface
normals and the correction is applied only along the directions the geometry
actually pins down. `1/3` pointed at a bare wall is the honest answer, not a
fault — turn towards a corner and it goes to `3/3`.

**`Depth vs Scan`** is a read-only second opinion, and the last two lines only
appear once a scan is loaded and depth is arriving. It samples up to 500 rays
from the current depth frame, puts them in map coordinates through the pose the
fusion is currently reporting, and measures how far each lands from the scanned
surface — nothing it finds is fed back into the pose. It re-evaluates agreement using the same depth sensor; it is correlated
with the registration and is not independent hardware evidence. Below 60 % agreement the panel turns amber,
below 30 % red.

Corrections are rate-limited by `lidar.max_slew_m_s` / `max_slew_deg_s`, so even
a large fix arrives as a crawl well under a millimetre per frame rather than a
visible slide.

### Depth must be on

`lidar.enabled: true` makes `launch_isaac.sh` set `VP_ENABLE_DEPTH=1`, which is
what makes the D455 publish depth at all. Without it there is no depth topic and
the fusion silently does nothing.

## On-set procedure

1. **Mount** the D455 rigidly to the rig. Any flex between the tracker and the
   taking lens shows up directly as jitter in the composite.
2. **Measure the rig offset** — where the taking lens's nodal point sits relative
   to the RealSense, in ROS body axes (X forward, Y left, Z up) — and enter it as
   `transform.rig_offset.translation_m`. Without this the virtual camera rotates
   about the tracker instead of about the lens, which reads as parallax error on
   pans.
3. **Start the pipeline**, hold the rig still briefly, then move slowly to let
   SLAM initialise. Check sensor readiness in Setup.
4. **Place the locator** at the physical floor/table marker centre and set its
   heading (+X toward the printed top, +Z up). Show the marker and room geometry,
   then enable **Operate → Auto lineup**.
5. **Wait for green Locked / Fusion active**, then move the camera. Watch for
   amber recovery or red tracking hold during the take. The marker can leave
   view after lineup.

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
| Dashboard stops responding when **maximized** | Restart through `./launch_isaac.sh` to load the current renderer. The viewport now uses Qt’s managed `paintGL()` framebuffer lifecycle, retains swap interval 0, and avoids repeated status-style layout invalidation. Regression checks cover framebuffer resizing and native window-manager replies in the Isaac container; see `VERSION.md` for results. |
| Dashboard goes unresponsive, worse the larger the window | Fixed in 1.3. The 3D viewport projected one line segment per Python call, so the 2000-point camera trail and the ground grid cost 35 ms a paint against a 16.7 ms budget and the GUI thread never caught up. If it recurs, check the paint cost before blaming the GPU — the mesh is drawn by OpenGL and is not usually the expensive part. |
| Virtual camera banks when you pan | The pre-fix symptom of the gimbal-lock bug. Confirm `pose_transform.py` is deployed and you are not running a stale `dashboard_isaac.py`. |
| Position drifts with the camera stationary | The stationary hold should pin this to ~1 mm; check `tracking.hold_enabled` and that the dashboard shows `HOLD (still)`. If the hold is engaging but drift persists while moving, the cause is scene content — see the row below. |
| Drift is much worse than usual, same lighting | Measured on this rig: with 0 % near-black pixels and texture gradient 0.854, per-sample jitter was 8.5 um; with 19.2 % near-black and gradient 0.670 — *identical mean brightness* — it was 154.5 um, 18x worse. Point the camera at evenly lit, textured geometry; large dark or blank regions are what hurt, not overall brightness. |
| **Pose is stable when still but diverges wildly (metres, impossible speeds) as soon as you move** | **Motion blur from an uncapped exposure.** The RealSense default `auto_exposure_limit` is 200 ms; at 90 fps every frame smears while moving, cuVSLAM loses visual tracking and falls back to IMU dead reckoning, which integrates into tens of metres of error. The launch scripts now cap it to 6 ms (`VP_AE_LIMIT_US`). Verify with `ros2 param get /camera/camera depth_module.auto_exposure_limit`. Also move more slowly — beyond roughly 2 rad/s no VIO will hold. |
| `magic_enum::magic_enum ... target was not found` when building | Do not build `isaac_ros_visual_slam` from source in this container. Install NVIDIA's prebuilt package: `apt-get install -y ros-humble-isaac-ros-visual-slam`. |
| Host `rs-enumerate-devices` finds 0 devices but the container finds the camera | Expected if the librealsense DKMS kernel patches are absent (`dpkg -l \| grep librealsense` empty). The host kernel leaves the D455's video interfaces 0–2 and its HID interface at `Driver=[none]`; the container's librealsense uses the libusb backend instead. It also means frame metadata (per-frame exposure/gain) is unavailable. Run `./install_realsense_dkms.sh` and replug to fix. |
| Camera dot reads tens of metres out (e.g. `Z +94 m` in a 3 m room) | Scan units. The **S** box was left at 1.00 with a centimetre scan, so the locator and the anchor live in a 309 m frame. Set **S** to 0.01, or re-import so it is detected. |
| Fusion locks onto the wrong part of the room | It found lookalike geometry. Narrow `lidar.hint_radius_m`, place the locator more accurately, or show the camera more of the room. The panel reports `solved N mm off hint` so a large jump is visible. |
| `best match is N mm from the hint, outside the search region` | Working as intended — the scan does not match the camera view anywhere near your hint. Move the locator closer to where the camera really is, point at more of the room, or widen `lidar.hint_radius_m`. |
| Camera dot sits exactly on the locator | Fixed in 1.3 — the locator is a hint and the fusion solves from it. If you still see it, the solve was refused (the panel says why) and the pose fell back to the hint alone. |
| `could not place the camera on the scan` | The hint is too far out, or the view is too featureless. Move the locator closer to where the camera really is, or point at more geometry. Refusing is deliberate: a wrong lock is worse than none. |
| Map Fusion says `Not placed` and never changes | No depth. `lidar.enabled` must be `true` so `launch_isaac.sh` sets `VP_ENABLE_DEPTH=1`; check `ros2 topic hz /camera/depth/image_rect_raw`. |
| Map Fusion never converges, fitness stays 0.00 | Scan units. Look at the `[LiDAR] Map ready: ... N x N x N m` line — if the room reads as hundreds of metres it is a centimetre export; set the mesh-alignment **S** slider (or `lidar.map_scale`) to `0.01`. |
| `view spans only 0.02 m — too flat or too close to match` | The camera is inches from a flat surface, so there is no structure to register against. Back away until more of the room is in frame. The marker anchor still holds the pose; it is just not scan-verified. |
| `Axes constrained: 1/3` or `2/3` | Working as intended: the view is too featureless to pin down every axis (one flat wall). Turn towards a corner. The unconstrained directions are left uncorrected rather than guessed at. |
| Correction seems very slow to arrive | By design — `lidar.max_slew_m_s` (default 0.02) caps how fast a fix may move the camera, so a 100 mm drift takes ~5 s to come out invisibly. Raise it if you want faster convergence and can accept visible motion. |
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

```bash
python3 -m unittest test_lidar_fusion -v        # map fusion
```

The pure-maths half runs anywhere; the ICP half needs open3d and skips without
it, so the authoritative run is inside the container. Two further tools check the
fusion against a real scan rather than a synthetic one:

```bash
# inside the container
python3 bench_fusion.py           # accuracy + per-fix timing vs Room01
python3 smoke_locator_lock.py     # the whole locator lock-on path, end to end
```

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
* **The map fusion corrects the frame, not the sample.** Each ICP run is an
  absolute measurement of `T_slam_to_map`, blended geodesically on SE(3). An
  earlier version nudged each incoming position by `icp.transformation[:3, 3]`,
  which is where the map *origin* went, not the camera — a rigid transform moves
  a point by `(R - I)p + t`, and with the camera 5 m out one degree of ICP
  rotation puts 87 mm into the term that was being ignored. It also threw
  rotation away entirely and fed its own output back into the estimator.
* **Under-constrained geometry is detected, not guessed at.** A correction is
  projected onto the subspace the matched surface normals actually constrain, so
  a camera looking at one flat wall gets its distance-to-wall corrected and
  nothing else.
* **The D455's error model is used.** Stereo range error grows as `z²`, so depth
  samples are range-gated to 0.3–6 m and thinned by `1/σ²`; point density then
  tracks information content and the noisy far field cannot outvote the near
  field.
* **The maths is isolated and tested.** `pose_transform.py` and `freed.py` have
  no ROS/Qt/OpenCV imports. Keep new geometry there, not in the GUI.
