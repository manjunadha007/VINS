# Version 1.3 — in development

Working tree. Frozen snapshots: `versions/v1.2/`, `versions/v1.1/` (see
[versions/README.md](versions/README.md)).

Nothing has changed from 1.2 yet — this is the 1.2 code with the version opened
for the next round of work.

## Inherited from 1.2 (all hardware-measured)

| Property | Value |
|---|---|
| Stereo IR into SLAM | 89.9 Hz (640x360x90) |
| IMU into SLAM | 198.4 Hz, header dt 5.002 ms +/- 0.003 |
| IMU calibration | accel 9.811 m/s2 vs gravity 9.807 |
| SLAM output | 89.9 Hz, cuVSLAM 1.65-2.3 ms/frame |
| **Stationary drift** | **0.70 mm / 20 s** (raw 27.89 mm, 97.5 % suppressed) |
| Same, degraded scene | 1.03 mm / 20 s (raw 27.87 mm, 96.3 % suppressed) |
| Real motion preserved by the hold | 100 % on recorded 90 s capture |
| Worst single-frame step (gate) | 146 mm, from 106854 mm unfiltered |
| Peak excursion, diverging take | 1.08 m, from 138.93 m unfiltered |
| pan vs -yaw, 8087 live poses | max error 0.000000000 deg, corr -1.000000 |
| FreeD encode, 8087 live poses | max error 1 LSB, 0 malformed |
| Tests | 92 host + 29 integration |

## Pipeline stages a pose passes through

    Isaac ROS odometry (90 Hz, camera_link in odom, x-fwd / y-left / z-up)
      -> PoseGate         absorb relocalisation snaps          (pose_transform.py)
      -> StationaryHold   pin output while the IMU says still  (pose_transform.py)
      -> PoseTransformer  ROS canonical -> Unreal, rig offset, ground zero
      -> PosePredictor    extrapolate to now from Isaac's twist
      -> PoseSmoother     optional, off by default
      -> freed.py         29-byte FreeD D1 packet -> UDP -> Unreal

## Candidate work for 1.3

Ranked by measured impact, highest first.

1. **Drift while moving is uncharacterised.** The one number still missing.
   `python3 verify_tracking.py --benchmark --distance 0.5` reports scale error and
   return-to-origin drift; that is what compares against a commercial tracker's
   spec. Needs a person to move the rig, so it cannot be automated here.

2. **Drift is dominated by what the camera looks at.** Measured directly: with
   0 % near-black pixels and texture gradient 0.854, per-sample jitter was 8.5 um.
   With 19.2 % near-black and gradient 0.670 — same mean brightness — jitter was
   154.5 um, 18x worse. An automatic scene-quality warning on the dashboard
   (dark-pixel fraction and texture gradient, computed from the frames already
   being received) would tell the operator when the volume is unsuitable before
   a take rather than after.

3. **librealsense DKMS kernel patches are not installed.** `dpkg -l | grep
   librealsense` is empty. The host kernel leaves the D455's video interfaces 0-2
   and its HID interface at `Driver=[none]`, so the host cannot see the camera at
   all and per-frame metadata (actual exposure and gain) is unreadable. Needs host
   sudo plus a physical replug.

4. **848x480 stereo does not resolve** on this D455 over its current USB path,
   failing with `Failed to resolve the request: Format: Y8, width: 848,
   height: 480` at both 60 and 30 fps; at that resolution the pair appears to be
   offered only as interleaved Y8I. The camera sits behind two chained hubs — a
   direct USB 3.x port is worth retrying. More pixels would reduce jitter.

5. **The Unreal Engine end is unverified.** Unreal is running on this machine
   (`UnrealEngineZen` in `/dev/shm`) but no LiveLink subject has been confirmed
   driven. `python3 calibrate_axes.py` settles the axis signs in two minutes
   without needing the camera or SLAM.

6. **ROS 2 CLI param services are unreliable in this container** — `ros2 param
   get` reports `Node not found` while topics flow normally. Already worked around
   by setting exposure declaratively in `isaac_vp.launch.py`; worth understanding
   rather than living with, since it makes runtime tuning awkward.
