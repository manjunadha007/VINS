# Version 1.2 — stationary drift eliminated

Frozen release snapshot. Previous: `versions/v1.1/`. Superseded by 1.3 (the working tree at the repository root).

## What changed from 1.1

**Stationary drift.** With the rig physically still, pure visual odometry creeps —
sub-pixel feature-match errors integrate into a slow slide, which on a locked-off
plate is exactly the artefact you notice. `pose_transform.StationaryHold` uses the
IMU to decide the rig is genuinely not moving (quiet gyro plus a steady 1 g), pins
the output while that holds, and absorbs the creep into a standing offset so
releasing produces no jump.

Measured on this rig, camera untouched on the bench, 20 s:

| | v1.1 (raw) | v1.2 (held) | |
|---|---|---|---|
| Max excursion | 27.89 mm | **0.70 mm** | 97.5 % reduction |
| Integrated path | 60.76 mm | **0.72 mm** | |

Replayed against the recorded 90 s motion capture, **100 % of real movement was
preserved** — the hold engages only when the IMU says the camera is still.

Entering the hold requires a full quiet window (0.3 s at 200 Hz); leaving it takes
a single sample over the release threshold. That asymmetry is deliberate: delaying
the start of a real move would be a worse artefact than the drift being removed.

Tuning lives in `tracking.yaml` under `tracking.hold_*`.

## Current measured performance

| Property | Value |
|---|---|
| Stereo IR into SLAM | 89.9 Hz (640x360x90) |
| IMU into SLAM | 198.5 Hz, header dt 5.002 ms +/- 0.003 |
| IMU calibration | accel 9.813 m/s2 vs gravity 9.807 |
| SLAM output | 89.9 Hz, cuVSLAM 1.65-2.3 ms/frame |
| **Stationary drift** | **0.70 mm / 20 s** |
| Worst single-frame step (gate) | 146 mm, from 106854 mm unfiltered |
| Peak excursion on a diverging take | 1.08 m, from 138.93 m unfiltered |
| pan vs -yaw, 8087 live poses | max error 0.000000000 deg, corr -1.000000 |
| FreeD encode, 8087 live poses | max error 1 LSB, 0 malformed |

## Still open

* **Absolute drift over a moving take** is not yet characterised. Run
  `verify_tracking.py --benchmark --distance 0.5` for the scale and
  return-to-origin numbers; that is the figure to compare against a commercial
  tracker's spec.
* **librealsense DKMS kernel patches are not installed** (`dpkg -l | grep
  librealsense` is empty). Consequence: no per-frame metadata, and the host
  cannot see the camera at all. Needs host sudo plus a replug.
* **848x480 stereo does not resolve** on this D455 over its current USB path
  (two chained hubs); 640x360x90 is in use. More pixels would help accuracy.
* **The Unreal Engine end is unverified.** Use `calibrate_axes.py` to confirm
  axis directions in your project.

## Layers, in the order a pose passes through them

    Isaac ROS odometry (90 Hz, camera_link in odom, x-fwd/y-left/z-up)
      -> PoseGate         absorb relocalisation snaps          (pose_transform.py)
      -> StationaryHold   pin output while the IMU says still  (pose_transform.py)
      -> PoseTransformer  ROS canonical -> Unreal, rig offset, ground zero
      -> PosePredictor    extrapolate to now from Isaac's twist
      -> PoseSmoother     optional, off by default
      -> freed.py         29-byte FreeD D1 packet -> UDP -> Unreal
