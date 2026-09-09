# Version 1.1 — validated working baseline

Snapshot of the pipeline in the state that was measured end to end on real
hardware (Intel RealSense D455 s/n 261822302756, RTX 3060, Isaac ROS 3.2.6).

## Verified on hardware

| Check | Measured |
|---|---|
| Stereo IR into SLAM | 89.9 Hz (640x360x90) |
| IMU into SLAM | 198.5-199.8 Hz, header dt 5.002 ms +/- 0.003 |
| IMU calibration | accel magnitude 9.813 m/s2 vs gravity 9.807 |
| SLAM output | 89.9 Hz, cuVSLAM 1.65-2.3 ms/frame |
| Stationary drift (pure VO) | 8.50 mm / 12 s |
| pan vs -yaw over 8087 live poses | max error 0.000000000 deg, corr -1.000000 |
| FreeD encode over 8087 live poses | max error 1 LSB, 0 malformed |

## What this version contains

* Correct ROS-canonical -> Unreal conversion (pose_transform.py). Replaces the
  P_w/P_b matrix pair that was permanently Euler-singular for a level camera and
  pinned FreeD pan at 0.0 while leaking pan motion onto roll.
* Auto-exposure capped to 6 ms, declaratively in isaac_vp.launch.py. The
  RealSense default of 200 ms smeared every frame during movement and made the
  pose diverge to 138 m / 1089 m/s.
* Pure visual odometry by default (enable_localization_n_mapping: False), so the
  pose never snaps from a loop closure.
* PoseGate: absorbs tracker snaps into a standing offset. On recorded hardware
  data this cut peak excursion from 138.93 m to 1.08 m and the worst single-frame
  step from 106854 mm to 146 mm.
* Isaac ROS owns sensor fusion; the bridge does not re-fuse.
* Network-configurable Unreal target, ArUco ground zero with camera_info
  intrinsics, latency prediction from Isaac's twist.

## Known limitation carried into 1.2

Stationary drift of ~8.5 mm per 12 s: with the rig physically still, the SLAM
estimate still creeps. Addressed in version 1.2.

## Tests at time of snapshot

81 host-side (test_pose_transform, test_freed_protocol) + 29 integration
(test_virtual_camera) — all passing.
