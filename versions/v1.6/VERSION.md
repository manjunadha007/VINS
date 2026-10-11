# v1.6 — 2026-10-11 — Camera keeps moving after lineup; looser scan matching

v1.5's pipeline with the camera no longer freezing after a lineup. Two fixes:
a VO-only pose stream is no longer throttled to one pose per second, and once a
lineup is accepted, live stereo/IMU motion carries on while scan checks are
missing or failing, shown as an amber **Auto lineup · Camera tracking**.

## What is new

**VO poses are no longer throttled to 1 Hz.** `vo_pose_callback` used to drop
every VO pose that arrived within 1 s of `last_vio_time`, which a VO-only
stream set itself, so with no odometry topic the output updated once a second.
The engine now takes a `source` ("odometry" / "vo") and ignores a VO pose only
while odometry is fresh (within `stale_timeout`). `accept_pose` returns whether
it took the pose, and the dashboard and headless bridge only stream FreeD when
it did.

**VIO fallback after lineup** (`fusion.allow_vio_fallback: true`, the default).
Once aligned, the supervisor reports `DEGRADED` with output allowed instead of
`VERIFYING` / holding when scan or RTK checks are unavailable. That covers both
the first few checks after a lineup and later scan dropouts. The dashboard shows
amber **Auto lineup · Camera tracking**; green still needs consecutive fresh
scan checks. Drift can accumulate until matching resumes. Stale or rejected
camera poses still hold the output. Setting `allow_vio_fallback: false` restores
v1.5's behaviour, with the `single_source_grace_s` timed hold.

**Scan matching gate lowered from 95 % to 75 %** (`lidar.min_verify_score:
0.75`; the code default moves from 0.55 to 0.75). This lets incomplete or noisy
scans lock. The search region, geometry, RMSE and correction limits still apply.

`LINEUP_LITE.md` and `README.md` document the new amber state and both
settings. `launch_isaac.sh` keeps an existing deployed
`~/workspaces/isaac_ros-dev/src/tracking.yaml`, so set both keys there too. On
2026-10-11 the deployed copy already had them.

## Limitations

* **Not validated live.** The dashboard was restarted on the new code at 10:32
  on 2026-10-11, but the camera node and SLAM had stopped by 10:36 while the
  dashboard was still open; this was not investigated. No motion, lineup, or
  drift figures have been measured on hardware for this version.
* **The 75 % gate is inside the band v1.5 measured for wrong locks.** v1.5's
  `tracking.yaml` recorded that on the D455 in Room01_Scaled_UE01, solves
  380–450 mm from the true pose scored 0.87–0.91 (correct ones 0.98), which is
  why it sat at 0.95. v1.6's new test only checks that a correct lineup on an
  imperfect scan is accepted; nothing re-tests the wrong-lock case. Check the
  overlay after every lineup, especially around repeated room geometry. If a
  confident wrong lock shows up, raise `min_verify_score` back towards 0.95.
* **Amber "Camera tracking" can run unvalidated for as long as scan checks keep
  failing.** It is pure VIO in scan coordinates from the last accepted
  alignment.
* v1.5's limits still apply (noisy ~1 s startup-check window, `test_ui.py`
  abort on exit with a live pipeline).

## Verifying this snapshot

    cd versions/v1.6
    python3 run_offline_tests.py
    python3 -m unittest test_map_viewer test_marker_lineup test_channel_box test_check_pipeline

Inside the Isaac container on 2026-10-11 the working tree this was frozen from
passed 423 unit tests across 13 modules (v1.5: 416; `test_fusion_policy` 26 → 32,
`test_lidar_fusion` 88 → 89), plus `smoke_gizmo.py`, `smoke_lineup.py` and
`smoke_locator_lock.py`.
