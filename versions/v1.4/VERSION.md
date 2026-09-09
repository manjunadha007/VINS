# v1.4 — 2026-09-09 — Marker-referenced absolute lineup, validated in motion

Places the camera in a LiDAR scan from a printed ArUco marker and holds it
there while the camera moves. Validated end to end on a D455 against a room
scan, against an independent reference rather than against itself.

## Headline result

With the rig moving through 1.13 m of real 3-D travel, the camera dot tracked
the physical camera to a **median 10 mm / worst 26 mm**, with a **0.988 scale
ratio** over the moving legs (1.000 would be exact). Every one of the ten
checkpoints was scored against a separate hint-free global registration of the
same depth frame, and all ten cleared a 0.90 reliability gate, so the figures
do not rest on the fusion agreeing with itself.

Lineup itself: **99 % scan agreement, rank 3, 13–21 mm from the marker hint**,
locking in roughly a second, with the camera dot live ~0.7 s later.

## What is new

`Find marker in scan · place Locator` (marker panel) removes the step that
caused most failures. Hand-placing the locator asks the operator to mark a spot
that is not in the scan; a wrong height or a 90° heading both look plausible on
screen and are worth ~750–950 mm of camera error, which is past the 500 mm
search region and so rejects every time. The button instead runs a hint-free
global registration (`scan_localizer.py`), derives the marker's pose from the
camera pose, and places the locator — height and heading included. It seeds the
locator only: the solve still passes through the normal verification gates, so
a bad global fix is rejected rather than trusted.

Also: File menu (Import / Save Scene / Delete Scan), a per-object transform
channel box, marker-assisted lineup from frozen timestamp-paired observations,
and map fusion that works on any scan imported at runtime — the fusion worker
now starts whether or not a scan was preloaded, which is what makes an empty
`lidar.map_file` usable.

## Configuration changed from measurement

| Key | Was | Now | Why |
|---|---|---|---|
| `lidar.coarsest_radius_m` | 1.00 | 0.60 | At 1.00 the coarse ICP stage swung a hint accurate to 10 mm by 36.6° onto wrong geometry and failed 4/4; a metre-wide correspondence search spans most of a 3 m room. |
| `lidar.min_verify_score` | 0.55 | 0.95 | Solves 380–450 mm from truth still scored 0.87–0.91 and were **accepted**; correct ones score 0.98. A wrong lock anchors the whole fusion to the wrong place. |
| `lidar.map_file` | a scan path | `""` | Nothing preloads; import at runtime and the fusion picks it up. |

## Limitations

* **Scan coverage decides where this works.** The lineup needs ≥ 95 % of the
  view within 5 cm of the scan. Where the reference scan covered the room, that
  held at 96–99 %; aimed at unscanned furniture it fell to 41 %, and both the
  fusion and the independent reference failed together. Expect to re-press.
* **Under sustained motion the supervisor drops to DEGRADED** when a fresh scan
  check does not land inside `fusion.scan_timeout_s` (0.8 s). Output continues
  and accuracy held (10–21 mm at those checkpoints), but FreeD may stutter as
  the state oscillates.
* **The camera dot is dead until a lineup validates.** With `fusion.enabled`
  and `require_lineup`, `camera_pose_ros_corrected()` returns nothing until the
  supervisor allows output, so the 3-D view gives no feedback while waiting.
  A frozen dot looks identical to a crash.
* Marker size must be measured, not assumed. The reference marker measured
  90.4 mm against a nominal 100 mm; everything scales with it.

## Verifying this snapshot

    cd versions/v1.4
    python3 run_offline_tests.py                                  # 123 tests
    python3 -m unittest test_map_viewer test_marker_lineup test_channel_box
    python3 smoke_gizmo.py

123 offline (24 Open3D/environment skips) and 57 viewport/marker/channel tests
pass without ROS. The three `verify_marker_*.py` / `verify_motion_tracking.py`
tools need open3d, a live D455 and a scan, so they run only in the container.
