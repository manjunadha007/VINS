# v1.7.1 — 2026-11-14 — Lineup recovers by itself, steady Locator rotations

v1.7's pipeline with the lineup no longer lost for good after a tracking jump.
When a marker leaves the frame and tracking jumps, Auto lineup now finds the
saved lineup again by itself. Locator rotations are measured only from steady,
clear views.

## What is new

**Lineup recovery** (`dashboard.py`, `tracking_engine.py`). In v1.7 a tracking
resync or "Camera moved too far during hold" switched Auto lineup off and left
FreeD frozen until someone lined up again by hand. Now Auto lineup keeps the
marker layout and shows **Auto lineup · Recovering** while FreeD output is
held. It needs 3 fresh marker-and-scan checks that agree to 3 cm / 1.5°,
arriving at most 1.2 s apart (`fusion.recovery_consistency_m`,
`fusion.recovery_consistency_deg`). FreeD then eases back to the camera at the
recovery slew rate instead of jumping. A view that can't be checked (markers
oblique, depth looking off the scan) doesn't reset the count. Turning Auto
lineup on while the dashboard asks for a re-lineup is no longer cancelled.

**Steady Locator rotations** (`marker_map.py`). With two or more assigned
Locators, a marker smaller than 30 px across or seen more than 65° off face-on
is not measured. A Locator is turned only when 3 frames in a row agree to 3°,
then to their mean, once per lineup. The lineup runs once two measured markers
are in view, using only those. New `aruco` settings: `orientation_samples`,
`orientation_agree_deg`, `orientation_min_marker_px`,
`orientation_max_view_deg`.

**FreeD button shows what is sent.** **FreeD output** now reads Live,
Live · unverified, Held, Recovering, Waiting or Paused, and is amber whenever
it isn't Live. v1.7 said "Enabled" while it sent a frozen pose.

**Stereo no longer waits for depth** (`realsense_streams.py`). In alternating
projector mode the splitter joined stereo and depth into one exposure, so a
late depth frame dropped a stereo frame. They are now routed separately: stereo
went from 29–35 Hz to 44.9 Hz.

**Pose gate and prediction** (`pose_transform.py`, `tracking_engine.py`).
- **Short frame gaps:** the jump limit now allows normal travel across them,
  up to `tracking.gate_max_gap_s`. The allowance can't grow during a run of
  rejections.
- **Rotated resyncs:** a resync that absorbs a rotation now applies it to
  later translation too.
- **Pose history after a resync:** older poses are dropped, so an image is
  never paired with a pose from before the resync.
- **Prediction:** Isaac reports velocity in the camera's own frame; it is now
  rotated into the world before extrapolating. Before, prediction drifted
  sideways after a pan.
- **Gate log:** each `[Gate]` line now carries sensor and host timestamps.

**Dashboard.**
- **Camera dot:** the dot in the 3D view follows the live camera even while
  FreeD is held, and turns amber when Unreal isn't getting that pose.
- **Reset Scene:** a new button (and File menu item) clears the lineup and
  starts Auto lineup again. It keeps the scan, scale, Locators and marker IDs.
- **Camera panel:** it no longer freezes on the last overlay frame when the
  lineup is lost.
- **Log:** `[SLAM] ✅ Tracking` no longer repeats on every packet.

## Seen live on 2026-11-14

Outdoors on the terrace in daylight, against `Terras_scan02.obj` and
`scene03.json`, launched with `VP_EMITTER_MODE=off`:

* **Startup check:** IR 86–90 Hz per side, IMU 400 Hz, odometry about 98 Hz,
  depth 84 Hz. Earlier the same day in alternating mode: stereo 44.9 Hz.
* **Scale check:** 1.024× inside ±3.3 %, so the scan scale was kept at
  0.021024 m per unit.
* **Lineup:** the floor marker (ID 1) was skipped as too small or edge-on. The
  wall Locators turned only 2° (ID 2) and 1° (ID 0), and the lineup locked at
  94 % of the view on the scan, 47 mm median miss.
* **Recovery:** between 13:26 and 13:32 tracking jumped five times: 3.3 m,
  124 mm, 59 mm, a 12 s burst of 60–370 mm jumps, and 6.0 m. After each, the
  camera was placed back on the scan from the markers at 84–89 % on the scan
  and 41–49 mm median miss, with no manual lineup.
* **Before these fixes**, the 12:53 recording showed the problems: recovery
  stuck at 1/3–2/3 for 30 s, and the camera panel frozen.

## Limitations

* **The marker layout fits the scan only just.** Locks pass at a 41–49 mm
  median miss against the 50 mm limit (`lidar.survey_max_median_m`), and the
  scale is only known to ±3.3 %.
* **Outdoors, cuVSLAM jumps** (up to 6 m) when the view leaves the terrace for
  distant buildings or sky. Each jump holds FreeD until 3 checks agree. Keep
  some terrace in frame while panning.
* **Fast moves may be cut off:** one burst rejected poses at 3.1–4.2 m/s
  against the 3 m/s gate. Whether those were real moves is not known.
* **GNSS** is SBAS differential, not RTK, with weak signal (16–33 dB-Hz). Its
  1 Hz output flickers "Signal stale" against the 0.8 s timeout; not changed.
* **Not measured:** how accurately FreeD tracks a moving camera against ground
  truth.
* v1.7's limits still apply: the GoPro preview abort (`gopro.auto_start` off in
  the deployed config), and the scan overlay drawing only outlines.

## Verifying this snapshot

Inside the Isaac image, with open3d, numpy<2 and PyOpenGL on `PYTHONPATH` and
`QT_QPA_PLATFORM=offscreen`:

    cd versions/v1.7.1
    python3 -m unittest $(ls test_*.py | sed 's/\.py$//' | grep -v -x -e test_freed -e test_layout -e test_ui)
    python3 smoke_gizmo.py && python3 smoke_lineup.py && python3 smoke_locator_lock.py

On 2026-11-14 this snapshot passed 601 unit tests in place (1 skipped: no
OpenGL context) and all three smoke scripts. `smoke_lineup.py` reads
`/workspaces/isaac_ros-dev/Room01_Scaled_UE01.obj` and `smoke_locator_lock.py`
reads `Room01_Scaled_UE.obj` there.
