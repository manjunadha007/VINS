# v1.5 — 2026-09-27 — Hardware-checked startup, IMU fix inside the container

v1.4's tracking, made to start reliably on a freshly recreated Isaac container
and to refuse to open the dashboard until the camera, IMU and SLAM are actually
streaming. Validated live on the D455, both direct to the PC and through a
5 Gbps hub.

## Headline result

From a cold, freshly recreated container, `./launch_isaac.sh` provisioned
everything it needed (NumPy pin, cuVSLAM, patched librealsense) and the new
startup check passed with every stream live:

| Stream | Direct, USB 3.2 | Through hub, 5 Gbps |
|---|---|---|
| Left IR | 79.7 Hz | 86.8 / 81.6 Hz |
| Right IR | 90.0 Hz | 89.3 / 89.9 Hz |
| IR calibration | 89.9 Hz | 90.0 / 89.9 Hz |
| IMU | 399.8 Hz | 400.2 / 399.7 Hz |
| SLAM odometry | 100.4 Hz | 97.1 / 99.0 Hz |
| Depth | 80.5 Hz | 87.6 / 81.7 Hz |
| Depth calibration | 86.8 Hz | 88.3 / 88.2 Hz |

The IMU runs at 400 Hz alongside stereo and depth **without host-side
librealsense DKMS** — the in-container patch is sufficient. On the hub run an
ArUco Auto lineup placed the camera on the scan 16 mm / 1.33° from the marker
hint, with 100 % of the view on the scan, median miss 23 mm, in 214 ms. That is
distance from the hint, not an accuracy figure against truth; v1.4's motion
validation is still the accuracy reference.

## What is new

**The D455 IMU no longer takes the cameras down.** The Isaac image ships
librealsense 2.55.1, whose RSUSB reader predates firmware 5.16's 38-byte IMU
report. When the IMU stalls, librealsense aborts the whole session, so the IR
and depth topics go to 0 Hz with no error naming the IMU. `ensure_realsense_imu.sh`
builds 2.55.1 with upstream PR #12857 (`patches/`, Apache 2.0 license included)
inside the container, keeps the original library for rollback, replaces it
atomically, and verifies the checksum on every later launch. The IMU stays on;
disabling it would break visual-inertial SLAM.

**Startup fails loudly instead of half-booting.** Each was previously a step
that could fail silently and leave a pipeline that looked healthy:

* `check_pipeline.py` replaces `sleep 15`. It waits for sustained, advancing
  left/right IR, calibration, IMU and odometry (plus depth when required),
  rejects frozen or repeated timestamps, and prints the SLAM log on failure.
* The `numpy<2` pin runs on every launch and verifies `cv2` / `cv_bridge`
  import. The image ships NumPy 2.x, against which ROS Humble's OpenCV fails
  with `_ARRAY_API not found` and the dashboard dies before its window opens.
* The cuVSLAM component is checked on every launch and installed when missing.
  Without it the camera streams, the dashboard opens and the scan loads, but no
  camera pose ever arrives. The recreated container on 2026-09-27 was missing
  it; this guard installed it.
* `cleanup()` now actually stops SLAM when the dashboard closes. Its
  `pkill -f` patterns matched the `bash -c` running them, so the first pkill
  killed its own shell and `component_container` kept holding the camera and
  GPU after "Pipeline shut down" was printed.

**Tracking.**

* The AR overlay and the image lineup score use the pose recorded at the
  frame's own timestamp (`engine.pose_at`), not the newest pose. Pairing a
  fresh pose with an older picture slid the scan across the image by however
  far the rig had moved — invisible while parked, drift while moving.
* Marker heading seeds every 10° instead of 15°. At 15° the seed error reached
  121 mm with the rig 0.9 m from the marker, and the solve was rejected with a
  message pointing at scan scale; 10° keeps it under ~80 mm out to ~1.2 m.

Also: a compact dashboard header; README corrections (848x480x90 works over USB
3 — the earlier failure was a USB 2 link); `smoke_locator_lock.py` drives the
Auto lineup path that replaced `_on_anchored`; the snapshot now includes
`Room01_Scaled_UE.obj`, which `launch_isaac.sh` copies into a fresh workspace.

## Limitations

* **First launch on a fresh container takes a few minutes** and needs network:
  it apt-installs cuVSLAM and compiles librealsense. Later launches verify both
  and start in about a minute.
* **The startup check's rates come from a ~1 s window**, so they are noisy.
  Left IR read 80–87 Hz across three runs while right IR held 89–90 Hz. Watch
  the dashboard over minutes before reading anything into one launch.
* **The `cleanup()` fix is proven in the container, not yet on a real
  dashboard close.** Every run on 2026-09-27 was started before the fix.
* **`test_ui.py` can abort on exit while a live pipeline is running.** It
  builds the real dashboard, which subscribes to the live topics, then exits
  without shutting ROS down; a late image callback hits the deleted window
  (`terminate called without an active exception`). It passes with no camera
  streaming.
* **"No USB log lines at all" when plugging in the camera means cable or port,
  not drivers.** A long active optical cable showed nothing; a short cable
  enumerated at USB 3.2 immediately.
* v1.4's limits still apply: the lineup needs ~95 % of the view within 5 cm of
  the scan, the supervisor can oscillate into DEGRADED under sustained motion,
  and the camera dot shows nothing until a lineup validates.

## Verifying this snapshot

    cd versions/v1.5
    python3 run_offline_tests.py                                  # 131 tests
    python3 -m unittest test_map_viewer test_marker_lineup test_channel_box test_check_pipeline

On the host (no ROS, no Open3D): 131 offline tests (24 environment skips) and
63 viewport/marker/channel/startup-check tests pass. Inside the Isaac container
the same snapshot passes 419 unit tests across 14 modules, plus
`smoke_lineup.py`, `smoke_locator_lock.py`, `smoke_gizmo.py` and
`verify_marker_lineup.py`. Note that `verify_marker_lineup.py` imports from
`/workspaces/isaac_ros-dev/src`, i.e. the deployed copy, not this directory.
