# Camera tracking project assessment — 2026-09-08

The current build supports supervised, scan-relative camera tracking with one D455, plus an optional calibrated GNSS position check. It is a prototype requiring live calibration and measurement, not a demonstrated Vicon or CinCraft equivalent.

## What was inspected

The root Python modules, tracking configuration, launch/deployment scripts, scan, and regression suites were reviewed. `versions/`, `v1.31/`, scratch scripts and ROS 1 prototypes are historical material; current development uses the root ROS 2 pipeline. This workspace has no Git metadata.

The supplied `Room01_Scaled_UE.obj` contains 96,777 vertices and 32,314 faces. At `map_scale: 0.01`, its bounding dimensions are **3.0946 × 2.2923 × 2.9394 m** in file-axis order. Bounds alone do not prove which axis is vertical; verify floor, scale and orientation during lineup. The scan was not changed.

## Active data path

```text
D455 stereo IR + IMU → Isaac ROS VSLAM → accepted timestamped camera poses
D455 depth + depth K + paired camera pose → scan registration → gradual world correction
ZED-F9P GGA + GST → fix/accuracy/correction checks → surveyed antenna-position check
                                      ↓
                              evidence supervisor
                                      ↓
                   shared held/validated pose → FreeD + viewport
```

Fast motion still comes from camera tracking. The scan estimates a slowly changing world alignment. GNSS corroborates position; it does not feed raw latitude/longitude into FreeD, add an arbitrary first-fix origin, or independently provide orientation.

| State | Output behavior |
|---|---|
| WAITING / ALIGNING / VERIFYING | No initial packet until camera data, lineup and supporting evidence exist |
| VALIDATED | Fresh camera motion supported by consecutive scan checks |
| RTK_ASSISTED | Fresh camera motion supported by calibrated RTK position agreement; rotation remains camera-derived |
| DEGRADED | Brief camera-only fallback after supporting checks fail or expire |
| HOLD | Last output pose held exactly; no continued prediction |
| RELINEUP | Recovery displacement exceeds the configured limit; repeat lineup |

Defaults: three consecutive supporting samples, scan/GNSS timeout 0.8 s, camera staleness 0.25 s, then at most 1 s of unsupported motion after support expires. A scan check with insufficient translation observability cannot claim full validation. A rejected current check replaces the green status of an older accepted check.

Depth pairing uses accepted pose timestamps in the ROS clock domain, bounded to 35 ms. Missing depth camera_info or a matching pose prevents supervised matching. Scan workers discard stale work and obsolete lineup generations. Survey-point poses are depth-verified without being relocated by ICP; runtime corrections cannot worsen an already verified match. Correction slew cannot accumulate through an outage. Small output recovery is rate limited; large recovery requires explicit lineup.

These are health and agreement gates, not a covariance-based multi-sensor estimator. An accepted scan match can still be wrong in repeated geometry. The translation-rank heuristic is not a full six-axis uncertainty estimate.

## Operator workflow

Start with [Camera Lineup Lite](LINEUP_LITE.md): freeze, pick corresponding 3D/2D points, solve, inspect residuals and overlay, then depth-verify and apply. The quick locator match and manual adjustment tools remain available.

The redesigned dashboard provides large camera and scan views, three live input cards, explicit output mode, an Operate tab and a Setup tab. Uninstalled extra cameras, UWB and IR inputs are identified as planned.

Run `python3 tracking_preflight.py` for a read-only inventory. In this session’s default Python environment, ROS 2, cv_bridge and Open3D are absent. An existing `lidar` environment has Open3D, and was used for registration testing; it does not contain the full GUI/ROS stack. Video device nodes were present, but no stable GNSS serial-by-id device was found. This does not prove which physical cameras or receivers are connected.

## Inputs still needed

| Input | Why it matters |
|---|---|
| Live ROS image/pose timing and real camera_info | Synthetic tests cannot establish delivery latency or validate camera extrinsics |
| Physical distance/axis verification of the room scan | File names and bounds cannot guarantee metre-scale world alignment |
| Taking lens nodal offset and lens calibration | D455 motion alone does not define a separate production camera’s optical centre or lens |
| Measured capture-to-render latency and frame synchronization | Matching positions without matching exposure/render time still gives sliding composites |
| Outdoor correction source and GNSS calibration | Current user setup does not yet include known RTK corrections or a surveyed map relation |
| Recorded trajectories and reference measurements | Needed to quantify translation, rotation, jitter, drift, latency and reacquisition |

[RTK_SETUP.md](RTK_SETUP.md) explains RTK from the beginning. GNSS stays disabled/reference-only until configured. No correction service was purchased, no receiver configuration was written, and no hardware calibration values were guessed.

## Outdoor and future hardware limits

The D455 depth stream and stereo tracking use the same optical pair. They share occlusion, lighting and lens-contamination failures; counting them as independent redundant cameras would overstate reliability. A saved LiDAR mesh is a reference map, not a live sensor that can observe motion during total camera blockage.

The [RealSense datasheet](https://realsenseai.com/wp-content/uploads/dlm_uploads/2025/08/Intel-RealSense-D400-Series-Datasheet-August-2025.pdf) identifies the D456 as the D455-related model with an IP65 enclosure. It does not establish an all-weather rating for this D455 rig. Protection from water, condensation and debris, plus real tests in sunlight, darkness, rain and fog, are still needed for the stated outdoor ambition. Neither software fusion nor GNSS supplies missing six-degree-of-freedom optical information during complete camera loss.

The [ZEISS CinCraft Scenario description](https://www.zeiss.com/photonics-and-optics/us/cinematography/cincraft/scenario.html) is a useful reference for natural-feature/marker tracking, setup and lens integration. Its operating claims do not establish this project’s performance.

Three future cameras need rigid camera-to-rig extrinsics, synchronized timestamps, bandwidth budgeting and per-camera quality checks before contributing to the same estimator. UWB needs surveyed anchors, tag lever arms and non-line-of-sight rejection. IR illumination/markers need a detection model and measured camera relation. Those hardware adapters are not implemented by this update.

## Deployment and verification

`launch_isaac.sh` copies the new fusion, GNSS and lineup modules plus the bundled scan to the ROS workspace. Existing workspace configuration is preserved. To enable this update on an existing installation, merge the project’s `fusion` and expanded `gnss` sections into the **active** workspace configuration, and verify its scan path. The relative bundled path resolves beside the loaded configuration. Set the actual Unreal workstation IP; localhost is only correct when Unreal runs on the same host.

Host regressions cover coordinates, FreeD, engine behavior, GNSS parsing, dropout/recovery, depth pairing, surveyed lineup solves and Qt workflow. Final verification:

- Main host suite: 241 tests, 11 dependency skips, no failures.
- Offline ROS/FreeD wiring: 105 tests, 20 Open3D skips, no failures.
- Those 20 synthetic-room Open3D tests passed separately in the existing `lidar` environment.
- Three additional `test_room_scan.py` regressions passed against the supplied OBJ: preserve the survey solution through verification and repeated stationary checks, reject a clearly wrong pose, and reject complete depth loss.
- Python compilation and launch-script syntax checks passed.

The supplied-room test raycasts ideal depth from the mesh; it does not use a real D455 recording. In the diagnostic run, the verified pose had 100% thresholded depth agreement, a 19 mm median distance to sampled scan points, and about 3.7 mm stationary position drift after 20 continuous checks. These are fixture-specific diagnostics, not measured device accuracy. A rough locator hint previously produced a wrong ICP local minimum despite high agreement; the Lite workflow now preserves the survey pose during depth verification and rejects runtime corrections that worsen an already verified match. Repeated geometry still makes scan agreement alone insufficient proof of correct localization.

The dashboard and Lite dialog were rendered offscreen with inert ROS stubs. This checks UI construction and interactions; production GPU scene rendering still needs a display check. These tests do not establish live D455 accuracy, successful satellite RTK, or weather tolerance.

LTC in this dashboard is display-only. It is not genlock, a timestamp synchronization implementation, or proof of exposure synchronization.
