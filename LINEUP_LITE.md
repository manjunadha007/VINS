# Automatic camera lineup

The dashboard now has one **Operate → Auto lineup** checkbox. It uses the
manually placed locator as the known position and orientation of the physical
floor/table marker. Marker settings and scan details are in **Setup**.

## Marker-assisted lineup for a floor or table

1. Place the existing **DICT_4X4_50, 10 cm ArUco marker** flat and stationary
   in the scanned environment. Load the scan and confirm its scale in metres.
2. Select **Locator** in the outliner. Use the Channel Box to place it at the
   **marker centre**, including the correct floor/table height.
3. Set its local **+X toward the top of the original printed marker**, and
   **+Z upward**. For a level floor/table, Rotation X and Y are zero; Rotation Z
   sets the heading. Position alone cannot establish the camera heading.
4. In **Setup**, keep **DICT_4X4_50** and **0.1000 m**. Select the ID shown in
   the preview, or use **One visible marker** when exactly one matching marker
   is in view. Width includes the black border and excludes the white paper margin.
5. Hold the camera still, show the marker plus surrounding room geometry, and
   check **Auto lineup**. Fresh calibrated IR, depth and timestamp-paired camera
   poses are required. The background search starts on a fresh detection.
6. Wait for **green: Auto lineup · Locked**, with **Fusion active** underneath.
   An accepted initial match starts live motion immediately and first appears
   amber: **Auto lineup · Camera tracking**, while consecutive fresh scan checks
   validate tracking. A stored alignment alone cannot turn it green.
7. Move the physical camera. The 3D dot and its six Camera channels follow the
   supervised camera pose in the displayed scan coordinates, including the
   configured rig offset. FreeD uses that same pose with the configured Unreal
   axis conversion. The marker can leave view; stereo/IMU supplies motion while
   depth-to-scan matching continues to verify and refine alignment. Missing or
   rejected scan checks turn the indicator amber; fresh camera motion continues
   using the last accepted alignment.

| Checkbox state | Meaning |
| --- | --- |
| Unchecked / grey | Auto acquisition off. Place the locator and enable it. |
| Amber / Waiting or Matching scan | Waiting for a fresh detection or checking its pose against the scan. |
| Amber / Camera tracking | Live stereo/IMU motion in the scan coordinates; scan checks are unavailable or recovering. Drift can accumulate until matching resumes. |
| Amber / Validating | Initial checks or scan recovery are incomplete; tracking is not fully validated. |
| Green / Locked | Fresh camera tracking and consecutive valid scan checks; fusion is active. |
| Red / Tracking held | Tracking is stale, rejected, or needs another lineup. The last safe camera pose is held. |
| Unchecked / Retry | The solve or setup was rejected. Read the reason, correct it, then enable again. |

The checkbox stays checked after a successful match without solving every later
marker sighting again. Unchecking cancels acquisition and retains an installed
alignment. Uncheck and enable again to realign. Moving the locator/scan or a
tracking resynchronization invalidates an in-flight result and the checkbox's
lock context. A close-up showing only a flat floor may not constrain all three
position axes; include walls, corners or fixed furniture.

**Camera** channels are live readouts in this dashboard, so dragging the dot
cannot take over tracking with a frozen preview. Locator, scan, trail and world
reference transforms remain editable in their Channel Boxes. Scan overlay is
available in **Setup** and turns on automatically after an accepted match.
**File → Save** records the locator transform and marker settings.

The marker supplies a camera pose at its captured image timestamp, including the
measured camera-to-marker distance. The search is centred on that camera pose;
its default bounds are ±0.50 m and 20°. It does not move the locator to a guessed
marker location. The scan match uses the depth frame's own timestamp-paired pose.
An accepted result installs the SLAM-to-scan transform, which is applied to every
subsequent camera pose. Rejected, cancelled or obsolete results leave the existing
alignment intact.

## Matching tolerance and live motion

`lidar.min_verify_score: 0.75` requires **75% scan agreement**, reduced from 95%.
A depth point counts as agreeing when it falls within
`lidar.verify_threshold_m: 0.10` (10 cm) of the scan. This measures surface
overlap, not camera-pose accuracy. It allows missing or inaccurate scan areas;
the search bounds, geometry checks and correction limits still apply. Inspect
the overlay after lineup, especially around repeated room geometry.

`fusion.allow_vio_fallback: true` keeps fresh camera motion running after lineup
even before the first three runtime scan checks pass. Scan failures do not freeze
the camera or require another lineup. Stale or rejected **camera poses** still
hold the last output; a large move during that hold still requires realignment.
Set `allow_vio_fallback: false` to require consecutive external checks before
output and restore the timed hold controlled by `single_source_grace_s`.

Restart through `./launch_isaac.sh` after updating. The launcher preserves the
deployed `~/workspaces/isaac_ros-dev/src/tracking.yaml`; if that file already
exists, set its `lidar.min_verify_score` to `0.75` and
`fusion.allow_vio_fallback` to `true` as well.

The pose calculation follows [OpenCV's ArUco pose-estimation workflow](https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html).

## Earlier Camera Lineup Lite solver

The survey-point solver remains in `lineup_dialog.py` and `lineup_solver.py`;
its button and the quick/manual/global-search alternatives have been removed
from the dashboard. Six consistent non-coplanar pairs are sufficient for the
Lite solver. Using 8–12 accurate pairs across different depths allows outlier
rejection; more points on one flat wall cannot resolve a planar rejection.
