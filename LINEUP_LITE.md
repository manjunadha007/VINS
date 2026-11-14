# Automatic camera lineup

The dashboard has one **Operate → Auto lineup** checkbox. Use a single Locator,
or duplicate it and assign a different detected ArUco ID to each Locator for
joint matching and ongoing marker support. Marker settings and scan details
are in **Setup**.

The markers are **DICT_4X4_250** (IDs 0–249), all printed at the
`aruco.marker_size_m` width (default 10 cm). Every marker in the frame is
detected; the camera preview lists their IDs.

**Measure the printed square, don't trust the PDF.** Print at **Actual size /
100%**: a printer's "Fit to page" shrank 176 mm squares to about 167 mm on this
rig. Measure across the black border, excluding the white paper, and enter it as
**Setup → Square width**. Distance from a marker scales with this width, so a 10%
error puts every marker 10% too close or too far. Auto lineup compares the width
against depth and refuses when they disagree by more than 10%: "Marker 2 measures
about 16.7 cm across from depth, but Square width is 10.0 cm".

## Every detected marker: automatic placement

You don't need to survey every marker by hand. Put one Locator on one marker
and run Auto lineup. Once it turns green (**Locked**):

1. The lineup marker's ID is given to its Locator, and the Locator moves onto
   the marker the scan found.
2. Every other marker the camera sees, with no Locator yet, is placed in the
   scan from the locked camera. Each one gets its own Locator (named
   `Locator N · ID n` in the outliner) and joins the joint solve and the 2 Hz
   marker support. Your outliner selection doesn't change.

A marker is placed only after `auto_place_min_sightings` (6) sightings agree on
one pose within 3 cm / 6°, and no clearly different pose is nearly as well
supported. A small printed square has two possible poses that fit its corners
(the planar PnP ambiguity). With the camera parked, both stay steady, so
nothing is placed until the camera moves a little and the wrong one drifts
away. Squares smaller than 24 px in the image are ignored. Placement pauses
while a marker support check is running and resumes straight after.

Placed markers carry the camera's error at the moment they were placed. They
hold later tracking to this lineup; they cannot make it more accurate than the
lineup was. Use **Save marker layout** to keep them: the next session loads
the layout and lines up from every marker at once, with no scan search.
Unchecking Auto lineup stops placement and keeps the Locators already placed.
Set `aruco.auto_place_markers: false` to turn this off.

With several markers in view and no Locator assigned yet, choose the ID of the
marker under your Locator in **Setup → Marker ID**. "One visible marker"
refuses, and lists the visible IDs, because it can't tell which one you meant.

**Where you set the ID decides how the marker is used.** An ID in **Setup →
Marker ID** means "the marker is somewhere near this Locator": the scan is
searched for it. An ID assigned to the Locator in the outliner (**Assign ID**)
means "this Locator *is* the marker, exactly": no search, and the hand-placed
position must be right to about a centimetre. With two or more assigned markers
in view, each rotation is measured from the camera (see **Locator
rotations are checked**). For a first lineup with a single marker, leave the
Locator unassigned.

**Floor, table or wall.** The lineup marker can lie flat or be fixed upright to
a wall. The SLAM frame is gravity-aligned (IMU fusion), so the marker's tilt
is measured: within 45° of flat it is taken as lying flat, otherwise as upright
on a wall. Either way, the way it faces and its exact spot within the search
radius are found from the scan. On a wall, put the Locator at about the
marker's height: the search grid runs along the floor, so it covers a wrong
spot along or across the wall but only about a decimetre of wrong height.
Leave the Locator's rotation alone; it isn't used.

## Multiple markers: duplicate and assign Locators

1. Load the scan and confirm its scale in metres. Place stationary ArUco
   markers with different IDs around the scanned space. Measure each printed
   square, including the black border and excluding the white paper margin.
2. Select **Locator** in the viewport outliner. Place its centre at the physical
   marker's position in the scan. The hand-set orientation is not trusted.
   Auto lineup derives **+X toward the printed top, +Z out of the printed
   face** from the camera and depth. It keeps the placed centre fixed.
3. Show that marker in the live camera. Choose its **dictionary / ID** in the
   detected-marker dropdown below the outliner, enter its printed width in
   metres, and click **Assign ID**. The Locator's row now shows its ID.
4. Click **Duplicate Locator**. The duplicate starts at the same position with
   **no ID assigned**. Move and rotate it to the next marker, show that marker
   to the camera, and assign its ID and width. Repeat for additional markers.
   Select any Locator row to edit its own transform. An ID cannot belong to
   two Locators. **Clear ID** removes an assignment; **Remove** deletes the
   selected Locator while retaining at least one Locator.
5. Show two or more assigned markers together, with surrounding scanned
   geometry, then enable **Operate → Auto lineup**. The solver combines their
   detected corners and known poses into one camera pose, which must also pass
   depth-to-scan verification. Unassigned Locators and unknown IDs are ignored.
6. Move the camera. Visible assigned markers support the alignment at up to
   **2 Hz**, with gradual corrections; stereo/IMU supplies continuous motion.
   Markers can leave and re-enter view. Their absence does not pin the camera
   to the first frame. Scan matching continues independently.

Two visible registered markers must agree. With three, one inconsistent marker
can be excluded. One visible registered marker can still work when its pose is
unambiguous and the depth view provides sufficient scan geometry. Repeated IDs,
stale frames and conflicting poses are rejected. Survey the positions carefully:
a flat floor cannot expose an entire layout shifted sideways along that floor.
Use separated markers so their combined view covers more of the image. Larger
printed squares can help at a distance; adding IDs cannot recover unreadable
corners or missing depth.

**Locator rotations are checked.** Rotations are easy to get wrong by hand. A
duplicate of the floor Locator stays flat when moved onto a wall, and
Rotation X = 90 stands it up with its printed top sideways. Either way the
markers "disagree". So before solving, Auto lineup works out each assigned
marker's rotation from the camera, trusting only the Locator **positions**:

- Three depth-backed marker centres determine the camera's full rotation,
  including any tilt between SLAM and the scan. With two centres, gravity
  supplies the missing axis.
- Each marker's face comes from the depth plane around it, and its printed
  top from its corners. The measured tilt is preserved.
- Every visible Locator gets a measured rotation, including corrections below
  10°. Its placed position is preserved exactly. An unclear orientation keeps
  acquisition waiting instead of letting the entered angles affect the solve.
  Later marker support uses only orientations that have been measured.

The check needs markers spread across the floor plan, not stacked one above
another. If the Locator positions don't fit what the camera sees (more than
10 cm off), nothing turns and the status says what's wrong, as specifically as
it can:

- Two swapped IDs: "The Locators of markers 1 and 2 look swapped".
- One Locator off, with the others agreeing: "Marker 2's Locator is 30 cm from
  where markers 1 and 3 put it: X 0.800, Y 2.000, Z 1.300 m". These are
  Channel Box coordinates, so you can type them in.
- Otherwise, the Locator furthest off.

Each attempt also logs every marker's Locator position next to where the
camera puts it ("[ArUco] Marker 1: Locator at … the camera puts it at …").

A wall marker with no depth around it can be ambiguous: from about its own
height, its two possible PnP poses both stand upright, facing tens of degrees
apart. Show its surrounding surface before acquisition can continue.

Earlier tests of the thresholded, level-mount correction used a 2 m room
corner, 0.4 px corner noise, 1.2% depth noise with 20% dropouts, 1° gravity
error and Locators placed to 1 cm. Across 395 random views, that method turned
every wrong Locator to within 4.6° and never turned a correct one.
With 3 cm placement error and 2° gravity error, 26 of 387 views were refused
as misplaced, the worst turn was 9° off, and still no correct Locator turned.

**Setup → Save marker layout / Load marker layout** stores and restores assigned
Locators, IDs, sizes and rotations. Layout coordinates are in scan metres before
the scan's display transform: load the matching scan and confirm its scale before
restoring a layout. **File → Save** also includes all Locators and assignments.
Moving an assigned Locator, changing its assignment, moving the scan or resetting
tracking cancels the active marker context; verify the placement and enable
Auto lineup again. Clearing every assignment restores the single-marker workflow.

Joint estimation follows [OpenCV's multi-marker board workflow](https://docs.opencv.org/4.13.0/db/da9/tutorial_aruco_board_detection.html),
using each Locator's known 3D corners and its matching detected image corners.

## Marker-assisted lineup from one Locator

1. Place a **DICT_4X4_250, 10 cm ArUco marker** flat and stationary
   in the scanned environment. Load the scan and confirm its scale in metres.
2. Select **Locator** in the outliner. Use the Channel Box to place it at the
   **marker centre**, including the correct floor/table height.
3. Set its local **+X toward the top of the original printed marker**, and
   **+Z upward**. For a level floor/table, Rotation X and Y are zero; Rotation Z
   sets the heading. Position alone cannot establish the camera heading.
4. In **Setup**, keep **DICT_4X4_250** and **0.1000 m**. Select the ID shown in
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
| Green / Locked | Fresh camera tracking and consecutive valid scan or mapped-marker checks; fusion is active. |
| Red / Tracking held | Tracking is stale, rejected, or needs another lineup. The last safe camera pose is held. |
| Unchecked / Retry | The solve or setup was rejected. Read the reason, correct it, then enable again. |

With no assigned IDs, the checkbox stays checked after a successful match without
solving every later marker sighting again. Assigned layouts instead supply
periodic marker checks and gradual corrections. Unchecking cancels acquisition
and marker support, and retains an installed
alignment. Uncheck and enable again to realign. Moving the locator/scan or a
tracking resynchronization invalidates an in-flight result and the checkbox's
lock context. A close-up showing only a flat floor may not constrain all three
position axes; include walls, corners or fixed furniture.

**Camera** channels are live readouts in this dashboard, so dragging the dot
cannot take over tracking with a frozen preview. Locator, scan, trail and world
reference transforms remain editable in their Channel Boxes. Scan overlay is
available directly below the **Live Camera / Scan Overlay** preview and turns
on automatically after an accepted match.
The overlay draws a yellow mesh wireframe at camera resolution with backface
culling on. Adjust **Transparency** below **Show scan wireframe overlay**:
0% gives solid lines, 100% shows only the camera image (default: 20%).
**File → Save** records the locator transform and marker settings.

The marker supplies a camera pose at its captured image timestamp, including the
measured camera-to-marker distance. The scan is searched for the marker anywhere
within `aruco.lineup_search_radius_m` (**1.0 m**) of the Locator, along the floor,
at every heading. Starts are laid every `lineup_search_step_m` (15 cm), ranked
cheaply, and the best 24 go through the same ICP screen as the Locator's own. When
placements match about equally well, the one nearer the Locator wins
(`lidar.hint_weight`). If they still tie, the lineup is refused as ambiguous. Each
start may move at most `lidar.hint_radius_m` / `hint_radius_deg` (0.50 m / 20°).
On success the Locator moves onto the marker the scan found, and the status says
how far it moved. The scan match uses the depth frame's own timestamp-paired pose.

Measured against the Room01 scan with simulated D455 depth noise, a 10 cm
marker, and a gravity-aligned SLAM frame 1.8° off true (32 floor and 24 wall
trials per Locator offset):

| Locator off by | Floor: correct / wrong / refused | Wall: correct / wrong / refused | Previous search, floor: correct / wrong |
| --- | --- | --- | --- |
| 0 m | 28 / 0 / 4 | 21 / 0 / 3 | 26 / 2 |
| 0.3 m | 22 / 0 / 10 | 19 / 0 / 5 | 19 / 5 |
| 0.6 m | 23 / 0 / 9 | 20 / 0 / 4 | 0 / 10 |
| 0.9 m | 20 / 0 / 12 | 18 / 0 / 6 | 0 / 5 |

The previous search accepted wrong poses up to 2.5 m off. A lineup takes about
1.5 s (at most 3 s), up from about 0.5 s. Every refusal was "ambiguous":
show a corner or fixed furniture as well as the marker.
An accepted result installs the SLAM-to-scan transform, which is applied to every
subsequent camera pose. Rejected, cancelled or obsolete results leave the existing
alignment intact.

## Scan scale

With three hand-placed Locators assigned to their physical marker IDs, enable
**Auto lineup** (or **Setup → Scale and line up from markers**). It gathers
12 seconds of timestamp-paired RealSense depth and marker observations. Hold
each marker and its surrounding surface in view for several frames; the three
centres must form a well-spaced triangle. They may all be in one room corner.

The marker centres get their range from depth, independently of the printed
width setting. A 3D similarity fit measures one scale, rotation and translation
between that metric triangle and the Locators. The remaining room depth must
agree with the proposed scan within 8 cm for at least 60% of gathered points.
Only then is the scan scale divided by the fitted ratio. Every Locator moves
with that uniform scaling, keeping its original spot on the scan; no Locator is
individually repositioned. Printed marker widths and the camera's metre units
are unchanged. Auto lineup then continues against the metric scan.

Missing depth, nearly collinear centres, inconsistent triangle proportions or
insufficient surface agreement leave the scan and placements unchanged.
Cancelling Auto lineup also cancels an unfinished calibration. **File → Save**
preserves the resulting scale and Locators. These triangle constraints allow
large import-unit corrections as well as small errors in scan size.

Without three markers, the separate **Measure scan scale** tool uses a wider
depth sweep after an initial lineup, as described below.

A phone or photogrammetry scan can come out a few percent too small or too large.
A lineup from one corner can't show it: three surfaces meeting at a point look
the same at any size, so the fit slides the camera to make up for it. The lineup
still "works", but markers sit where the depth puts them, not where they are on
the scan.

The D455's depth is metric. It measured 16.7 cm tape-measured markers at
16.5–17.0 cm. **Setup → Scan scale** uses that to measure the scan:

1. Line up first. A rough lineup is enough; the measurement starts from it.
2. Press **Measure scan scale** and pan slowly across the room for 12 s
   (`lidar.scale_sweep_s`). The camera must see walls on opposite sides, or
   furniture. Press again to stop early.
3. Each depth frame is placed by its tracked pose. The whole sweep is then fitted
   to the scan with rotation, translation and one scale free.
4. The result is one of:
   - "The scan is 5.0% smaller than the room… Apply sets the scan scale from
     0.01 to 0.010526". **Apply scale** rescales the scan, the same as the
     viewport's **S** slider, and moves every Locator with it, so each stays on
     the same spot of the scan. Run Auto lineup again afterwards. To keep the
     scale after a restart, use **File → Save**, or set `lidar.map_scale` in
     tracking.yaml.
   - "No change needed". The scan is within about 2% of the depth, the camera's
     own accuracy, so there's nothing to correct.
   - A refusal. "This sweep can't measure scale" means it saw only a corner or one
     wall: pan wider. Other refusals mean the depth doesn't land on the scan, or the
     scale is implausible: check the lineup, the scan and its units.

The measurement is refused rather than guessed when the sweep can't pin scale
down. It reports a "lever": how much of the scale fit sliding the camera can't
mimic. Simulated sweeps across a room had 1.3–1.5 m; a corner had 0.02 m. On the
rig a corner had 0.16–0.20 m, from clutter. At 0.16 m a free fit still shrank
that corner to its 0.65× limit while 81% of the depth "fitted". So scale is held,
and the measurement refused, below 0.5 m.

In simulation (a 4.2 × 3.4 m room scanned only to 1.2 m, with a table, a
cabinet and a chair the scan never saw, 1% depth noise, tracking 6° off
gravity, lineups 10 cm / 2.5° off), 16 random sweeps with scans 10% too small
to 10% too large were all measured within 0.05%. Each took up to 3 s.

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
