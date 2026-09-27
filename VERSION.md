# 2026-09-27 — v1.5 snapshot: hardware-checked startup, IMU fix in the container

Frozen as `versions/v1.5/`. v1.4's tracking, made to start reliably on a
freshly recreated Isaac container and to refuse to open the dashboard until the
camera, IMU and SLAM are actually streaming.

The D455 IMU no longer takes the cameras down. The Isaac image's librealsense
2.55.1 predates firmware 5.16's 38-byte IMU report; when the IMU stalls the
whole session aborts and IR and depth drop to 0 Hz. `ensure_realsense_imu.sh`
rebuilds it inside the container with upstream PR #12857. Validated live: IMU at
400 Hz alongside stereo, depth and SLAM, with no host-side DKMS.

Startup now fails loudly. `check_pipeline.py` replaces `sleep 15` and waits for
sustained, advancing stereo, calibration, IMU and odometry. The numpy<2 pin and
the cuVSLAM install run on every launch instead of silently once at cold start;
the recreated container on 2026-09-27 was missing cuVSLAM and this installed it.
`cleanup()` now really stops SLAM: its pkill patterns used to kill their own
shell first.

The check passed on the D455 both direct to the PC and through a 5 Gbps hub
(IR 80-90 Hz, depth 80-88 Hz, IMU 400 Hz, odometry 97-100 Hz). The AR overlay
and lineup score now use the pose at each frame's own timestamp, and marker
heading seeds moved from 15 to 10 degrees so a rig 0.9 m back still locks.

Known limits: a fresh container's first launch takes a few minutes and needs
network; the check's rates come from a ~1 s window; the cleanup fix has not yet
been seen on a real dashboard close; `test_ui.py` can abort on exit while a live
pipeline is streaming.

# 2026-09-09 — v1.4 snapshot: marker-referenced absolute lineup

Frozen as `versions/v1.4/`. A printed ArUco marker now places the camera inside
a LiDAR scan and the fusion holds it there while the camera moves.

`Find marker in scan · place Locator` locates the camera in the scan with no
prior, derives the marker's pose from it and places the locator — height and
heading included. That removes the step that caused most lineup failures: a
hand-placed locator asks the operator to mark a spot the scan does not contain,
and a wrong height or a 90 degree heading are each worth 750-950 mm of camera
error, past the 500 mm search region, so every attempt rejects. The global fix
seeds the locator only; the solve still passes the normal verification gates.

Validation on a D455 against a room scan, scored against an independent
hint-free registration of the same depth frames rather than against the fusion
itself: over 1.13 m of real travel the camera dot tracked to a median 10 mm,
worst 26 mm, scale ratio 0.988. Ten of ten checkpoints cleared a 0.90
reliability gate. The lineup locks at 99 percent scan agreement, rank 3, within
about a second.

Two configuration values changed from measurement, not preference.
`lidar.coarsest_radius_m` 1.00 -> 0.60: at 1.00 the coarse ICP stage swung a
10 mm-accurate hint 36.6 degrees onto wrong geometry and failed every attempt.
`lidar.min_verify_score` 0.55 -> 0.95: solves 380-450 mm from truth still scored
0.87-0.91 and were accepted, while correct ones score 0.98.

Known limits: the lineup needs about 95 percent of the view within 5 cm of the
scan, so it only works where the scan actually covers the room; under sustained
motion the supervisor drops in and out of DEGRADED when a scan check misses its
0.8 s window; and the camera dot receives nothing at all until a lineup
validates, so a waiting dot is indistinguishable from a crash.

# 2026-09-09 — Dashboard maximize / restore

The 3D viewport and Lite survey overlays now paint through Qt's managed
`paintGL()` lifecycle. Qt retains ownership of framebuffer recreation and
composition during window resizing; unchanged mesh buffers are reused.
The 60 Hz status refresh no longer reapplies identical stylesheets or alternates
the scan panel's styles twice per tick, avoiding repeated layout invalidation.
The existing zero swap-interval setting remains in place.

Validation: 32 viewport/lineup tests (one environment-dependent skip), plus
105 offline ROS/FreeD tests (20 Open3D skips). A real X11 window in the Isaac
container completed five maximize/restore cycles with the room mesh and
synthetic camera frames; the largest GUI heartbeat gap was 39 ms and native
window-manager ping replies took about 5 ms while maximized. Framebuffer tests
also cover sizes up to 2560 × 1440. These isolated checks do not replay the
full live tracking workload.

Restart through `./launch_isaac.sh` to deploy and load the updated modules.

# 2026-09-08 — Supervised tracking and Camera Lineup Lite

Adds a frozen-image 2D/3D survey correspondence workflow with robust PnP,
reprojection review and depth validation before applying the camera alignment.
Adds timestamp-paired depth matching, explicit fallback/hold/relineup states,
calibrated GNSS position checks and a redesigned operator dashboard.

GNSS corrections, hardware synchronization and all-weather accuracy are not
claimed. See LINEUP_LITE.md, RTK_SETUP.md and PROJECT_ASSESSMENT.md for operation,
limitations and calibration requirements. Historical release notes follow.

# Version 1.3 — in development

Working tree. Frozen snapshots: `versions/v1.2/`, `versions/v1.1/` (see
[versions/README.md](versions/README.md)).

## Changed in 1.3

**Map fusion rebuilt: the camera's pose is now solved against the LiDAR scan in
real time, in position and rotation.**

The old path could not work. It estimated a position *offset* rather than a
transform, and every piece of that was wrong:

* it read `icp.transformation[:3, 3]` as the camera correction. That is where the
  map *origin* moved; a rigid transform displaces a point by `(R - I)p + t`. With
  the camera 5 m out, one degree of ICP rotation puts 87 mm into the term being
  ignored — larger than the whole 100 mm rejection gate, so genuine corrections
  were discarded and whatever survived pointed somewhere arbitrary.
* rotation was discarded entirely — only a 3-vector came back.
* the caller subtracted the already-applied correction from a measurement taken
  *through* an already-corrected pose, double-counting the feedback.
* `max_correction_m` (0.10) was passed as ICP's correspondence radius, so an
  error bigger than the rejection threshold could never be observed at all.
* `initial_icp_iterations` was read from config and never used; there was no
  initial registration step, so with no ArUco marker `is_aligned` never became
  true and the entire path was dead code.
* `depth_K` was initialised to `None` and never assigned — there was no
  `/camera/depth/camera_info` subscription — so depth was deprojected with the
  infra1 focal length, correct only by coincidence.
* `bridge_headless.py` composed the seed as `T_slam_marker @ inv(T_map_marker)`,
  the inverse of what it needed, and built `T_map_marker` from raw command values
  as though Unreal centimetres were metres and (pan, tilt, roll) were ROS
  (roll, pitch, yaw).
* **and none of it ever ran**: `launch_isaac.sh` auto-enabled depth with
  `grep 'lidar.*enabled.*[tT]rue'`, which needs all three words on one line. In
  YAML they never are, so `VP_ENABLE_DEPTH` stayed 0, the D455 published no depth
  topic, and the fusion had nothing to match against.

What it does now — estimate the *transform*, not the sample:

```
T_map_cam(t)  =  T_slam_to_map  @  T_slam_cam(t)
                 └ slow, fused ┘   └ fast, VIO at 90 Hz ┘
```

Each ICP run is an absolute measurement of `T_slam_to_map`; fusion is a geodesic
blend on SE(3) toward it, rate-limited by the camera's own displacement. All the
fast motion stays in the VIO term, so the drift term can be filtered hard at no
latency cost. The fused transform drives `PoseTransformer._zero`, so the FreeD
stream, the 3D-viewport camera dot and the status readout cannot disagree.

* `lidar_map_matcher.py` — rewritten. Coarse-to-fine point-to-plane ICP over a
  cached map pyramid; measurement gating on fitness, inliers, RMSE and the
  camera-frame magnitude of the correction; SE(3) fusion with slew limits.
* **Degeneracy handling.** ICP on one flat wall slides along it while reporting
  a confident fitness. Observability per axis is measured from the spread of
  matched surface normals and the correction is projected onto the constrained
  subspace. Reported as `Axes constrained: n/3`.
* **The D455's error model is used.** `σ_z = z²·σ_d/(f·b)` with the 95 mm
  baseline: depth is range-gated to 0.3–6 m and thinned by `1/σ²`, so point
  density tracks information content and the noisy far field cannot outvote the
  near field.
* **The locator is a hint, not a target — and so is the marker.** Both entry
  points now call `solve_from_hint()`: the hint says only "the camera is
  somewhere around here" and the pose is registered from the depth cloud.
  Previously the ArUco path installed the marker-to-locator mapping verbatim
  (`set_initial_alignment`, no ICP), which pinned the camera exactly onto the
  locator the instant you pressed anchor and left continuous fusion to creep to
  the truth at the 20 mm/s slew limit — tens of seconds. That was the camera dot
  snapping to the locator.
  Measured against `Room01_Scaled_UE.obj` with the hint deliberately wrong:
  19 mm → 3.2 mm, 235 mm → 10.0 mm, 490 mm → 2.9 mm, 714 mm → 5.3 mm, all in
  0.2–0.8 s; 1034 mm is refused rather than faked. Both the button and the
  status panel report how far the answer moved off the hint.
* **Lock-on is verified independently of ICP.** ICP's `fitness` is the fraction
  of points with a correspondence inside the search radius, so at a wide radius
  it reads ~1.0 almost anywhere — on this scan a solution 530 mm out of place
  scored 0.75, so it cannot decide whether a lock is real. Acceptance now asks
  the honest question: with this solution applied, what fraction of the depth
  image lands within `verify_threshold_m` of the scanned surface (shown as
  `On scan`), ranked by median miss because the fraction alone cannot separate
  4 mm from 180 mm in a cluttered room.
* **848x480x90 and a 400 Hz gyro.** The launch file's note that 848x480 "does
  NOT resolve as a stereo Y8 pair on this D455" was wrong -- almost certainly
  measured on a USB 2 link, where librealsense truncates the profile list.
  Verified on the same unit over USB 3.2: both imagers and depth open at
  848x480x90, and all three run at 89.87 Hz with the IMU at 398.9 Hz. Focal
  length goes 337 -> 429.95 px (measured from `camera_info`) and pixel count
  1.77x, so stereo depth error drops by 1.28x at every range and cuVSLAM has
  substantially more texture to track.
* **The ArUco fallback intrinsics were for a different stream.** `fx 389.386,
  cx 319.544, cy 241.506` -- a 640x480 profile, not the 640x360 the launch file
  actually requested. Only used when `camera_info` never arrives, but a marker
  solved through them would have come out mis-scaled. Replaced with values read
  off this unit at 848x480.
* **Marker heading was undefined for wall-mounted markers.** `try_anchor`'s
  level-horizon path took the heading from the marker's +Y axis ("up" on the
  print). That is horizontal for a marker lying face-up on a table -- the case
  this rig uses, which is why it went unnoticed -- but points straight up for
  one taped to a wall, making `atan2(0, 0)` undefined. The same sighting
  measured +0.1 deg under one set of intrinsics and -180 deg under another, and
  180 deg out puts the camera on the *opposite side* of the anchor. Heading now
  comes from whichever of the marker's up-axis and normal is actually
  horizontal, with the normal oriented away from the camera so the answer does
  not depend on the detector's corner convention. Table markers are unaffected:
  they still take the +Y branch, exactly as before.
* **The map cloud is now sampled reproducibly.** Open3D's
  `sample_points_uniformly` takes no seed and is not reproducible (verified on
  0.19: two calls on the same mesh return different clouds), so every load gave
  the fusion a slightly different problem. Measured on the real scan, the same
  lock-on landed at 7 mm on one run and 151 mm on the next purely from that.
  Replaced with `sample_mesh_surface()` -- area-weighted triangle choice, then
  a uniform barycentric point, seeded. The same smoke test now returns
  9.1 mm / 0.30 deg on every run. Note this makes the behaviour *reproducible*,
  not correct: the 150 mm second basin is still there, and it is more evidence
  that the scan wants re-exporting with its edges preserved.
* **One gizmo, driven by the outliner selection.** The 3D viewport's
  translate/heading gizmo attaches to whichever object is selected -- camera
  (cyan), locator (amber), or the scan alignment (green) -- through a small
  target registry rather than a copy of the picking and dragging code per
  object. `Trail` and `World Origin` deliberately have no target: neither has a
  transform to edit, so no gizmo is shown rather than one that does nothing.
  Dragging the camera opens a manual lineup; the live pose is held off during a
  drag so it cannot snap the dot back. A `✥ Gizmo: ON/OFF` toggle hides it and
  disables picking. The outliner went from 75 px to 150 px, which is what it
  takes to show all five entries -- at 75 px the two most useful were the ones
  scrolled out of sight.
* **Dragging the scan edits the mesh alignment**, with the toolbar boxes still
  the single source of truth. Two things this turned up: the alignment
  spinboxes held only 2 decimals, so a dragged translation could not be
  represented and the next edit of any box silently snapped it back by up to
  5 mm (now 4 decimals for translation, 3 for angles, with a read-back so the
  boxes and the transform cannot disagree); and yaw was *clamped* at +/-180
  rather than wrapped, which for a periodic quantity left the heading and the
  pivot describing different rotations and swung the scan across the room.
* **LiDAR overlay on the camera view**, toggleable, at ~6 Hz.
* **Manual first-frame lineup** (`LineupSession` in `image_alignment.py`, panel
  in the dashboard). Freeze a frame, nudge the virtual camera in its own axes
  until the scan's projected edges sit on the picture, commit — and continuous
  fusion carries on from that pose. Frozen rather than live because the fusion
  corrects the *frame*, so a pose dialled on one frame stays valid for every
  pose after it; that invariant is what the tests pin. Coarse/fine steps, undo,
  reset, and an optional AUTO that offers the edge solver's answer as a
  suggestion rather than imposing it. The arithmetic is Qt-free and unit
  tested; the commit is exact by construction.
* **Camera-image lineup against the scan** (`image_alignment.py`) — the
  automatic form of a 3DEqualizer lineup. Ray-casts the scan from the current
  pose, extracts silhouettes, and measures how far they sit from the nearest
  edge in the live IR frame via a distance transform; Levenberg-Marquardt on
  SE(3) minimises that. Shipped as a **read-only check plus AR overlay**:
  `Image lineup: N% of the scan's edges on the picture`.

  Refinement (`image_align_refine`) is off, and the reason is measured rather
  than assumed. The score is a genuine measurement — it peaks at the true pose
  in all three translation axes and in yaw, which is what earns it a place on
  the panel. But the objective's *minimum* sits 30-60 mm from the true pose on
  this scan (cost 9.33 at truth against 8.84 at a pose 59 mm away), while depth
  ICP already solves to ~3 mm, so enabling it would make the answer worse.

  Three things were fixed along the way and are worth recording, because each
  was a real defect in the objective rather than the solver:
    - thresholding depth and normal steps directly fired on photogrammetry
      noise — 15 548 "edges" out of 57 600 pixels against the image's 3 311.
      Running the same Canny used on the image, over inverse-depth and
      headlight-shaded renders, brought the two into the same range.
    - averaging residuals only over edges that found a match made the cost
      incomparable between poses, and the solver exploited exactly that: it
      moved to a pose rendering fewer, easier edges and reported a lower cost
      than the truth. Every model edge now counts, clamped at the search radius
      rather than dropped.
    - fixed-damping Gauss-Newton halved the error in two iterations then
      oscillated for eighteen more; a distance transform is piecewise linear
      with kinks, so steps must be accepted only when the cost falls.
  Creases are off by default: on a photogrammetry mesh they contributed 4 000
  mostly meaningless edges against 676 useful silhouettes.
* **The hint bounds the search, not just seeds it.** `lidar.hint_radius_m` /
  `hint_radius_deg` (±0.5 m / ±20°) cap how far from the hint an answer may be.
  Found on hardware: gated at ±2 m / ±45°, a close-up of a table corner locked
  onto geometry 1.1 m and 42° away, reported success, and left continuous
  fusion permanently asking for a 409 mm correction the runtime gate refused —
  `Solved off hint: 1105mm / 42.02°`. A match outside the region is now refused
  with a message saying to move the locator, show more room, or widen the
  region.
* **Ranked on match fraction, not median distance.** From an ambiguous
  viewpoint a solution 233 mm out of place had the *better* median (29 mm vs
  38 mm) because it aligned some structure very tightly; it gave itself away
  only in the tail (86 % of points within 100 mm against 94 %). Ranking on the
  median chose the wrong pose. Candidates are now ranked on the fraction within
  `verify_threshold_m`, plus a light prior favouring the hint (`hint_weight`)
  to break the ties a cluttered room produces constantly.
* **Yaw search swings about the displayed vertical.** It was swinging about the
  camera's own up axis; the operator's heading error comes from the locator's
  heading ring, which turns about the displayed vertical. For a camera tilted
  down at a table the two are far apart, and a 15° heading error recovered only
  to 7.9°. The axis now comes from the mesh alignment (`R_align^T @ [0,0,1]`),
  so it follows however the operator has the scan oriented, rather than
  assuming the SLAM frame is gravity-aligned.
* **ICP runs on its own thread.** A 100-200 ms fix inside the ROS depth
  callback blocks the single-threaded executor, so odometry is not delivered
  while it works — measured as 89 Hz of SLAM arriving and 16 Hz of FreeD going
  out. `LidarMapMatcher.submit()` hands the newest frame to a worker (single
  slot, newest wins — matching a stale frame corrects for where the camera
  *was*) and returns; front-ends poll `fix_counter`.
* **Lock quality and live status are separate.** A rejected live fix reports a
  zero verification by construction (the gates fire before verification runs)
  and was overwriting the lock's own figures, so a good lock displayed
  `On scan: 0%`.
* **Yaw multi-start.** Heading is what a hand-placed hint gets most wrong and
  what ICP recovers least well. Candidates around the hint are screened on the
  coarse pyramid, a shortlist is refined at full resolution, and a hint that is
  already good exits the search immediately. Refining only the single leader was
  measurably losing good solutions.
* **The scale slider no longer silently beats the config.** `lidar.map_scale`
  was being overwritten by the viewport's **S** box default of 1.0 the first
  time anything touched the alignment, because the dashboard reads the slider
  and pushes it to the matcher. On real hardware that loaded a centimetre scan
  as a 309 m room: every lock was refused (`fitness 0.00`) and the camera read
  `X -51 m, Y -50 m, Z +94 m` while sitting 0.35 m from the marker. Config now
  drives the slider at startup, importing a scan auto-detects its units (picking
  the most room-like reading where more than one is plausible, and never
  overriding a value the operator set), and `MapViewerWidget.set_alignment_scale()`
  keeps the two in step. The **S** box also gained the decimals to represent
  0.01 at all — it was limited to two.
* **ICP scoring moved off the GIL.** `_verify()` and `validate_depth()` scored
  points through Open3D's one-point-per-call KD-tree API, i.e. a 600-iteration
  Python loop on the ROS callback thread. That starved the GUI thread's 60 Hz
  FreeD timer to ~17 Hz with SLAM arriving at 89 Hz. Both now use
  `scipy.spatial.cKDTree.query()`, which does the batch in C and releases the
  GIL.
* **The 3D viewport no longer saturates the GUI thread.** The dashboard froze
  with the window enlarged, and the cause was one Python call per line segment.
  `_project_segment` costs ~13 us; the camera trail is `TRAIL_MAX` = 2000
  segments and the ground grid another 486, every frame, in a viewport
  repainted from the 60 Hz UI timer. Measured on the 32k-face room scan at
  1900x1040, with the trail at its steady-state length: **35.4 ms per paint,
  28 fps**, against a 16.7 ms budget — plus 7.2 ms in each of the Top and Side
  views, which draw the same trail with the same per-point loop. ~50 ms of
  painting demanded per 16.7 ms tick is a UI that cannot respond, and window
  size only decided how far past the edge it already was (the same paint at
  700x450 was 34.2 ms — the size term is ~1 ms; the trail is the other 26).

  `_project_segments` now clips and projects every segment in one numpy pass,
  the trail draws as a single batched `drawLines`, and the ortho views project
  their points and trail as arrays. Same viewport, same pixels: the batched
  projector is asserted equal to the scalar one segment-for-segment in
  `test_map_viewer.py`, and the rendered grid + trail are pixel-identical
  (0 differing pixels of 267,961 lit).

  **The batching must not reach BLAS.** Written the obvious way, projection is
  `pts @ v` — an (N, 3) array against a 3-vector. numpy hands that to OpenBLAS,
  which wakes one thread per core; measured at 1900x1040 with the full trail,
  that is **1462 % CPU** — fourteen cores — to save 0.01 ms of wall time, and
  since the viewport projects on every paint the pool never settles. It looks
  like a win on a latency benchmark while starving the GUI thread, the ROS
  callbacks and SLAM of the machine. `_dot3()` does the same arithmetic as
  explicit column products and stays on one core. Whether a GEMV of this shape
  threads at all is a property of the numpy build — the container's does, the
  host's numpy 2.2 does not — so `ProjectionStaysOnOneCore` measures a control
  and skips where the fault cannot appear.

  Measured at 1900x1040, full trail, per paint / process CPU:

  | | paint | CPU |
  |---|---|---|
  | before | 32.8 ms | 100 % (one core) |
  | batched via `@` | 8.0 ms | **1462 %** |
  | batched via `_dot3` | **4.4 ms** | **37 %** |
* **Map fusion has its own status panel.** It was appended to the marker
  diagnostics box, whose `setFixedHeight(100)` clipped it — so the one readout
  that explains a refused lock was invisible.
* **The marker diagnostics panel is gone.** It reported PnP distance, depth
  distance, residual and bearing continuously, which cost an IPPE_SQUARE solve
  on the image callback at 5 Hz whenever a marker was in frame; all of it was
  advisory, and `try_anchor` re-solves independently at the moment it matters.
  Anchoring is unchanged. The `Depth vs Scan` agreement readout, which had been
  sharing that box, moved into the Map Fusion panel — it checks the fusion, so
  it belongs with it.
* **The viewport repaints at 30 Hz, not 60.** Pose arrives on the dashboard's
  UI timer, which runs at 60 Hz because the FreeD stream needs that cadence.
  Every arrival pushed into all three views and repainted them, so a
  monitoring viewport was being drawn as often as the camera stream is sent.
  `MapViewerWidget.update_camera_pose` still takes every pose; only the push
  into the views is coalesced (`VIEW_REFRESH_HZ`).
* **A stationary camera no longer fills the trail.** A point was appended on
  every refresh whether or not the camera had moved, so holding still built
  `TRAIL_MAX` copies of one position inside 33 s — 2000 segments redrawn in
  three views, every frame, showing nothing. `TRAIL_MIN_STEP_M` (2 mm) gates
  on distance from the last point, so slow movement still records, spaced.

  Measured on the real dashboard window at 1920x1080, driven at a true 60 Hz:

  | | paints/s (3D + ortho) | process CPU |
  |---|---|---|
  | moving, full trail — before | 60 + 120 | 59.8 % |
  | moving, full trail — after | 21 + 43 | **23.5 %** |
  | held still — before | 60 + 120 | 33.8 % (trail grew to 360 in 6 s) |
  | held still — after | 23 + 46 | **12.7 %** (trail stays at 1) |

* **The Outliner is drag-resizable.** A grip along its bottom edge changes
  the panel's height; the list takes the slack, the buttons stay pinned under
  it. It also fixes the panel clipping itself — the list was a fixed 150 px
  inside a panel fixed at 135 px, so 'World Origin' and the locator entry sat
  behind the Gizmo button. Default 210 px shows all five rows; the floor is
  whatever the contents actually need (`minimumSizeHint`, ~148 px) so it can
  never be squashed into overlap again, and the ceiling is the viewport, with
  the dragged height remembered when a smaller window forces it down. The long
  locator entry elides instead of raising a horizontal scrollbar.
* **The maximized window froze because the GUI thread blocked in the GL buffer
  swap.** Reported as "works windowed, freezes as soon as I maximize", and the
  process measured *healthy* throughout: painting, timers firing, FreeD out at
  60 Hz, pixels changing, ~50 % CPU. What it was not doing was answering X
  messages. Two stacks a second apart, taken while frozen, were identical:

  ```
  pthread_cond_wait
  xcb_wait_for_special_event          <- waiting on a Present event
  libGLX_nvidia.so.595.84  (x5)
  QPlatformBackingStore::composeAndFlush   <- compositing the QOpenGLWidget
  ...
  QEventLoop::exec
  ```

  A `QOpenGLWidget` inside a widget hierarchy is presented through
  `composeAndFlush`, which ends in a GLX swap. At the default swap interval of
  1 that swap waits for the compositor's Present event, and while parked there
  the thread is *not in the event loop*: input dies and the window manager
  reports "not responding", while timers still fire in the gaps and the
  viewport still repaints — which is why every CPU and paint metric looked
  fine. `_NET_WM_PING`, the protocol the WM itself uses to decide an app has
  hung, was the measurement that separated them: no reply from the dashboard's
  window, 10 ms from mutter's frame beside it.

  `QSurfaceFormat.setSwapInterval(0)` at import in `map_viewer.py`. Nothing
  here needs the retrace — repaints are already capped by `VIEW_REFRESH_HZ`,
  and tearing in a monitoring viewport costs nothing against a dead UI. The
  context reports the interval honoured (0) on this driver.

  Why only when *maximized*: an exactly screen-sized window takes a different
  presentation path from one merely dragged to nearly that size. Dragging the
  edges to fill the screen never froze; clicking maximize always did.
* **The gizmo works like Unreal's: one manipulator at a time.** It used to
  draw the heading ring and the translate arrows together, so a click near the
  centre moved or turned the object depending on which handle happened to be
  nearer the cursor. There are now two modes, **Move** and **Rotate**, with
  buttons under the Outliner and **W** / **E** to switch (ignored while flying,
  where W and E are forward and up). Only the active mode is drawn or
  pickable.
  * **Rotate is new.** Three world-aligned rings turn about world X, Y and Z —
    red, green, blue, matching the translate arrows. The old gizmo could only
    yaw. Rings are world-aligned rather than object-aligned so that what you
    drag matches what the alignment boxes and the pose readout say. A ring
    follows the cursor whichever side of the object you view it from: the
    drag's sign comes from whether that axis points toward the camera or away,
    which the old yaw-only code did not need and did not have.
  * **Orientation is now a full (roll, pitch, yaw) per target** instead of a
    lone heading, for the locator, the camera dot and the scan.
    `set_locator_transform` used to discard roll and pitch, which would have
    thrown away whatever the new gizmo produced on the next external set.
  * **Rotating the scan turns it about its own centroid**, on any axis, and
    writes roll/pitch/yaw back to the alignment boxes. It applies the *delta*
    rotation, so a repeated drag does not turn it twice; asserted against an
    exact rotation in `test_map_viewer.py`.
  * Fixed alongside: `_axis_screen_dir` measured its pixels-per-metre at the
    *locator* whichever object was selected, so dragging the camera or the
    scan scaled by the wrong point whenever they were apart.
  * The Outliner's default height went 210 → 240 px to fit the mode row.
* **The gizmo no longer jumps to the locator.** Two faults, both making the
  outliner lie about what the gizmo would move. `_setup_outliner` called
  `setCurrentRow(0)` *before* connecting `currentRowChanged`, so the handler
  never ran: from startup the list highlighted "Camera" while `_gizmo_target`
  was still `__init__`'s `'locator'`. And `mousePressEvent` selected the
  Locator row on *any* handle grab, so a drag switched the target to the
  locator and the list jumped to match — which is what the operator saw as
  the gizmo "switching to locator" on the second attempt to move the camera.
  The signal is now connected before the default row is chosen, and a grab
  selects the row of the target the gizmo is already on. The default gizmo is
  unchanged (the locator); only the highlight now tells the truth about it.
* **The ArUco anchor buttons are gone.** "ANCHOR TO ARUCO GROUND ZERO" and
  "RESET ANCHOR" were removed from the left column, and with them the only
  things that armed anchoring in the dashboard — so `_anchor_aruco`,
  `_reset_anchor`, `_on_anchored`, the `request_ground_zero` plumbing and the
  `try_anchor` branch in the image callback went too rather than sit there
  unreachable. Marker *detection* stays: the camera view still outlines
  markers and reports `MARKER f/id DETECTED`. Anchoring itself is untouched in
  `tracking_engine.py` and still reachable over the wire through
  `handle_command("anchor_aruco" / "anchor_lidar" / "set_anchor")`, which is
  how `bridge_headless.py` drives it.
* **The viewport's Clip Planes panel is gone.** The Near/Far sliders were
  removed from the 3D view; the clip range itself stays (projection and the
  GL projection matrix both need it) at the same defaults, 5 cm and 10 km,
  now fixed in `PerspView.__init__` with `set_near_clip()` / `set_far_clip()`
  left as the programmatic way to change them. The Outliner keeps its corner.
* **The Map Fusion panel sizes to its content.** It is variable-length — five
  base lines, plus image lineup, plus a rejection reason, plus the Depth vs
  Scan pair, and the `Lock` line wraps at this column width. Measured at the
  real 380 px width the worst case is 230 px, so any fixed height clips it;
  that is how the readout got lost the first time. `setMinimumHeight` plus a
  height-for-width size policy now lets it grow, and the left column is a
  scroll area, so it scrolls instead of pushing anything.
* **Scan units.** `T_align` carries the viewport's scale slider in its rotation
  block, making it a *similarity* — `invert_transform` on it is off by the square
  of the scale (at the 0.01 a centimetre scan needs, essentially 100 % wrong).
  `decompose_similarity()` / `invert_similarity()` split it; the matcher holds
  its map in metres via `map_scale` and every pose composition uses the rigid
  part. The loader prints the scan's size in metres and warns when it is not
  plausibly a room.
* `pose_transform.py` gains `so3_log/exp`, `se3_log/exp`, `se3_interpolate`,
  `decompose_similarity`, `invert_similarity`, and now owns `T_BODY_OPTICAL`
  (there were three copies).
* `TrackingEngine.apply_map_alignment()` is the single entry point; both
  front-ends use it, and `reset_pose` drops the fix.
* New `tracking.yaml` keys under `lidar:` — separate ICP radii from the
  rejection gate, slew limits, the D455 depth band, and `map_scale`.

Measured on `Room01_Scaled_UE.obj` (3.1 × 2.3 × 2.9 m), three viewpoints, depth
ray-cast with realistic stereo noise: 137 mm drift → 19 mm and 3° → 0.3° within
20 fixes, at 9 ms per fix against a 200 ms budget at 5 Hz. Tests:
`test_lidar_fusion.py` (52), plus `bench_fusion.py` and `smoke_locator_lock.py`
for real-scan verification.


**LiDAR-scan anchoring.** Anchoring can now map the marker onto a *chosen*
transform instead of always onto the world origin, so tracking output can be
expressed in the coordinate frame of a LiDAR-scanned set imported into Unreal.

The workflow: a printed ArUco marker is taped to a known spot in the physical
set; an anchor Actor sits at that same spot in the Unreal scan; the operator
points the camera at the marker and presses anchor; from then on the streamed
pose is in the scan's frame.

What was wrong before: both anchor paths called
`set_ground_zero_from_transform(T_marker)`, which pins the marker to the origin.
The anchor Actor's transform *was* being received over UDP 11114 by
`bridge_headless.py`, but it only ever reached the LiDAR ICP matcher — never the
FreeD output. So the virtual camera came out offset by exactly the anchor Actor's
position, which looks perfect whenever that Actor sits near the origin and is
silently wrong everywhere else.

* `PoseTransformer.set_ground_zero_from_transform(T_marker, T_target=None)` —
  `_zero = T_target @ inv(T_marker)`. Default None keeps the old origin
  behaviour, so nothing else changed.
* `PoseTransformer.unreal_to_ros_transform()` — converts an Unreal Actor
  transform (centimetres, left-handed, pan/tilt/roll) into ROS canonical metres
  by exactly inverting this instance's own output mapping, signs and trim
  included. A sign flipped in tracking.yaml therefore moves the anchor and the
  live pose together instead of putting them in different frames.

**One engine, two front-ends.** `tracking_engine.TrackingEngine` now owns the
pose pipeline, marker solving, anchoring, FreeD encoding and the Unreal command
protocol. `dashboard.py` and `bridge_headless.py` are thin front-ends over it and
delegate through properties. They had each grown their own copy of the pose path
and the anchoring maths in subtly different code.

**The dashboard gained the inbound command channel** (UDP 11114) it never had, so
an anchor triggered from Unreal reaches the GUI too. If the port is already held
(usually by `bridge_headless.py`) it says so and keeps tracking.

**Depth cross-check on the marker solve.** The solve now uses
`SOLVEPNP_IPPE_SQUARE` rather than the general iterative solver, and where a
depth frame is available it measures distance to the marker independently. Four
coplanar corners are pose-ambiguous, so a mono solve can settle on the mirrored
pose — and a flipped anchor takes the whole virtual set with it. On disagreement
beyond `aruco.depth_tolerance_m` the anchor is **refused and stays armed** rather
than anchoring wrongly.

Verified on live hardware, camera streaming stereo + depth + IMU:

    anchor_lidar (250, -400, 120) cm pan 30  ->  target ROS [2.5, 4.0, 1.2] m
    inconsistent sighting  -> refused: "depth disagrees with PnP by 988mm"
    consistent sighting    -> anchored, PnP 1.988 m vs depth 1.984 m, residual 4 mm
    camera pose out        -> (60.3, -458.2, 120.3) cm pan +17.05

Those output numbers were checked by hand against the expected
`T_target @ inv(T_marker) @ T_camera`: (60.6, -458.3, 120.2) cm pan +17.1.

Tests: 133 host-side + 29 integration; 162 all-in inside the container.

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
