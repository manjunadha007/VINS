# v1.7 — 2026-11-10 — Multi-marker layouts, scan scale, depth preview, GoPro view

v1.6's pipeline with several ArUco markers working together, a depth-measured
scan scale, alternating IR projection and a live depth preview. The dashboard
also drops its TOP/FRONT views for a GoPro preview beside the 3D view.

## What is new

**Several markers, one solve** (`marker_map.py`). Duplicate the Locator, place
each copy on a physical marker and **Assign ID**: visible assigned markers are
solved together and then support tracking at up to 2 Hz with gradual
corrections. Only the placed centres are trusted. Each Locator's rotation is
measured from the marker corners and the surrounding depth, and the placed
position is never moved. If the positions don't fit what the camera sees, the
status names the culprit: two swapped IDs, or one Locator and the spot the
other markers put it, in Channel Box coordinates. **Save / Load marker layout**
keeps the assignments.

**Markers place themselves.** Line up on one marker with a single unassigned
Locator. Once it is green, every other marker in view gets its own Locator
after 6 agreeing sightings and joins the joint solve. The single-marker search
now covers 1 m around the Locator at every heading, instead of ±0.5 m / 20°.
Markers default to **DICT_4X4_250**. Auto lineup also refuses when the
**Square width** disagrees with the depth-measured width by more than 10 %.

**Scan scale from depth** (`scan_scale.py`). With three assigned markers, 12 s
of paired depth and marker sightings fit one uniform scale to the scan. The
fit is checked against the room's depth before it is applied, and every
Locator scales with the scan.

**Alternating projection and a depth preview** (`realsense_streams.py`,
`depth_view.py`, `projector_control.py`). Capture is 848×480 at 90 fps, split
by the camera's emitter metadata into about 45 Hz projector-off stereo for
SLAM and 45 Hz projector-on depth for scan matching. Frame parity is never
used. **Depth points / Auto lineup** shows usable depth and how many position
axes the view constrains. **IR PROJECTOR: ON/OFF** switches projection live.
The dashboard uses Qt's `xcb_egl` backend because the window went blank under
NVIDIA GLX.

**GoPro preview** (`gopro_stream.py`). **GoPro / Camera view** sits beside the
3D view. It finds a USB-connected HERO10 from its network link, runs webcam
mode at 720p with the linear lens, and leaves webcam mode on Stop or close.
It is display only. The TOP (XY) and FRONT (XZ) views and `OrthoView` are
removed.

## Seen live on 2026-11-10

On the D455 through the powered Acer hub, against `Terras_scan02.obj`:

* The startup check passed on every launch: IR 44.9 Hz per side, IMU 400 Hz,
  odometry 49–51 Hz, depth 45 Hz.
* The width check caught squares entered as 10 cm that depth measured at
  16.2 cm; the width was then set to 16.7 cm. Three hand-placed, ID-assigned
  Locators were refused because marker 1's sat 57 cm from where the others
  put it.
* Starting again from one unassigned Locator, Auto lineup locked at 99 % scan
  agreement with 3/3 axes. The other two markers placed
  themselves, giving 3 mapped markers in support. Depth vs scan: 99.8 %
  agreement, 27 mm mean.
* GoPro HERO10 over USB: 1280×720 at 30 fps, first picture 3.6 s after
  start, decoding at 10 % of one core, out of webcam mode 0.5 s after stop.

## Limitations

* **The dashboard aborted once with `stack smashing detected`**, on the main
  (Qt) thread, about 5 s after the first GoPro frames arrived. The relaunch
  ran over 80 s with the preview on. The same message had been seen in this
  dashboard before the GoPro existed, and the cause is unknown. The deployed
  `tracking.yaml` sets `gopro: auto_start: false` until it is found. Don't
  run the preview during a take: a crash also stops FreeD.
* **The camera/SLAM launch died once** (08:03:31, a `python3` segfault in
  `librmw_fastrtps_shared_cpp.so`) while the dashboard stayed open. This was
  not investigated. A relaunch restored it.
* **The scan overlay draws only the scan's outlines**, with crease edges off
  as before. From 62 positions inside `Terras_scan02`, it drew nothing in
  52 % of views: plain walls and floor have no outline to draw.
* **The GoPro doesn't charge while streaming**, and drops off USB when the hub
  sags. Keep the power bank on the Acer hub's USB-C IN.
* v1.6's limits still apply: the 0.75 scan gate sits inside the band v1.5
  measured for wrong locks, and amber "Camera tracking" can run unvalidated
  while scan checks fail.

## Verifying this snapshot

Inside the Isaac image, with open3d, numpy<2 and PyOpenGL on `PYTHONPATH` and
`QT_QPA_PLATFORM=offscreen`:

    cd versions/v1.7
    python3 -m unittest $(ls test_*.py | sed 's/\.py$//' | grep -v -x -e test_freed -e test_layout -e test_ui)
    python3 smoke_gizmo.py && python3 smoke_lineup.py && python3 smoke_locator_lock.py

On 2026-11-10 the tree this was frozen from passed 561 unit tests (1 skipped:
no OpenGL context) and all three smoke scripts. `smoke_lineup.py` reads
`/workspaces/isaac_ros-dev/Room01_Scaled_UE01.obj` and `smoke_locator_lock.py`
reads `Room01_Scaled_UE.obj` there.
