#!/usr/bin/env python3
"""
Tests for the 3D viewport's batched screen projection (map_viewer.py).

    QT_QPA_PLATFORM=offscreen python3 -m unittest test_map_viewer

`PerspView` used to project one line segment per Python call. At ~13 us each
that is nothing for a single line and 26 ms a frame for the 2000-point camera
trail plus the 486-segment grid -- which put one paint at 35 ms against a
16.7 ms UI budget, so the GUI thread never caught up and the dashboard stopped
responding (worst when the window is large, because every other cost grows too).

`_project_segments` does the same maths for every segment at once. The whole
fix rests on it being *exactly* equivalent to the per-segment version it
replaced, so that is what is asserted here, over randomised camera poses and
clip ranges including the degenerate cases the scalar code special-cased:
zero-length segments and segments parallel to the view plane.
"""
import os
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QPoint, Qt
    import map_viewer
    _HAS_QT = True
except Exception:                                    # pragma: no cover
    _HAS_QT = False

_APP = None


def setUpModule():
    global _APP
    if _HAS_QT:
        _APP = QApplication.instance() or QApplication([])


@unittest.skipUnless(_HAS_QT, "PyQt5 not available")
class ProjectSegmentsMatchesScalar(unittest.TestCase):
    """The batched projector must agree with `_project_segment`, segment for
    segment: same acceptances, same rejections, same screen coordinates."""

    def setUp(self):
        self.view = map_viewer.PerspView()
        self.view.resize(1280, 720)

    def _random_camera(self, rng):
        v = self.view
        v._orb_yaw = rng.uniform(-180, 180)
        v._orb_pitch = rng.uniform(-89, 89)
        v._orb_dist = 10 ** rng.uniform(-1, 2)
        v._orb_center = rng.normal(0, 5, 3)
        v._near = 10 ** rng.uniform(-3, 0)
        v._far = v._near * 10 ** rng.uniform(0.5, 4)
        return v._eye_basis()

    def test_matches_segment_for_segment(self):
        rng = np.random.default_rng(7)
        compared = 0
        for _ in range(120):
            basis = self._random_camera(rng)
            eye, fwd, _, _ = basis
            A = rng.normal(0, 20, (40, 3))
            B = A + rng.normal(0, 20, (40, 3))
            A[0], B[0] = A[0], A[0]                  # zero length
            B[1] = A[1] + np.cross(fwd, [0., 0., 1.]) * 3.0   # view-plane parallel

            screen, keep = self.view._project_segments(A, B, basis)
            for i in range(len(A)):
                old = self.view._project_segment(A[i], B[i], basis)
                if old is None:
                    self.assertFalse(keep[i], "batched kept a rejected segment")
                    continue
                self.assertTrue(keep[i], "batched rejected an accepted segment")
                np.testing.assert_allclose(screen[i], old, rtol=1e-9, atol=1e-7)
                compared += 1
        self.assertGreater(compared, 1000)

    def test_empty_input(self):
        basis = self.view._eye_basis()
        screen, keep = self.view._project_segments(
            np.zeros((0, 3)), np.zeros((0, 3)), basis)
        self.assertEqual(len(keep), 0)
        self.assertEqual(screen.shape, (0, 2, 2))

    def test_polyline_skips_clipped_segments(self):
        """A trail running from behind the camera to in front of it yields
        fewer lines than points, and never a line per rejected segment."""
        basis = self.view._eye_basis()
        eye, fwd, _, _ = basis
        pts = [eye + fwd * d for d in np.linspace(-5.0, 5.0, 50)]
        lines = self.view._polyline_lines(pts, basis)
        self.assertGreater(len(lines), 0)
        self.assertLess(len(lines), len(pts) - 1)



@unittest.skipUnless(_HAS_QT, "PyQt5 not available")
class FramebufferResizeTests(unittest.TestCase):
    """Resize must preserve GPU geometry and present a nonblank scene."""

    def test_mesh_survives_maximize_sized_framebuffer_recreation(self):
        from PyQt5.QtGui import QImage
        view = map_viewer.PerspView()
        view.outliner_widget.hide()
        view._gizmo_enabled = False
        view.set_mesh(np.array([[-2., -2., 0.], [2., -2., 0.], [0., 2., 0.]]),
                      np.array([[0, 1, 2]]))
        view.resize(640, 480)
        view.show()
        QApplication.processEvents()
        try:
            if not view.isValid():
                self.skipTest("No OpenGL context on this test platform")
            resources = None
            for width, height in ((640, 480), (1920, 1080), (800, 600),
                                  (2560, 1440), (640, 480)):
                view.resize(width, height)
                QApplication.processEvents()
                frame = view.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
                dpr = view.devicePixelRatioF()
                # A real window manager can clamp the requested size to the
                # screen's available area (e.g. excluding GNOME's top bar).
                self.assertAlmostEqual(frame.width(), view.width() * dpr, delta=1)
                self.assertAlmostEqual(frame.height(), view.height() * dpr, delta=1)
                bits = frame.bits()
                bits.setsize(frame.byteCount())
                rgba = np.frombuffer(bits, np.uint8).reshape(
                    frame.height(), frame.bytesPerLine() // 4, 4)
                # Background is dark blue (< 60). The shaded triangle must
                # actually reach the framebuffer, not just issue GL calls.
                mesh = (rgba[:, :, 2] > 60) & (
                    rgba[:, :, 2] > rgba[:, :, 0].astype(float) * 1.2)
                self.assertGreater(int(mesh.sum()), 100)
                current = (view._gl_prog, view._gl_vbo, view._gl_ebo)
                if resources is not None:
                    self.assertEqual(current, resources,
                                     "Resizing should reuse mesh buffers")
                resources = current
        finally:
            view.close()

def _cores_per_wall_second(fn, n):
    """CPU-seconds burned per wall-second. ~1.0 single-threaded, ~ncores if a
    threaded BLAS is doing the work (process_time counts every thread)."""
    fn()
    t0, c0 = time.perf_counter(), time.process_time()
    for _ in range(n):
        fn()
    return (time.process_time() - c0) / max(time.perf_counter() - t0, 1e-9)


@unittest.skipUnless(_HAS_QT, "PyQt5 not available")
class ProjectionStaysOnOneCore(unittest.TestCase):
    """Projection must not dispatch to a threaded BLAS.

    `pts @ v` for an (N, 3) array against a 3-vector is a GEMV, and numpy
    hands it to OpenBLAS, which wakes one thread per core. For this shape
    that measured ~1460 % CPU to save ~0.01 ms of wall time -- and because
    the viewport projects on every paint, the pool never settles, so the
    machine has nothing left for the GUI thread and the dashboard stops
    responding. The regression is invisible to a latency benchmark: the
    batched version looks *faster* while pegging every core. So assert the
    cost, not the speed.

    Whether a GEMV threads at all depends on the numpy build -- the Isaac
    container's does, the host's numpy 2.2 does not -- so this measures a
    control first and skips where the fault could not show up anyway.
    """

    def test_projection_is_not_multithreaded(self):
        rng = np.random.default_rng(3)
        view = map_viewer.PerspView()
        view.resize(1900, 1040)
        basis = view._eye_basis()
        A = rng.normal(0, 10, (4000, 3))
        B = A + 1.0

        # Measure the subject FIRST, before anything has touched BLAS. Once
        # the pool is awake its threads keep spinning for a while, which is
        # the whole reason this is so damaging -- and it would contaminate
        # this measurement exactly as it starves the GUI thread in the app.
        got = _cores_per_wall_second(
            lambda: view._project_segments(A, B, basis), 200)

        rel = rng.normal(0, 10, (4000, 3))
        vec = np.array([0.3, 0.5, 0.8])
        control = _cores_per_wall_second(lambda: rel @ vec, 3000)
        if control < 2.5:
            self.skipTest(f"this numpy does not thread a (N,3) GEMV "
                          f"({control:.1f} cores); regression not detectable here")

        self.assertLess(got, 2.5,
                        f"projection burns {got:.1f} cores of CPU per wall-second "
                        f"(a threaded GEMV costs {control:.1f} here) -- it is "
                        f"dispatching to BLAS again")


@unittest.skipUnless(_HAS_QT, "PyQt5 not available")
class OutlinerResizes(unittest.TestCase):
    """The Outliner panel is drag-resizable vertically.

    It floats over the GL surface as an absolutely positioned child, so its
    geometry is set by hand in `_layout_outliner` rather than by a layout --
    which is what let it be pinned to 135 px around a 150 px list, clipping
    'World Origin' and the locator behind the buttons.
    """

    def setUp(self):
        self.view = map_viewer.PerspView()
        self.view.resize(1200, 800)
        self.view.show()
        QApplication.processEvents()

    def tearDown(self):
        self.view.close()

    def test_default_shows_every_entry(self):
        lst = self.view.outliner_list
        row_h = lst.sizeHintForRow(0)
        self.assertGreaterEqual(lst.height(), row_h * lst.count(),
                                "default height clips an outliner entry")

    def test_drag_grows_and_shrinks(self):
        before = self.view.outliner_widget.height()
        self.view._drag_outliner_height(+120)
        self.assertEqual(self.view.outliner_widget.height(), before + 120)
        self.view._drag_outliner_height(-60)
        self.assertEqual(self.view.outliner_widget.height(), before + 60)

    def test_never_squashed_below_what_the_contents_need(self):
        self.view._drag_outliner_height(-10000)
        panel = self.view.outliner_widget
        self.assertGreaterEqual(panel.height(),
                                panel.minimumSizeHint().height())
        # the buttons must still be inside the panel, not overlapped away
        for btn in (self.view.btn_gizmo, self.view.btn_center):
            self.assertLessEqual(btn.y() + btn.height(), panel.height())

    def test_never_taller_than_the_viewport(self):
        self.view._drag_outliner_height(+10000)
        panel = self.view.outliner_widget
        self.assertLessEqual(panel.y() + panel.height(), self.view.height())

    def test_requested_height_survives_a_smaller_window(self):
        """Shrinking the window must not silently forget the drag."""
        self.view._drag_outliner_height(+200)
        tall = self.view.outliner_widget.height()
        self.view.resize(1200, 260)
        QApplication.processEvents()
        self.assertLess(self.view.outliner_widget.height(), tall)
        self.view.resize(1200, 800)
        QApplication.processEvents()
        self.assertEqual(self.view.outliner_widget.height(), tall)


@unittest.skipUnless(_HAS_QT, "PyQt5 not available")
class MultipleLocators(unittest.TestCase):
    def setUp(self):
        self.view = map_viewer.PerspView()
        self.addCleanup(self.view.deleteLater)

    def select(self, key):
        for i in range(self.view.outliner_list.count()):
            if self.view.outliner_list.item(i).data(Qt.UserRole) == key:
                self.view.outliner_list.setCurrentRow(i)
                return
        self.fail(f"Missing locator {key}")

    def test_duplicate_keeps_pose_but_needs_its_own_id(self):
        original = np.eye(4)
        original[:3,3] = [1,2,3]
        self.view.set_locator_transform(original)
        self.view.assign_locator_marker("DICT_4X4_50", 7, .2)
        duplicate = self.view.duplicate_locator()
        self.assertIsNotNone(duplicate)
        np.testing.assert_allclose(self.view.get_locator_transform(), original)
        self.assertIsNone(self.view.locator_entries()[duplicate]["marker"])
        with self.assertRaisesRegex(ValueError, "already belongs"):
            self.view.assign_locator_marker("DICT_4X4_50", 7, .2)
        self.view.assign_locator_marker("DICT_4X4_50", 8, .15)
        self.view._edit_channel(0, 4.)
        self.view._edit_channel(5, 30.)
        self.select("locator")
        np.testing.assert_allclose(self.view.get_locator_transform(), original)
        self.select(duplicate)
        self.assertAlmostEqual(self.view.get_locator_transform()[0,3], 4.)
        self.assertAlmostEqual(self.view._locator_rpy[2], 30.)
        self.assertEqual(self.view.locator_entries()["locator"]["marker"]["id"], 7)

    def test_auto_placed_marker_gets_its_own_locator_without_taking_the_selection(self):
        self.view.assign_locator_marker("DICT_4X4_250", 3, .1)
        selected = self.view.get_locator_transform()
        T = np.eye(4)
        T[:3, 3] = [1, -2, .2]
        key = self.view.add_marker_locator(T, "DICT_4X4_250", 120, .1)
        self.assertEqual(self.view._active_locator_id, "locator")
        np.testing.assert_allclose(self.view.get_locator_transform(), selected)
        entry = self.view.locator_entries()[key]
        np.testing.assert_allclose(entry["transform"], T)
        self.assertEqual(entry["marker"], dict(dictionary="DICT_4X4_250", id=120, size_m=.1))
        with self.assertRaisesRegex(ValueError, "already belongs"):
            self.view.add_marker_locator(T, "DICT_4X4_250", 3, .1)
        self.assertEqual(len(self.view.locator_entries()), 2)
        self.select(key)
        self.assertIn("ID 120", self.view.outliner_list.currentItem().text())

    def test_gizmo_row_selection_stays_on_duplicate_during_drag(self):
        duplicate = self.view.duplicate_locator()
        self.view._select_gizmo_row("locator")
        self.assertEqual(self.view.outliner_list.currentItem().data(Qt.UserRole), duplicate)
        self.view._gizmo_set_pos([5,6,7])
        self.view._gizmo_changed()
        np.testing.assert_allclose(self.view.get_locator_transform("locator"), np.eye(4))
        np.testing.assert_allclose(self.view.get_locator_transform(duplicate)[:3,3], [5,6,7])

    def test_assign_from_detected_id_and_remove_duplicate(self):
        self.view.set_detected_markers([("DICT_4X4_50", 12, np.zeros((4,2)))])
        self.view.locator_width.setValue(.25)
        self.view._assign_detected_marker()
        self.assertEqual(self.view.locator_entries()["locator"]["marker"],
                         dict(dictionary="DICT_4X4_50", id=12, size_m=.25))
        duplicate = self.view.duplicate_locator()
        self.view.remove_locator()
        self.assertNotIn(duplicate, self.view.locator_entries())
        self.assertEqual(self.view._active_locator_id, "locator")
        self.assertEqual(self.view.locator_entries()["locator"]["marker"]["id"], 12)
        self.view.set_detected_markers([])
        self.assertIsNone(self.view.detected_marker.currentData())
        self.view.clear_locator_marker()
        self.assertIsNone(self.view.locator_entries()["locator"]["marker"])


@unittest.skipUnless(_HAS_QT, "PyQt5 not available")
class ViewPushesAreRateLimited(unittest.TestCase):
    """Pose arrives at the UI timer's 60 Hz; the views must not repaint that
    often. Each push repaints the 3D view (then also two ortho views), and
    at 1920x1080 with a full trail that measured 60 % of a core against a
    16.7 ms budget -- the GUI thread had nothing left to answer events with.
    """

    def setUp(self):
        self.mv = map_viewer.MapViewerWidget()

    def tearDown(self):
        self.mv.deleteLater()

    def _feed(self, n, step, dt):
        pushes = []
        real = self.mv.persp_view.set_cam
        self.mv.persp_view.set_cam = lambda *a, **k: (pushes.append(1),
                                                      real(*a, **k))[1]
        q = np.array([0., 0., 0., 1.])
        for i in range(n):
            self.mv.update_camera_pose(np.array([i * step, 0., 0.]), q)
            time.sleep(dt)
        return len(pushes)

    def test_pushes_far_below_the_pose_rate(self):
        """60 poses over ~1 s must not become 60 repaints."""
        pushes = self._feed(60, 0.01, 1 / 60)
        self.assertLessEqual(pushes, 40, "view pushes are not rate limited")
        self.assertGreater(pushes, 5, "rate limiting starved the view entirely")

    def test_a_still_camera_does_not_fill_the_trail(self):
        """Held still, every pose used to append a duplicate point until the
        trail was TRAIL_MAX copies of one position -- drawn every frame in
        three views."""
        q = np.array([0., 0., 0., 1.])
        for _ in range(300):
            self.mv.update_camera_pose(np.array([1.0, 2.0, 0.3]), q)
        self.assertEqual(len(self.mv._trail), 1)

    def test_movement_still_records(self):
        """The filter is a distance gate, not a rate limit: slow movement
        still lands on the trail, just spaced by TRAIL_MIN_STEP_M."""
        q = np.array([0., 0., 0., 1.])
        step = self.mv.TRAIL_MIN_STEP_M / 4      # a quarter step per pose
        for i in range(200):
            self.mv.update_camera_pose(np.array([i * step, 0., 0.]), q)
        self.assertGreater(len(self.mv._trail), 40)
        self.assertLess(len(self.mv._trail), 60)

    def test_clearing_the_trail_reaches_the_views(self):
        q = np.array([0., 0., 0., 1.])
        for i in range(50):
            self.mv.update_camera_pose(np.array([i * 0.05, 0., 0.]), q)
        self.mv._push_trail()
        self.assertGreater(len(self.mv.persp_view._trail), 1)
        self.mv._trail.clear()
        self.mv._push_trail()
        self.assertEqual(len(self.mv.persp_view._trail), 0)


@unittest.skipUnless(_HAS_QT, "PyQt5 not available")
class GizmoStaysOnTheSelectedObject(unittest.TestCase):
    """Dragging a gizmo must not change which object it belongs to.

    Two faults did that. `_setup_outliner` called `setCurrentRow(0)` before
    connecting `currentRowChanged`, so the handler never ran and the list
    showed "Camera" while `_gizmo_target` was still its `__init__` default,
    'locator'. And `mousePressEvent` selected the Locator row on *any* handle
    grab, so a drag switched the target to the locator and the outliner
    jumped to match -- the operator saw the gizmo "switch to locator" when
    they tried to move the camera a second time.
    """

    def setUp(self):
        # Deliberately not shown: this is selection logic, and painting a
        # QOpenGLWidget under the offscreen platform has no GL context.
        self.view = map_viewer.PerspView()
        self.view.resize(1200, 800)

    def tearDown(self):
        self.view.deleteLater()

    def _select(self, prefix):
        lst = self.view.outliner_list
        for row in range(lst.count()):
            if lst.item(row).text().startswith(prefix):
                lst.setCurrentRow(row)
                break

    def _selected(self):
        item = self.view.outliner_list.currentItem()
        return item.text() if item else ""

    def test_startup_selection_matches_the_gizmo(self):
        spec = self.view.GIZMO_TARGETS[self.view._gizmo_target]
        self.assertTrue(self._selected().startswith(spec['prefix']),
                        f"outliner says {self._selected()!r} but the gizmo is "
                        f"on {self.view._gizmo_target!r}")

    def test_grabbing_a_handle_keeps_the_target(self):
        for prefix, expect in (("Camera", "camera"),
                               ("Locator", "locator"),
                               ("Point Cloud", "mesh")):
            with self.subTest(target=prefix):
                self._select(prefix)
                self.assertEqual(self.view._gizmo_target, expect)
                # two grabs in a row: the second is where the operator saw it
                for _ in range(2):
                    self.view._select_gizmo_row(self.view._gizmo_target)
                    self.view._begin_locator_drag('x', QPoint(600, 400))
                    self.view._active_gizmo = None
                    self.assertEqual(self.view._gizmo_target, expect)
                    self.assertTrue(self._selected().startswith(prefix))

    def test_selecting_a_target_moves_that_target(self):
        """The drag must act on the selected object, not on the locator."""
        self._select("Camera")
        before_cam = np.asarray(self.view._gizmo_pos(), dtype=float).copy()
        self._select("Locator")
        before_loc = np.asarray(self.view._gizmo_pos(), dtype=float).copy()
        self._select("Camera")
        self.view._gizmo_set_pos(before_cam + np.array([1.0, 0., 0.]))
        self._select("Locator")
        np.testing.assert_allclose(
            np.asarray(self.view._gizmo_pos(), dtype=float), before_loc,
            err_msg="moving the camera gizmo also moved the locator")


@unittest.skipUnless(_HAS_QT, "PyQt5 not available")
class GizmoModes(unittest.TestCase):
    """Translate and rotate are separate manipulators, one at a time.

    The old gizmo drew the heading ring and the translate arrows together, so
    a click near the centre could move or turn the object depending on which
    handle happened to be nearer the cursor. Unreal switches manipulator
    instead (W / E), and only the active one is drawn or pickable.
    """

    def setUp(self):
        self.view = map_viewer.PerspView()
        self.view.resize(1200, 800)
        self.view._orb_center = np.zeros(3)
        self.view._orb_dist = 6.0
        self.view._orb_yaw, self.view._orb_pitch = 35.0, 25.0

    def _pickable(self):
        found = set()
        for x in range(0, 1200, 8):
            for y in range(0, 800, 8):
                h = self.view._pick_gizmo_handle(QPoint(x, y))
                if h:
                    found.add(h)
        return found

    def test_default_is_translate(self):
        self.assertEqual(self.view.gizmo_mode(), "translate")
        self.assertTrue(self.view.btn_move.isChecked())
        self.assertFalse(self.view.btn_rotate.isChecked())

    def test_only_the_active_mode_is_pickable(self):
        self.view.set_gizmo_mode("translate")
        self.assertEqual(self._pickable(), {"x", "y", "z", "plane_xy"})
        self.view.set_gizmo_mode("rotate")
        self.assertEqual(self._pickable(), {"rot_x", "rot_y", "rot_z"})

    def test_buttons_track_the_mode(self):
        self.view.set_gizmo_mode("rotate")
        self.assertTrue(self.view.btn_rotate.isChecked())
        self.assertFalse(self.view.btn_move.isChecked())

    def test_each_ring_turns_exactly_one_axis(self):
        basis = self.view._eye_basis()
        for key, idx, _axis, _col in self.view.ROT_RINGS:
            with self.subTest(ring=key):
                self.view.set_gizmo_mode("rotate")
                self.view._gizmo_set_rpy(np.zeros(3))
                o, _ok = self.view._project_pts(
                    np.asarray(self.view._gizmo_pos(), float).reshape(1, 3), basis)
                start = QPoint(int(o[0][0]) + 60, int(o[0][1]))
                self.view._begin_locator_drag(key, start)
                self.view._drag_locator(QPoint(int(o[0][0]) + 42,
                                               int(o[0][1]) + 42))
                rpy = self.view._gizmo_rpy()
                turned = [i for i in range(3) if abs(rpy[i]) > 1e-6]
                self.assertEqual(turned, [idx])
                self.view._active_gizmo = None

    def test_translating_does_not_rotate(self):
        self.view.set_gizmo_mode("translate")
        self.view._gizmo_set_rpy(np.array([10., 20., 30.]))
        basis = self.view._eye_basis()
        p0 = np.asarray(self.view._gizmo_pos(), float).copy()
        o, _ok = self.view._project_pts(p0.reshape(1, 3), basis)
        self.view._begin_locator_drag("x", QPoint(int(o[0][0]), int(o[0][1])))
        self.view._drag_locator(QPoint(int(o[0][0]) + 50, int(o[0][1])))
        self.assertFalse(np.allclose(p0, self.view._gizmo_pos()))
        np.testing.assert_allclose(self.view._gizmo_rpy(), [10., 20., 30.])

    def test_full_orientation_survives_a_transform_round_trip(self):
        """Roll and pitch used to be dropped by set_locator_transform, which
        would have silently discarded whatever the rotate gizmo produced."""
        self.view._gizmo_target = "locator"
        self.view._gizmo_set_rpy(np.array([12., -34., 56.]))
        T = self.view.get_locator_transform()
        other = map_viewer.PerspView()
        other.set_locator_transform(T)
        np.testing.assert_allclose(other._locator_rpy, [12., -34., 56.],
                                   atol=1e-6)

    def test_w_and_e_switch_but_not_while_flying(self):
        from PyQt5.QtGui import QKeyEvent
        from PyQt5.QtCore import QEvent
        press = lambda k: QKeyEvent(QEvent.KeyPress, k, Qt.NoModifier)
        self.view.keyPressEvent(press(Qt.Key_E))
        self.assertEqual(self.view.gizmo_mode(), "rotate")
        self.view.keyPressEvent(press(Qt.Key_W))
        self.assertEqual(self.view.gizmo_mode(), "translate")
        # holding RMB is fly mode: W/E are forward/up there, not mode keys
        self.view._drag_btn = Qt.RightButton
        self.view.keyPressEvent(press(Qt.Key_E))
        self.assertEqual(self.view.gizmo_mode(), "translate")
        self.assertIn(Qt.Key_E, self.view._keys_held)


@unittest.skipUnless(_HAS_QT, "PyQt5 not available")
class ScanRotatesAboutItsCentroid(unittest.TestCase):
    """Rotating the scan's gizmo edits the scan-alignment transform.

    It has to turn the cloud about its own centroid -- rotating about the
    world origin would swing a room-scale scan clear across the map -- and it
    must apply the *delta*, or a second drag would double the rotation.
    """

    def test_all_three_axes_rotate_in_place(self):
        from pose_transform import rpy_to_matrix
        mv = map_viewer.MapViewerWidget()
        rng = np.random.default_rng(0)
        pts = rng.normal(0, 0.4, (400, 3)) + np.array([3.0, 2.0, 0.5])
        mv.map_points_3d = pts
        mv._aligned_cache = None
        mv._update_transform()
        before = mv._aligned_centroid().copy()

        mv.persp_view._gizmo_target = "mesh"
        mv.persp_view._mesh_gizmo_rpy = np.array([15., -10., 40.])
        mv._on_persp_mesh_moved()

        np.testing.assert_allclose(mv._aligned_centroid(), before, atol=2e-3)
        self.assertAlmostEqual(mv.roll, 15.0, places=3)
        self.assertAlmostEqual(mv.pitch, -10.0, places=3)
        self.assertAlmostEqual(mv.yaw, 40.0, places=3)

        R = rpy_to_matrix(*np.radians([15., -10., 40.]))
        expect = (R @ (pts - pts.mean(axis=0)).T).T + before
        got = mv._get_aligned()
        np.testing.assert_allclose(got, expect, atol=2e-3)

        # a repeat of the same drag must not turn it twice
        mv.persp_view._mesh_gizmo_rpy = np.array([15., -10., 40.])
        mv._on_persp_mesh_moved()
        np.testing.assert_allclose(mv._get_aligned(), got, atol=2e-3)
        mv.deleteLater()


if __name__ == "__main__":
    unittest.main()
