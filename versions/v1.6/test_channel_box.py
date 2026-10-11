"""Channel-box transforms through the real outliner, widgets and view callbacks."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from map_viewer import MapViewerWidget
from pose_transform import matrix_to_quaternion, rpy_to_matrix


class ChannelBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.viewer = MapViewerWidget()
        self.view = self.viewer.persp_view
        self.view.resize(1000, 800)
        self.fields = self.view.channel_box.fields

    def tearDown(self):
        self.viewer.close()
        self.viewer.deleteLater()
        self.app.sendPostedEvents(None, QEvent.DeferredDelete)

    def select(self, target):
        self.view._select_gizmo_row(target)

    def edit(self, **values):
        for key, value in values.items():
            self.fields[key].setValue(value)

    def test_each_outliner_item_has_independent_channels(self):
        self.viewer.set_map_points(np.array([[-1., -1., 0.], [1., 1., 0.]]))
        for n, target in enumerate(self.view.GIZMO_TARGETS, 1):
            self.select(target)
            self.edit(tx=n, ty=-n, tz=n / 10, rx=10*n, ry=-5*n, rz=15*n)
        for n, target in enumerate(self.view.GIZMO_TARGETS, 1):
            self.select(target)
            np.testing.assert_allclose([f.value() for f in self.fields.values()],
                                       [n, -n, n / 10, 10*n, -5*n, 15*n], atol=1e-4)

    def test_locator_numeric_edits_reach_transform_and_ortho_views(self):
        changed = Mock()
        camera = Mock()
        self.viewer.locator_changed.connect(changed)
        self.viewer.camera_moved.connect(camera)
        self.select("locator")
        self.edit(tx=1.25, ty=-2.5, tz=3.75, rx=12, ry=-24, rz=36)
        T = changed.call_args.args[0]
        np.testing.assert_allclose(T[:3, 3], [1.25, -2.5, 3.75])
        np.testing.assert_allclose(T[:3, :3], rpy_to_matrix(*np.radians([12, -24, 36])))
        self.assertEqual(self.viewer.top_view._locator, (1.25, -2.5))
        self.assertEqual(self.viewer.side_view._locator, (1.25, 3.75))
        camera.assert_not_called()

    def test_scan_channels_move_and_rotate_about_its_centre(self):
        points = np.array([[2., 1., 0.], [4., 1., 0.], [4., 3., 2.], [2., 3., 2.]])
        self.viewer.set_alignment_scale(0.5)
        self.viewer.set_map_mesh(points, np.array([[0, 1, 2], [0, 2, 3]]))
        self.edit(tx=3.25, ty=-1.5, tz=2, rx=25, ry=-10, rz=70)
        R = rpy_to_matrix(*np.radians([25, -10, 70]))
        expected = (points - points.mean(axis=0)) @ R.T * 0.5 + [3.25, -1.5, 2]
        np.testing.assert_allclose(self.viewer._get_aligned(), expected, atol=2e-4)
        self.assertEqual(self.viewer.alignment_scale, 0.5)
        np.testing.assert_allclose(self.view._locator_pos, 0)

    def test_empty_and_locked_scan_explain_why_channels_are_disabled(self):
        self.select("mesh")
        self.assertTrue(all(not f.isEnabled() for f in self.fields.values()))
        self.assertIn("Import", self.view.channel_box.status.text())
        self.viewer.set_map_points(np.array([[0., 0., 0.], [2., 2., 2.]]))
        self.viewer._store_pose()
        before = self.viewer.T_align.copy()
        self.assertTrue(all(not f.isEnabled() for f in self.fields.values()))
        self.edit(tx=99)  # even a programmatic signal cannot bypass the lock
        np.testing.assert_array_equal(self.viewer.T_align, before)
        self.assertIn("locked", self.view.channel_box.status.text())
        self.viewer._reset_pose()
        self.assertTrue(all(f.isEnabled() for f in self.fields.values()))

    def test_camera_edits_emit_full_orientation_without_rounding_other_channels(self):
        self.select("camera")
        pos = [1.12345678, 2.34567891, 3.56789123]
        self.view.set_cam(pos, rpy_deg=[11.123456, -22.345678, 33.567891])
        changed = Mock()
        self.viewer.camera_moved.connect(changed)
        self.edit(rx=45)
        T = changed.call_args.args[0]
        np.testing.assert_array_equal(T[:3, 3], pos)
        np.testing.assert_array_equal(self.viewer.cam_pose, T)
        np.testing.assert_allclose(T[:3, :3],
                                  rpy_to_matrix(*np.radians([45, -22.345678, 33.567891])))

    def test_live_camera_lock_blocks_edits_and_drag_without_freezing_pose(self):
        self.select("camera")
        self.view._active_gizmo = "x"
        self.view.set_camera_edit_locked(True)
        self.assertIsNone(self.view._active_gizmo)
        self.assertFalse(self.view.gizmo_active)
        changed = Mock()
        self.viewer.camera_moved.connect(changed)
        self.edit(tx=123)
        changed.assert_not_called()
        self.view.set_cam([1, 2, 3], rpy_deg=[12, -24, 36])
        np.testing.assert_allclose([f.value() for f in self.fields.values()],
                                   [1, 2, 3, 12, -24, 36])
        self.assertTrue(all(not f.isEnabled() for f in self.fields.values()))
        self.select("locator")
        self.edit(tx=4.5, rz=90)
        self.assertAlmostEqual(self.viewer.get_locator_transform()[0, 3], 4.5)
        self.assertTrue(self.view.gizmo_active)

    def test_external_poses_refresh_all_six_channels_without_emitting_edits(self):
        self.select("camera")
        changed = Mock()
        self.viewer.camera_moved.connect(changed)
        q = matrix_to_quaternion(rpy_to_matrix(*np.radians([15, -30, 65])))
        self.viewer.update_camera_pose(np.array([1., 2., 3.]), q)
        np.testing.assert_allclose([f.value() for f in self.fields.values()],
                                   [1, 2, 3, 15, -30, 65], atol=1e-3)
        changed.assert_not_called()

    def test_live_updates_preserve_unfinished_typing_then_commit_only_that_axis(self):
        self.select("camera")
        self.viewer.resize(1000, 1000)
        self.viewer.show()
        self.app.processEvents()
        field = self.fields["tx"]
        self.view.channel_scroll.ensureWidgetVisible(field)
        field.setFocus()
        self.app.processEvents()
        field.lineEdit().selectAll()
        QTest.keyClicks(field, "12.75")
        self.assertTrue(field.lineEdit().isModified())
        self.view.set_cam([9., 8., 7.], rpy_deg=[6., 5., 4.])
        self.assertIn("12.75", field.text())
        QTest.keyClick(field, Qt.Key_Return)
        np.testing.assert_allclose(self.view._cam_pos, [12.75, 8., 7.])
        np.testing.assert_allclose(self.view._cam_rpy, [6., 5., 4.])

    def test_switching_selection_does_not_apply_unfinished_text_to_new_item(self):
        self.select("camera")
        self.fields["tx"].lineEdit().setText("123")
        self.fields["tx"].lineEdit().setModified(True)
        self.select("locator")
        self.fields["tx"].interpretText()
        np.testing.assert_array_equal(self.view._locator_pos, np.zeros(3))
        np.testing.assert_array_equal(self.view._cam_pos, np.zeros(3))

    def test_gizmo_drag_refreshes_numeric_channels(self):
        from PyQt5.QtCore import QPoint
        self.select("locator")
        self.view._begin_locator_drag("x", QPoint(500, 400))
        self.view._drag_locator(QPoint(560, 400))
        self.assertNotEqual(self.fields["tx"].value(), 0)
        self.assertAlmostEqual(self.fields["tx"].value(), self.view._locator_pos[0], places=4)

    def test_trail_display_transform_preserves_recorded_and_future_samples(self):
        self.viewer._trail.extend([(1., 0., 0.), (2., 0., 0.)])
        self.viewer._push_trail()
        self.select("trail")
        self.edit(tx=3, ty=4, tz=5, rz=90)
        np.testing.assert_allclose(self.view._trail, [(1, 0, 0), (2, 0, 0)])
        np.testing.assert_allclose(self.viewer.top_view._trail, [(3, 5), (3, 6)])
        np.testing.assert_allclose(self.viewer.side_view._trail, [(3, 5), (3, 5)])
        self.viewer._trail.append((3., 0., 0.))
        self.viewer._push_trail()
        self.view._center_on_selection()
        np.testing.assert_allclose(self.view._orb_center, [3, 7, 5])
        np.testing.assert_allclose(self.viewer.top_view._trail[-1], [3, 7])

    def test_world_channels_transform_rendered_grid_without_moving_scan_or_camera(self):
        self.viewer.set_map_points(np.array([[1., 2., 3.]]))
        before = self.viewer._get_aligned().copy()
        self.select("world")
        self.edit(tx=2, ty=3, tz=4, rx=90)
        # The renderer must project the transformed plane, not just store numbers.
        with patch.object(self.view, "_project_segments",
                          wraps=self.view._project_segments) as projected:
            self.view._draw_grid(Mock(), self.view._eye_basis())
        self.assertEqual(projected.call_count, 3)
        for call in projected.call_args_list:
            for ends in call.args[:2]:
                np.testing.assert_allclose(ends[:, 1], 3, atol=1e-10)
        self.view._center_on_selection()
        np.testing.assert_allclose(self.view._orb_center, [2, 3, 4])
        np.testing.assert_array_equal(self.viewer._get_aligned(), before)
        np.testing.assert_array_equal(self.view._cam_pos, np.zeros(3))

    def test_cleared_selection_disables_channels(self):
        self.view.outliner_list.setCurrentRow(-1)
        self.assertTrue(all(not f.isEnabled() for f in self.fields.values()))
        self.assertFalse(self.view.gizmo_active)

    def test_small_viewport_can_scroll_to_last_rotation_field(self):
        self.viewer.resize(650, 600)
        self.viewer.show()
        self.viewer.v_split.setSizes([200, 350])
        self.app.processEvents()
        scroll = self.view.channel_scroll
        field = self.fields["rz"]
        scroll.ensureWidgetVisible(field)
        self.app.processEvents()
        self.assertGreater(scroll.verticalScrollBar().maximum(), 0)
        rect = scroll.viewport().rect()
        self.assertTrue(rect.contains(field.mapTo(scroll.viewport(), field.rect().center())))


if __name__ == "__main__":
    unittest.main()
