#!/usr/bin/env python3
"""
Tests for the camera-image lineup (image_alignment.py).

Needs open3d and cv2, so it skips on the host and runs in the container:

    docker exec -u admin -w /workspaces/isaac_ros-dev/src \
        isaac_ros_dev-x86_64-container python3 -m unittest test_image_alignment

What is asserted here is deliberately narrower than what the module can do.
The lineup is trustworthy as a *measurement* -- its score peaks at the true
pose, which is what makes it useful as an on-set check and an AR overlay -- but
its optimiser is not yet trustworthy enough to move the pose (see
`test_score_peaks_at_the_true_pose` and the note in tracking.yaml). So the
tests pin the measurement, not a convergence claim the module has not earned.
"""
import math
import unittest

import numpy as np

from image_alignment import EdgeAligner, _HAS_CV2, _HAS_O3D
from pose_transform import (T_BODY_OPTICAL, invert_transform,
                            make_transform, so3_exp)

_AVAILABLE = _HAS_CV2 and _HAS_O3D


def _room_mesh():
    """A 6 x 5 x 2.8 m box: six planes, so silhouettes and creases are exact."""
    lo = np.array([-3.0, -2.5, 0.0])
    hi = np.array([3.0, 2.5, 2.8])
    v = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                  [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                  [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]], dtype=float)
    f = np.array([[0, 2, 1], [0, 3, 2],        # floor
                  [4, 5, 6], [4, 6, 7],        # ceiling
                  [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6],
                  [3, 0, 4], [3, 4, 7]], dtype=np.int32)
    # A pillar, so the view has silhouettes rather than only wall creases.
    p_lo, p_hi = np.array([0.6, -0.4, 0.0]), np.array([1.0, 0.0, 2.8])
    pv = np.array([[p_lo[0], p_lo[1], p_lo[2]], [p_hi[0], p_lo[1], p_lo[2]],
                   [p_hi[0], p_hi[1], p_lo[2]], [p_lo[0], p_hi[1], p_lo[2]],
                   [p_lo[0], p_lo[1], p_hi[2]], [p_hi[0], p_lo[1], p_hi[2]],
                   [p_hi[0], p_hi[1], p_hi[2]], [p_lo[0], p_hi[1], p_hi[2]]])
    pf = f.copy() + len(v)
    return np.vstack([v, pv]), np.vstack([f, pf]).astype(np.int32)


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    if abs(float(np.dot(fwd, up))) > 0.99:
        up = (0.0, 1.0, 0.0)
    left = np.cross(np.asarray(up, float), fwd); left /= np.linalg.norm(left)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([fwd, left, np.cross(fwd, left)])
    T[:3, 3] = eye
    return T


class TestLineupSession(unittest.TestCase):
    """The frozen-frame manual lineup. Pure arithmetic -- no cv2, no open3d."""

    def _session(self, T_slam_cam=None, T_map_cam=None):
        from image_alignment import LineupSession
        T_slam_cam = (np.eye(4) if T_slam_cam is None
                      else np.asarray(T_slam_cam, float))
        T_map_cam = (np.eye(4) if T_map_cam is None
                     else np.asarray(T_map_cam, float))
        return LineupSession(gray=None, K=None, T_slam_cam=T_slam_cam,
                             T_map_cam=T_map_cam)

    # ── the commit, which is the whole point ─────────────────────────────────
    def test_commit_places_the_camera_exactly_where_it_was_dialled(self):
        """T_slam_to_map must reproduce the dialled pose from the SLAM pose."""
        T_slam_cam = make_transform((2.4, -1.1, 0.3),
                                    (0.0, 0.0, 0.3826834, 0.9238795))  # 45 deg
        T_map_cam = make_transform((0.5, 1.2, 1.4),
                                   (0.0, 0.0, -0.1305262, 0.9914449))  # -15 deg
        s = self._session(T_slam_cam, T_map_cam)
        T_slam_to_map = s.slam_to_map()
        np.testing.assert_allclose(T_slam_to_map @ T_slam_cam, T_map_cam,
                                   atol=1e-12)

    def test_commit_holds_for_later_poses_not_just_the_frozen_one(self):
        """A pose dialled on one frame must stay valid as the camera moves.

        This is why freezing a frame is legitimate: the fusion corrects the
        frame, so the same T_slam_to_map carries every subsequent VIO pose.
        """
        T_slam_from_map = make_transform((2.4, -1.1, 0.3),
                                         (0.0, 0.0, 0.2588190, 0.9659258))
        truth_at_freeze = make_transform((0.5, 1.2, 1.4), (0, 0, 0, 1))
        s = self._session(T_slam_from_map @ truth_at_freeze, truth_at_freeze)
        T_slam_to_map = s.slam_to_map()

        # The camera then moves somewhere else entirely.
        truth_later = make_transform((-0.8, 0.4, 1.35),
                                     (0.0, 0.0, 0.7071068, 0.7071068))
        slam_later = T_slam_from_map @ truth_later
        np.testing.assert_allclose(T_slam_to_map @ slam_later, truth_later,
                                   atol=1e-12)

    # ── nudging ──────────────────────────────────────────────────────────────
    def test_nudge_moves_along_the_cameras_own_axes(self):
        # Camera yawed 90 deg: its "forward" is world +Y.
        T = make_transform((0.0, 0.0, 0.0),
                           (0.0, 0.0, 0.7071068, 0.7071068))
        s = self._session(T_map_cam=T)
        s.nudge(forward=1)
        np.testing.assert_allclose(s.T_map_cam[:3, 3],
                                   [0.0, s.COARSE[0], 0.0], atol=1e-9)

    def test_nudge_directions_are_independent(self):
        s = self._session()
        s.nudge(forward=1, left=2, up=-1)
        m = s.COARSE[0]
        np.testing.assert_allclose(s.T_map_cam[:3, 3], [m, 2 * m, -m], atol=1e-12)

    def test_fine_steps_are_smaller(self):
        coarse = self._session(); coarse.nudge(forward=1)
        fine = self._session(); fine.fine = True; fine.nudge(forward=1)
        self.assertLess(abs(fine.T_map_cam[0, 3]), abs(coarse.T_map_cam[0, 3]))
        self.assertAlmostEqual(fine.T_map_cam[0, 3], fine.FINE[0], places=12)

    def test_pan_turns_about_the_cameras_up_axis(self):
        s = self._session()
        s.nudge(pan=1)
        yaw = math.degrees(math.atan2(s.T_map_cam[1, 0], s.T_map_cam[0, 0]))
        self.assertAlmostEqual(yaw, s.COARSE[1], places=9)
        np.testing.assert_allclose(s.T_map_cam[:3, 3], np.zeros(3), atol=1e-12,
                                   err_msg="a pure rotation moved the camera")

    def test_rotation_stays_a_rotation_after_many_nudges(self):
        s = self._session()
        for _ in range(50):
            s.nudge(pan=1, tilt=-1, roll=1)
        R = s.T_map_cam[:3, :3]
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)

    # ── undo / reset / reporting ─────────────────────────────────────────────
    def test_undo_steps_back(self):
        s = self._session()
        s.nudge(forward=1); s.nudge(left=1)
        s.undo()
        np.testing.assert_allclose(s.T_map_cam[:3, 3], [s.COARSE[0], 0, 0],
                                   atol=1e-12)

    def test_undo_on_an_untouched_session_is_harmless(self):
        s = self._session()
        s.undo()
        np.testing.assert_allclose(s.T_map_cam, np.eye(4), atol=1e-12)

    def test_reset_returns_to_the_seed_and_is_undoable(self):
        s = self._session()
        s.nudge(forward=3, pan=2)
        s.reset()
        np.testing.assert_allclose(s.T_map_cam, np.eye(4), atol=1e-12)
        s.undo()
        self.assertGreater(abs(s.T_map_cam[0, 3]), 0.0)

    def test_reports_how_far_the_operator_dialled(self):
        s = self._session()
        s.nudge(forward=5)
        s.nudge(pan=3)
        moved_m, moved_deg = s.moved_from_start
        self.assertAlmostEqual(moved_m, 5 * s.COARSE[0], places=9)
        self.assertAlmostEqual(moved_deg, 3 * s.COARSE[1], places=6)

    def test_overlay_and_score_are_safe_without_an_aligner(self):
        s = self._session()
        self.assertIsNone(s.overlay())
        self.assertIsNone(s.score())
        self.assertIsNone(s.auto_refine())


@unittest.skipUnless(_AVAILABLE, "needs open3d + cv2 (run in the container)")
class TestEdgeAligner(unittest.TestCase):

    K = np.array([[337.0, 0.0, 320.0], [0.0, 337.0, 180.0], [0.0, 0.0, 1.0]])
    W, H = 640, 360

    @classmethod
    def setUpClass(cls):
        import open3d as o3d
        cls.o3d = o3d
        cls.vertices, cls.triangles = _room_mesh()
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(cls.vertices),
            o3d.utility.Vector3iVector(cls.triangles))
        cls.scene = o3d.t.geometry.RaycastingScene()
        cls.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        cls.truth = _look_at((-2.0, -1.5, 1.4), (2.5, 2.0, 1.0))

    def _aligner(self, **cfg):
        a = EdgeAligner(cfg)
        self.assertTrue(a.set_mesh(self.vertices, self.triangles))
        return a

    def _picture(self, T_body):
        """A grayscale frame: headlight shading of the room, plus noise."""
        o3d = self.o3d
        opt = T_body @ T_BODY_OPTICAL
        rays = self.scene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(self.K),
            extrinsic_matrix=o3d.core.Tensor(invert_transform(opt)),
            width_px=self.W, height_px=self.H)
        ans = self.scene.cast_rays(rays)
        z = ans["t_hit"].numpy()
        n = ans["primitive_normals"].numpy()
        d = rays.numpy()[:, :, 3:6]
        d = d / np.linalg.norm(d, axis=2, keepdims=True)
        img = np.clip(np.abs(np.sum(n * (-d), axis=2)) * 200 + 25, 0, 255)
        img[~np.isfinite(z)] = 0
        rng = np.random.default_rng(0)
        return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)

    # ── the measurement, which is what this is trusted for ───────────────────
    def test_score_peaks_at_the_true_pose(self):
        """The lineup score must be a real measurement of agreement.

        This is what earns it a place on the dashboard: at the true pose more
        of the scan's edges sit on the picture than at any nearby pose, in
        both translation and rotation.
        """
        a = self._aligner()
        gray = self._picture(self.truth)
        at_truth = a.score(gray, self.K, self.truth)
        self.assertIsNotNone(at_truth)

        for axis in range(3):
            for offset in (-0.12, 0.12):
                T = self.truth.copy()
                T[axis, 3] += offset
                off = a.score(gray, self.K, T)
                self.assertLess(off["score"], at_truth["score"],
                                f"axis {axis} offset {offset} scored higher "
                                f"than the truth")

        for deg in (-3.0, 3.0):
            T = self.truth.copy()
            T[:3, :3] = so3_exp(np.array([0.0, 0.0, 1.0]) * math.radians(deg)) @ T[:3, :3]
            self.assertLess(a.score(gray, self.K, T)["score"],
                            at_truth["score"], f"{deg} deg scored higher")

    def test_a_good_pose_scores_well(self):
        a = self._aligner()
        s = a.score(self._picture(self.truth), self.K, self.truth)
        self.assertGreater(s["score"], 0.5)
        self.assertLess(s["median_px"], 4.0)
        self.assertGreater(s["n_model_edges"], 100)

    def test_score_needs_an_alignment_to_mean_anything(self):
        a = self._aligner()
        gray = self._picture(self.truth)
        far = self.truth.copy()
        far[:3, 3] += np.array([1.5, 1.2, 0.0])
        self.assertLess(a.score(gray, self.K, far)["score"],
                        a.score(gray, self.K, self.truth)["score"])

    # ── guards ───────────────────────────────────────────────────────────────
    def test_unusable_image_is_refused_not_guessed(self):
        a = self._aligner()
        blank = np.full((self.H, self.W), 128, dtype=np.uint8)
        res = a.align(blank, self.K, self.truth)
        self.assertFalse(res.ok)
        self.assertIn("edges in the image", res.reason)

    def test_no_mesh_means_unavailable(self):
        a = EdgeAligner({})
        self.assertFalse(a.available)
        self.assertFalse(a.align(self._picture(self.truth), self.K,
                                 self.truth).ok)

    def test_disabled_by_config(self):
        a = EdgeAligner({"image_align_enabled": False})
        a.set_mesh(self.vertices, self.triangles)
        self.assertFalse(a.available)

    def test_refinement_never_makes_a_large_move(self):
        """Whatever it does, it may not leap: the seed comes from depth ICP."""
        a = self._aligner(image_align_max_move_m=0.05,
                          image_align_max_move_deg=2.0)
        gray = self._picture(self.truth)
        start = self.truth.copy()
        start[:3, 3] += np.array([0.20, -0.15, 0.05])
        res = a.align(gray, self.K, start)
        if res.ok:
            self.assertLessEqual(res.moved_m, 0.05 + 1e-9)
            self.assertLessEqual(res.moved_deg, 2.0 + 1e-9)
        else:
            self.assertTrue("move" in res.reason or "line up" in res.reason)

    def test_align_leaves_the_input_pose_untouched(self):
        a = self._aligner()
        start = self.truth.copy()
        before = start.copy()
        a.align(self._picture(self.truth), self.K, start)
        np.testing.assert_array_equal(start, before)

    # ── the overlay, which is how an operator actually checks it ─────────────
    def test_overlay_draws_edges_on_the_frame(self):
        a = self._aligner()
        gray = self._picture(self.truth)
        canvas = a.overlay(gray, self.K, self.truth)
        self.assertIsNotNone(canvas)
        self.assertEqual(canvas.shape[2], 3)
        self.assertEqual(canvas.shape[1], a.work_width)
        painted = np.count_nonzero(np.all(canvas == (0, 255, 255), axis=2))
        self.assertGreater(painted, 50, "no scan edges drawn")

    def test_scale_is_applied_to_the_mesh(self):
        """A centimetre scan must be installed in metres, like the ICP path."""
        a = EdgeAligner({})
        self.assertTrue(a.set_mesh(self.vertices * 100.0, self.triangles,
                                   scale=0.01))
        s = a.score(self._picture(self.truth), self.K, self.truth)
        self.assertIsNotNone(s)
        self.assertGreater(s["score"], 0.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
