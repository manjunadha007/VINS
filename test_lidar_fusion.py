#!/usr/bin/env python3
"""
Tests for the map-relative pose fusion (lidar_map_matcher.py).

Split in two deliberately:

  * everything up to `TestSyntheticRoomICP` is pure numpy and runs anywhere,
    including the host conda env where open3d is not installed. That is where
    the maths that was actually wrong lives, so it is the part that must stay
    cheap to run.
  * `TestSyntheticRoomICP` needs open3d and is skipped without it. Run it in
    the Isaac container:

        docker exec -u admin isaac_ros_dev-x86_64-container \
            python3 /workspaces/isaac_ros-dev/src/test_lidar_fusion.py
"""
import math
import unittest

import numpy as np

from lidar_map_matcher import (D455_BASELINE_M, LidarMapMatcher, MapMeasurement,
                               depth_sigma_m, _HAS_O3D)
from pose_transform import (T_BODY_OPTICAL, decompose_similarity,
                            invert_similarity, invert_transform,
                            make_transform, so3_exp, so3_log)


def matcher(**overrides):
    """A matcher with no map loaded -- enough for the pure-maths methods."""
    cfg = {"map_file": "", "voxel_size": 0.05}
    cfg.update(overrides)
    return LidarMapMatcher(cfg)


def rigid(rpy_deg=(0, 0, 0), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.radians(rpy_deg))
    T[:3, 3] = t
    return T


# ── the bug that broke the fusion ────────────────────────────────────────────

class TestCameraDelta(unittest.TestCase):
    """`_camera_delta` is what the old code got wrong by reading T[:3, 3]."""

    def test_pure_translation_matches_translation_column(self):
        # The one case where the old approach happened to be right, which is
        # why the bug survived: with no rotation, (R - I)p vanishes.
        T_a = np.eye(4)
        T_b = rigid(t=(0.03, -0.01, 0.02))
        p = make_transform((5.0, 2.0, 1.5), (0, 0, 0, 1))
        moved, deg = LidarMapMatcher._camera_delta(T_a, T_b, p)
        self.assertAlmostEqual(moved, np.linalg.norm([0.03, -0.01, 0.02]), places=9)
        self.assertAlmostEqual(deg, 0.0, places=9)

    def test_rotation_about_origin_moves_a_distant_camera(self):
        # 1 degree of yaw, camera 5 m out. The translation column is exactly
        # zero here, so the old code reported "no correction" while the camera
        # had in fact moved 87 mm.
        T_a = np.eye(4)
        T_b = rigid(rpy_deg=(0, 0, 1.0))
        p = make_transform((5.0, 0.0, 0.0), (0, 0, 0, 1))

        self.assertAlmostEqual(np.linalg.norm(T_b[:3, 3]), 0.0, places=12,
                               msg="translation column is zero for this case")

        moved, deg = LidarMapMatcher._camera_delta(T_a, T_b, p)
        self.assertAlmostEqual(moved, 2 * 5.0 * math.sin(math.radians(0.5)),
                               places=9)
        self.assertGreater(moved, 0.08)
        self.assertAlmostEqual(deg, 1.0, places=6)

    def test_delta_scales_with_distance_from_origin(self):
        T_a, T_b = np.eye(4), rigid(rpy_deg=(0, 0, 1.0))
        near = LidarMapMatcher._camera_delta(
            T_a, T_b, make_transform((1.0, 0, 0), (0, 0, 0, 1)))[0]
        far = LidarMapMatcher._camera_delta(
            T_a, T_b, make_transform((10.0, 0, 0), (0, 0, 0, 1)))[0]
        self.assertAlmostEqual(far / near, 10.0, places=6)

    def test_rotation_reported_independently_of_position(self):
        T_a, T_b = np.eye(4), rigid(rpy_deg=(0.4, -0.3, 2.0))
        expected = math.degrees(np.linalg.norm(so3_log(T_b[:3, :3])))
        for pos in ((0, 0, 0), (7.0, -3.0, 2.0)):
            _, deg = LidarMapMatcher._camera_delta(
                T_a, T_b, make_transform(pos, (0, 0, 0, 1)))
            self.assertAlmostEqual(deg, expected, places=9)

    def test_handles_missing_pose(self):
        moved, deg = LidarMapMatcher._camera_delta(np.eye(4), rigid(t=(1, 0, 0)),
                                                   None)
        self.assertAlmostEqual(moved, 1.0, places=9)


# ── fusion: convergence, no oscillation, rate limiting ───────────────────────

class TestFusion(unittest.TestCase):

    def _run(self, m, T_true, T_cam, ticks, fitness=1.0, dt=0.2):
        now = 0.0
        history = []
        for _ in range(ticks):
            now += dt
            meas = MapMeasurement(ok=True, T_slam_to_map=T_true.copy(),
                                  fitness=fitness)
            m._fuse(meas, T_cam, now)
            err = LidarMapMatcher._camera_delta(m.T_slam_to_map, T_true, T_cam)[0]
            history.append(err)
        return history

    def test_converges_toward_the_measurement(self):
        m = matcher(blend_alpha=0.3, max_slew_m_s=10.0, max_slew_deg_s=360.0)
        T_true = rigid(rpy_deg=(0, 0, 0.5), t=(0.10, -0.04, 0.02))
        T_cam = make_transform((2.0, 1.0, 1.4), (0, 0, 0, 1))
        history = self._run(m, T_true, T_cam, 60)
        self.assertLess(history[-1], 1e-4)

    def test_error_decreases_monotonically_no_oscillation(self):
        # The old caller computed `remaining = correction - already_applied`
        # against a measurement that already saw the applied part, so the
        # residual overshot and changed sign. Estimating the transform means
        # each measurement is absolute and the blend can only shrink the error.
        m = matcher(blend_alpha=0.4, max_slew_m_s=10.0, max_slew_deg_s=360.0)
        T_true = rigid(rpy_deg=(0.2, 0.1, 0.8), t=(0.12, 0.05, -0.03))
        T_cam = make_transform((3.0, -2.0, 1.2), (0, 0, 0, 1))
        history = self._run(m, T_true, T_cam, 40)
        for previous, current in zip(history, history[1:]):
            self.assertLessEqual(current, previous + 1e-12)

    def test_result_stays_a_rigid_transform(self):
        m = matcher(blend_alpha=0.25, max_slew_m_s=10.0, max_slew_deg_s=360.0)
        T_true = rigid(rpy_deg=(3.0, -2.0, 5.0), t=(0.3, 0.2, -0.1))
        self._run(m, T_true, make_transform((1, 1, 1), (0, 0, 0, 1)), 25)
        R = m.T_slam_to_map[:3, :3]
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)

    def test_slew_limit_caps_camera_movement_per_second(self):
        m = matcher(blend_alpha=1.0, max_slew_m_s=0.02, max_slew_deg_s=1.0)
        T_true = rigid(t=(1.0, 0.0, 0.0))          # a metre of error, at once
        T_cam = make_transform((0.0, 0.0, 0.0), (0, 0, 0, 1))
        dt = 0.2
        before = m.T_slam_to_map.copy()
        m._fuse(MapMeasurement(ok=True, T_slam_to_map=T_true, fitness=1.0),
                T_cam, dt)
        moved, _ = LidarMapMatcher._camera_delta(before, m.T_slam_to_map, T_cam)
        self.assertLessEqual(moved, 0.02 * dt + 1e-9)

    def test_slew_limit_caps_rotation_per_second(self):
        m = matcher(blend_alpha=1.0, max_slew_m_s=10.0, max_slew_deg_s=1.0)
        T_true = rigid(rpy_deg=(0, 0, 30.0))
        T_cam = make_transform((0.0, 0.0, 0.0), (0, 0, 0, 1))
        dt = 0.2
        before = m.T_slam_to_map.copy()
        m._fuse(MapMeasurement(ok=True, T_slam_to_map=T_true, fitness=1.0),
                T_cam, dt)
        _, deg = LidarMapMatcher._camera_delta(before, m.T_slam_to_map, T_cam)
        self.assertLessEqual(deg, 1.0 * dt + 1e-9)

    def test_low_fitness_moves_less_than_high_fitness(self):
        T_true = rigid(t=(0.05, 0, 0))
        T_cam = make_transform((1.0, 0, 0), (0, 0, 0, 1))
        moved = {}
        for fitness in (0.35, 0.95):
            m = matcher(blend_alpha=0.5, max_slew_m_s=10.0, max_slew_deg_s=360.0)
            before = m.T_slam_to_map.copy()
            m._fuse(MapMeasurement(ok=True, T_slam_to_map=T_true,
                                   fitness=fitness), T_cam, 0.2)
            moved[fitness] = LidarMapMatcher._camera_delta(
                before, m.T_slam_to_map, T_cam)[0]
        self.assertLess(moved[0.35], moved[0.95])

    def test_fusion_corrects_rotation_not_just_position(self):
        # The old implementation returned a 3-vector and could not do this
        # at all.
        m = matcher(blend_alpha=0.5, max_slew_m_s=10.0, max_slew_deg_s=360.0)
        T_true = rigid(rpy_deg=(0.0, 0.0, 2.0))
        self._run(m, T_true, make_transform((1, 0, 0), (0, 0, 0, 1)), 40)
        yaw = math.degrees(so3_log(m.T_slam_to_map[:3, :3])[2])
        self.assertAlmostEqual(yaw, 2.0, places=4)


# ── degenerate geometry ──────────────────────────────────────────────────────

class TestObservabilityConstraint(unittest.TestCase):

    def test_single_wall_keeps_only_the_normal_direction(self):
        m = matcher(observability_eps=0.04)
        # All matched normals along +X: only X translation is observable.
        m._obs_basis = np.eye(3)
        eigvals = np.array([0.0, 0.0, 1.0])         # eigh order: ascending
        T_fix = rigid(t=(0.05, 0.05, 0.05))
        out = m._constrain(T_fix, eigvals)
        np.testing.assert_allclose(out[:3, 3], [0.0, 0.0, 0.05], atol=1e-12)

    def test_corner_geometry_passes_through_unchanged(self):
        m = matcher(observability_eps=0.04)
        m._obs_basis = np.eye(3)
        T_fix = rigid(t=(0.05, -0.02, 0.01))
        out = m._constrain(T_fix, np.array([0.33, 0.33, 0.34]))
        np.testing.assert_allclose(out[:3, 3], T_fix[:3, 3], atol=1e-12)

    def test_constraint_respects_a_rotated_eigenbasis(self):
        m = matcher(observability_eps=0.04)
        # Observable only along a diagonal direction.
        v = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
        w = np.array([1.0, -1.0, 0.0]) / math.sqrt(2)
        u = np.array([0.0, 0.0, 1.0])
        m._obs_basis = np.column_stack([w, u, v])
        out = m._constrain(rigid(t=(0.1, 0.1, 0.1)), np.array([0.0, 0.0, 1.0]))
        # Result must be parallel to v, with v's component preserved.
        np.testing.assert_allclose(out[:3, 3], v * (v @ np.array([0.1, 0.1, 0.1])),
                                   atol=1e-12)

    def test_rotation_is_never_suppressed(self):
        m = matcher(observability_eps=0.04)
        m._obs_basis = np.eye(3)
        T_fix = rigid(rpy_deg=(0, 0, 1.0), t=(0.05, 0.05, 0.05))
        out = m._constrain(T_fix, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(out[:3, :3], T_fix[:3, :3], atol=1e-12)


# ── D455 depth conditioning ──────────────────────────────────────────────────

class TestDepthModel(unittest.TestCase):

    def test_matches_the_published_accuracy_spec(self):
        # Intel quotes "<2 % of range at 4 m" for the D455. fx at 640x360 with
        # the 87 deg depth FOV is ~337 px.
        fx = 320.0 / math.tan(math.radians(87.0 / 2))
        sigma = depth_sigma_m(4.0, fx, D455_BASELINE_M)
        self.assertLess(sigma / 4.0, 0.02)

    def test_error_grows_with_the_square_of_range(self):
        fx = 337.0
        self.assertAlmostEqual(
            depth_sigma_m(6.0, fx) / depth_sigma_m(3.0, fx), 4.0, places=9)


class TestDeprojection(unittest.TestCase):

    K = np.array([[337.0, 0.0, 320.0],
                  [0.0, 337.0, 180.0],
                  [0.0, 0.0, 1.0]])

    def _depth(self, value_mm, shape=(360, 640)):
        return np.full(shape, value_mm, dtype=np.uint16)

    def test_rejects_out_of_range_samples(self):
        m = matcher(min_range_m=0.3, max_range_m=6.0)
        self.assertEqual(len(m._deproject(self._depth(100), self.K)), 0,
                         "10 cm is inside the D455 blind zone")
        self.assertEqual(len(m._deproject(self._depth(9000), self.K)), 0,
                         "9 m is past the usable stereo range")
        self.assertGreater(len(m._deproject(self._depth(2000), self.K)), 0)

    def test_rejects_zero_holes(self):
        m = matcher()
        self.assertEqual(len(m._deproject(self._depth(0), self.K)), 0)

    def test_far_points_are_thinned_relative_to_near_points(self):
        m = matcher(weight_ref_range_m=2.0, min_keep_fraction=0.0)
        near = len(m._deproject(self._depth(1000), self.K))
        far = len(m._deproject(self._depth(5000), self.K))
        self.assertGreater(near, far * 3,
                           "5 m points carry ~1/39th the information of 1 m ones")

    def test_deprojection_geometry_is_correct(self):
        m = matcher(min_keep_fraction=1.0)
        depth = np.zeros((360, 640), dtype=np.uint16)
        depth[180, 320] = 2000          # principal point, 2 m
        pts = m._deproject(depth, self.K)
        self.assertEqual(len(pts), 1)
        # Optical frame: x right, y down, z forward. On the principal ray, only
        # z is non-zero.
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 2.0], atol=1e-9)

    def test_offset_pixel_deprojects_to_the_right_ray(self):
        m = matcher(min_keep_fraction=1.0)
        depth = np.zeros((360, 640), dtype=np.uint16)
        depth[180, 420] = 1000          # 100 px right of the principal point
        pts = m._deproject(depth, self.K)
        self.assertEqual(len(pts), 1)
        np.testing.assert_allclose(pts[0], [100.0 / 337.0, 0.0, 1.0], atol=1e-6)

    def test_rejects_a_degenerate_intrinsic_matrix(self):
        m = matcher()
        self.assertEqual(len(m._deproject(self._depth(2000), np.zeros((3, 3)))), 0)

    def test_is_deterministic(self):
        a = matcher()._deproject(self._depth(3000), self.K)
        b = matcher()._deproject(self._depth(3000), self.K)
        np.testing.assert_array_equal(a, b)


class TestPoseCoercion(unittest.TestCase):

    def test_accepts_a_4x4(self):
        T = rigid(rpy_deg=(0, 0, 30), t=(1, 2, 3))
        np.testing.assert_allclose(LidarMapMatcher._as_transform(T), T)

    def test_accepts_a_position_quaternion_pair(self):
        T = LidarMapMatcher._as_transform(((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)))
        np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)

    def test_accepts_ros_message_objects(self):
        class V:
            x, y, z = 1.0, 2.0, 3.0

        class Q:
            x, y, z, w = 0.0, 0.0, 0.0, 1.0

        T = LidarMapMatcher._as_transform((V(), Q()))
        np.testing.assert_allclose(T[:3, 3], [1, 2, 3])

    def test_none_and_junk_are_survivable(self):
        self.assertIsNone(LidarMapMatcher._as_transform(None))
        self.assertIsNone(LidarMapMatcher._as_transform(5))


# ── the frame identity the dashboard relies on ───────────────────────────────

class TestFrameComposition(unittest.TestCase):
    """`_zero == T_align @ T_slam_to_map` is what lets ICP drive the output."""

    def test_map_transform_composes_to_the_ground_zero_the_anchor_would_set(self):
        from pose_transform import PoseTransformer

        T_align = rigid(rpy_deg=(0, 0, 25.0), t=(1.5, -0.5, 0.0))
        T_slam_marker = rigid(rpy_deg=(0, 0, 10.0), t=(2.0, 1.0, 0.5))
        T_locator = rigid(rpy_deg=(0, 0, -5.0), t=(0.3, 0.2, 0.0))

        # What the ArUco path sets as ground zero.
        tf = PoseTransformer({})
        tf.set_ground_zero_from_transform(T_slam_marker, T_target=T_locator)

        # What the map path derives as SLAM->map from the same sighting.
        T_slam_to_map = (invert_transform(T_align) @ T_locator
                         @ invert_transform(T_slam_marker))

        np.testing.assert_allclose(T_align @ T_slam_to_map, tf._zero, atol=1e-12)

    def test_corrected_pose_applies_the_full_transform(self):
        m = matcher()
        m.T_slam_to_map = rigid(rpy_deg=(0, 0, 90.0), t=(1.0, 0.0, 0.0))
        pos, quat = m.corrected_pose((2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        # 90 deg yaw about origin sends (2,0,0) to (0,2,0), then +1 in x.
        np.testing.assert_allclose(pos, [1.0, 2.0, 0.0], atol=1e-9)


# ── map units and the mesh-alignment scale ───────────────────────────────────

class TestSimilarityAlignment(unittest.TestCase):
    """`T_align` carries the operator's scale slider, so it is not rigid."""

    def _similarity(self, scale, rpy_deg=(0, 0, 0), t=(0, 0, 0)):
        T = np.eye(4)
        T[:3, :3] = so3_exp(np.radians(rpy_deg)) * scale
        T[:3, 3] = t
        return T

    def test_decomposes_scale_and_rotation(self):
        for scale in (1.0, 0.01, 100.0, 0.5):
            T = self._similarity(scale, (10.0, -20.0, 35.0), (1.5, -2.0, 0.75))
            got_scale, rigid = decompose_similarity(T)
            self.assertAlmostEqual(got_scale, scale, places=12)
            R = rigid[:3, :3]
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(rigid[:3, 3], T[:3, 3], atol=1e-12)

    def test_similarity_inverse_round_trips(self):
        for scale in (1.0, 0.01, 100.0):
            T = self._similarity(scale, (5.0, 15.0, -40.0), (2.0, 1.0, -3.0))
            np.testing.assert_allclose(invert_similarity(T) @ T, np.eye(4),
                                       atol=1e-9)

    def test_plain_rigid_inverse_is_wrong_once_scale_is_present(self):
        # The failure mode this exists to prevent: 0.01 is exactly the scale a
        # centimetre scan needs, and invert_transform is off by ~1.0 there.
        T = self._similarity(0.01, (5.0, 15.0, -40.0), (2.0, 1.0, -3.0))
        residual = np.abs(invert_transform(T) @ T - np.eye(4)).max()
        self.assertGreater(residual, 0.5)
        self.assertLess(np.abs(invert_similarity(T) @ T - np.eye(4)).max(), 1e-9)

    def test_identity_scale_leaves_the_matrix_alone(self):
        T = self._similarity(1.0, (12.0, 0.0, 30.0), (1.0, 2.0, 3.0))
        scale, rigid = decompose_similarity(T)
        self.assertAlmostEqual(scale, 1.0, places=12)
        np.testing.assert_allclose(rigid, T, atol=1e-12)

    def test_degenerate_matrix_is_survivable(self):
        scale, rigid = decompose_similarity(np.zeros((4, 4)))
        self.assertEqual(scale, 1.0)
        np.testing.assert_allclose(rigid, np.eye(4))


class TestMapScale(unittest.TestCase):

    def test_scale_is_recorded_and_change_detected(self):
        m = matcher(map_scale=1.0)
        self.assertTrue(m.set_map_scale(0.01))
        self.assertEqual(m.map_scale, 0.01)
        self.assertFalse(m.set_map_scale(0.01), "no-op change must not rebuild")
        self.assertFalse(m.set_map_scale(0.0), "a zero scale is meaningless")
        self.assertFalse(m.set_map_scale(-1.0))

    def test_default_scale_is_metres(self):
        self.assertEqual(matcher().map_scale, 1.0)

    def test_a_centimetre_scan_is_flagged(self):
        # A room-scale scan left in centimetres reads as ~300 m across.
        m = matcher(map_scale=1.0)
        room_cm = np.random.default_rng(0).uniform(-150, 150, size=(500, 3))
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m._check_units(room_cm)
        self.assertIn("0.01", buf.getvalue())

    def test_a_correctly_scaled_scan_is_silent(self):
        m = matcher(map_scale=0.01)
        room_cm = np.random.default_rng(0).uniform(-150, 150, size=(500, 3))
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m._check_units(room_cm)
        self.assertEqual(buf.getvalue(), "")


class TestMeshSampling(unittest.TestCase):
    """The map cloud must be the same every time the same scan is loaded.

    Open3D's `sample_points_uniformly` takes no seed and is not reproducible.
    Measured on the real scan, that alone moved a lock-on between 7 mm and
    151 mm on otherwise identical runs -- the fusion was solving a slightly
    different problem each load.
    """

    @staticmethod
    def _box():
        v = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                      [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]])
        f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int64)
        return v, f

    def test_same_seed_gives_the_same_cloud(self):
        from lidar_map_matcher import sample_mesh_surface
        v, f = self._box()
        a = sample_mesh_surface(v, f, 2000, seed=0)
        b = sample_mesh_surface(v, f, 2000, seed=0)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        from lidar_map_matcher import sample_mesh_surface
        v, f = self._box()
        self.assertFalse(np.allclose(sample_mesh_surface(v, f, 2000, seed=0),
                                     sample_mesh_surface(v, f, 2000, seed=1)))

    def test_points_lie_on_the_surface(self):
        from lidar_map_matcher import sample_mesh_surface
        v, f = self._box()
        pts = sample_mesh_surface(v, f, 4000)
        # Every point on a unit cube's surface has a coordinate at 0 or 1.
        on_face = np.isclose(pts, 0.0, atol=1e-9) | np.isclose(pts, 1.0, atol=1e-9)
        self.assertTrue(np.all(on_face.any(axis=1)))
        self.assertTrue(np.all(pts >= -1e-9) and np.all(pts <= 1 + 1e-9))

    def test_sampling_is_area_weighted_not_triangle_weighted(self):
        """A big face must get proportionally more points than a small one."""
        from lidar_map_matcher import sample_mesh_surface
        v = np.array([[0., 0., 0.], [10., 0., 0.], [10., 10., 0.], [0., 10., 0.],
                      [0., 0., 5.], [1., 0., 5.], [1., 1., 5.], [0., 1., 5.]])
        f = np.array([[0, 1, 2], [0, 2, 3],          # 100 m^2 at z=0
                      [4, 5, 6], [4, 6, 7]], dtype=np.int64)   # 1 m^2 at z=5
        pts = sample_mesh_surface(v, f, 10100)
        big = int(np.count_nonzero(pts[:, 2] < 0.5))
        self.assertGreater(big / len(pts), 0.95,
                           "small far-off faces are being over-sampled")

    def test_degenerate_inputs_are_survivable(self):
        from lidar_map_matcher import sample_mesh_surface
        v, f = self._box()
        self.assertEqual(len(sample_mesh_surface(v, np.empty((0, 3), np.int64), 10)), 0)
        self.assertEqual(len(sample_mesh_surface(v, f, 0)), 0)
        flat = np.zeros((3, 3))
        self.assertEqual(
            len(sample_mesh_surface(flat, np.array([[0, 1, 2]], np.int64), 10)), 0)


# ── scan units: config vs the viewport's scale slider ────────────────────────

class _StubViewer:
    """Just enough MapViewerWidget for the scale-reconciliation logic."""

    def __init__(self, scale=1.0):
        self.scale = scale
        self.log = []

    @property
    def alignment_scale(self):
        return self.scale

    def set_alignment_scale(self, value):
        if value <= 0.0 or abs(value - self.scale) < 1e-12:
            return False
        self.scale = float(value)
        self.log.append(self.scale)
        return True


class TestScalePrecedence(unittest.TestCase):
    """The viewport slider defaulting to 1.0 must not silently beat config.

    This is the bug that made the fusion unusable on real hardware: the
    dashboard reads the slider and pushes it to the matcher, so a configured
    `map_scale: 0.01` was overwritten by the slider's default 1.0 and the
    fusion was handed a 309 m map to match a 3 m depth cloud against. The
    camera then read tens of metres out and every lock was refused.
    """

    @staticmethod
    def _dashboard():
        """dashboard.py needs ROS; skip where it is not importable.

        run_offline_tests.py installs stand-ins, and the Isaac container has
        the real thing, so this is covered in both places that matter.
        """
        try:
            import dashboard
        except Exception as exc:                      # pragma: no cover
            raise unittest.SkipTest(f"dashboard not importable: {exc}")
        return dashboard

    def _detect(self, viewer, extent_units):
        _d = self._dashboard()
        gui = _d.HyperFusionApp.__new__(_d.HyperFusionApp)
        gui.map_viewer = viewer
        pts = np.array([[0.0, 0.0, 0.0], [extent_units, extent_units * 0.9,
                                          extent_units * 0.7]])
        _d.HyperFusionApp._autodetect_map_scale(gui, pts)
        return viewer.scale

    def test_centimetre_scan_is_detected(self):
        self.assertAlmostEqual(self._detect(_StubViewer(1.0), 309.5), 0.01)

    def test_millimetre_scan_is_detected(self):
        self.assertAlmostEqual(self._detect(_StubViewer(1.0), 3095.0), 0.001)

    def test_metre_scan_is_left_alone(self):
        self.assertAlmostEqual(self._detect(_StubViewer(1.0), 3.1), 1.0)

    def test_an_operator_set_scale_is_never_overridden(self):
        viewer = _StubViewer(0.02)
        self.assertAlmostEqual(self._detect(viewer, 309.5), 0.02)
        self.assertEqual(viewer.log, [])

    def test_matcher_follows_the_slider(self):
        m = matcher(map_scale=1.0)
        scale, rigid = decompose_similarity(
            np.diag([0.01, 0.01, 0.01, 1.0]))
        self.assertTrue(m.set_map_scale(scale))
        self.assertAlmostEqual(m.map_scale, 0.01)
        np.testing.assert_allclose(rigid, np.eye(4), atol=1e-12)


# ── engine integration: what the dashboard camera dot actually shows ─────────

class TestEngineIntegration(unittest.TestCase):
    """A fused fix must move the FreeD output and the on-screen dot together."""

    def _engine(self):
        import tracking_config
        from tracking_engine import TrackingEngine
        return TrackingEngine(tracking_config.DEFAULTS, log=lambda *a: None)

    def test_map_alignment_places_the_camera_where_the_map_says(self):
        eng = self._engine()
        # Camera sits at (3, 1, 1.5) in the SLAM frame. The fusion has worked
        # out the SLAM frame is 20 cm out and yawed 90 degrees from the scan.
        eng.accept_pose((3.0, 1.0, 1.5), (0.0, 0.0, 0.0, 1.0),
                        (0, 0, 0), (0, 0, 0))
        T = rigid(rpy_deg=(0, 0, 90.0), t=(0.2, 0.0, 0.0))
        eng.apply_map_alignment(T)

        position, _ = eng.camera_pose_ros_corrected()
        # 90 deg yaw sends (3,1,1.5) to (-1,3,1.5), then +0.2 in x.
        np.testing.assert_allclose(position, [-0.8, 3.0, 1.5], atol=1e-9)

    def test_the_dot_and_the_freed_stream_agree(self):
        eng = self._engine()
        eng.accept_pose((2.0, -1.0, 1.2), (0.0, 0.0, 0.0, 1.0),
                        (0, 0, 0), (0, 0, 0))
        eng.apply_map_alignment(rigid(rpy_deg=(0, 0, 30.0), t=(0.5, 0.1, 0.0)))

        ros_position, _ = eng.camera_pose_ros_corrected()
        unreal, _ = eng.camera_pose_now()
        # ros_to_unreal only mirrors Y and scales; magnitudes must line up.
        self.assertAlmostEqual(unreal.x, ros_position[0], places=9)
        self.assertAlmostEqual(unreal.y, -ros_position[1], places=9)
        self.assertAlmostEqual(unreal.z, ros_position[2], places=9)

    def test_map_alignment_folds_in_the_mesh_alignment(self):
        eng = self._engine()
        eng.accept_pose((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                        (0, 0, 0), (0, 0, 0))
        T_slam_to_map = rigid(t=(0.1, 0.0, 0.0))
        T_align = rigid(rpy_deg=(0, 0, 90.0), t=(0.0, 0.0, 0.0))
        eng.apply_map_alignment(T_slam_to_map, T_align)

        position, _ = eng.camera_pose_ros_corrected()
        # (1,0,0) -> +0.1 in x -> (1.1,0,0) -> yaw 90 -> (0, 1.1, 0)
        np.testing.assert_allclose(position, [0.0, 1.1, 0.0], atol=1e-9)

    def test_correcting_rotation_changes_the_output_angles(self):
        eng = self._engine()
        eng.accept_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                        (0, 0, 0), (0, 0, 0))
        before, _ = eng.camera_pose_now()
        eng.apply_map_alignment(rigid(rpy_deg=(0, 0, 5.0)))
        after, _ = eng.camera_pose_now()
        # pan_sign is -1 by default, so +5 deg of ROS yaw is -5 deg of pan.
        self.assertAlmostEqual(after.pan - before.pan, -5.0, places=6)

    def test_reset_pose_drops_the_map_alignment(self):
        eng = self._engine()
        eng.accept_pose((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                        (0, 0, 0), (0, 0, 0))
        eng.apply_map_alignment(rigid(t=(0.5, 0.0, 0.0)))
        self.assertTrue(eng.map_aligned)

        eng.handle_command({"cmd": "reset_pose"})
        self.assertFalse(eng.map_aligned)
        position, _ = eng.camera_pose_ros_corrected()
        np.testing.assert_allclose(position, [1.0, 0.0, 0.0], atol=1e-12)

    def test_status_reports_map_alignment(self):
        eng = self._engine()
        self.assertFalse(eng.status_dict()["map_aligned"])
        eng.apply_map_alignment(np.eye(4))
        self.assertTrue(eng.status_dict()["map_aligned"])


# ── end to end, needs open3d ─────────────────────────────────────────────────

@unittest.skipUnless(_HAS_O3D, "open3d not installed (run inside the container)")
class TestSyntheticRoomICP(unittest.TestCase):
    """A full recover-the-injected-drift test against a synthetic room."""

    @classmethod
    def setUpClass(cls):
        import open3d as o3d
        # A 6 x 5 x 2.8 m room: floor, ceiling and four walls. Corners give
        # ICP all three translation axes.
        pts = []
        for x in np.linspace(-3, 3, 120):
            for y in np.linspace(-2.5, 2.5, 100):
                pts += [[x, y, 0.0], [x, y, 2.8]]
        for x in np.linspace(-3, 3, 120):
            for z in np.linspace(0, 2.8, 60):
                pts += [[x, -2.5, z], [x, 2.5, z]]
        for y in np.linspace(-2.5, 2.5, 100):
            for z in np.linspace(0, 2.8, 60):
                pts += [[-3.0, y, z], [3.0, y, z]]
        cls.room = np.array(pts)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cls.room)
        cls.cloud = cloud

    def _matcher(self, **kw):
        cfg = {"map_file": "", "voxel_size": 0.05, "min_fitness": 0.2,
               "max_correction_m": 1.0, "max_correction_deg": 20.0,
               "icp_rate_hz": 1000.0, "hint_radius_m": 2.0,
               "hint_radius_deg": 45.0}
        cfg.update(kw)
        m = LidarMapMatcher(cfg)
        m.set_map_cloud(self.cloud)
        return m

    def _render_depth(self, T_map_cam, K=None, shape=(360, 640)):
        """Ray-cast the synthetic room into a depth image from T_map_cam.

        Analytic box intersection rather than a rasteriser: the room is six
        planes, so the exact range along each pixel ray is a closed form.
        """
        K = self.K if K is None else K
        h, w = shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        vv, uu = np.meshgrid(np.arange(h, dtype=float),
                             np.arange(w, dtype=float), indexing="ij")
        dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)],
                        axis=-1).reshape(-1, 3)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        T_map_optical = T_map_cam @ T_BODY_OPTICAL
        R = T_map_optical[:3, :3]
        origin = T_map_optical[:3, 3]
        rays = dirs @ R.T

        lo = np.array([-3.0, -2.5, 0.0])
        hi = np.array([3.0, 2.5, 2.8])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - origin) / rays
            t2 = (hi - origin) / rays
            t_far = np.minimum(np.maximum(t1, t2), 1e9).min(axis=1)
        t_far[~np.isfinite(t_far)] = 0.0

        # Range along the ray -> Z in the optical frame, which is what a depth
        # image stores.
        z = t_far * dirs[:, 2]
        z[(t_far <= 0) | (z <= 0.3) | (z > 6.0)] = 0.0
        return (z.reshape(h, w) * 1000.0).astype(np.uint16)

    K = np.array([[337.0, 0.0, 320.0],
                  [0.0, 337.0, 180.0],
                  [0.0, 0.0, 1.0]])

    def test_imperfect_scan_accepts_lineup_and_continuous_checks(self):
        from tracking_config import load_config
        from pathlib import Path
        cfg = load_config(str(Path(__file__).with_name("tracking.yaml")), verbose=False)["lidar"]
        cfg = dict(cfg, map_scale=1.0)  # synthetic room coordinates are already metres
        camera = make_transform((0, 0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(camera)
        # A strip of the room differs from the saved scan by 40 cm.
        depth[:, :80] += 400
        matcher = self._matcher(**dict(cfg, seed_yaw_steps=1))
        match = matcher.solve_from_hint(depth, self.K, camera, np.eye(4))
        self.assertTrue(match.ok, match.reason)
        self.assertGreaterEqual(match.verify_score, 0.75)
        self.assertLess(match.verify_score, 0.95)
        distance, angle = matcher._camera_delta(match.T_slam_to_map, np.eye(4), camera)
        self.assertLess(distance, 0.02)
        self.assertLess(angle, 0.3)
        matcher.accept_hint_solution(match, "test")
        update = matcher.update_depth(depth, self.K, camera, timestamp=100.)
        self.assertTrue(update.ok, update.reason)
        # Explicit strict settings still work; low agreement is never relabelled.
        strict = self._matcher(**dict(cfg, min_verify_score=0.95, seed_yaw_steps=1))
        rejected = strict.solve_from_hint(depth, self.K, camera, np.eye(4))
        self.assertFalse(rejected.ok)
        self.assertIn("need 95%", rejected.reason)

    def test_recovers_injected_translation_drift(self):
        m = self._matcher()
        # Truth: camera at (0, 0, 1.5) looking down +X. SLAM believes it is
        # 12 cm off, which is exactly the drift the fusion must find.
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)

        drift = rigid(t=(0.12, -0.06, 0.03))
        m.set_initial_alignment(drift, source="test")
        meas = m._measure(depth, self.K, T_true, m.T_slam_to_map,
                          stages=((2, 0.5, 60), (1, 0.1, 60)),
                          max_translation_m=1.0, max_rotation_deg=20.0,
                          constrain=True)

        self.assertTrue(meas.ok, meas.reason)
        residual = LidarMapMatcher._camera_delta(
            meas.T_slam_to_map, np.eye(4), T_true)[0]
        self.assertLess(residual, 0.02,
                        f"12 cm drift not recovered: {residual * 1000:.0f} mm left")

    def test_recovers_injected_rotation_drift(self):
        m = self._matcher()
        T_true = make_transform((0.5, 0.3, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)

        drift = rigid(rpy_deg=(0.0, 0.0, 2.0))
        m.set_initial_alignment(drift, source="test")
        meas = m._measure(depth, self.K, T_true, m.T_slam_to_map,
                          stages=((2, 0.5, 60), (1, 0.1, 60)),
                          max_translation_m=1.0, max_rotation_deg=20.0,
                          constrain=True)

        self.assertTrue(meas.ok, meas.reason)
        _, residual_deg = LidarMapMatcher._camera_delta(
            meas.T_slam_to_map, np.eye(4), T_true)
        self.assertLess(residual_deg, 0.3,
                        f"2 deg drift not recovered: {residual_deg:.2f} deg left")

    def test_continuous_fusion_walks_the_error_down(self):
        m = self._matcher(blend_alpha=0.5, max_slew_m_s=1.0,
                          max_slew_deg_s=30.0, icp_rate_hz=5.0)
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        m.set_initial_alignment(rigid(t=(0.10, 0.05, 0.0)), source="test")

        start = LidarMapMatcher._camera_delta(m.T_slam_to_map, np.eye(4), T_true)[0]
        for i in range(15):
            m.update_depth(depth, self.K, T_true, timestamp=i * 1.0)
        end = LidarMapMatcher._camera_delta(m.T_slam_to_map, np.eye(4), T_true)[0]

        self.assertLess(end, start * 0.25,
                        f"error only went {start * 1000:.0f} -> {end * 1000:.0f} mm")

    def test_floor_marker_hint_refines_on_scan_then_continues_fusion(self):
        from marker_lineup import (MarkerObservation, matcher_snapshot,
                                   solve_marker_observation)
        from pose_transform import invert_transform
        from test_tracking_engine import project_marker
        import time
        live = self._matcher(hint_radius_m=0.5, hint_radius_deg=20.0)
        camera = rigid(rpy_deg=(0, 35, 20), t=(0, -0.7, 1.5))
        marker_centre = camera[:3, 3] + camera[:3, 0] * (1.5 / -camera[2, 0])
        locator_true = rigid(rpy_deg=(0, 0, 30), t=marker_centre)
        local_marker = rigid(rpy_deg=(0, 0, -90))
        optical_marker = invert_transform(camera @ T_BODY_OPTICAL) @ locator_true @ local_marker
        corners = project_marker(self.K, 0.1, optical_marker)
        self.assertTrue(np.all(corners > 0) and np.all(corners < (640, 360)))
        slam_map = rigid(rpy_deg=(0, 0, 40), t=(2.4, -1.3, 0.4))
        # The locator is an imperfect indication: 20 cm and 3 degrees off.
        locator_guess = rigid(rpy_deg=(0, 0, 3), t=(0.2, -0.1, 0.02)) @ locator_true
        observation = MarkerObservation(
            "DICT_4X4_50", 7, corners, self.K, 0.1,
            slam_map @ camera, slam_map @ camera, locator_guess,
            self._render_depth(camera), self.K, np.array([0., 0., 1.]), time.monotonic())
        measurement = solve_marker_observation(observation, matcher_snapshot(live))
        self.assertTrue(measurement.ok, measurement.reason)
        self.assertFalse(live.is_aligned, "background search changed live alignment")
        live.accept_hint_solution(measurement, "aruco", observation.captured_at)
        solved = live.T_slam_to_map @ slam_map @ camera
        self.assertLess(np.linalg.norm(solved[:3, 3] - camera[:3, 3]), 0.03)
        self.assertGreater(np.linalg.norm(solved[:3, 3] - marker_centre), 1)
        # No further marker sighting is supplied. New depth + SLAM keeps fusing.
        moved = camera.copy()
        moved[:3, 3] += [0.1, 0.05, 0]
        later = live.update_depth(self._render_depth(moved), self.K, slam_map @ moved,
                                  timestamp=time.monotonic())
        self.assertIsNotNone(later)
        self.assertTrue(later.ok, later.reason)
        corrected = live.T_slam_to_map @ slam_map @ moved
        self.assertLess(np.linalg.norm(corrected[:3, 3] - moved[:3, 3]), 0.03)

    def test_multiple_surveyed_markers_verify_against_real_room_depth(self):
        from dataclasses import replace
        from marker_map import MappedObservation, solve_mapped_observation
        from pose_transform import invert_transform
        from test_tracking_engine import project_marker
        observation, camera, slam_map, locator = self._unknown_heading_observation()
        markers = []
        for i, offset in enumerate((-.35, .35)):
            target = locator.copy()
            target[:3,3] += offset * camera[:3,1]
            optical = invert_transform(camera @ T_BODY_OPTICAL) @ target @ rigid(rpy_deg=(0,0,-90))
            markers.append(replace(observation, marker_id=7+i, marker_size_m=.2,
                T_map_locator=target, corners=project_marker(self.K, .2, optical)))
        matcher = self._matcher()
        result = solve_mapped_observation(MappedObservation(tuple(markers)), matcher)
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(len(result.marker_keys), 2)
        self.assertGreater(result.verify_score, .75)
        np.testing.assert_allclose(result.T_slam_to_map, invert_transform(slam_map), atol=1e-5)
        self.assertFalse(matcher.is_aligned, 'verification must not install its result')
        # Shift off the observed floor: depth must veto the wrong survey height.
        for marker in markers:
            marker.T_map_locator[2,3] += 1.
        result = solve_mapped_observation(MappedObservation(tuple(markers)), matcher)
        self.assertFalse(result.ok)
        self.assertIn('scan verification', result.reason)

    def _unknown_heading_observation(self, heading=137, symmetric=False):
        from marker_lineup import MarkerObservation
        from pose_transform import invert_transform
        from test_tracking_engine import project_marker
        import time
        camera = rigid(rpy_deg=(0, 35, 20), t=(0, -.7, 1.5))
        centre = camera[:3, 3] + camera[:3, 0] * (1.5 / -camera[2, 0])
        if symmetric:
            camera = rigid(rpy_deg=(0, 36.2538, 0), t=(-1.5, 0, 1.1))
            centre = np.array([0., 0., 0.])
        actual_locator = rigid(rpy_deg=(0, 0, heading), t=centre)
        optical_marker = invert_transform(camera @ T_BODY_OPTICAL) @ actual_locator @ rigid(rpy_deg=(0, 0, -90))
        corners = project_marker(self.K, .1, optical_marker)
        slam_map = rigid(rpy_deg=(5, -8, 43), t=(2.4, -1.3, .4))
        position_only = rigid(t=centre)  # operator never entered any rotation
        observation = MarkerObservation('DICT_4X4_50', 0, corners, self.K, .1,
            slam_map @ camera, slam_map @ camera, position_only,
            self._render_depth(camera), self.K, np.array([0., 0., 1.]), time.monotonic())
        return observation, camera, slam_map, actual_locator

    def test_unknown_marker_heading_is_solved_without_relaxing_position_bounds(self):
        from marker_lineup import solve_marker_observation
        for heading in (91, 137, -173):
            with self.subTest(heading=heading):
                obs, camera, slam_map, locator = self._unknown_heading_observation(heading)
                m = self._matcher(hint_radius_m=.5, hint_radius_deg=20)
                result = solve_marker_observation(obs, m)
                self.assertTrue(result.ok, result.reason)
                self.assertFalse(m.is_aligned, 'search must wait for GUI context validation')
                distance, angle = m._camera_delta(np.linalg.inv(slam_map), result.T_slam_to_map, obs.T_slam_image)
                self.assertLess(distance, .03)
                self.assertLess(angle, 1.)
                np.testing.assert_allclose(result.T_map_locator[:3, 3], locator[:3, 3])
                delta = result.T_map_locator[:3, :3].T @ locator[:3, :3]
                self.assertLess(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1))), 1.)

    def test_wall_marker_is_found_upright_on_its_wall(self):
        """A marker taped to a wall: face level, printed top up, heading unknown."""
        from marker_lineup import MarkerObservation, solve_marker_observation
        from pose_transform import invert_transform
        from test_tracking_engine import project_marker
        import time
        camera = rigid(rpy_deg=(0, 25, 62), t=(.8, 1.2, 1.4))
        fwd = camera[:3, 0]
        centre = camera[:3, 3] + fwd * ((2.5 - camera[1, 3]) / fwd[1])   # on the y = 2.5 wall
        wall = np.eye(4)
        wall[:3, :3] = np.column_stack(([0, 0, 1.], [-1, 0, 0.], [0, -1, 0.]))  # top up, face into the room
        wall[:3, 3] = centre
        optical = invert_transform(camera @ T_BODY_OPTICAL) @ wall @ rigid(rpy_deg=(0, 0, -90))
        corners = project_marker(self.K, .1, optical)
        self.assertTrue(np.all(corners > 0) and np.all(corners < (640, 360)))
        slam_map = rigid(rpy_deg=(0, 0, 43), t=(2.4, -1.3, .4))   # gravity-aligned SLAM
        for offset in ((0, 0, 0), (.45, -.3, 0)):
            with self.subTest(offset=offset):
                obs = MarkerObservation('DICT_4X4_250', 5, corners, self.K, .1,
                    slam_map @ camera, slam_map @ camera, rigid(t=centre + offset),
                    self._render_depth(camera), self.K, np.array([0., 0., 1.]), time.monotonic())
                m = self._matcher(hint_radius_m=.5, hint_radius_deg=20)
                result = solve_marker_observation(obs, m, search_radius_m=1.)
                self.assertTrue(result.ok, result.reason)
                distance, angle = m._camera_delta(np.linalg.inv(slam_map), result.T_slam_to_map, obs.T_slam_image)
                self.assertLess(distance, .03)
                self.assertLess(angle, 1.)
                self.assertLess(np.linalg.norm(result.T_map_marker[:3, 3] - centre), .03)
                np.testing.assert_allclose(result.T_map_marker[:3, :3], wall[:3, :3], atol=.02)

    def test_symmetric_room_does_not_invent_a_marker_heading(self):
        from marker_lineup import solve_marker_observation
        obs, _, _, _ = self._unknown_heading_observation(symmetric=True)
        m = self._matcher(hint_radius_m=.5, hint_radius_deg=20)
        result = solve_marker_observation(obs, m)
        self.assertFalse(result.ok)
        self.assertIn('ambiguous', result.reason)
        self.assertFalse(m.is_aligned)

    def test_unknown_heading_cannot_move_a_misplaced_marker_across_the_room(self):
        from marker_lineup import solve_marker_observation
        obs, _, _, _ = self._unknown_heading_observation()
        obs.T_map_locator[0, 3] += 10.
        m = self._matcher(hint_radius_m=.5, hint_radius_deg=20)
        for radius in (None, 1.):
            result = solve_marker_observation(obs, m, search_radius_m=radius)
            self.assertFalse(result.ok)
        self.assertFalse(m.is_aligned)

    def test_wider_search_finds_a_marker_away_from_the_locator(self):
        from marker_lineup import solve_marker_observation
        obs, camera, slam_map, locator = self._unknown_heading_observation(137)
        placed = obs.T_map_locator[:3, 3].copy()
        obs.T_map_locator[:3, 3] += [.55, -.45, 0]   # 0.71 m off, between grid points
        m = self._matcher(hint_radius_m=.5, hint_radius_deg=20)
        self.assertFalse(solve_marker_observation(obs, m).ok,
                         "the default radius is lidar.hint_radius_m")
        result = solve_marker_observation(obs, m, search_radius_m=1.)
        self.assertTrue(result.ok, result.reason)
        self.assertFalse(m.is_aligned, 'search must wait for GUI context validation')
        distance, angle = m._camera_delta(np.linalg.inv(slam_map), result.T_slam_to_map, obs.T_slam_image)
        self.assertLess(distance, .03)
        self.assertLess(angle, 1.)
        self.assertLess(np.linalg.norm(result.T_map_marker[:3, 3] - placed), .03)
        delta = result.T_map_marker[:3, :3].T @ locator[:3, :3]
        self.assertLess(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1))), 1.)
        self.assertAlmostEqual(result.marker_error_m, .71, delta=.03)
        np.testing.assert_allclose(result.T_map_locator[:3, 3], obs.T_map_locator[:3, 3])

    def test_solve_from_hint_ignores_where_the_hint_actually_was(self):
        """The reported symptom: the camera must not land on the locator."""
        m = self._matcher()
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        # Operator drops the locator 25 cm and 3 deg off where the camera is.
        hint = rigid(rpy_deg=(0, 0, 3.0), t=(0.25, -0.15, 0.05))
        meas = m.solve_from_hint(depth, self.K, T_true, hint, source="test")

        self.assertTrue(meas.ok, meas.reason)
        self.assertTrue(m.is_aligned)

        to_truth, to_truth_deg = LidarMapMatcher._camera_delta(
            m.T_slam_to_map, np.eye(4), T_true)
        to_hint, _ = LidarMapMatcher._camera_delta(
            m.T_slam_to_map, hint, T_true)
        self.assertLess(to_truth, 0.03, "did not solve to the true pose")
        self.assertLess(to_truth_deg, 0.5)
        self.assertGreater(to_hint, 0.15,
                           "solution sat on the hint instead of solving")

    def test_solution_is_independent_of_the_hint(self):
        """Two different hints for the same view must give the same answer."""
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = None
        answers = []
        for t, yaw in [((0.20, -0.10, 0.04), 2.5), ((-0.18, 0.12, -0.05), -3.0)]:
            m = self._matcher()
            if depth is None:
                depth = self._render_depth(T_true)
            meas = m.solve_from_hint(depth, self.K, T_true,
                                     rigid(rpy_deg=(0, 0, yaw), t=t),
                                     source="test")
            self.assertTrue(meas.ok, meas.reason)
            answers.append(m.T_slam_to_map)
        spread, spread_deg = LidarMapMatcher._camera_delta(
            answers[0], answers[1], T_true)
        self.assertLess(spread, 0.02, "answer moved with the hint")
        self.assertLess(spread_deg, 0.5)

    def test_reports_how_far_it_moved_off_the_hint(self):
        m = self._matcher()
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        hint = rigid(rpy_deg=(0, 0, 3.0), t=(0.25, -0.15, 0.05))
        meas = m.solve_from_hint(depth, self.K, T_true, hint, source="test")
        self.assertTrue(meas.ok, meas.reason)
        expected, _ = LidarMapMatcher._camera_delta(hint, np.eye(4), T_true)
        # translation_m is measured against the hint the operator placed.
        self.assertAlmostEqual(meas.translation_m, expected, delta=0.03)
        self.assertGreater(m.get_diagnostics()["hint_offset_mm"], 100.0)

    def test_verification_scores_a_good_solve_highly(self):
        m = self._matcher()
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        meas = m.solve_from_hint(depth, self.K, T_true,
                                 rigid(t=(0.1, 0.0, 0.0)), source="test")
        self.assertTrue(meas.ok, meas.reason)
        self.assertGreater(meas.verify_score, 0.9)
        self.assertLess(meas.verify_median_m, 0.05)

    def test_a_match_outside_the_search_region_is_refused(self):
        """Reported on hardware: a close-up of a table corner locked onto
        geometry 1.1 m and 42 deg from the hint, reported success, and left
        continuous fusion stuck asking for a correction it could not make."""
        m = self._matcher(hint_radius_m=0.25, hint_radius_deg=10.0)
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        # A hint 1.2 m out: the true pose is outside the stated search region,
        # so there is no acceptable answer and it must say so.
        meas = m.solve_from_hint(depth, self.K, T_true,
                                 rigid(t=(1.2, 0.4, 0.0)), source="test")
        self.assertFalse(meas.ok)
        self.assertFalse(m.is_aligned)
        self.assertIn("search region", meas.reason)

    def test_widening_the_region_admits_the_same_match(self):
        """The region is a policy, not a limitation of the solver."""
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        hint = rigid(t=(1.2, 0.4, 0.0))
        tight = self._matcher(hint_radius_m=0.25, hint_radius_deg=10.0)
        self.assertFalse(tight.solve_from_hint(depth, self.K, T_true, hint,
                                               source="test").ok)
        wide = self._matcher(hint_radius_m=2.0, hint_radius_deg=45.0)
        self.assertTrue(wide.solve_from_hint(depth, self.K, T_true, hint,
                                             source="test").ok)

    def test_search_region_is_reported(self):
        m = self._matcher(hint_radius_m=0.4, hint_radius_deg=15.0)
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        m.solve_from_hint(depth, self.K, T_true, rigid(t=(0.1, 0.0, 0.0)),
                          source="test")
        d = m.get_diagnostics()
        self.assertAlmostEqual(d["search_region_m"], 0.4)
        self.assertAlmostEqual(d["search_region_deg"], 15.0)

    def test_a_rejected_live_fix_does_not_wipe_the_lock_quality(self):
        """The panel read 'On scan: 0%' for a lock that had verified fine,
        because a rejected continuous fix reports 0 and overwrote it."""
        m = self._matcher()
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        self.assertTrue(m.solve_from_hint(depth, self.K, T_true,
                                          rigid(t=(0.05, 0.0, 0.0)),
                                          source="test").ok)
        lock_score = m.get_diagnostics()["lock_verify_score"]
        self.assertGreater(lock_score, 0.5)

        # Force a rejected continuous fix (gate fires before verification).
        m.max_correction_m = 1e-6
        m.last_icp_time = -1e9
        rejected = m.update_depth(depth, self.K, T_true, timestamp=1e6)
        self.assertIsNotNone(rejected)
        self.assertFalse(rejected.ok)
        self.assertAlmostEqual(m.get_diagnostics()["lock_verify_score"],
                               lock_score, places=9)

    def test_a_hopeless_hint_is_refused_not_accepted(self):
        """Better to say "not on the scan" than to sit on a wrong hint."""
        m = self._matcher()
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        # Well outside the room entirely.
        meas = m.solve_from_hint(depth, self.K, T_true,
                                 rigid(rpy_deg=(0, 0, 140.0), t=(9.0, -7.0, 3.0)),
                                 source="test")
        self.assertFalse(meas.ok)
        self.assertFalse(m.is_aligned)

    def test_yaw_multistart_recovers_heading_error(self):
        m = self._matcher()
        T_true = make_transform((0.2, 0.1, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        # 18 deg of heading error: beyond what plain ICP recovers here.
        meas = m.solve_from_hint(depth, self.K, T_true,
                                 rigid(rpy_deg=(0, 0, 18.0), t=(0.1, 0.05, 0.0)),
                                 source="test")
        self.assertTrue(meas.ok, meas.reason)
        _, to_truth_deg = LidarMapMatcher._camera_delta(
            m.T_slam_to_map, np.eye(4), T_true)
        self.assertLess(to_truth_deg, 1.0)

    def test_align_from_seed_is_still_available(self):
        self.assertIs(LidarMapMatcher.align_from_seed,
                      LidarMapMatcher.solve_from_hint)

    def test_rate_limit_skips_off_tick_calls(self):
        m = self._matcher(icp_rate_hz=5.0)
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        m.set_initial_alignment(np.eye(4), source="test")
        self.assertIsNotNone(m.update_depth(depth, self.K, T_true, timestamp=100.0))
        self.assertIsNone(m.update_depth(depth, self.K, T_true, timestamp=100.05))
        self.assertIsNotNone(m.update_depth(depth, self.K, T_true, timestamp=100.5))

    def test_worker_returns_immediately_and_still_fuses(self):
        """ICP must not run on the sensor thread.

        Measured on hardware: a 100-200 ms fix inside the ROS depth callback
        blocked the single-threaded executor, so 89 Hz of SLAM produced only
        16 Hz of FreeD. submit() hands the frame over and returns.
        """
        import time as _t
        m = self._matcher(icp_rate_hz=1000.0)
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        m.set_initial_alignment(rigid(t=(0.05, 0.02, 0.0)), source="test")
        self.assertTrue(m.start_worker())
        self.assertFalse(m.start_worker(), "start_worker must be idempotent")
        try:
            # A blocking call would take far longer than this.
            t0 = _t.time()
            for i in range(5):
                m.submit(depth, self.K, T_true, timestamp=_t.monotonic())
            handoff_ms = (_t.time() - t0) * 1000.0
            self.assertLess(handoff_ms, 40.0,
                            f"submit() blocked for {handoff_ms:.0f} ms")

            deadline = _t.time() + 20.0
            while m.fix_counter == 0 and _t.time() < deadline:
                _t.sleep(0.02)
            self.assertGreater(m.fix_counter, 0, "worker produced no fix")
        finally:
            m.stop_worker()

    def test_submit_without_a_worker_runs_inline(self):
        m = self._matcher(icp_rate_hz=1000.0)
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        m.set_initial_alignment(rigid(t=(0.05, 0.0, 0.0)), source="test")
        self.assertTrue(m.submit(depth, self.K, T_true, timestamp=1.0))
        self.assertGreater(m.fix_counter, 0)

    def test_submit_is_ignored_until_aligned(self):
        m = self._matcher()
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        self.assertFalse(m.submit(self._render_depth(T_true), self.K, T_true))

    def test_stop_worker_is_safe_to_call_twice(self):
        m = self._matcher()
        m.start_worker()
        m.stop_worker()
        m.stop_worker()

    def test_unaligned_matcher_does_nothing(self):
        m = self._matcher()
        T_true = make_transform((0.0, 0.0, 1.5), (0, 0, 0, 1))
        depth = self._render_depth(T_true)
        self.assertIsNone(m.update_depth(depth, self.K, T_true, timestamp=1.0))


class TestOverlayFramePairing(unittest.TestCase):
    """The overlay must be drawn from the pose belonging to its own frame.

    Pairing the newest odometry sample with a picture that is a frame or two
    old slides the scan across the image by however far the camera travelled
    in between. The error is zero while the rig is parked, which is what made
    a single-frame lineup look correct while live tracking drifted.
    """

    K = np.array([[430., 0., 424.], [0., 430., 240.], [0., 0., 1.]])
    POSE = ((0.0, 0.0, 1.5), (0, 0, 0, 1))

    class _Recorder:
        """Stands in for the edge aligner, keeping what it was asked to draw."""
        available = True

        def __init__(self):
            self.calls = []

        def _record(self, gray, K, T_map_cam):
            self.calls.append((gray, K, np.asarray(T_map_cam, dtype=float).copy()))

        def overlay(self, gray, K, T_map_cam, colour=(0, 255, 255), opacity=0.8):
            self.opacity = opacity
            self._record(gray, K, T_map_cam)
            return gray

        def score(self, gray, K, T_map_cam):
            self._record(gray, K, T_map_cam)
            return 0.5

    def _matcher(self):
        m = matcher()
        m.edge_aligner = self._Recorder()
        m.is_aligned = True
        m.T_slam_to_map = rigid((0, 0, 30), (1.0, -2.0, 0.5))
        return m

    @staticmethod
    def _gray(fill=0):
        return np.full((480, 848), fill, dtype=np.uint8)

    def test_overlay_asks_for_the_pose_at_the_frames_own_stamp(self):
        m = self._matcher()
        m.set_image(self._gray(), self.K, 1234.5)
        asked = []

        def pose_at(stamp):
            asked.append(stamp)
            return self.POSE

        self.assertIsNotNone(m.image_overlay(pose_at))
        self.assertEqual(asked, [1234.5])

    def test_overlay_uses_the_paired_pose_not_a_later_one(self):
        m = self._matcher()
        m.set_image(self._gray(), self.K, 10.0)
        m.image_overlay(lambda stamp: self.POSE)
        used = m.edge_aligner.calls[-1][2]
        paired = make_transform(*self.POSE)
        np.testing.assert_allclose(used, m.T_slam_to_map @ paired, atol=1e-12)
        # The pose 400 mm further on -- what `latest_pose` would have supplied
        # mid-move -- must not be what got drawn.
        moved = make_transform((0.4, 0.0, 1.5), (0, 0, 0, 1))
        self.assertFalse(np.allclose(used, m.T_slam_to_map @ moved, atol=1e-3))

    def test_overlay_passes_opacity_without_changing_the_alignment(self):
        m = self._matcher()
        m.set_image(self._gray(), self.K, 10.0)
        alignment = m.T_slam_to_map.copy()
        m.image_overlay(lambda stamp: self.POSE, opacity=0.35)
        self.assertEqual(m.edge_aligner.opacity, 0.35)
        np.testing.assert_array_equal(m.T_slam_to_map, alignment)

    def test_unpairable_frame_draws_nothing_rather_than_the_wrong_pose(self):
        m = self._matcher()
        m.set_image(self._gray(), self.K, 10.0)
        self.assertIsNone(m.image_overlay(lambda stamp: None))
        self.assertEqual(m.edge_aligner.calls, [])

    def test_lineup_score_is_paired_the_same_way(self):
        m = self._matcher()
        m.set_image(self._gray(), self.K, 77.0)
        asked = []

        def pose_at(stamp):
            asked.append(stamp)
            return self.POSE

        self.assertIsNotNone(m.image_lineup(pose_at))
        self.assertEqual(asked, [77.0])
        self.assertIsNone(m.image_lineup(lambda stamp: None))

    def test_a_plain_pose_is_still_accepted(self):
        """Tools and tests that already hold one pose keep working."""
        m = self._matcher()
        m.set_image(self._gray(), self.K, 5.0)
        T = make_transform(*self.POSE)
        self.assertIsNotNone(m.image_overlay(T))
        np.testing.assert_allclose(m.edge_aligner.calls[-1][2],
                                   m.T_slam_to_map @ T, atol=1e-12)

    def test_frame_and_stamp_are_replaced_together(self):
        """One atomic tuple: no reader sees a new stamp on the old picture."""
        m = self._matcher()
        second = self._gray(7)
        m.set_image(self._gray(), self.K, 1.0)
        m.set_image(second, self.K, 2.0)
        seen = []
        m.image_overlay(lambda stamp: (seen.append(stamp), self.POSE)[1])
        self.assertEqual(seen, [2.0])
        self.assertIs(m.edge_aligner.calls[-1][0], second)

    def test_no_frame_yet_is_not_an_error(self):
        m = self._matcher()
        self.assertIsNone(m.image_overlay(lambda stamp: self.POSE))
        self.assertIsNone(m.image_lineup(lambda stamp: self.POSE))



try:                                    # the dashboard needs rclpy + PyQt5,
    import dashboard as _dashboard      # which only the container provides
    _HAS_DASHBOARD = True
except Exception:                       # pragma: no cover - host test runs
    _dashboard = None
    _HAS_DASHBOARD = False


@unittest.skipUnless(_HAS_DASHBOARD, "dashboard needs rclpy/PyQt5")
class TestDashboardOverlayWiring(unittest.TestCase):
    """`_refresh_ar_overlay` must hand the matcher a per-frame pose lookup.

    Locks the wiring, not just the matcher's contract: the regression was the
    dashboard passing `engine.latest_pose` -- one pose, sampled when the
    overlay happened to run -- instead of `engine.pose_at`, which answers for
    the frame actually being drawn on.
    """

    class _Engine:
        def __init__(self):
            self.asked = []

        def pose_at(self, stamp, **kw):
            self.asked.append(stamp)
            return (0.0, 0.0, 1.5), (0, 0, 0, 1)

        @property
        def latest_pose(self):
            raise AssertionError("the overlay must not use latest_pose")

    class _Matcher:
        is_aligned = True

        def __init__(self):
            self.handed = None

        def image_overlay(self, pose, opacity=0.8):
            self.handed = pose
            self.opacity = opacity
            # Stand in for the stored frame: ask about one known stamp.
            return pose(4242.0)

    def _stub(self):
        stub = _dashboard.HyperFusionApp.__new__(_dashboard.HyperFusionApp)
        stub._ar_overlay_on = True
        stub._lineup = None
        stub._ar_frame_n = 9            # the next tick is the drawing one
        stub._ar_overlay = None
        stub.engine = self._Engine()
        stub.lidar_matcher = self._Matcher()
        return stub

    def test_overlay_is_drawn_from_the_pose_at_the_frames_stamp(self):
        stub = self._stub()
        _dashboard.HyperFusionApp._refresh_ar_overlay(stub)
        self.assertIsNotNone(stub.lidar_matcher.handed,
                             "the matcher was never asked to draw")
        self.assertTrue(callable(stub.lidar_matcher.handed),
                        "a per-frame pose lookup must be handed over, not one pose")
        self.assertEqual(stub.engine.asked, [4242.0],
                         "the pose must be looked up at the frame's own stamp")
        self.assertIsNotNone(stub._ar_overlay)

    def test_transparency_slider_changes_the_next_overlay_and_keeps_its_pose_pair(self):
        from PyQt5.QtWidgets import QApplication, QMainWindow
        app = QApplication.instance() or QApplication([])
        stub = self._stub()
        QMainWindow.__init__(stub)
        panel = stub._build_ar_overlay_panel()
        try:
            self.assertEqual(stub.ar_transparency.value(), 20)
            self.assertTrue(stub.ar_transparency.isEnabled())
            stub.ar_transparency.setValue(65)
            stub._refresh_ar_overlay()
            self.assertAlmostEqual(stub.lidar_matcher.opacity, 0.35)
            self.assertIn("65%", stub.lbl_ar_transparency.text())
            self.assertEqual(stub.engine.asked, [4242.0])
            stub.btn_ar.setChecked(False)
            self.assertFalse(stub.ar_transparency.isEnabled())
            self.assertIsNone(stub._ar_overlay)
            stub.btn_ar.setChecked(True)
            self.assertTrue(stub.ar_transparency.isEnabled())
            self.assertEqual(stub.ar_transparency.value(), 65)
        finally:
            panel.close()
            stub.deleteLater()
            app.processEvents()


if __name__ == "__main__":
    unittest.main(verbosity=2)
