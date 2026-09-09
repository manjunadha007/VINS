"""Floor-marker pose, synchronized capture and isolated scan-search tests."""
import copy
import time
import unittest
from unittest.mock import Mock
import numpy as np
import cv2
from marker_lineup import (MarkerObservation, capture_marker_observation,
                           floor_marker_hint, matcher_snapshot, solve_marker_observation)
from pose_transform import (T_BODY_OPTICAL, invert_transform, make_transform,
                            matrix_to_quaternion, rpy_to_matrix)
from lidar_map_matcher import LidarMapMatcher, MapMeasurement
from test_tracking_engine import base_config, project_marker
from tracking_engine import TrackingEngine


def fixture():
    K = np.array([[500., 0., 424.], [0., 500., 240.], [0., 0., 1.]])
    locator = np.eye(4)
    locator[:3, :3] = rpy_to_matrix(0, 0, np.radians(37))
    locator[:3, 3] = [4, 2, 0.7]
    local_marker = np.eye(4)
    local_marker[:3, :3] = rpy_to_matrix(0, 0, -np.pi / 2)
    optical_marker = np.eye(4)
    optical_marker[:3, :3] = cv2.Rodrigues(np.array([2.6, 0.2, 0.1]))[0]
    optical_marker[:3, 3] = [0.15, 0.1, 1.3]
    camera = locator @ local_marker @ invert_transform(optical_marker) @ invert_transform(T_BODY_OPTICAL)
    slam_map = np.eye(4)
    slam_map[:3, :3] = rpy_to_matrix(0, 0, np.radians(43))
    slam_map[:3, 3] = [2.4, 1.7, 0.1]
    T_slam_image = slam_map @ camera
    moved = np.eye(4)
    moved[:3, :3] = rpy_to_matrix(0, 0, 0.1)
    moved[:3, 3] = [0.15, 0, 0]
    T_slam_depth = T_slam_image @ moved
    corners = project_marker(K, 0.1, optical_marker)
    obs = MarkerObservation("DICT_4X4_50", 7, corners, K, 0.1,
                            T_slam_image, T_slam_depth, locator,
                            np.full((480, 848), 1300, np.uint16), K.copy(),
                            np.array([0., 0., 1.]), time.monotonic())
    return obs, invert_transform(slam_map), camera


class FloorMarkerTests(unittest.TestCase):
    def test_known_camera_pose_is_recovered_from_marker_at_nonzero_location(self):
        obs, truth, camera = fixture()
        hint = floor_marker_hint(obs)
        np.testing.assert_allclose(hint, truth, atol=1e-7)
        np.testing.assert_allclose(hint @ obs.T_slam_image, camera, atol=1e-7)
        self.assertGreater(np.linalg.norm(camera[:3, 3] - obs.T_map_locator[:3, 3]), 1)
        np.testing.assert_allclose(
            obs.T_map_locator[:3, :3].T @ (hint @ obs.T_slam_image)[:3, :3],
            obs.T_map_locator[:3, :3].T @ camera[:3, :3], atol=1e-7)

    def test_locator_heading_changes_camera_position_and_heading_together(self):
        obs, truth, camera = fixture()
        delta = np.eye(4)
        delta[:3, :3] = rpy_to_matrix(0, 0, np.pi / 2)
        delta[:3, 3] = [1, -2, 0.5]
        obs.T_map_locator = delta @ obs.T_map_locator
        np.testing.assert_allclose(floor_marker_hint(obs), delta @ truth, atol=1e-7)

    def test_tiny_marker_is_rejected(self):
        obs, _, _ = fixture()
        obs.corners = np.array([[10, 10], [11, 10], [11, 11], [10, 11]], float)
        with self.assertRaisesRegex(ValueError, "too small"):
            floor_marker_hint(obs)

    def test_scan_search_uses_depth_pose_and_preserves_capture_time(self):
        obs, truth, _ = fixture()
        measurement = MapMeasurement(ok=True, T_slam_to_map=truth, observable_rank=3)
        matcher = Mock()
        matcher.solve_from_hint.return_value = measurement
        result = solve_marker_observation(obs, matcher, estimate_heading=False)
        args = matcher.solve_from_hint.call_args.args
        np.testing.assert_array_equal(args[2], obs.T_slam_depth)
        np.testing.assert_allclose(args[3], truth, atol=1e-7)
        self.assertEqual(result.monotonic_timestamp, obs.captured_at)


class MarkerHeadingGeometryTests(unittest.TestCase):
    def test_heading_sweep_keeps_marker_position_and_orbits_camera(self):
        from marker_lineup import marker_heading_seeds
        obs, _, _ = fixture()
        seeds = list(marker_heading_seeds(obs))
        positions = []
        for seed, slam_locator, yaw in seeds:
            transformed = seed @ slam_locator
            np.testing.assert_allclose(transformed[:3, 3], obs.T_map_locator[:3, 3], atol=1e-12)
            np.testing.assert_allclose(transformed[:3, 2], obs.up_map, atol=1e-12)
            positions.append((seed @ obs.T_slam_image)[:3, 3])
        self.assertEqual({angle for _, _, angle in seeds}, set(range(0, 360, 15)))
        self.assertGreater(np.linalg.norm(positions[0]-positions[12]), .5)

    def test_input_locator_rotation_is_ignored_and_tilted_scan_up_is_respected(self):
        from marker_lineup import marker_heading_seeds
        obs, _, _ = fixture()
        first = list(marker_heading_seeds(obs))
        obs.T_map_locator[:3, :3] = rpy_to_matrix(.7, -.9, 2.3)
        second = list(marker_heading_seeds(obs))
        for (a, _, _), (b, _, _) in zip(first, second):
            np.testing.assert_allclose(a, b)
        obs.up_map = rpy_to_matrix(.4, -.3, 1.2) @ np.array([0., 0., 1.])
        for seed, slam_locator, _ in marker_heading_seeds(obs):
            np.testing.assert_allclose((seed @ slam_locator)[:3, 2], obs.up_map, atol=1e-12)

    def test_coarse_success_cannot_escape_a_failed_fine_check(self):
        from unittest.mock import patch
        obs, _, _ = fixture()
        m = LidarMapMatcher({})
        m._ready = Mock(return_value=True)
        m._ensure_pyramid = Mock()
        m._deproject = Mock(return_value=np.zeros((100, 3)))
        m._map_tree = Mock()
        m._map_tree.query.return_value = (np.zeros(100), np.zeros(100, int))
        def measure(depth, K, pose, prior, stages, **kwargs):
            if stages[-1][0] == 1:
                return MapMeasurement(reason="Fine scan verification rejected")
            return MapMeasurement(ok=True, T_slam_to_map=prior, observable_rank=3)
        m._measure = Mock(side_effect=measure)
        result = solve_marker_observation(obs, m)
        self.assertFalse(result.ok)
        self.assertIn("Fine scan verification rejected", result.reason)
        self.assertFalse(m.is_aligned)


class MarkerCaptureTests(unittest.TestCase):
    def setUp(self):
        self.obs, self.truth, _ = fixture()
        self.engine = TrackingEngine(base_config(), log=lambda *a: None)
        self.engine.set_intrinsics(self.obs.K)
        for stamp, T in ((10.0, self.obs.T_slam_image), (10.1, self.obs.T_slam_depth)):
            self.engine.accept_pose(T[:3, 3], matrix_to_quaternion(T[:3, :3]),
                                    [0, 0, 0], [0, 0, 0], stamp=stamp)
        self.engine.set_depth(self.obs.depth, self.obs.depth_K, stamp=10.1)
        self.received = time.monotonic()
        self.detections = [(self.obs.family, 7, self.obs.corners)]

    def capture(self, **kwargs):
        args = dict(engine=self.engine, detections=self.detections,
                    image_stamp=10.0, image_received=self.received,
                    image_size=(848, 480), family=self.obs.family, marker_id=7,
                    marker_size_m=0.1, T_display_locator=self.obs.T_map_locator,
                    T_display_map=np.eye(4))
        args.update(kwargs)
        return capture_marker_observation(**args)

    def test_captures_each_frames_own_pose_and_copies_sensor_arrays(self):
        captured, reason = self.capture()
        self.assertIsNotNone(captured, reason)
        np.testing.assert_allclose(captured.T_slam_image, self.obs.T_slam_image, atol=1e-14)
        np.testing.assert_allclose(captured.T_slam_depth, self.obs.T_slam_depth, atol=1e-15)
        np.testing.assert_allclose(floor_marker_hint(captured), self.truth, atol=1e-7)
        self.obs.depth[:] = 0
        self.obs.corners[:] = 0
        self.assertGreater(captured.depth.max(), 0)
        self.assertGreater(captured.corners.max(), 0)

    def test_display_alignment_is_removed_before_solving(self):
        display = np.eye(4)
        display[:3, :3] = rpy_to_matrix(*np.radians([5, -8, 22]))
        display[:3, 3] = [10, -5, 2]
        captured, reason = self.capture(T_display_map=display,
                                       T_display_locator=display @ self.obs.T_map_locator)
        self.assertIsNotNone(captured, reason)
        np.testing.assert_allclose(floor_marker_hint(captured), self.truth, atol=1e-7)

    def test_wrong_id_or_dictionary_cannot_anchor(self):
        self.assertIsNone(self.capture(marker_id=8)[0])
        self.assertIsNone(self.capture(family="DICT_4X4_1000")[0])

    def test_unspecified_id_requires_one_marker(self):
        captured, reason = self.capture(marker_id=-1)
        self.assertEqual(captured.marker_id, 7, reason)
        self.detections.append((self.obs.family, 8, self.obs.corners + 100))
        captured, reason = self.capture(marker_id=-1)
        self.assertIsNone(captured)
        self.assertIn("More than one", reason)

    def test_stale_or_unpaired_frames_and_uncalibrated_intrinsics_are_refused(self):
        for overrides in (dict(image_received=self.received-2), dict(image_stamp=1.0)):
            self.assertIsNone(self.capture(**overrides)[0])
        self.engine.intrinsics_from_camera_info = False
        self.assertIsNone(self.capture()[0])
        self.engine.intrinsics_from_camera_info = True
        self.engine._depth_sample = None
        self.assertIsNone(self.capture()[0])

    def test_invalid_size_or_out_of_image_marker_is_refused(self):
        self.assertIsNone(self.capture(marker_size_m=0)[0])
        self.detections[0][2][0] = [-1, 30]
        self.assertIsNone(self.capture()[0])

    def test_capture_never_changes_output_alignment(self):
        self.capture()
        self.assertFalse(self.engine.is_anchored)
        self.assertFalse(self.engine.map_aligned)


class ScanSnapshotTests(unittest.TestCase):
    def test_snapshot_alignment_and_caches_are_private(self):
        live = LidarMapMatcher({})
        live._map_native = np.array([[1., 2., 3.]])
        live._map_pyramid = {1: object()}
        original = dict(live.diagnostics)
        snapshot = matcher_snapshot(live)
        self.assertIs(snapshot._map_native, live._map_native)
        self.assertIsNone(snapshot.edge_aligner)
        snapshot.T_slam_to_map[0, 3] = 10
        snapshot.diagnostics["lock_ok"] = True
        snapshot._map_pyramid.clear()
        self.assertEqual(live.T_slam_to_map[0, 3], 0)
        self.assertEqual(live.diagnostics, original)
        self.assertEqual(len(live._map_pyramid), 1)

    def test_verified_result_commits_with_original_capture_timestamp(self):
        obs, truth, _ = fixture()
        live = LidarMapMatcher({})
        generation = live._generation
        result = MapMeasurement(ok=True, T_slam_to_map=truth,
                                observable_rank=3, verify_score=0.9)
        live.accept_hint_solution(result, "aruco", captured_at=obs.captured_at)
        self.assertTrue(live.is_aligned)
        self.assertGreater(live._generation, generation)
        np.testing.assert_allclose(live.T_slam_to_map, truth)
        self.assertEqual(live.last_measurement.monotonic_timestamp, obs.captured_at)

    def test_rejected_result_cannot_be_committed(self):
        live = LidarMapMatcher({})
        with self.assertRaises(ValueError):
            live.accept_hint_solution(MapMeasurement(reason="bad scan"), "aruco")
        self.assertFalse(live.is_aligned)


if __name__ == "__main__":
    unittest.main()
