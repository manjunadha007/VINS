"""Actual bundled-scan regressions; run in the environment with Open3D."""
from pathlib import Path
import time
import unittest

import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None

from pose_transform import T_BODY_OPTICAL, invert_transform
from tracking_config import load_config
from lidar_map_matcher import LidarMapMatcher


@unittest.skipIf(o3d is None, "Open3D is required for scan raycasting")
class RoomScanRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            str(Path(__file__).with_name("tracking.yaml")), verbose=False)["lidar"]
        # Use the checked-in scan even if the operator has selected another map.
        cls.config["map_file"] = str(Path(__file__).with_name("Room01_Scaled_UE.obj"))
        cls.config["map_scale"] = 0.01
        mesh = o3d.io.read_triangle_mesh(cls.config["map_file"])
        mesh.scale(cls.config["map_scale"], center=(0, 0, 0))
        eye = np.array([-.2, 1.34, .19])
        forward = np.array([1.4, 1., .19]) - eye
        forward /= np.linalg.norm(forward)
        left = np.cross([0, 1, 0], forward)
        left /= np.linalg.norm(left)
        cls.T = np.eye(4)
        cls.T[:3, :3] = np.column_stack(
            [forward, left, np.cross(forward, left)])
        cls.T[:3, 3] = eye
        cls.K = np.array([[429.948, 0, 423.497],
                          [0, 429.948, 241.663], [0, 0, 1.]])
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        rays = scene.create_rays_pinhole(
            o3d.core.Tensor(cls.K),
            o3d.core.Tensor(invert_transform(cls.T @ T_BODY_OPTICAL)),
            848, 480)
        z = scene.cast_rays(rays)["t_hit"].numpy()
        valid = np.isfinite(z) & (z > .3) & (z < 6)
        cls.depth = np.uint16(np.where(valid, z, 0) * 1000)
        cls.matcher = LidarMapMatcher(cls.config)

    def test_survey_pose_survives_depth_verification_and_continuous_checks(self):
        # Regression: free ICP refinement once moved this known pose toward
        # another surface despite reporting high scan agreement.
        measurement = self.matcher.verify_pose(
            self.depth, self.K, self.T, np.eye(4))
        self.assertTrue(measurement.ok, measurement.reason)
        self.assertEqual(measurement.observable_rank, 3)
        np.testing.assert_array_equal(measurement.T_slam_to_map, np.eye(4))
        self.matcher.set_initial_alignment(
            measurement.T_slam_to_map, source="survey regression")
        start = time.monotonic()
        for i in range(20):
            self.matcher.update_depth(
                self.depth, self.K, self.T, start + i * .21)
        drift = np.linalg.norm(
            (self.matcher.T_slam_to_map @ self.T)[:3, 3] - self.T[:3, 3])
        self.assertLess(drift, .01, f"Stationary camera drifted {drift:.4f} m")

    def test_total_depth_occlusion_is_rejected(self):
        measurement = self.matcher.verify_pose(
            np.zeros_like(self.depth), self.K, self.T, np.eye(4))
        self.assertFalse(measurement.ok)
        self.assertIn("Too few depth", measurement.reason)

    def test_clearly_wrong_survey_position_is_rejected(self):
        wrong = np.eye(4)
        wrong[0, 3] = 5.0
        measurement = self.matcher.verify_pose(
            self.depth, self.K, self.T, wrong)
        self.assertFalse(measurement.ok)


if __name__ == "__main__":
    unittest.main()
