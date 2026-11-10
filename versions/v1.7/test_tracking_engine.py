#!/usr/bin/env python3
"""
Tests for tracking_engine.TrackingEngine — the LiDAR-anchor workflow.

No ROS and no Qt; OpenCV is needed only for the marker-solve tests, which skip
without it.

    python3 -m unittest test_tracking_engine -v

The workflow under test:

    a printed ArUco marker is taped to a known spot in the physical set
    the same set is LiDAR-scanned and imported into Unreal
    an anchor Actor sits in Unreal at that same spot
    the operator points the camera at the marker and presses anchor
    -> tracking output is thereafter expressed in the scan's coordinate frame
"""
import copy
import math
import unittest

import numpy as np

import freed
import tracking_config
from pose_transform import (UNREAL_UNITS_PER_METRE, invert_transform,
                            make_transform, rpy_to_matrix)
from tracking_engine import T_BODY_OPTICAL, TrackingEngine

try:
    import cv2
    HAVE_CV2 = True
except ImportError:                                   # pragma: no cover
    HAVE_CV2 = False


def base_config(**tracking):
    cfg = copy.deepcopy(tracking_config.DEFAULTS)
    cfg["tracking"].update({"gate_enabled": False, "hold_enabled": False})
    cfg["tracking"].update(tracking)
    return cfg


def engine(**tracking):
    return TrackingEngine(base_config(**tracking), log=lambda *a, **k: None)


def project_marker(K, marker_size, T_optical_marker):
    """Image corners of a marker at a given pose in the camera optical frame."""
    s = marker_size / 2.0
    obj = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
                   dtype=np.float64)
    R = T_optical_marker[:3, :3]
    t = T_optical_marker[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    img, _ = cv2.projectPoints(obj, rvec, t, K, np.zeros((5, 1)))
    return img.reshape(4, 2)


class TestPosePipeline(unittest.TestCase):
    def test_pose_reaches_the_wire(self):
        e = engine()
        e.accept_pose((1.0, -2.0, 0.5), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0),
                      stamp=1.0)
        d = freed.decode_freed_packet(e.freed_packet())
        self.assertAlmostEqual(d["x_m"], 1.0, delta=1e-3)
        self.assertAlmostEqual(d["y_m"], 2.0, delta=1e-3)   # Y mirrors
        self.assertAlmostEqual(d["z_m"], 0.5, delta=1e-3)

    def test_nothing_before_the_first_pose(self):
        self.assertIsNone(engine().freed_packet())

    def test_status_reports_tracking(self):
        e = engine()
        self.assertFalse(e.status_dict()["tracking"])
        e.accept_pose((0, 0, 0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0), stamp=1.0)
        self.assertTrue(e.status_dict()["tracking"])
        self.assertEqual(e.slam_status, "TRACKING")


class TestArucoConfiguration(unittest.TestCase):
    def test_new_list_form_preserves_all_families(self):
        cfg = {"dictionaries": list(tracking_config.ARUCO_4X4_FAMILIES)}
        self.assertEqual(tracking_config.aruco_dictionary_names(cfg),
                         tracking_config.ARUCO_4X4_FAMILIES)

    def test_printed_4x4_250_set_is_the_default_and_configured_family(self):
        from pathlib import Path
        self.assertEqual(tracking_config.aruco_dictionary_names({}), ("DICT_4X4_250",))
        cfg = tracking_config.load_config(
            str(Path(tracking_config.__file__).with_name("tracking.yaml")), verbose=False)
        self.assertEqual(tracking_config.aruco_dictionary_names(cfg["aruco"]), ("DICT_4X4_250",))

    def test_legacy_singular_form_remains_supported(self):
        self.assertEqual(tracking_config.aruco_dictionary_names(
            {"dictionary": "DICT_4X4_250"}), ("DICT_4X4_250",))


class TestUnrealAnchorTarget(unittest.TestCase):
    """The defect this workflow change fixes."""

    def test_anchor_target_converts_from_unreal_centimetres(self):
        e = engine()
        T = e.set_anchor_target_from_unreal_units((500.0, 200.0, 100.0), (0.0, 0.0, 0.0))
        np.testing.assert_allclose(T[:3, 3], [5.0, -2.0, 1.0], atol=1e-9)

    def test_without_a_target_the_marker_becomes_the_origin(self):
        e = engine()
        e.accept_pose((3.0, -1.0, 0.5), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0),
                      stamp=1.0)
        T_marker = make_transform((3.0, -1.0, 0.5), (0, 0, 0, 1))
        e.transformer.set_ground_zero_from_transform(T_marker)
        d = freed.decode_freed_packet(e.freed_packet())
        self.assertAlmostEqual(d["x_m"], 0.0, delta=1e-3)
        self.assertAlmostEqual(d["y_m"], 0.0, delta=1e-3)

    def test_with_a_target_the_marker_becomes_the_anchor(self):
        """Camera sitting on the marker must report the anchor Actor's transform."""
        e = engine()
        e.accept_pose((3.0, -1.0, 0.5), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0),
                      stamp=1.0)
        e.set_anchor_target_from_unreal_units((250.0, -400.0, 120.0), (30.0, 0.0, 0.0))
        T_marker = make_transform((3.0, -1.0, 0.5), (0, 0, 0, 1))
        e.transformer.set_ground_zero_from_transform(
            T_marker, T_target=e.anchor_target)

        d = freed.decode_freed_packet(e.freed_packet())
        self.assertAlmostEqual(d["x_m"] * UNREAL_UNITS_PER_METRE, 250.0, delta=0.5)
        self.assertAlmostEqual(d["y_m"] * UNREAL_UNITS_PER_METRE, -400.0, delta=0.5)
        self.assertAlmostEqual(d["z_m"] * UNREAL_UNITS_PER_METRE, 120.0, delta=0.5)
        self.assertAlmostEqual(d["yaw_deg"], 30.0, delta=0.05)

    def test_offset_from_the_marker_is_preserved(self):
        e = engine()
        e.set_anchor_target_from_unreal_units((100.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        T_marker = make_transform((2.0, 0.0, 0.0), (0, 0, 0, 1))
        e.transformer.set_ground_zero_from_transform(
            T_marker, T_target=e.anchor_target)
        # 0.5 m further along +X than the marker.
        e.accept_pose((2.5, 0.0, 0.0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0),
                      stamp=1.0)
        d = freed.decode_freed_packet(e.freed_packet())
        self.assertAlmostEqual(d["x_m"] * UNREAL_UNITS_PER_METRE, 150.0, delta=0.5)


class TestDashboardLocatorAnchor(unittest.TestCase):
    """set_anchor_target_ros(): the dashboard's own 3D-viewport locator
    driving the anchor target directly, in ROS metres -- no Unreal-unit
    conversion, unlike set_anchor_target_from_unreal_units()."""

    def test_sets_the_anchor_target_directly_in_ros_metres(self):
        e = engine()
        T = make_transform((5.0, -2.0, 1.0), (0, 0, 0, 1))
        out = e.set_anchor_target_ros(T, source="dashboard_locator")
        np.testing.assert_allclose(out[:3, 3], [5.0, -2.0, 1.0])
        self.assertEqual(e.anchor_target_source, "dashboard_locator")

    def test_marker_snaps_to_wherever_the_locator_is(self):
        e = engine()
        e.set_anchor_target_ros(make_transform((5.0, -2.0, 1.0), (0, 0, 0, 1)))
        e.accept_pose((3.0, -1.0, 0.5), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0),
                      stamp=1.0)
        T_marker = make_transform((3.0, -1.0, 0.5), (0, 0, 0, 1))
        e.transformer.set_ground_zero_from_transform(T_marker, T_target=e.anchor_target)
        d = freed.decode_freed_packet(e.freed_packet())
        self.assertAlmostEqual(d["x_m"], 5.0, delta=1e-3)
        self.assertAlmostEqual(d["y_m"], 2.0, delta=1e-3)   # Y mirrors
        self.assertAlmostEqual(d["z_m"], 1.0, delta=1e-3)

    def test_clear_anchor_target_removes_it(self):
        e = engine()
        e.set_anchor_target_ros(make_transform((1.0, 0.0, 0.0), (0, 0, 0, 1)))
        self.assertIsNotNone(e.anchor_target)
        e.clear_anchor_target()
        self.assertIsNone(e.anchor_target)
        self.assertIsNone(e.anchor_target_source)


class TestCameraPoseRosCorrected(unittest.TestCase):
    """camera_pose_ros_corrected(): the frame the dashboard's own 3D
    viewport should draw the camera dot in -- ROS metres, ground-zero
    applied, same frame as the locator and the marker solve."""

    def test_none_before_the_first_pose(self):
        self.assertIsNone(engine().camera_pose_ros_corrected())

    def test_unanchored_matches_the_raw_pose(self):
        e = engine()
        e.accept_pose((1.0, -2.0, 0.5), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0),
                      stamp=1.0)
        pos, quat = e.camera_pose_ros_corrected()
        np.testing.assert_allclose(pos, [1.0, -2.0, 0.5], atol=1e-9)

    def test_anchored_lands_on_the_locator(self):
        e = engine()
        e.set_anchor_target_ros(make_transform((5.0, -2.0, 1.0), (0, 0, 0, 1)))
        e.accept_pose((3.0, -1.0, 0.5), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0),
                      stamp=1.0)
        T_marker = make_transform((3.0, -1.0, 0.5), (0, 0, 0, 1))
        e.transformer.set_ground_zero_from_transform(T_marker, T_target=e.anchor_target)
        # Camera sits exactly on the marker, which now maps onto the locator.
        pos, quat = e.camera_pose_ros_corrected()
        np.testing.assert_allclose(pos, [5.0, -2.0, 1.0], atol=1e-6)


@unittest.skipUnless(HAVE_CV2, "OpenCV not installed")
class TestMarkerSolve(unittest.TestCase):
    def setUp(self):
        self.e = engine()
        self.e.accept_pose((0.0, 0.0, 0.0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0),
                           stamp=1.0)

    def _sighting(self, distance_m=1.0, yaw_deg=0.0):
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(0.0, math.radians(yaw_deg), 0.0)
        T[:3, 3] = [0.0, 0.0, distance_m]
        return project_marker(self.e.aruco_K, self.e.marker_size, T), T

    def test_recovers_the_distance(self):
        corners, _ = self._sighting(distance_m=1.25)
        sol = self.e.solve_marker_pose(corners)
        self.assertTrue(sol.ok, sol.reason)
        self.assertAlmostEqual(sol.distance_pnp_m, 1.25, delta=0.01)

    def test_marker_lands_in_front_of_the_camera(self):
        """Optical z-forward must become body x-forward, not sideways."""
        corners, _ = self._sighting(distance_m=2.0)
        sol = self.e.solve_marker_pose(corners)
        self.assertTrue(sol.ok, sol.reason)
        np.testing.assert_allclose(sol.T_world_marker[:3, 3], [2.0, 0.0, 0.0],
                                   atol=0.02)

    def test_no_pose_yet_is_refused_not_crashed(self):
        e = engine()
        corners, _ = self._sighting()
        sol = e.solve_marker_pose(corners)
        self.assertFalse(sol.ok)
        self.assertIn("no SLAM pose", sol.reason)

    def test_depth_agreeing_is_accepted_and_flagged(self):
        corners, _ = self._sighting(distance_m=1.5)
        self.e.set_depth(self._depth_frame(1.5))
        sol = self.e.solve_marker_pose(corners)
        self.assertTrue(sol.ok, sol.reason)
        self.assertTrue(sol.used_depth)
        self.assertLess(sol.depth_residual_m, 0.05)

    def test_depth_disagreeing_refuses_to_anchor(self):
        """A flipped or mis-scaled solve must not become the scene anchor."""
        corners, _ = self._sighting(distance_m=1.5)
        self.e.set_depth(self._depth_frame(3.0))     # depth says twice as far
        sol = self.e.solve_marker_pose(corners)
        self.assertFalse(sol.ok)
        self.assertIn("depth disagrees", sol.reason)

        self.e.set_anchor_target_from_unreal_units((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.e.request_anchor()
        ok, message = self.e.try_anchor(corners)
        self.assertFalse(ok)
        self.assertFalse(self.e.is_anchored,
                         "anchored despite depth disagreeing with PnP")
        self.assertTrue(self.e.anchor_requested,
                        "a refused anchor should stay armed for a retry")

    def test_depth_holes_fall_back_to_mono(self):
        corners, _ = self._sighting(distance_m=1.5)
        self.e.set_depth(np.zeros((480, 640), dtype=np.uint16))   # all invalid
        sol = self.e.solve_marker_pose(corners)
        self.assertTrue(sol.ok, sol.reason)
        self.assertFalse(sol.used_depth)

    def _depth_frame(self, distance_m):
        K = self.e.aruco_K
        h = int(round(K[1, 2] * 2))
        w = int(round(K[0, 2] * 2))
        return np.full((h, w), int(distance_m * 1000), dtype=np.uint16)

    def test_anchor_through_the_engine(self):
        corners, _ = self._sighting(distance_m=1.0)
        self.e.set_anchor_target_from_unreal_units((300.0, 0.0, 50.0), (0.0, 0.0, 0.0))
        self.e.request_anchor()
        ok, message = self.e.try_anchor(corners)
        self.assertTrue(ok, message)
        self.assertFalse(self.e.anchor_requested)
        # Camera is 1 m behind the marker, so it reports 1 m behind the anchor.
        d = freed.decode_freed_packet(self.e.freed_packet())
        self.assertAlmostEqual(d["x_m"] * UNREAL_UNITS_PER_METRE, 200.0, delta=2.0)
        self.assertAlmostEqual(d["z_m"] * UNREAL_UNITS_PER_METRE, 50.0, delta=2.0)


class TestMarkerHeading(unittest.TestCase):
    """Heading must be well defined however the marker is mounted.

    It used to come from the marker's +Y axis alone, which is horizontal for a
    marker lying face-up on a table but vertical for one taped to a wall. In
    the wall case atan2(0, 0) is undefined and the heading landed on
    floating-point noise -- measured at +0.1 deg and -180 deg for the same
    sighting under two different sets of intrinsics. 180 deg out puts the
    camera on the opposite side of the anchor.
    """

    ANCHOR_UNREAL = (300.0, 0.0, 50.0)

    @staticmethod
    def _world_marker(normal, print_up):
        """Marker rotation in the WORLD frame, from its normal and print-up."""
        z = np.asarray(normal, float); z /= np.linalg.norm(z)
        y = np.asarray(print_up, float); y /= np.linalg.norm(y)
        return np.column_stack([np.cross(y, z), y, z])

    def _anchor_with(self, R_world_marker, position=(1.0, 0.0, 0.0)):
        """Anchor on a marker mounted as described, at `position` in body axes."""
        e = engine()
        e.accept_pose((0.0, 0.0, 0.0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0),
                      stamp=1.0)
        # The sighting is generated in the optical frame; convert.
        T_world = np.eye(4)
        T_world[:3, :3] = R_world_marker
        T_world[:3, 3] = np.asarray(position, dtype=float)
        self.marker_position = np.asarray(position, dtype=float)
        T_optical = invert_transform(T_BODY_OPTICAL) @ T_world
        corners = project_marker(e.aruco_K, e.marker_size, T_optical)
        e.set_anchor_target_from_unreal_units(self.ANCHOR_UNREAL, (0.0, 0.0, 0.0))
        e.request_anchor()
        ok, message = e.try_anchor(corners)
        self.assertTrue(ok, message)
        return freed.decode_freed_packet(e.freed_packet())

    def _distance_from_anchor_m(self, decoded):
        """3D distance between the reported camera and the anchor."""
        ax, ay, az = (self.ANCHOR_UNREAL[0] / UNREAL_UNITS_PER_METRE,
                      self.ANCHOR_UNREAL[1] / UNREAL_UNITS_PER_METRE,
                      self.ANCHOR_UNREAL[2] / UNREAL_UNITS_PER_METRE)
        return math.dist((decoded["x_m"], decoded["y_m"], decoded["z_m"]),
                         (ax, ay, az))

    def test_wall_marker_head_on_reads_heading_zero(self):
        """Pins the convention: head-on marker, anchor yaw 0, camera 1 m back."""
        d = self._anchor_with(self._world_marker((-1, 0, 0), (0, 0, 1)))
        self.assertAlmostEqual(d["x_m"] * UNREAL_UNITS_PER_METRE, 200.0, delta=2.0)
        self.assertAlmostEqual(d["z_m"] * UNREAL_UNITS_PER_METRE, 50.0, delta=2.0)

    def test_wall_marker_at_an_angle(self):
        # Turned 25 deg. The heading changes, so the camera lands somewhere
        # else around the anchor -- but it is still exactly 1 m away from it,
        # which is the invariant that must hold whatever the convention.
        turn = math.radians(25.0)
        R = self._world_marker((-math.cos(turn), math.sin(turn), 0), (0, 0, 1))
        self.assertAlmostEqual(self._distance_from_anchor_m(self._anchor_with(R)),
                               1.0, delta=0.05)

    def test_table_marker_face_up(self):
        """The case that always worked, and must keep working.

        Placed below the camera and looked down on -- a face-up marker at the
        camera's own height is edge-on and has no solvable pose at all.
        """
        R = self._world_marker((0, 0, 1), (1, 0, 0))
        pos = (1.0, 0.0, -0.4)
        d = self._anchor_with(R, pos)
        self.assertAlmostEqual(self._distance_from_anchor_m(d),
                               float(np.linalg.norm(pos)), delta=0.05)

    def test_table_marker_rotated_on_the_table(self):
        R = self._world_marker((0, 0, 1), (0, 1, 0))
        pos = (1.0, 0.0, -0.4)
        d = self._anchor_with(R, pos)
        self.assertAlmostEqual(self._distance_from_anchor_m(d),
                               float(np.linalg.norm(pos)), delta=0.05)

    def test_heading_is_stable_across_intrinsics(self):
        """The same sighting must anchor the same way at any resolution."""
        results = []
        for fx, cx, cy in ((389.386, 319.544, 241.506),
                           (429.948, 423.497, 241.663)):
            e = engine()
            e.aruco_K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])
            e.accept_pose((0.0, 0.0, 0.0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0),
                          stamp=1.0)
            T = np.eye(4); T[:3, 3] = [0.0, 0.0, 1.0]
            corners = project_marker(e.aruco_K, e.marker_size, T)
            e.set_anchor_target_from_unreal_units(self.ANCHOR_UNREAL,
                                                  (0.0, 0.0, 0.0))
            e.request_anchor()
            ok, message = e.try_anchor(corners)
            self.assertTrue(ok, message)
            results.append(freed.decode_freed_packet(e.freed_packet())["x_m"])
        self.assertAlmostEqual(results[0], results[1], delta=0.02)


class TestCommandProtocol(unittest.TestCase):
    def test_ping(self):
        self.assertEqual(engine().handle_command({"cmd": "ping"})["reply"], "pong")

    def test_unknown_command_is_reported_not_raised(self):
        reply = engine().handle_command({"cmd": "nonsense"})
        self.assertFalse(reply["ok"])
        self.assertIn("unknown", reply["error"])

    def test_anchor_lidar_arms_and_sets_the_target(self):
        e = engine()
        reply = e.handle_command({"cmd": "anchor_lidar",
                                  "marker_map_position": [250, -400, 120],
                                  "marker_map_rotation": [30, 0, 0]})
        self.assertTrue(reply["ok"])
        self.assertTrue(e.anchor_requested)
        self.assertIsNotNone(e.anchor_target)
        np.testing.assert_allclose(reply["anchor_ros_m"], [2.5, 4.0, 1.2],
                                   atol=1e-6)

    def test_anchor_aruco_clears_any_target(self):
        e = engine()
        e.set_anchor_target_from_unreal_units((100.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        e.handle_command({"cmd": "anchor_aruco"})
        self.assertIsNone(e.anchor_target)
        self.assertTrue(e.anchor_requested)

    def test_reset_clears_ground_zero_and_target(self):
        e = engine()
        e.set_anchor_target_from_unreal_units((100.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        e.transformer.set_ground_zero_from_transform(
            make_transform((1.0, 0, 0), (0, 0, 0, 1)), T_target=e.anchor_target)
        e.handle_command({"cmd": "reset_pose"})
        self.assertFalse(e.is_anchored)
        self.assertIsNone(e.anchor_target)

    def test_accepts_raw_json_bytes(self):
        self.assertTrue(engine().handle_command(b'{"cmd": "ping"}')["ok"])

    def test_set_offset(self):
        e = engine()
        e.handle_command({"cmd": "set_offset", "tx": 1.0, "ty": 2.0})
        self.assertTrue(e.is_anchored)

    def test_status_dict_is_json_serialisable(self):
        import json
        e = engine()
        e.accept_pose((1.0, 0, 0), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0), stamp=1.0)
        e.freed_packet()
        json.dumps(e.status_dict())      # must not raise


class TestFrameConstant(unittest.TestCase):
    def test_body_optical_maps_optical_forward_to_body_forward(self):
        """Optical +Z (forward) must become body +X (forward)."""
        v = T_BODY_OPTICAL[:3, :3] @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(v, [1.0, 0.0, 0.0], atol=1e-12)

    def test_optical_right_becomes_body_right(self):
        """Optical +X (right) -> body -Y, because body +Y is left."""
        v = T_BODY_OPTICAL[:3, :3] @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [0.0, -1.0, 0.0], atol=1e-12)

    def test_optical_down_becomes_body_down(self):
        v = T_BODY_OPTICAL[:3, :3] @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 0.0, -1.0], atol=1e-12)

    def test_is_a_proper_rotation(self):
        R = T_BODY_OPTICAL[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
