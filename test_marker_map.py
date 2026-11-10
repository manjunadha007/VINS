"""Joint marker pose, layout integrity and continuous correction regressions."""
from dataclasses import replace
import copy
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np
import cv2

from marker_map import (MAX_MARKERS, MappedObservation, MappedMeasurement, MarkerSurvey,
                        capture_mapped_observation, layout_entry, level_locators_from_view,
                        solve_mapped_observation, marker_similarity, MarkerSightings,
                        validate_layout, locator_transform, save_layout, load_layout,
                        layout_signature)
from marker_lineup import MarkerObservation
from lidar_map_matcher import LidarMapMatcher, MapMeasurement
from pose_transform import (T_BODY_OPTICAL, make_transform, invert_transform,
                            rpy_to_matrix, matrix_to_rpy, matrix_to_quaternion, so3_exp)
from tracking_engine import TrackingEngine
from test_tracking_engine import base_config, project_marker


def fixture(count=3, distance=2.5):
    K = np.array([[500., 0, 424], [0, 500., 240], [0, 0, 1.]])
    camera = make_transform((1., -2., 1.2), matrix_to_quaternion(rpy_to_matrix(.1, -.2, .4)))
    slam_map = make_transform((2., 3., -.5), matrix_to_quaternion(rpy_to_matrix(.03, .04, -.6)))
    T_depth = slam_map @ camera @ make_transform((.04, -.01, .02), (0,0,0,1))
    to_locator = np.eye(4)
    to_locator[:3,:3] = rpy_to_matrix(0, 0, np.pi/2)
    markers, records = [], []
    # Depth agrees with where each marker is: the lineup checks marker size
    # against depth, and one flat 2.5 m plane put the 2.9 m marker 14% off.
    depth = np.full((480, 848), 2500, np.uint16)
    for i in range(count):
        optical = np.eye(4)
        optical[:3,:3] = rpy_to_matrix(np.pi-.2, .05*i, .13*i)
        optical[:3,3] = [-.45 + .45*i, .1*(i % 2), distance + .2*i]
        quad = project_marker(K, .15 + .025*i, optical)
        cv2.fillConvexPoly(depth, np.round(quad).astype(np.int32) * 1, int(optical[2, 3] * 1000))
        cv2.polylines(depth, [np.round(quad).astype(np.int32)], True, int(optical[2, 3] * 1000), 25)
    for i in range(count):
        optical = np.eye(4)
        optical[:3,:3] = rpy_to_matrix(np.pi-.2, .05*i, .13*i)
        optical[:3,3] = [-.45 + .45*i, .1*(i % 2), distance + .2*i]
        locator = camera @ T_BODY_OPTICAL @ optical @ to_locator
        size = .15 + .025*i
        corners = project_marker(K, size, optical)
        markers.append(MarkerObservation("DICT_4X4_50", 7+i, corners, K, size,
            slam_map @ camera, T_depth, locator, depth.copy(),
            K.copy(), np.array([0.,0.,1.]), 100.))
        records.append(dict(dictionary="DICT_4X4_50", id=7+i, size_m=size,
            position_m=locator[:3,3].tolist(), rotation_deg=np.degrees(matrix_to_rpy(locator[:3,:3])).tolist()))
    return MappedObservation(tuple(markers)), invert_transform(slam_map), records


#: Locator position and rotation (deg) of three markers in a room corner.
ROOM_MARKERS = {1: ((2.0, .9, 1.4), (0, -90, 0)),     # on the x = 2 wall, facing -X
                2: ((.8, 2.0, 1.3), (0, -90, 90)),    # on the y = 2 wall, facing -Y
                3: ((1.2, 1.0, 0.), (0, 0, 30))}      # on the floor, printed top 30 deg round


def room_fixture(rotations=None, positions=None, noise_px=0., depth=True):
    """A D455 looking into a room corner, ~2 m from a floor and two wall markers.

    SLAM is gravity-aligned, as with IMU fusion, and the depth image is
    rendered from the floor and both walls. `rotations` and `positions`
    replace the Locators' by ID, as an operator might have set them.
    """
    K = np.array([[430., 0, 424], [0, 430., 240], [0, 0, 1.]])
    T_map_slam = make_transform((1., -2., 0.), matrix_to_quaternion(rpy_to_matrix(0, 0, .7)))
    camera = make_transform((.1, -.1, 1.3), matrix_to_quaternion(rpy_to_matrix(0, np.radians(20), np.radians(48))))
    optical = camera @ T_BODY_OPTICAL
    v, u = np.mgrid[0:480, 0:848]
    rays = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(u.shape)), -1) @ optical[:3, :3].T
    image = np.zeros(u.shape)
    for axis, value in ((0, 2.), (1, 2.), (2, 0.)):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (value - optical[axis, 3]) / rays[..., axis]
        hit = optical[:3, 3] + t[..., None] * rays
        inside = (t > 0) & np.all(hit[..., :2] <= 2.001, axis=-1) & (hit[..., 2] >= -.001)
        image = np.where(inside & ((image == 0) | (t < image)), t, image)
    image = (image * 1000 * depth).astype(np.uint16)
    to_locator = np.eye(4)
    to_locator[:3, :3] = rpy_to_matrix(0, 0, np.pi / 2)
    rng = np.random.default_rng(7)
    markers, records = [], []
    for fid, (position, rpy) in ROOM_MARKERS.items():
        T_map_locator = make_transform(position, matrix_to_quaternion(rpy_to_matrix(*np.radians(rpy))))
        corners = project_marker(K, .167, invert_transform(optical) @ T_map_locator @ invert_transform(to_locator))
        corners += rng.normal(0, noise_px, (4, 2))
        record = dict(dictionary="DICT_4X4_250", id=fid, size_m=.167,
                      position_m=list((positions or {}).get(fid, position)),
                      rotation_deg=list((rotations or {}).get(fid, rpy)))
        markers.append(MarkerObservation("DICT_4X4_250", fid, corners, K, .167,
            invert_transform(T_map_slam) @ camera, invert_transform(T_map_slam) @ camera,
            locator_transform(record), image, K.copy(), np.array([0., 0., 1.]), 100.))
        records.append(record)
    return MappedObservation(tuple(markers)), T_map_slam, records


class JointSolveTests(unittest.TestCase):
    def matcher(self, rank=1, ok=True):
        matcher = Mock()
        matcher._camera_delta = LidarMapMatcher._camera_delta
        matcher.verify_pose.side_effect = lambda depth, K, pose, alignment: MapMeasurement(
            ok=ok, T_slam_to_map=alignment.copy(), verify_score=.85, observable_rank=rank,
            fitness=.85, reason="" if ok else "depth disagrees")
        return matcher

    def test_multiple_sizes_rotations_and_distances_solve_one_pose(self):
        for distance in (1.5, 2.5, 4.):
            with self.subTest(distance=distance):
                obs, truth, _ = fixture(distance=distance)
                matcher = self.matcher()
                result = solve_mapped_observation(obs, matcher)
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(len(result.marker_keys), 3)
                np.testing.assert_allclose(result.T_slam_to_map, truth, atol=1e-6)
                self.assertEqual(result.monotonic_timestamp, 100.)
                for call in matcher.verify_pose.call_args_list:
                    np.testing.assert_allclose(call.args[2], obs.T_slam_depth)

    def test_one_bad_marker_is_excluded_from_three(self):
        obs, truth, _ = fixture()
        obs.markers[2].corners += [90., 20.]
        result = solve_mapped_observation(obs, self.matcher())
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.marker_keys, (("DICT_4X4_50", 7), ("DICT_4X4_50", 8)))
        self.assertEqual(result.rejected_marker_keys, (("DICT_4X4_50", 9),))
        np.testing.assert_allclose(result.T_slam_to_map, truth, atol=1e-6)

    def test_two_conflicting_markers_cannot_become_single_marker_success(self):
        obs, _, _ = fixture(2)
        obs.markers[1].T_map_locator[:3,3] += [1., 0., 0.]
        result = solve_mapped_observation(obs, self.matcher())
        self.assertFalse(result.ok)
        self.assertIn("disagree", result.reason)

    def test_scan_can_veto_a_perfect_marker_solve(self):
        obs, _, _ = fixture()
        result = solve_mapped_observation(obs, self.matcher(ok=False))
        self.assertFalse(result.ok)
        self.assertIn("depth disagrees", result.reason)

    def test_partial_visibility_and_single_marker_geometry_gate(self):
        obs, truth, _ = fixture()
        result = solve_mapped_observation(MappedObservation(obs.markers[:2]), self.matcher())
        self.assertTrue(result.ok, result.reason)
        np.testing.assert_allclose(result.T_slam_to_map, truth, atol=1e-6)
        single = MappedObservation(obs.markers[:1])
        self.assertFalse(solve_mapped_observation(single, self.matcher(rank=1)).ok)
        matcher = self.matcher(rank=3)
        matcher.verify_pose.side_effect = lambda depth, K, pose, alignment: MapMeasurement(
            ok=np.linalg.norm(alignment-truth) < .02, T_slam_to_map=alignment,
            observable_rank=3, verify_score=.9, fitness=.9)
        result = solve_mapped_observation(single, matcher)
        self.assertTrue(result.ok, result.reason)

    def test_indistinguishable_single_marker_branches_are_rejected(self):
        obs, _, _ = fixture(1)
        result = solve_mapped_observation(obs, self.matcher(rank=3))
        self.assertFalse(result.ok)
        self.assertIn("ambiguous", result.reason)

    def test_subpixel_corner_noise_still_combines_markers(self):
        obs, truth, _ = fixture()
        rng = np.random.default_rng(4)
        for marker in obs.markers:
            marker.corners += rng.normal(0, .15, (4,2))
        result = solve_mapped_observation(obs, self.matcher())
        self.assertTrue(result.ok, result.reason)
        distance, angle = LidarMapMatcher._camera_delta(truth, result.T_slam_to_map, obs.markers[0].T_slam_image)
        self.assertLess(distance, .04)
        self.assertLess(angle, 1)


class LayoutCaptureTests(unittest.TestCase):
    def setUp(self):
        self.obs, _, self.records = fixture()
        self.engine = TrackingEngine(base_config(), log=lambda *a: None)
        m = self.obs.markers[0]
        self.engine.set_intrinsics(m.K)
        for stamp, T in ((10., m.T_slam_image), (10.1, m.T_slam_depth)):
            self.engine.accept_pose(T[:3,3], matrix_to_quaternion(T[:3,:3]), [0,0,0], [0,0,0], stamp=stamp)
        self.engine.set_depth(m.depth, m.depth_K, stamp=10.1)
        self.received = time.monotonic()
        self.detections = [(m.family, m.marker_id, m.corners) for m in self.obs.markers]

    def capture(self, **kw):
        args = dict(engine=self.engine, detections=self.detections, image_stamp=10.,
            image_received=self.received, image_size=(848,480), records=self.records,
            T_display_map=np.eye(4))
        args.update(kw)
        return capture_mapped_observation(**args)

    def test_known_ids_use_individual_surveyed_poses_and_sizes(self):
        self.detections.append(("DICT_4X4_50", 49, self.detections[0][2]))
        result, reason = self.capture()
        self.assertIsNotNone(result, reason)
        self.assertEqual(len(result.markers), 3)
        for i, m in enumerate(result.markers):
            np.testing.assert_allclose(m.T_map_locator, self.obs.markers[i].T_map_locator, atol=1e-12)
            self.assertEqual(m.marker_size_m, self.records[i]["size_m"])
            np.testing.assert_allclose(m.T_slam_image, self.obs.markers[0].T_slam_image)
            np.testing.assert_allclose(m.T_slam_depth, self.obs.T_slam_depth)
        self.detections[0][2][:] = 0
        self.assertGreater(result.markers[0].corners.max(), 0)

    def test_display_transform_does_not_change_surveyed_coordinates(self):
        T = make_transform((3,4,5), matrix_to_quaternion(rpy_to_matrix(.2,.3,.4)))
        result, reason = self.capture(T_display_map=T)
        self.assertIsNotNone(result, reason)
        np.testing.assert_allclose(result.markers[0].T_map_locator, locator_transform(self.records[0]), atol=1e-12)

    def test_depth_scale_capture_is_independent_of_the_entered_print_width(self):
        self.records[0]["size_m"] *= .5
        normal, reason = self.capture()
        self.assertIsNone(normal)
        self.assertIn("Square width", reason)
        observation, reason = self.capture(check_marker_width=False)
        self.assertIsNotNone(observation, reason)
        self.assertEqual(observation.markers[0].marker_size_m, self.records[0]["size_m"])

    def test_duplicate_printed_id_is_rejected(self):
        self.detections.append(self.detections[0])
        result, reason = self.capture()
        self.assertIsNone(result)
        self.assertIn("twice", reason)

    def test_stale_or_unpaired_image_is_rejected(self):
        self.assertIsNone(self.capture(image_received=self.received-1)[0])
        self.assertIsNone(self.capture(image_stamp=5)[0])

    def test_layout_roundtrip_and_invalid_definitions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"markers.json"
            save_layout(path, self.records)
            self.assertEqual(load_layout(path), self.records)
        for change in (dict(id=-1), dict(id=50), dict(size_m=float("nan")),
                       dict(position_m=[1,2]), dict(rotation_deg=[0,0,float("inf")])):
            records = copy.deepcopy(self.records)
            records[0].update(change)
            with self.assertRaises(ValueError):
                validate_layout(records)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_layout(self.records + self.records[:1])
        records = copy.deepcopy(self.records)
        records[0]["position_m"][0] += 1e-12
        self.assertEqual(layout_signature(records), layout_signature(self.records))
        records[0]["rotation_deg"][2] += 360
        self.assertEqual(layout_signature(records), layout_signature(self.records))


class ContinuousSupportTests(unittest.TestCase):
    def measurement(self, stamp=100., distance=.1):
        scan = MapMeasurement(ok=True, T_slam_to_map=make_transform((distance,0,0),(0,0,0,1)),
                              fitness=.85, verify_score=.85, observable_rank=1)
        scan.monotonic_timestamp = stamp
        return MappedMeasurement(scan, [("DICT_4X4_50",7),("DICT_4X4_50",8)], [], .2)

    def test_correction_is_gradual_and_does_not_reset_alignment(self):
        matcher = LidarMapMatcher({})
        matcher.is_aligned = True
        generation = matcher._generation
        with patch("time.monotonic", return_value=100.1):
            result = self.measurement()
            self.assertTrue(matcher.submit_marker_fix(result, np.eye(4), generation))
            self.assertGreater(matcher.T_slam_to_map[0,3], 0)
            self.assertLessEqual(matcher.T_slam_to_map[0,3], .00401)
            self.assertEqual(matcher._generation, generation)
            self.assertFalse(matcher.submit_marker_fix(result, np.eye(4), generation))

    def test_stale_obsolete_and_large_corrections_are_rejected(self):
        for stamp, generation, distance in ((98.,0,.1), (100.,-1,.1), (100.,0,1.)):
            matcher = LidarMapMatcher({})
            matcher.is_aligned = True
            with patch("time.monotonic", return_value=100.1):
                self.assertFalse(matcher.submit_marker_fix(self.measurement(stamp, distance), np.eye(4), generation))
            np.testing.assert_array_equal(matcher.T_slam_to_map, np.eye(4))

    def test_two_markers_support_partial_scan_without_claiming_scan_rank(self):
        cfg = base_config()
        cfg["fusion"]["enabled"] = True
        engine = TrackingEngine(cfg, log=lambda *a: None)
        engine.apply_map_alignment(np.eye(4))
        with patch("time.monotonic", return_value=100.) as mono:
            for i in range(3):
                mono.return_value = 100 + .3*i
                engine.accept_pose((i*.1,0,0),(0,0,0,1),(0,0,0),(0,0,0),stamp=mono.return_value)
                engine.observe_scan(self.measurement(mono.return_value))
                engine.freed_packet()
            status = engine.fusion_status()
            self.assertEqual(status["state"], "VALIDATED")
            self.assertEqual(status["supported_by"], ["markers"])
            self.assertEqual(status["sources"]["markers"]["rank"], 1)
            mono.return_value += 2
            engine.accept_pose((.5,0,0),(0,0,0,1),(0,0,0),(0,0,0),stamp=mono.return_value)
            self.assertEqual(engine.fusion_status()["state"], "DEGRADED")
            engine.freed_packet()
            self.assertAlmostEqual(engine.camera_pose_ros_corrected()[0][0], .5)


def rotation_error_deg(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1)))


class MarkerSurveyTests(unittest.TestCase):
    """Placing detected but unsurveyed markers from the locked camera."""
    K = np.array([[500., 0, 424], [0, 500., 240], [0, 0, 1.]])
    KEY = ("DICT_4X4_250", 120)

    def setUp(self):
        self.truth = make_transform((1.3, -.4, .02), matrix_to_quaternion(rpy_to_matrix(0, 0, .7)))
        to_marker = np.eye(4)
        to_marker[:3, :3] = rpy_to_matrix(0, 0, -np.pi / 2)
        self.T_map_marker = self.truth @ to_marker

    def view(self, survey, i, size=.1, distance=1.2, swing_deg=4., now=None):
        """Sighting i from a camera swinging around the marker, as a moving rig does."""
        optical = np.eye(4)
        optical[:3, :3] = rpy_to_matrix(np.pi - .6, np.radians(swing_deg * i), .1 * i)
        optical[:3, 3] = [.05, -.03, distance]
        corners = project_marker(self.K, size, optical)
        T_map_optical = self.T_map_marker @ invert_transform(optical)
        return survey.observe(self.KEY, corners, self.K, size, T_map_optical,
                              now=.2 * i if now is None else now)

    def test_moving_camera_places_marker_at_its_surveyed_pose(self):
        survey = MarkerSurvey(min_sightings=6)
        for i in range(5):
            self.assertTrue(self.view(survey, i))
            self.assertEqual(survey.ready(), [])
        self.view(survey, 5)
        (key, size, T), = survey.ready()
        self.assertEqual((key, size), (self.KEY, .1))
        self.assertLess(np.linalg.norm(T[:3, 3] - self.truth[:3, 3]), .002)
        self.assertLess(rotation_error_deg(T, self.truth), .5)
        self.assertEqual(survey.ready(), [], "a placed marker is handed out once")
        entry = layout_entry(*key, size, T)
        np.testing.assert_allclose(locator_transform(entry), T, atol=1e-9)

    def test_small_marker_in_the_image_is_not_used(self):
        survey = MarkerSurvey(min_marker_px=24)
        self.assertFalse(self.view(survey, 0, size=.04, distance=2.5))
        self.assertEqual(survey.ready(), [])

    def test_two_steady_pnp_branches_wait_for_the_view_to_settle_them(self):
        survey = MarkerSurvey(min_sightings=6, max_age_s=10.)
        flipped = self.truth.copy()
        flipped[:3, :3] = rpy_to_matrix(.5, 0, 0) @ self.truth[:3, :3]
        # Parked camera: both branches agree with themselves every frame.
        survey._tracks[self.KEY] = [(i * .2, .1, [self.truth.copy(), flipped.copy()]) for i in range(12)]
        self.assertEqual(survey.ready(), [])
        # Moving: the wrong branch swings with the viewpoint and stops agreeing.
        swinging = []
        for i in range(12):
            wrong = self.truth.copy()
            wrong[:3, :3] = rpy_to_matrix(.3 + .25 * i, 0, 0) @ self.truth[:3, :3]
            swinging.append((5 + i * .2, .1, [self.truth.copy(), wrong]))
        survey._tracks[self.KEY] = swinging
        (_, _, T), = survey.ready()
        np.testing.assert_allclose(T, self.truth, atol=1e-9)

    def test_disagreeing_sightings_are_never_averaged_into_a_placement(self):
        survey = MarkerSurvey(min_sightings=6, tolerance_m=.03)
        track = []
        for i in range(12):
            T = self.truth.copy()
            T[0, 3] += .1 * (i % 3)          # three places 10 cm apart
            track.append((i * .2, .1, [T]))
        survey._tracks[self.KEY] = track
        self.assertEqual(survey.ready(), [])

    def test_old_sightings_and_a_new_width_start_over(self):
        survey = MarkerSurvey(min_sightings=6, max_age_s=2.)
        for i in range(5):
            self.view(survey, i)
        self.view(survey, 5, now=60.)
        self.assertEqual(survey.ready(), [])
        for i in range(6, 11):
            self.view(survey, i, size=.15, now=60. + .2 * i)
        self.assertEqual(survey.ready(), [], "a different width is a different marker")
        survey.clear()
        self.assertEqual(survey._tracks, {})

    def test_layout_holds_every_id_of_the_250_dictionary(self):
        records = [dict(dictionary="DICT_4X4_250", id=i, size_m=.1,
                        position_m=[i * .2, 0, 0], rotation_deg=[0, 0, 0]) for i in range(MAX_MARKERS)]
        self.assertEqual(len(validate_layout(records)), 250)
        with self.assertRaisesRegex(ValueError, "0 to 249"):
            validate_layout([dict(records[0], id=250)])
        with self.assertRaisesRegex(ValueError, "at most 250"):
            validate_layout(records + [dict(records[0], dictionary="DICT_4X4_1000", id=900)])


class MarkerSimilarityTests(unittest.TestCase):
    """Scale, heading and offset from SLAM marker centres to roughly placed Locators."""
    centres = np.array([[0., 0., 0.], [3., .5, 0.], [1., 2.5, 1.2]])

    def test_heading_scale_and_offset_are_recovered_for_a_tilted_scan(self):
        for up in (np.array([0., 0., 1.]), np.array([.1, -.05, 1.]) / np.linalg.norm([.1, -.05, 1.])):
            with self.subTest(up=up):
                from marker_lineup import _turn_onto
                R = so3_exp(up * .8) @ _turn_onto(np.array([0., 0., 1.]), up)
                targets = .95 * self.centres @ R.T + [1., -2., .3]
                scale, R_fit, t_fit, miss = marker_similarity(self.centres, targets, up)
                self.assertAlmostEqual(scale, .95, places=9)
                np.testing.assert_allclose(R_fit, R, atol=1e-9)
                np.testing.assert_allclose(t_fit, [1., -2., .3], atol=1e-9)
                np.testing.assert_allclose(miss, 0, atol=1e-9)

    def test_a_misplaced_locator_misses_most(self):
        targets = self.centres + [.5, .5, 0.]
        targets[2] += [.6, 0., 0.]
        scale, _, _, miss = marker_similarity(self.centres, targets, [0., 0., 1.])
        # The fit spreads one Locator's 0.6 m over all three; it still misses most.
        self.assertEqual(int(np.argmax(miss)), 2)
        self.assertGreater(miss.min(), .15)

    def test_markers_above_one_another_are_refused(self):
        stacked = np.array([[1., 1., 0.], [1., 1.1, 1.], [1.05, 1., 2.]])
        with self.assertRaisesRegex(ValueError, "further apart across the floor"):
            marker_similarity(stacked, stacked, [0., 0., 1.])


class MarkerSightingsTests(unittest.TestCase):
    def test_scale_requires_depth_even_when_pnp_only_sightings_are_steady(self):
        sightings = MarkerSightings()
        for _ in range(3):
            with patch("marker_map.seen_marker", return_value=([], None, np.array([1., 2., 3.]))):
                sightings.add(Mock(family="DICT_4X4_50", marker_id=7))
        self.assertEqual(len(sightings.centres()), 1)
        self.assertEqual(sightings.centres(require_depth=True), {})

    def test_room_markers_are_found_where_they_are(self):
        obs, T_map_slam, _ = room_fixture(noise_px=.2)
        sightings = MarkerSightings()
        for _ in range(3):
            for marker in obs.markers:
                self.assertTrue(sightings.add(marker))
        centres = sightings.centres()
        self.assertEqual(sorted(key[1] for key in centres), [1, 2, 3])
        for fid, (position, _) in ROOM_MARKERS.items():
            seen = T_map_slam[:3, :3] @ centres[("DICT_4X4_250", fid)] + T_map_slam[:3, 3]
            np.testing.assert_allclose(seen, position, atol=.02)

    def test_depth_backed_sightings_win_and_unsteady_or_rare_ones_do_not_count(self):
        sightings = MarkerSightings(min_sightings=3, max_spread_m=.05)
        script = {1: [((1., 0, 0), True)] * 3 + [((1.5, 0, 0), False)] * 4,   # depth wins
                  2: [((0, 1., 0), True), ((0, 1.2, 0), True), ((0, 1.4, 0), True)],  # unsteady
                  3: [((0, 0, 1.), True)] * 2}                                  # too few
        for fid, seen in script.items():
            for centre, backed in seen:
                with patch("marker_map.seen_marker",
                           return_value=([], object() if backed else None, np.array(centre))):
                    sightings.add(Mock(family="DICT_4X4_50", marker_id=fid))
        self.assertEqual(sightings.counts(), {("DICT_4X4_50", 1): 7, ("DICT_4X4_50", 2): 3,
                                              ("DICT_4X4_50", 3): 2})
        centres = sightings.centres()
        self.assertEqual(list(centres), [("DICT_4X4_50", 1)])
        np.testing.assert_allclose(centres[("DICT_4X4_50", 1)], [1., 0, 0])


class LocatorLevellingTests(unittest.TestCase):
    """Assigned Locators turned to how the camera sees their markers."""

    def truth(self, fid):
        position, rpy = ROOM_MARKERS[fid]
        return make_transform(position, matrix_to_quaternion(rpy_to_matrix(*np.radians(rpy))))

    def turned(self, obs):
        corrections, note = level_locators_from_view(obs)
        return {key[1]: (T, error, on_wall) for key, T, error, on_wall in corrections}, note

    def test_measured_orientations_ignore_all_entered_angles_including_small_errors(self):
        reference = None
        for rotations in ({}, {1: (0, -85, 4), 3: (2, 3, 35)},
                          {1: (37, 110, -147), 2: (-63, 24, 178), 3: (145, -71, 0)}):
            obs, truth, records = room_fixture(rotations, noise_px=.1)
            corrections, note = level_locators_from_view(obs, min_error_deg=0., snap_mounts=False)
            self.assertEqual(len(corrections), 3, note)
            solved = {}
            for key, T, _, _ in corrections:
                fid = key[1]
                solved[fid] = T
                self.assertLess(rotation_error_deg(T, self.truth(fid)), 1.)
                np.testing.assert_array_equal(T[:3, 3], ROOM_MARKERS[fid][0])
                records[fid - 1] = layout_entry(*key, .167, T)
            if reference is not None:
                for fid in solved:
                    np.testing.assert_array_equal(solved[fid], reference[fid])
            reference = solved
            result = solve_mapped_observation(obs.with_layout(records), JointSolveTests.matcher(self))
            self.assertTrue(result.ok, result.reason)
            distance, angle = LidarMapMatcher._camera_delta(truth, result.T_slam_to_map, obs.T_slam_depth)
            self.assertLess(distance, .02)
            self.assertLess(angle, 1.)

    def test_measured_surface_tilt_is_preserved_instead_of_forcing_level_mounts(self):
        obs, _, _ = room_fixture(noise_px=.1)
        tilt = make_transform((.2, .1, -.3), matrix_to_quaternion(rpy_to_matrix(.1, -.16, .25)))
        changed = []
        for m in obs.markers:
            T = tilt @ m.T_map_locator
            T[:3, :3] = np.eye(3)       # only the centre was placed accurately
            changed.append(replace(m, T_map_locator=T))
        corrections, note = level_locators_from_view(MappedObservation(tuple(changed)),
                                                     min_error_deg=0., snap_mounts=False)
        self.assertEqual(len(corrections), 3, note)
        for key, T, _, _ in corrections:
            expected = tilt @ self.truth(key[1])
            np.testing.assert_array_equal(T[:3, 3], expected[:3, 3])
            self.assertLess(rotation_error_deg(T, expected), 1.)

    def test_depth_triangle_handles_tilted_tracking_after_scale_calibration(self):
        obs, _, _ = room_fixture({1: (0, 0, 0), 2: (90, 0, 0)}, noise_px=.15)
        tilt = make_transform((.3, -.2, .1), matrix_to_quaternion(rpy_to_matrix(.15, -.2, .4)))
        obs = MappedObservation(tuple(replace(m, T_slam_image=tilt @ m.T_slam_image,
                                             T_slam_depth=tilt @ m.T_slam_depth) for m in obs.markers))
        turned, note = self.turned(obs)
        self.assertEqual(sorted(turned), [1, 2], note)
        for fid, (T, _, _) in turned.items():
            self.assertLess(rotation_error_deg(T, self.truth(fid)), 2)
            np.testing.assert_array_equal(T[:3, 3], ROOM_MARKERS[fid][0])

    def test_flat_and_sideways_wall_locators_are_stood_up_printed_top_up(self):
        # As on the rig: one duplicated from the floor Locator and left flat,
        # one stood up with Rotation X = 90, so its printed top is sideways.
        for noise in (0., .3):
            with self.subTest(noise_px=noise):
                obs, truth, records = room_fixture({1: (0, 0, 0), 2: (90, 0, 0)}, noise_px=noise)
                self.assertIn("disagree", solve_mapped_observation(obs, JointSolveTests.matcher(self)).reason)
                turned, note = self.turned(obs)
                self.assertEqual(sorted(turned), [1, 2], note)
                for fid in (1, 2):
                    T, error, on_wall = turned[fid]
                    self.assertTrue(on_wall)
                    self.assertAlmostEqual(error, 90, delta=3)
                    self.assertLess(rotation_error_deg(T, self.truth(fid)), 2)
                    np.testing.assert_array_equal(T[:3, 3], ROOM_MARKERS[fid][0])
                for key, T, _, _ in level_locators_from_view(obs)[0]:
                    index = next(i for i, r in enumerate(records) if r["id"] == key[1])
                    records[index] = layout_entry(*key, .167, T)
                result = solve_mapped_observation(obs.with_layout(records), JointSolveTests.matcher(self))
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(len(result.marker_keys), 3)
                distance, angle = LidarMapMatcher._camera_delta(truth, result.T_slam_to_map, obs.markers[0].T_slam_image)
                self.assertLess(distance, .03)
                self.assertLess(angle, 1)

    def test_floor_locator_turns_to_its_printed_top(self):
        turned, _ = self.turned(room_fixture({3: (0, 0, 120)})[0])
        self.assertEqual(list(turned), [3])
        T, error, on_wall = turned[3]
        self.assertFalse(on_wall)
        self.assertAlmostEqual(error, 90, delta=2)
        self.assertLess(rotation_error_deg(T, self.truth(3)), 2)

    def test_locators_already_right_or_slightly_off_are_left_alone(self):
        self.assertEqual(self.turned(room_fixture(noise_px=.3)[0]), ({}, ""))
        self.assertEqual(self.turned(room_fixture({1: (0, -85, 4)})[0])[0], {})

    def test_without_depth_only_unambiguous_markers_turn(self):
        # From about its own height, a wall marker's other PnP branch stands
        # upright too, facing ~55 deg off: without depth it can't be told.
        obs, _, _ = room_fixture({1: (0, 0, 0), 2: (90, 0, 0), 3: (0, 0, 120)}, depth=False)
        turned, note = self.turned(obs)
        self.assertEqual(list(turned), [3])
        self.assertLess(rotation_error_deg(turned[3][0], self.truth(3)), 2)
        self.assertEqual(note, "Could not check the rotation of marker(s) 1, 2: "
                               "keep the wall or floor around them in the depth view.")

    def test_misplaced_locator_is_named_with_where_the_others_put_it(self):
        obs, _, _ = room_fixture({1: (0, 0, 0)}, positions={2: (.8, 2.0, 1.6)})
        display = make_transform((10., 0., 0.), (0, 0, 0, 1))
        lines = []
        corrections, note = level_locators_from_view(obs, T_display_map=display, log=lines.append)
        self.assertEqual(corrections, [])
        self.assertEqual(note, "Marker 2's Locator is 30 cm from where markers 1 and 3 put it: "
                               "X 10.800, Y 2.000, Z 1.300 m. Move it there if their Locators are right.")
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[1].startswith("Marker 2: Locator at X 10.800, Y 2.000, Z 1.600 m; "
                                            "the camera puts it at "), lines[1])
        two = MappedObservation(obs.markers[1:])
        turned, note = self.turned(two)
        self.assertEqual(turned, {})
        self.assertIn("Markers 2 and 3", note)
        self.assertIn("-1.30 m in height, but their Locators are 1.08 m and -1.60 m", note)

    def test_swapped_ids_are_named(self):
        (p1, _), (p2, _) = ROOM_MARKERS[1], ROOM_MARKERS[2]
        obs, _, _ = room_fixture({1: (0, 0, 0)}, positions={1: p2, 2: p1})
        self.assertEqual(self.turned(obs), ({}, "The Locators of markers 1 and 2 look swapped: each marker "
                                                "is where the other's Locator is. Swap their IDs."))

    def test_uniform_scale_error_is_identified_without_blaming_locator_positions(self):
        # Every pair disagrees, as with a scan imported at the wrong scale.
        centre = np.mean([p for p, _ in ROOM_MARKERS.values()], axis=0)
        obs, _, _ = room_fixture(positions={fid: centre + 1.5 * (np.array(p) - centre)
                                            for fid, (p, _) in ROOM_MARKERS.items()})
        _, note = self.turned(obs)
        self.assertIn("scan scale mismatch (1.500x", note)
        self.assertIn("keep their placements on the scan", note)

    def test_one_marker_or_markers_above_one_another_are_not_checked(self):
        obs, _, _ = room_fixture({1: (0, 0, 0)})
        turned, note = self.turned(MappedObservation(obs.markers[:1]))
        self.assertEqual(turned, {})
        self.assertIn("two or more", note)
        stacked = (replace(obs.markers[0], T_map_locator=obs.markers[1].T_map_locator.copy()), obs.markers[1])
        stacked[0].T_map_locator[2, 3] += .5
        turned, note = self.turned(MappedObservation(stacked))
        self.assertEqual(turned, {})
        self.assertIn("further apart", note)


if __name__ == "__main__":
    unittest.main()
