"""Scan scale measured from a panning depth sweep."""
import time
import unittest
import numpy as np

from scan_scale import (DepthSweep, measure_scale, voxel_down,
                        fit_locator_scale, verify_locator_scale)
from pose_transform import (T_BODY_OPTICAL, invert_transform, make_transform,
                            matrix_to_quaternion, rpy_to_matrix)

ROOM = np.array([4.2, 3.4, 2.6])                       # walls and ceiling, metres
SCANNED = [((2.4, 1.8, 0.), (3.4, 2.6, .75)),          # table
           ((3.6, .2, 0.), (4.2, 1.0, 1.6))]           # cabinet
CLUTTER = [((1.0, 1.9, 0.), (1.4, 2.3, .9))]           # a chair the scan never saw
K = np.array([[110., 0, 80], [0, 110., 60], [0, 0, 1.]])


def pose(position, rpy_deg):
    return make_transform(position, matrix_to_quaternion(rpy_to_matrix(*np.radians(rpy_deg))))


def render(T_world_optical, noise=.01, rng=None):
    """Depth (optical-frame points) of the real room, chair and all, to 5 m."""
    v, u = np.mgrid[0:120, 0:160]
    rays = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(u.shape)), -1).reshape(-1, 3)
    d = rays @ T_world_optical[:3, :3].T
    o = T_world_optical[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        exits = np.where(d > 0, (ROOM - o) / d, -o / d)
        t = np.nanmin(np.where(exits > 0, exits, np.inf), axis=1)
        for lo, hi in SCANNED + CLUTTER:
            a, b = (np.array(lo) - o) / d, (np.array(hi) - o) / d
            near, far = np.minimum(a, b).max(axis=1), np.maximum(a, b).min(axis=1)
            t = np.where((near > 0) & (near < far) & (near < t), near, t)
    z = t * (1 + (rng.normal(0, noise, t.shape) if rng is not None else 0))
    keep = np.isfinite(z) & (z > .3) & (z < 5)
    return rays[keep] * z[keep, None]


def scan_points(scale, T_map_world, top=1.2, step=.02):
    """The room as a phone scan sees it: no ceiling, walls cut at `top`, `scale` off."""
    g = lambda a, b: np.arange(a, b + 1e-9, step)
    faces = []
    for lo, hi in [((0, 0, 0), (ROOM[0], ROOM[1], 0))]:
        x, y = np.meshgrid(g(lo[0], hi[0]), g(lo[1], hi[1]))
        faces.append(np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))))
    for axis, value in ((0, 0.), (0, ROOM[0]), (1, 0.), (1, ROOM[1])):
        other = 1 - axis
        a, z = np.meshgrid(g(0, ROOM[other]), g(0, top))
        p = np.zeros((a.size, 3))
        p[:, axis], p[:, other], p[:, 2] = value, a.ravel(), z.ravel()
        faces.append(p)
    for lo, hi in SCANNED:
        lo, hi = np.array(lo), np.minimum(np.array(hi), [9, 9, top])
        for axis in range(3):
            for value in (lo[axis], hi[axis]):
                rest = [i for i in range(3) if i != axis]
                a, b = np.meshgrid(g(lo[rest[0]], hi[rest[0]]), g(lo[rest[1]], hi[rest[1]]))
                p = np.zeros((a.size, 3))
                p[:, axis], p[:, rest[0]], p[:, rest[1]] = value, a.ravel(), b.ravel()
                faces.append(p)
    world = np.vstack(faces)
    return (scale * world) @ T_map_world[:3, :3].T + T_map_world[:3, 3]


class LocatorScaleTests(unittest.TestCase):
    def test_depth_triangle_recovers_large_scale_errors_in_a_single_corner(self):
        from marker_map import seen_marker
        from test_marker_map import room_fixture, ROOM_MARKERS
        from scan_localizer import depth_to_cloud
        obs, truth, _ = room_fixture(noise_px=.15)
        centres = np.array([seen_marker(m)[2] for m in obs.markers])
        physical = np.array([p for p, _ in ROOM_MARKERS.values()])
        sample = obs.markers[0]
        sweep = DepthSweep()
        sweep.add(depth_to_cloud(sample.depth, sample.depth_K, stride=4),
                  sample.T_slam_depth @ T_BODY_OPTICAL)
        # Independently sampled wall/floor surfaces, not the depth points.
        xy, z = np.arange(0, 2.001, .025), np.arange(0, 2.501, .025)
        a, b = np.meshgrid(xy, xy)
        c, d = np.meshgrid(xy, z)
        room = np.vstack((np.column_stack((a.ravel(), b.ravel(), np.zeros(a.size))),
                          np.column_stack((np.full(c.size, 2.), c.ravel(), d.ravel())),
                          np.column_stack((c.ravel(), np.full(c.size, 2.), d.ravel()))))
        rotation = rpy_to_matrix(.16, -.1, .4)
        for scale in (.01, .5, 1., 1.5, 100.):
            with self.subTest(scale=scale):
                targets = scale * (physical @ rotation.T + [1., -2., .4])
                fit = fit_locator_scale(centres, targets)
                self.assertTrue(fit["ok"], fit["reason"])
                self.assertAlmostEqual(fit["scale"] / scale, 1., delta=.003)
                np.testing.assert_allclose(fit["T"][:3, :3] / fit["scale"],
                                           rotation @ truth[:3, :3], atol=.003)
                result = verify_locator_scale(sweep.points(),
                    scale * (room @ rotation.T + [1., -2., .4]), fit)
                self.assertTrue(result["ok"], result["reason"])
                self.assertGreater(result["inliers"], .85)
                # A different scan must fail even if its marker triangle fits.
                wrong = verify_locator_scale(sweep.points(),
                    scale * (room @ rotation.T + [10., -2., .4]), fit)
                self.assertFalse(wrong["ok"])
                self.assertIn("depth agreement", wrong["reason"])

    def test_bad_correspondences_and_degenerate_triangles_are_refused(self):
        centres = np.array([[0., 0., 0.], [4., .5, .2], [1., 2.5, 1.2]])
        for bad in (centres[:2], centres * np.nan,
                    np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]), centres / 100):
            self.assertFalse(fit_locator_scale(bad, bad)["ok"])
        for targets in (centres * [1.5, 1., 1.], centres + [[0, 0, 0], [0, 0, 0], [.6, 0, 0]],
                        centres[[1, 0, 2]]):
            result = fit_locator_scale(centres, targets)
            self.assertFalse(result["ok"])
            self.assertIn("one uniform scale", result["reason"])

    def test_missing_and_invalid_depth_cannot_verify_scale(self):
        centres = np.array([[0., 0., 0.], [4., .5, .2], [1., 2.5, 1.2]])
        fit = fit_locator_scale(centres, centres)
        for source, target in ((np.empty((0, 3)), centres), (centres, centres * np.nan)):
            result = verify_locator_scale(source, target, fit)
            self.assertFalse(result["ok"])
            self.assertIn("Too few", result["reason"])


class ScanScaleTests(unittest.TestCase):
    # The rig's tracking frame is tilted 6 deg from gravity; the fit must not care.
    T_slam_world = pose((.3, -.2, .1), (4, -4.5, 57))
    T_map_world = pose((-2., -1.5, 0.), (0, 0, 23))

    def sweep(self, yaws, position=(1.6, 1.4, 1.1), pitch=15):
        rng = np.random.default_rng(3)
        sweep = DepthSweep()
        cameras = []
        for yaw in yaws:
            T_world_optical = pose(position, (0, pitch, yaw)) @ T_BODY_OPTICAL
            sweep.add(render(T_world_optical, rng=rng), self.T_slam_world @ T_world_optical)
            cameras.append(self.T_slam_world @ T_world_optical)
        return sweep, cameras

    def seed(self, scale, camera, offset=(.1, -.05, .03), turn_deg=(1, -1, 3)):
        """A lineup's rigid alignment: right where the camera is, off elsewhere."""
        truth = self.T_map_world.copy()
        truth[:3, :3] *= scale
        truth = truth @ invert_transform(self.T_slam_world)
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(*np.radians(turn_deg)) @ truth[:3, :3] / scale
        here = camera[:3, 3]
        T[:3, 3] = truth[:3, :3] @ here + truth[:3, 3] - T[:3, :3] @ here + offset
        return T

    def measure(self, scale, yaws=range(0, 360, 30), **seed):
        sweep, cameras = self.sweep(yaws)
        return measure_scale(sweep.points(), scan_points(scale, self.T_map_world),
                             self.seed(scale, cameras[0], **seed))

    def test_scan_too_small_or_too_large_is_measured(self):
        for scale in (.95, 1.0, 1.06):
            with self.subTest(scale=scale):
                started = time.monotonic()
                result = self.measure(scale)
                self.assertTrue(result["ok"], result["reason"])
                self.assertAlmostEqual(result["scale"], scale, delta=.003)
                # The camera's own 1% depth accuracy dominates the fit's.
                self.assertAlmostEqual(result["sigma"], .01, delta=.001)
                self.assertGreater(result["lever_m"], .5)
                self.assertLess(time.monotonic() - started, 20)

    def test_lineup_off_by_decimetres_and_degrees_still_converges(self):
        result = self.measure(.95, offset=(.2, .15, -.05), turn_deg=(2, -2, 6))
        self.assertTrue(result["ok"], result["reason"])
        self.assertAlmostEqual(result["scale"], .95, delta=.003)

    def test_scan_almost_ten_percent_off_is_reached(self):
        # At scale 1 the far walls start ~20 cm out, beyond the fit's reach;
        # the starts either side of 1 reach them.
        for scale in (.905, 1.095):
            with self.subTest(scale=scale):
                result = self.measure(scale)
                self.assertTrue(result["ok"], result["reason"])
                self.assertAlmostEqual(result["scale"], scale, delta=.003)

    def test_a_markers_similarity_starts_the_fit_without_a_lineup(self):
        # Rough Locators: the markers' fit has the scale 3% out, the heading
        # 3 degrees and the offset 15 cm, about SLAM's origin, not the camera.
        sweep, _ = self.sweep(range(0, 360, 30))
        truth = self.T_map_world.copy()
        truth[:3, :3] *= .95
        truth = truth @ invert_transform(self.T_slam_world)
        seed = np.eye(4)
        seed[:3, :3] = rpy_to_matrix(0, 0, np.radians(3)) @ truth[:3, :3] * (.98 / .95)
        seed[:3, 3] = truth[:3, 3] + (.1, -.1, .05)
        result = measure_scale(sweep.points(), scan_points(.95, self.T_map_world), seed)
        self.assertTrue(result["ok"], result["reason"])
        self.assertAlmostEqual(result["scale"], .95, delta=.003)
        np.testing.assert_allclose(result["T"], truth, atol=.02)

    def test_half_a_turn_is_enough(self):
        result = self.measure(.95, yaws=range(0, 180, 30))
        self.assertTrue(result["ok"], result["reason"])
        self.assertAlmostEqual(result["scale"], .95, delta=.003)

    def test_a_corner_alone_cannot_tell_scale(self):
        # Looking into one corner, as on the rig, even panning across both walls:
        # scale is held rather than left to shrink the cloud, and refused.
        for yaws in ((214, 221, 228), (180, 221, 262)):
            with self.subTest(yaws=yaws):
                result = self.measure(.95, yaws=yaws)
                self.assertFalse(result["ok"])
                self.assertIn("can't measure scale", result["reason"])
                self.assertLess(result["lever_m"], .15)
                self.assertEqual(result["scale"], 1.)
                self.assertGreater(result["inliers"], .7)

    def test_wrong_scan_or_lost_lineup_is_refused(self):
        result = self.measure(.95, offset=(1.5, 1.2, 0.))
        self.assertFalse(result["ok"])
        self.assertTrue("land near the scan" in result["reason"] or "fits the scan" in result["reason"],
                        result["reason"])

    def test_too_few_points(self):
        result = measure_scale(np.zeros((10, 3)), scan_points(1., self.T_map_world), np.eye(4))
        self.assertFalse(result["ok"])
        self.assertIn("pan across more", result["reason"])

    def test_sweep_places_frames_by_their_pose_and_thins_overlap(self):
        sweep = DepthSweep(voxel=.05)
        points = np.array([[0., 0., 1.], [0., .01, 1.], [.5, 0., 2.]])
        T = make_transform((1., 2., 3.), (0, 0, 0, 1))
        self.assertTrue(sweep.add(points, T))
        self.assertTrue(sweep.add(points, T))
        self.assertFalse(sweep.add(np.empty((0, 3)), T))
        self.assertEqual(sweep.frames, 2)
        merged = sweep.points()
        self.assertEqual(len(merged), 2)
        np.testing.assert_allclose(sorted(merged[:, 2]), [4., 5.])
        self.assertEqual(len(voxel_down(np.empty((0, 3)), .05)), 0)


if __name__ == "__main__":
    unittest.main()
