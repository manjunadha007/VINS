#!/usr/bin/env python3
"""Depth-coverage preview: synthetic rooms with a known number of constrained axes.

    python3 -m unittest test_depth_view -v
"""
import unittest

import numpy as np

from depth_view import AMBER, GREEN, RED, depth_coverage, draw_depth_points, verdict

W, H = 848, 480
K = np.array([[430.0, 0, 424.0], [0, 430.0, 240.0], [0, 0, 1]])


def render(planes, noise_px=0.0, seed=0):
    """Depth (mm) of the nearest plane n.X = d along each pixel ray, optical frame."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    depth = np.full((H, W), np.inf)
    for normal, offset in planes:
        denom = rays @ np.asarray(normal, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = offset / denom
        t[(denom <= 1e-9) | (t <= 0)] = np.inf
        depth = np.minimum(depth, t)            # ray z component is 1, so t is depth
    if noise_px:
        # Stereo error grows with z^2: sigma_z = z^2 * sigma_d / (f * b), D455 b = 95 mm.
        rng = np.random.default_rng(seed)
        depth = depth + rng.normal(0, 1, depth.shape) * depth ** 2 * noise_px / (K[0, 0] * 0.095)
    depth[~np.isfinite(depth)] = 0
    return np.clip(depth * 1000, 0, 65535).astype(np.uint16)


FLOOR = ((0, 1, 0), 1.2)        # optical y points down; camera 1.2 m above the floor
BACK_WALL = ((0, 0, 1), 3.0)
SIDE_WALL = ((1, 0, 0), 1.0)


class DepthCoverageTests(unittest.TestCase):
    def test_room_corner_constrains_all_three_axes(self):
        cov = depth_coverage(render([FLOOR, BACK_WALL, SIDE_WALL]), K)
        self.assertEqual(cov.rank, 3)
        self.assertGreater(cov.coverage, 0.95)
        self.assertAlmostEqual(float(cov.eigvals.sum()), 1.0, places=5)

    def test_floor_and_one_wall_leave_an_axis_free(self):
        cov = depth_coverage(render([FLOOR, BACK_WALL]), K)
        self.assertEqual(cov.rank, 2)

    def test_single_wall_constrains_one_axis(self):
        cov = depth_coverage(render([((0, 0, 1), 2.0)]), K)
        self.assertEqual(cov.rank, 1)

    def test_stereo_noise_does_not_fake_extra_axes(self):
        # 0.3 px disparity noise is several times the D455's nominal figure.
        for planes, rank in (([((0, 0, 1), 2.0)], 1), ([FLOOR, BACK_WALL], 2)):
            cov = depth_coverage(render(planes, noise_px=0.3), K)
            self.assertEqual(cov.rank, rank, planes)

    def test_no_depth_is_reported_not_crashed(self):
        cov = depth_coverage(np.zeros((H, W), np.uint16), K)
        self.assertEqual(cov.coverage, 0.0)
        self.assertEqual(cov.rank, 0)
        self.assertEqual(len(cov.u), 0)
        self.assertIs(verdict(cov)[0], RED)

    def test_depth_outside_the_matcher_band_is_excluded(self):
        far = render([((0, 0, 1), 8.0)])          # past max_range_m
        self.assertEqual(depth_coverage(far, K).coverage, 0.0)
        near = render([((0, 0, 1), 0.2)])         # inside the D455 blind zone
        self.assertEqual(depth_coverage(near, K).coverage, 0.0)

    def test_partial_coverage(self):
        depth = render([FLOOR, BACK_WALL, SIDE_WALL])
        depth[:, : W // 2] = 0                     # texture-less left half
        cov = depth_coverage(depth, K)
        self.assertAlmostEqual(cov.coverage, 0.5, delta=0.02)
        self.assertTrue(np.all(cov.u >= W // 2))

    def test_unusable_inputs(self):
        self.assertIsNone(depth_coverage(np.zeros(5), K))
        self.assertIsNone(depth_coverage(np.zeros((H, W)), np.zeros((3, 3))))


class VerdictTests(unittest.TestCase):
    def test_levels(self):
        corner = depth_coverage(render([FLOOR, BACK_WALL, SIDE_WALL]), K)
        self.assertIs(verdict(corner)[0], GREEN)
        flat = depth_coverage(render([FLOOR, BACK_WALL]), K)
        self.assertIs(verdict(flat)[0], AMBER)
        self.assertIn("corner", verdict(flat)[2])
        self.assertIs(verdict(None)[0], RED)


class DrawTests(unittest.TestCase):
    def test_dots_only_where_depth_exists_and_size_is_kept(self):
        depth = render([FLOOR, BACK_WALL, SIDE_WALL])
        depth[:, : W // 2] = 0
        cov = depth_coverage(depth, K)
        image = np.full((H, W, 3), 60, np.uint8)
        out = draw_depth_points(image, cov)
        self.assertEqual(out.shape, image.shape)
        self.assertTrue(np.array_equal(image, np.full((H, W, 3), 60, np.uint8)), "input untouched")
        body = slice(0, H - 60)
        changed = np.any(out[body] != 60, axis=2)
        self.assertFalse(changed[:, : W // 2 - 8].any(), "no dots without depth")
        self.assertTrue(changed[:, W // 2 + 8:].any())

    def test_display_image_of_another_size(self):
        cov = depth_coverage(render([FLOOR, BACK_WALL, SIDE_WALL]), K)
        out = draw_depth_points(np.zeros((240, 424, 3), np.uint8), cov)
        self.assertEqual(out.shape, (240, 424, 3))
        self.assertTrue(np.any(out[:150] != 0))

    def test_no_frame_still_draws_the_status(self):
        out = draw_depth_points(np.zeros((H, W, 3), np.uint8), None)
        self.assertTrue(np.any(out[H - 50:] != 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
