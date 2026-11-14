import unittest
import numpy as np
import cv2
from lineup_solver import solve_lineup, pick_mesh_surface
from pose_transform import make_transform, T_BODY_OPTICAL, invert_transform, matrix_to_quaternion, rpy_to_matrix


class SurveySolveTests(unittest.TestCase):
    def setUp(self):
        self.K=np.array([[500.,0,424],[0,500,240],[0,0,1]])
        rng=np.random.default_rng(42)
        self.optical=np.column_stack([rng.uniform(-1,1,15),rng.uniform(-.6,.6,15),rng.uniform(2,5,15)])
        self.T=make_transform((1.2,-.3,1.4),matrix_to_quaternion(rpy_to_matrix(.1,-.2,.3)))
        world_optical=self.T @ T_BODY_OPTICAL
        self.world=self.optical @ world_optical[:3,:3].T + world_optical[:3,3]
        projected=self.optical @ self.K.T
        self.image=projected[:,:2]/projected[:,2:3]

    def test_known_camera_is_recovered_in_body_axes(self):
        result=solve_lineup(self.world,self.image,self.K,(848,480))
        self.assertTrue(result.ok,result.reason)
        np.testing.assert_allclose(result.T_map_cam,self.T,atol=1e-5)
        self.assertLess(result.rms_px,1e-5)

    def test_mismatched_pairs_are_identified(self):
        image=self.image.copy()
        image[:2] += (80,35)
        result=solve_lineup(self.world,image,self.K,(848,480))
        self.assertTrue(result.ok,result.reason)
        self.assertNotIn(0,result.inliers)
        self.assertNotIn(1,result.inliers)
        np.testing.assert_allclose(result.T_map_cam,self.T,atol=1e-4)

    def test_noisy_clicks_give_bounded_pose_error(self):
        image=self.image+np.random.default_rng(4).normal(0,.3,self.image.shape)
        result=solve_lineup(self.world,image,self.K,(848,480))
        self.assertTrue(result.ok,result.reason)
        self.assertLess(np.linalg.norm(result.T_map_cam[:3,3]-self.T[:3,3]),.02)
        self.assertLess(result.rms_px,1)

    def test_needs_six_pairs(self):
        self.assertFalse(solve_lineup(self.world[:5],self.image[:5],self.K).ok)

    def test_exactly_six_consistent_pairs_solve(self):
        result=solve_lineup(self.world[:6],self.image[:6],self.K,(848,480))
        self.assertTrue(result.ok,result.reason)
        self.assertEqual(len(result.inliers),6)
        np.testing.assert_allclose(result.T_map_cam,self.T,atol=1e-5)

    def test_six_pairs_with_a_mismatch_explain_rejection(self):
        image=self.image[:6].copy()
        image[0] += (80,35)
        result=solve_lineup(self.world[:6],image,self.K,(848,480))
        self.assertFalse(result.ok)
        self.assertIn("/6 (need 6)",result.reason)
        self.assertIn("all 6 must agree",result.reason)

    def test_planar_points_are_refused(self):
        points=self.world.copy(); points[:,2]=1
        self.assertIn("different depths",solve_lineup(points,self.image,self.K).reason)

    def test_duplicate_points_are_refused(self):
        points=self.world.copy(); points[0]=points[1]
        self.assertFalse(solve_lineup(points,self.image,self.K).ok)

    def test_nan_inputs_are_refused(self):
        image=self.image.copy(); image[0,0]=np.nan
        self.assertFalse(solve_lineup(self.world,image,self.K).ok)

    def test_bad_correspondences_do_not_produce_accepted_pose(self):
        image=self.image[::-1].copy()
        self.assertFalse(solve_lineup(self.world,image,self.K).ok)

    def test_points_outside_the_plate_are_refused(self):
        image=self.image.copy(); image[0,0]=-1
        self.assertFalse(solve_lineup(self.world,image,self.K,(848,480)).ok)


class SurfacePickTests(unittest.TestCase):
    def setUp(self):
        self.vertices=np.array([[-1,-1,2],[1,-1,2],[0,1,2],
                                [-1,-1,4],[1,-1,4],[0,1,4]],float)
        self.faces=np.array([[3,4,5],[0,1,2]])

    def test_nearest_surface_wins_over_hidden_surface(self):
        point=pick_mesh_surface(self.vertices,self.faces,(0,0,0),(0,0,1))
        np.testing.assert_allclose(point,(0,0,2))

    def test_miss_does_not_select_a_nearby_vertex(self):
        self.assertIsNone(pick_mesh_surface(self.vertices,self.faces,(5,5,0),(0,0,1)))

    def test_near_clip_is_respected(self):
        point=pick_mesh_surface(self.vertices,self.faces,(0,0,0),(0,0,1),near=3)
        np.testing.assert_allclose(point,(0,0,4))

    def test_back_facing_triangle_is_pickable(self):
        point=pick_mesh_surface(self.vertices,self.faces[:,::-1],(0,0,0),(0,0,1))
        np.testing.assert_allclose(point,(0,0,2))


if __name__ == "__main__":
    unittest.main()
