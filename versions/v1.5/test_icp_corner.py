import numpy as np
import open3d as o3d

# Map
points = []
for x in np.linspace(-2, 2, 40):
    for y in np.linspace(-2, 2, 40):
        points.append([x, y, 0]) # floor
        points.append([x, y, 3]) # ceiling
for x in np.linspace(-2, 2, 40):
    for z in np.linspace(0, 3, 30):
        points.append([x, -2, z])
        points.append([x, 2, z])
for y in np.linspace(-2, 2, 40):
    for z in np.linspace(0, 3, 30):
        points.append([-2, y, z])
        points.append([2, y, z])
map_cloud = o3d.geometry.PointCloud()
map_cloud.points = o3d.utility.Vector3dVector(np.array(points))
map_cloud.estimate_normals()

# Camera at corner
cam_points = []
for x in np.linspace(-2, -1, 10):
    for y in np.linspace(1, 2, 10):
        cam_points.append([x, y, 0])
for y in np.linspace(1, 2, 10):
    for z in np.linspace(0, 1, 10):
        cam_points.append([-2, y, z])
for x in np.linspace(-2, -1, 10):
    for z in np.linspace(0, 1, 10):
        cam_points.append([x, 2, z])
cam_cloud = o3d.geometry.PointCloud()
cam_cloud.points = o3d.utility.Vector3dVector(np.array(cam_points))

offset = np.array([0.05, 0.05, 0.05])
cam_cloud.translate(offset)

reg_p2p = o3d.pipelines.registration.registration_icp(
    cam_cloud, map_cloud, 0.2, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

print(reg_p2p.transformation[:3, 3])
