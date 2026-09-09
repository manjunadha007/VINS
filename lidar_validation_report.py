#!/usr/稼n/env python3
"""
lidar_validation_report.py

Standalone tool to visualize and validate LiDAR map matching accuracy.
Usage:
    python3 lidar_validation_report.py --map studio_scan.ply
    python3 lidar_validation_report.py --test
"""

import argparse
import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False

def run_test():
    if not _HAS_O3D:
        print("open3d is required. Please 'pip install open3d'")
        return

    print("Running LiDAR matcher integration test with synthetic data...")
    # Create a synthetic room map
    room_bounds = np.array([[-2, -2, 0], [2, 2, 3]])
    
    # 6 walls
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
    
    # Simulate a camera depth scan slightly offset (at a corner to constrain 3 axes)
    cam_cloud = o3d.geometry.PointCloud()
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
            
    cam_cloud.points = o3d.utility.Vector3dVector(np.array(cam_points))
    
    # Offset it by 5cm
    offset = np.array([0.05, 0.05, 0.05])
    cam_cloud.translate(offset)
    
    # Run ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        cam_cloud, map_cloud, 0.1, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    print(f"Test ICP fitness: {reg_p2p.fitness:.3f}")
    print(f"Test ICP correction: \n{reg_p2p.transformation[:3, 3]}")
    
    if np.linalg.norm(reg_p2p.transformation[:3, 3] - (-offset)) < 0.01:
        print("✅ Synthetic test passed!")
    else:
        print("❌ Synthetic test failed!")
    
def visualize(map_file):
    if not _HAS_O3D:
        print("open3d is required. Please 'pip install open3d'")
        return
        
    print(f"Loading map: {map_file}")
    cloud = o3d.io.read_point_cloud(map_file)
    if not cloud.has_points():
        print("Invalid or empty point cloud.")
        return
        
    print("Visualizing map... (close window to exit)")
    o3d.visualization.draw_geometries([cloud])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run synthetic validation test")
    parser.add_argument("--map", type=str, help="Path to LiDAR map file to view")
    args = parser.parse_args()
    
    if args.test:
        run_test()
    elif args.map:
        visualize(args.map)
    else:
        parser.print_help()
