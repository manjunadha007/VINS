#!/bin/bash
# Quick check: is the IMU actually publishing data?
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ros_noetic
source ~/vins_3input_simulation/catkin_ws/devel/setup.bash

echo "========================================="
echo "Checking if /camera/imu is publishing..."
echo "========================================="
timeout 5 rostopic echo /camera/imu -n 3 2>&1

echo ""
echo "========================================="
echo "IMU rate:"
echo "========================================="
timeout 5 rostopic hz /camera/imu --window 100 2>&1
