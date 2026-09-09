#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  LEGACY PATH — ROS 1 (Noetic) + VINS-Fusion, headless FreeD bridge
# ═══════════════════════════════════════════════════════════════════
# This path is ROS 1 end to end. It streams 6DOF to Unreal but has NO GUI:
# the HyperFusion dashboard is ROS 2 (rclpy) and subscribes to Isaac ROS
# /visual_slam/* topics, so it cannot run here. It used to be launched from
# this script, which crashed on `import rclpy` every time.
#
#   GUI / current path  -> ./launch_isaac.sh   (ROS 2 + Isaac ROS VSLAM)
#   this script         -> ROS 1 + VINS-Fusion + ros_vins_to_unreal.py
# ═══════════════════════════════════════════════════════════════════

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Automatically setup the conda environment and ROS workspace
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ros_noetic
export DISABLE_ROS1_EOL_WARNINGS=1
source ~/vins_3input_simulation/catkin_ws/devel/setup.bash

# Function to clean up background processes on exit
cleanup() {
    echo ""
    echo "==========================================="
    echo "Shutting down the HyperFusion pipeline..."
    echo "==========================================="
    kill $(jobs -p) 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

echo "==========================================="
echo "1/5 Starting ROS Core..."
echo "==========================================="
roscore &
sleep 3

echo "==========================================="
echo "2/5 Starting RealSense Camera Node..."
echo "==========================================="
roslaunch realsense2_camera rs_camera.launch enable_gyro:=true enable_accel:=true unite_imu_method:=linear_interpolation enable_sync:=true initial_reset:=true enable_infra1:=true enable_infra2:=true enable_color:=false enable_depth:=true infra_width:=640 infra_height:=480 infra_fps:=15 &
sleep 8

echo "==========================================="
echo "3/5 Disabling IR Emitter..."
echo "==========================================="
rosrun dynamic_reconfigure dynparam set /camera/stereo_module emitter_enabled 0

echo "==========================================="
echo "4/5 Starting ZUPT IMU Bias Calibrator (DISABLED for Pure Stereo VO)..."
echo "==========================================="
# To re-enable IMU bias calibration + dual gating, uncomment these lines AND
# point the VINS config at /camera/imu_corrected and /camera/infraN/gated.
# python3 "$DIR/zupt_imu_node.py" &
# echo "[ZUPT] Waiting 3s for calibration to begin..."
# sleep 3

echo "==========================================="
echo "5/5 Starting VINS-Fusion & FreeD Bridge..."
echo "==========================================="
rosrun vins vins_node ~/vins_3input_simulation/catkin_ws/src/vins_fusion/config/realsense_d435i/realsense_stereo_imu_config.yaml __name:=vins_estimator &
sleep 2
python3 "$DIR/ros_vins_to_unreal.py" &

echo ""
echo "✅ PIPELINE IS LIVE — streaming FreeD to Unreal on 127.0.0.1:11111."
echo "   (headless — run ./launch_isaac.sh for the dashboard GUI)"
echo "Press Ctrl+C at any time to shut everything down gracefully."
echo ""

wait
