#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Virtual Production Camera Tracking — ROS 2 + Isaac ROS Visual SLAM
#
#  RealSense D455 -> Isaac ROS VSLAM -> FreeD -> Unreal Engine
#
#  Assumes the Isaac ROS container is already running (start it with
#  ./test_new_pipeline.sh, or manually via isaac_ros_common/scripts/run_dev.sh).
#
#  Uses isaac_vp.launch.py by default: pure visual odometry (no loop-closure
#  snapping) with auto-exposure capped declaratively. Measured on this rig:
#  stationary drift 8.5 mm / 12 s, and unlike the stock config the pose is not
#  frozen by a zero-motion lock.
#
#  Set VP_SLAM_LAUNCH=nvidia to fall back to NVIDIA's shipped launch file.
# ═══════════════════════════════════════════════════════════════════
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WS="$HOME/workspaces/isaac_ros-dev"
CONTAINER="isaac_ros_dev-x86_64-container"

echo "==========================================="
echo "Virtual Production Camera Tracking"
echo "==========================================="

# ── Pre-flight ────────────────────────────────────────────────────────────────
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "❌ Container '${CONTAINER}' is not running."
    echo "   Start it first:"
    echo "     cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && ./scripts/run_dev.sh"
    echo "   or run ./test_new_pipeline.sh for a cold start."
    exit 1
fi

if [ ! -d "$WS/src" ]; then
    echo "❌ $WS/src not found. Run ./setup_isaac_ros.sh first."
    exit 1
fi

# Ensure we have access to the X11 display
xhost +local:root >/dev/null 2>&1

cleanup() {
    echo ""
    echo "Dashboard closed. Stopping SLAM node..."
    docker exec -u admin "$CONTAINER" /bin/bash -c \
        "pkill -f 'ros2 launch' || true; \
         pkill -9 -f 'component_container' || true; \
         pkill -9 -f 'realsense2_camera' || true"
    echo "✅ Pipeline shut down."
}
trap cleanup EXIT INT TERM

# ── 1/3 Deploy ────────────────────────────────────────────────────────────────
# $WS is bind-mounted into the container at /workspaces/isaac_ros-dev. The copy
# that lives there used to be hand-edited separately from this repo, so the two
# silently diverged; it is now generated on every launch. All of these modules
# must land in the same directory — the dashboard imports them by name.
echo "1/3 Deploying bridge to $WS/src ..."
for f in dashboard.py freed.py pose_transform.py tracking_config.py verify_tracking.py; do
    cp "$DIR/$f" "$WS/src/$f" || exit 1
done
# The dashboard is launched as dashboard_isaac.py for backwards compatibility.
cp "$DIR/dashboard.py" "$WS/src/dashboard_isaac.py"

# Operator config: do NOT clobber an edited copy inside the workspace.
if [ ! -f "$WS/src/tracking.yaml" ]; then
    cp "$DIR/tracking.yaml" "$WS/src/tracking.yaml"
    echo "    installed tracking.yaml (edit $WS/src/tracking.yaml to configure)"
else
    echo "    keeping existing $WS/src/tracking.yaml"
fi

cp "$DIR/isaac_vp.launch.py" "$WS/src/isaac_vp.launch.py"

# ── 2/3 SLAM ──────────────────────────────────────────────────────────────────
if [ "${VP_SLAM_LAUNCH:-vp}" = "nvidia" ]; then
    LAUNCH_CMD="ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense.launch.py"
    echo "2/3 Starting SLAM node (NVIDIA stock profile)..."
else
    LAUNCH_CMD="ros2 launch /workspaces/isaac_ros-dev/src/isaac_vp.launch.py"
    echo "2/3 Starting SLAM node (virtual-production profile: pure VO, AE capped)..."
fi

docker exec -u admin -d -w /workspaces/isaac_ros-dev "$CONTAINER" /bin/bash -c \
    "source /opt/ros/humble/setup.bash && source install/setup.bash && \
     export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$(find /workspaces/isaac_ros-dev/install -type d -name lib | grep gxf | tr '\n' ':') && \
     $LAUNCH_CMD"

echo "    waiting for SLAM to come up..."
sleep 15

# Cap auto-exposure. The RealSense default AE limit is 200 ms; at 90 fps that
# smears every frame while the rig moves, cuVSLAM loses visual tracking, falls
# back to IMU dead reckoning, and the pose diverges by tens of metres. Measured
# on this rig: capping to 6 ms improved BOTH contrast and sharpness with no
# meaningful brightness loss. Override with VP_AE_LIMIT_US if your set is darker.
if [ "${VP_SLAM_LAUNCH:-vp}" = "nvidia" ]; then
echo "    capping auto-exposure (AE limit ${VP_AE_LIMIT_US:-6000} us)..."
docker exec -u admin "$CONTAINER" /bin/bash -c \
    "source /opt/ros/humble/setup.bash && \
     ros2 param set /camera/camera depth_module.auto_exposure_limit_toggle true && \
     ros2 param set /camera/camera depth_module.auto_exposure_limit ${VP_AE_LIMIT_US:-6000} && \
     ros2 param set /camera/camera depth_module.auto_gain_limit_toggle true && \
     ros2 param set /camera/camera depth_module.auto_gain_limit ${VP_GAIN_LIMIT:-200}" \
    >/dev/null 2>&1 || echo "    ⚠️  could not set exposure limits"
fi

# ── 3/3 Bridge + dashboard ────────────────────────────────────────────────────
# VP_TRACKING_CONFIG points the bridge at the workspace copy of tracking.yaml so
# the operator edits one file in one place.
echo "3/3 Starting HyperFusion dashboard..."
docker exec -u admin -e DISPLAY="$DISPLAY" \
    -e VP_TRACKING_CONFIG=/workspaces/isaac_ros-dev/src/tracking.yaml \
    -w /workspaces/isaac_ros-dev "$CONTAINER" /bin/bash -c \
    "source /opt/ros/humble/setup.bash && source install/setup.bash && \
     python3 src/dashboard_isaac.py"
