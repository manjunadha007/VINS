#!/bin/bash
# Cold-start helper: brings the Isaac ROS container up from nothing (via tmux),
# builds VSLAM, then launches the pipeline. Use ./launch_isaac.sh instead once
# the container is already running.
echo "==========================================="
echo "Testing Isaac ROS Pipeline"
echo "==========================================="

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WS="$HOME/workspaces/isaac_ros-dev"

# We need a terminal multiplexer to start the container and run commands inside it.
# We will use tmux.
sudo apt-get install -y tmux

tmux new-session -d -s isaac_ros "cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && ./scripts/run_dev.sh -d ~/workspaces/isaac_ros-dev"
echo "Started Isaac ROS container in tmux session 'isaac_ros'."
echo "Waiting for container to be ready..."

# Wait until the container is running
while ! docker ps | grep -q "isaac_ros_dev-x86_64-container"; do
    sleep 5
    echo "Still waiting for container to start (might be building)..."
done

echo "Container is running! Checking Isaac ROS Visual SLAM is installed..."
# Do NOT build isaac_ros_visual_slam from source here. That build fails in this
# container (isaac_ros_nitros cannot find magic_enum::magic_enum), and because
# the launch used to be chained after it with &&, a failed build silently meant
# the pipeline never started at all. NVIDIA ships a prebuilt package on the
# Isaac ROS apt repo, which is the supported path for 3.x.
if ! docker exec -u admin isaac_ros_dev-x86_64-container /bin/bash -c \
        "test -f /opt/ros/humble/share/isaac_ros_visual_slam/launch/isaac_ros_visual_slam_realsense.launch.py"; then
    echo "❌ isaac_ros_visual_slam is not installed in the container. Install it with:"
    echo "   docker exec -u root isaac_ros_dev-x86_64-container \\"
    echo "     apt-get install -y ros-humble-isaac-ros-visual-slam"
    exit 1
fi

# Deploy first: the launch file we use lives in this repo.
echo "Deploying bridge to $WS/src ..."
for f in dashboard.py freed.py pose_transform.py tracking_config.py verify_tracking.py isaac_vp.launch.py; do
    cp "$DIR/$f" "$WS/src/$f" || exit 1
done
cp "$DIR/dashboard.py" "$WS/src/dashboard_isaac.py"
if [ ! -f "$WS/src/tracking.yaml" ]; then cp "$DIR/tracking.yaml" "$WS/src/tracking.yaml"; fi

echo "Launching SLAM (virtual-production profile: pure VO, AE capped)..."
docker exec -d -u admin -w /workspaces/isaac_ros-dev isaac_ros_dev-x86_64-container /bin/bash -c \
    "source /opt/ros/humble/setup.bash && ros2 launch src/isaac_vp.launch.py > /tmp/vslam.log 2>&1"

echo "Waiting for ROS 2 nodes to initialize..."
sleep 15

# Auto-exposure is capped declaratively inside isaac_vp.launch.py.

echo "Starting HyperFusion Dashboard..."
# NOTE: no -v here — that is a `docker run` flag and makes `docker exec` fail.
# X11 reaches the container through the mount run_dev.sh already set up.
docker exec -d -u admin -e DISPLAY=$DISPLAY \
    -e VP_TRACKING_CONFIG=/workspaces/isaac_ros-dev/src/tracking.yaml \
    -w /workspaces/isaac_ros-dev isaac_ros_dev-x86_64-container \
    /bin/bash -c "source /opt/ros/humble/setup.bash && source install/setup.bash && python3 src/dashboard_isaac.py"

echo "✅ Pipeline is LIVE!"
