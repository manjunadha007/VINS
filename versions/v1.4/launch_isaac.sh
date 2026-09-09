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

# Parse flags
HEADLESS=false
for arg in "$@"; do
    case "$arg" in
        --headless) HEADLESS=true ;;
    esac
done

echo "==========================================="
echo "Virtual Production Camera Tracking"
echo "==========================================="

# ── Pre-flight ────────────────────────────────────────────────────────────────
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "⚠️  Container '${CONTAINER}' is not running. Performing cold start..."
    
    if ! command -v tmux &> /dev/null; then
        echo "   Installing tmux (required for background container)..."
        sudo apt-get update && sudo apt-get install -y tmux
    fi

    # Kill any stale tmux session from a previous run
    tmux kill-session -t isaac_ros 2>/dev/null || true

    tmux new-session -d -s isaac_ros "cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && ./scripts/run_dev.sh -d ~/workspaces/isaac_ros-dev"
    echo "   Started Isaac ROS container in tmux session 'isaac_ros'."
    echo -n "   Waiting for container to be ready"

    while ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; do
        sleep 3
        echo -n "."
    done
    echo ""

    # Give the container a moment to finish its entrypoint before running exec
    echo "   Container is up! Waiting 10s for initialization..."
    sleep 10

    echo "   Installing required ROS packages & fixing NumPy..."
    # The container loses apt installs on fresh boot unless committed
    docker exec -u root "$CONTAINER" /bin/bash -c "apt-get update && apt-get install -y ros-humble-isaac-ros-visual-slam" >/dev/null 2>&1
    docker exec -u admin "$CONTAINER" /bin/bash -c "python3 -m pip install 'numpy<2'" >/dev/null 2>&1
    
    echo "✅ Cold start complete!"
fi

if [ ! -d "$WS/src" ]; then
    echo "❌ $WS/src not found. Run ./setup_isaac_ros.sh first."
    exit 1
fi

# ── Stop anything left from a previous run ────────────────────────────────────
# Without this, instances stack. Observed on this machine: three bridge_headless
# processes and three SLAM containers running at once, all streaming FreeD to
# Unreal from independent pose state. The cleanup trap below only fires when
# *this* script exits, so it cannot help with an earlier run that was killed or
# backgrounded.
#
# Killed by explicit PID rather than pkill, which did not reliably reap the
# component_container processes here.
echo "0/3 Stopping any previous run..."
STALE=$(docker exec -u root "$CONTAINER" /bin/bash -c \
    "ps -eo pid,cmd --no-headers | grep -E 'bridge_headless|dashboard_isaac|component_container|realsense2_camera_node' | grep -v grep | awk '{print \$1}'" 2>/dev/null | tr '\n' ' ')
if [ -n "${STALE// /}" ]; then
    echo "    killing: $STALE"
    docker exec -u root "$CONTAINER" /bin/bash -c "kill -9 $STALE" >/dev/null 2>&1 || true
    sleep 5
else
    echo "    nothing to stop"
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
for f in marker_lineup.py lineup_solver.py lineup_dialog.py fusion_supervisor.py gnss_receiver.py tracking_preflight.py dashboard.py freed.py pose_transform.py tracking_config.py verify_tracking.py map_viewer.py bridge_headless.py depth_probe.py lidar_map_matcher.py image_alignment.py tracking_engine.py anchor_cli.py bench_fusion.py smoke_locator_lock.py smoke_lineup.py smoke_gizmo.py test_lidar_fusion.py test_image_alignment.py verify_region.py; do
    cp "$DIR/$f" "$WS/src/$f" || exit 1
done
# Install the bundled scan beside the relative-path configuration.
if [ ! -f "$WS/src/Room01_Scaled_UE.obj" ]; then
    cp "$DIR/Room01_Scaled_UE.obj" "$WS/src/Room01_Scaled_UE.obj" || exit 1
fi
# The dashboard is launched as dashboard_isaac.py for backwards compatibility.
cp "$DIR/dashboard.py" "$WS/src/dashboard_isaac.py"

# Operator config: do NOT clobber an edited copy inside the workspace.
if [ ! -f "$WS/src/tracking.yaml" ]; then
    cp "$DIR/tracking.yaml" "$WS/src/tracking.yaml"
    echo "    installed tracking.yaml (edit $WS/src/tracking.yaml to configure)"
else
    echo "    keeping existing $WS/src/tracking.yaml; review fusion settings in the project config to enable supervised output"
fi

cp "$DIR/isaac_vp.launch.py" "$WS/src/isaac_vp.launch.py"

# Auto-enable depth if LiDAR map fusion is on.
#
# This used to be `grep -q 'lidar.*enabled.*[tT]rue'`, which requires "lidar",
# "enabled" and "true" to appear on ONE line. In YAML they never do -- `lidar:`
# is the block header and `enabled: true` is an indented child -- so the test
# never fired, VP_ENABLE_DEPTH stayed 0, isaac_vp.launch.py started the camera
# with enable_depth:=False, and the D455 published no depth topic at all. With
# no depth frames the map fusion has nothing to match against and silently does
# nothing, whatever the config says.
#
# Block-scoped instead: find the `lidar:` section, then its own `enabled:` key,
# ignoring comments.
lidar_fusion_enabled() {
    awk '
        /^[^[:space:]#]/          { in_lidar = ($0 ~ /^lidar:[[:space:]]*$/) }
        in_lidar && /^[[:space:]]*#/ { next }
        in_lidar && /^[[:space:]]+enabled:[[:space:]]*([tT]rue|[yY]es|on)([[:space:]]|#|$)/ { found = 1 }
        END                       { exit !found }
    ' "$1" 2>/dev/null
}

if lidar_fusion_enabled "$WS/src/tracking.yaml"; then
    echo "    LiDAR map fusion enabled in config -> VP_ENABLE_DEPTH=1"
    export VP_ENABLE_DEPTH=1
    echo "    ensuring open3d is installed in container..."
    docker exec -u admin "$CONTAINER" /bin/bash -c "python3 -c 'import open3d' 2>/dev/null || python3 -m pip install open3d" >/dev/null 2>&1
else
    echo "    LiDAR map fusion disabled in config -> depth stream stays off"
fi

# ── 2/3 SLAM ──────────────────────────────────────────────────────────────────
if [ "${VP_SLAM_LAUNCH:-vp}" = "nvidia" ]; then
    LAUNCH_CMD="ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense.launch.py"
    echo "2/3 Starting SLAM node (NVIDIA stock profile)..."
else
    LAUNCH_CMD="ros2 launch /workspaces/isaac_ros-dev/src/isaac_vp.launch.py"
    echo "2/3 Starting SLAM node (virtual-production profile: pure VO, AE capped)..."
fi

# Clear stale ROS 2 FastDDS shared memory locks that cause "Failed init_port" errors after unclean shutdowns
docker exec -u admin "$CONTAINER" /bin/bash -c "rm -f /dev/shm/fastrtps*" >/dev/null 2>&1

# Ensure Open3D is installed for LiDAR map matching / .obj import
echo "    ensuring Open3D is installed..."
docker exec -u admin "$CONTAINER" pip3 install -q "open3d>=0.18" 2>/dev/null || echo "    ⚠️  open3d install failed (non-fatal)"

# Ensure PyOpenGL is installed -- map_viewer.py's 3D viewport renders the
# imported mesh with real OpenGL (depth-buffered) instead of QPainter, so
# this import is required, not optional; without it dashboard_isaac.py
# fails at import time with ModuleNotFoundError before the window even opens.
echo "    ensuring PyOpenGL is installed..."
docker exec -u admin "$CONTAINER" pip3 install -q PyOpenGL 2>/dev/null || echo "    ⚠️  PyOpenGL install failed (non-fatal)"

# -e VP_ENABLE_DEPTH is required, not decorative: the launch file reads it with
# os.environ inside the container, and docker exec forwards nothing by default.
# Without this the variable is set on the host, silently never arrives, and
# depth stays off however you invoke the script.
docker exec -u admin -d -w /workspaces/isaac_ros-dev \
    -e VP_ENABLE_DEPTH="${VP_ENABLE_DEPTH:-0}" \
    "$CONTAINER" /bin/bash -c \
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
if [ "$HEADLESS" = true ]; then
    echo "3/3 Starting HyperFusion headless bridge (Unreal plugin mode)..."
    docker exec -u admin \
        -e VP_TRACKING_CONFIG=/workspaces/isaac_ros-dev/src/tracking.yaml \
        -w /workspaces/isaac_ros-dev "$CONTAINER" /bin/bash -c \
        "source /opt/ros/humble/setup.bash && source install/setup.bash && \
         python3 src/bridge_headless.py"
else
    if [ -z "${DISPLAY:-}" ]; then
        echo "❌ DISPLAY is not set, so the dashboard has no screen to open on."
        echo "   Either run from a graphical session, or use:  $0 --headless"
        exit 1
    fi
    echo "3/3 Starting HyperFusion dashboard (DISPLAY=$DISPLAY)..."
    docker exec -u admin -e DISPLAY="$DISPLAY" \
        -e VP_TRACKING_CONFIG=/workspaces/isaac_ros-dev/src/tracking.yaml \
        -w /workspaces/isaac_ros-dev "$CONTAINER" /bin/bash -c \
        "source /opt/ros/humble/setup.bash && source install/setup.bash && \
         python3 src/dashboard_isaac.py"
fi
