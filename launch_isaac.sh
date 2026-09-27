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

    # The ROS package install and the numpy<2 pin used to live here, ten
    # seconds after the container starts and silenced with >/dev/null 2>&1.
    # Both are now guards further down: they run on every launch, they run
    # late enough that the container's network is actually up, and they say
    # what they are doing instead of failing invisibly.
    
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
for f in marker_lineup.py lineup_solver.py lineup_dialog.py fusion_supervisor.py gnss_receiver.py tracking_preflight.py check_pipeline.py dashboard.py freed.py pose_transform.py tracking_config.py verify_tracking.py map_viewer.py bridge_headless.py depth_probe.py lidar_map_matcher.py image_alignment.py tracking_engine.py anchor_cli.py bench_fusion.py smoke_locator_lock.py smoke_lineup.py smoke_gizmo.py test_lidar_fusion.py test_image_alignment.py verify_region.py; do
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

# ── NumPy ABI guard ───────────────────────────────────────────────────────────
# The container image ships NumPy 2.x in /usr/local/lib/python3.10/dist-packages,
# but ROS Humble's cv_bridge and cv2 are compiled against the NumPy 1.x C ABI.
# With NumPy 2 ahead of them on sys.path they fail at import with
# "_ARRAY_API not found" -> "numpy.core.multiarray failed to import". That takes
# out map_viewer, lineup_dialog and bridge_headless, so the dashboard dies
# before its window ever opens -- while SLAM comes up fine, which makes it look
# like the pipeline "half booted".
#
# This pin used to sit in the cold-start branch above, silenced with
# >/dev/null 2>&1. Two ways that failed: a warm start (container already up)
# skipped it entirely, and on a cold start pip can run before the container has
# working network, in which case nothing reported it. It now runs on every
# launch, says what it is doing, and the imports are verified before anything
# that needs them is started.
#
# Keep it at numpy<2. cupy-cuda12x in this image wants numpy>=2.0 and pip will
# say so -- that warning is expected and harmless here, nothing in this project
# imports cupy, whereas the dashboard cannot start without cv_bridge and cv2.
echo "    checking NumPy ABI (cv_bridge/cv2 need numpy<2)..."
if ! docker exec -u admin "$CONTAINER" \
        python3 -c 'import numpy,sys; sys.exit(0 if numpy.__version__.split(".")[0] == "1" else 1)' 2>/dev/null; then
    echo "    NumPy 2.x is in front -- pinning to numpy<2 (takes a few seconds)..."
    docker exec -u admin "$CONTAINER" python3 -m pip install -q 'numpy<2' \
        || echo "    ⚠️  numpy<2 install reported an error"
fi
if ! docker exec -u admin "$CONTAINER" /bin/bash -c \
        "source /opt/ros/humble/setup.bash && python3 -c 'import numpy, cv2, cv_bridge'" 2>/dev/null; then
    echo "❌ cv_bridge/cv2 cannot be imported in the container, so the dashboard will not start."
    echo "   Reproduce the error with:"
    echo "     docker exec -u admin $CONTAINER /bin/bash -c 'source /opt/ros/humble/setup.bash && python3 -c \"import cv2, cv_bridge\"'"
    exit 1
fi
echo "    NumPy $(docker exec -u admin "$CONTAINER" python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null) -- cv_bridge/cv2 import OK"

# ── cuVSLAM component guard ───────────────────────────────────────────────
# The container is ephemeral, so ros-humble-isaac-ros-visual-slam has to be
# reinstalled into every fresh one. That install used to run once during cold
# start, silenced -- and when it failed nothing said so. The failure then hid
# for the rest of the session: the launch below still starts a
# component_container, but with no cuVSLAM component to load into it, so no
# /visual_slam node appears and /visual_slam/tracking/odometry never
# publishes. Everything else looks healthy -- the camera streams at 90 Hz, the
# dashboard opens, the scan loads -- while there is simply no camera pose, so
# tracking never validates and auto lineup waits forever for a fresh pose it
# cannot get. Observed here: the whole NITROS/GXF chain was missing too.
#
# Checked on every launch: a warm container that came up without it stays
# broken until something notices, and only this check ever will.
echo "    checking Isaac ROS cuVSLAM component..."
vslam_present() {
    docker exec -u admin "$CONTAINER" /bin/bash -c \
        "source /opt/ros/humble/setup.bash && ros2 component types 2>/dev/null | grep -q isaac_ros_visual_slam"
}
if ! vslam_present; then
    echo "    cuVSLAM component missing -- installing ros-humble-isaac-ros-visual-slam..."
    docker exec -u root "$CONTAINER" /bin/bash -c \
        "DEBIAN_FRONTEND=noninteractive apt-get update && \
         DEBIAN_FRONTEND=noninteractive apt-get install -y ros-humble-isaac-ros-visual-slam" \
        2>&1 | tail -5
fi
if ! vslam_present; then
    echo "❌ The cuVSLAM component is not in the container, so there will be no camera"
    echo "   pose at all: tracking cannot validate and auto lineup can never lock."
    echo "   Check the container's network, then install it by hand:"
    echo "     docker exec -u root $CONTAINER apt-get install -y ros-humble-isaac-ros-visual-slam"
    exit 1
fi
echo "    cuVSLAM component present"

# SDK 2.55.1's RSUSB IMU reader predates the firmware's 38-byte report format.
# Apply the upstream fix; disabling the IMU would break visual-inertial SLAM.
bash "$DIR/ensure_realsense_imu.sh" "$CONTAINER" || exit 1

# -e VP_ENABLE_DEPTH is required, not decorative: the launch file reads it with
# os.environ inside the container, and docker exec forwards nothing by default.
# Without this the variable is set on the host, silently never arrives, and
# depth stays off however you invoke the script.
docker exec -u admin -d -w /workspaces/isaac_ros-dev \
    -e VP_ENABLE_DEPTH="${VP_ENABLE_DEPTH:-0}" \
    "$CONTAINER" /bin/bash -c \
    "source /opt/ros/humble/setup.bash && source install/setup.bash && \
     export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$(find /workspaces/isaac_ros-dev/install -type d -name lib | grep gxf | tr '\n' ':') && \
     $LAUNCH_CMD > /tmp/hyperfusion-slam.log 2>&1"

DEPTH_CHECK_ARGS=()
case "${VP_ENABLE_DEPTH:-0}" in
    0|false|no|'') ;;
    *) DEPTH_CHECK_ARGS=(--require-depth) ;;
esac
if ! docker exec -u admin \
        -e VP_TRACKING_CONFIG=/workspaces/isaac_ros-dev/src/tracking.yaml \
        -w /workspaces/isaac_ros-dev "$CONTAINER" /bin/bash -c \
        'source /opt/ros/humble/setup.bash && source install/setup.bash &&
         python3 src/check_pipeline.py "$@"' pipeline-check "${DEPTH_CHECK_ARGS[@]}"; then
    echo "❌ Stereo / IMU / SLAM startup check failed. Recent camera/SLAM logs:"
    docker exec -u admin "$CONTAINER" tail -n 30 /tmp/hyperfusion-slam.log
    exit 1
fi

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
