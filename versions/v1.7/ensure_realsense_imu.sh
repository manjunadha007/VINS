#!/bin/bash
# Backport upstream librealsense PR #12857 to the Isaac image's SDK 2.55.1.
# Keeps the SDK ABI and the v1.4 stereo/depth/IMU configuration intact.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${1:-isaac_ros_dev-x86_64-container}"
docker cp "$DIR/patches/librealsense-2.55.1-rsusb-imu.patch" \
    "$CONTAINER:/tmp/hyperfusion-rsusb-imu.patch" >/dev/null

docker exec -i -u root "$CONTAINER" /bin/bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail
LIBRARY=$(ldd /opt/ros/humble/lib/librealsense2_camera.so | \
    awk '$1 ~ /^librealsense2\.so/ && $2 == "=>" {print $3; exit}')
if [[ -z "$LIBRARY" || ! -f "$LIBRARY" ]]; then
    echo "Cannot locate the camera driver's librealsense library."
    exit 1
fi
LIBRARY=$(readlink -f "$LIBRARY")
API_VERSION=$(python3 - "$LIBRARY" <<'PY'
import ctypes, sys
sdk = ctypes.CDLL(sys.argv[1])
error = ctypes.c_void_p()
sdk.rs2_get_api_version.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
sdk.rs2_get_api_version.restype = ctypes.c_int
version = sdk.rs2_get_api_version(ctypes.byref(error))
if error.value:
    raise SystemExit("Cannot read librealsense API version")
print(version)
PY
)
if [[ "$API_VERSION" != 25501 ]]; then
    echo "    librealsense API $API_VERSION: SDK 2.55.1 backport does not apply"
    exit 0
fi

PATCH=/tmp/hyperfusion-rsusb-imu.patch
STATE=/opt/hyperfusion/realsense-rsusb-imu
PATCH_HASH=$(sha256sum "$PATCH" | cut -d ' ' -f 1)
if [[ -f "$STATE/installed.sha256" ]] && \
        [[ "$(head -n 1 "$STATE/installed.sha256")" == "$PATCH_HASH" ]] && \
        tail -n 1 "$STATE/installed.sha256" | sha256sum --check --status; then
    echo "    librealsense 2.55.1: upstream RSUSB IMU fix verified"
    exit 0
fi

SOURCE=/tmp/librealsense-imu-fix-source
BUILD=/tmp/librealsense-imu-fix-build
COMMIT=e196cefa896e312d79c2df400c7623aa1e9c62ac
if [[ ! -d "$SOURCE" ]]; then
    git clone --depth 1 --branch v2.55.1 \
        https://github.com/realsenseai/librealsense.git "$SOURCE"
fi
if [[ "$(git -C "$SOURCE" rev-parse HEAD)" != "$COMMIT" ]]; then
    echo "Unexpected librealsense source revision at $SOURCE; refusing to patch it."
    exit 1
fi
if ! git -C "$SOURCE" apply --reverse --check "$PATCH" 2>/dev/null; then
    git -C "$SOURCE" apply --check "$PATCH"
    git -C "$SOURCE" apply "$PATCH"
fi

echo "    Building librealsense 2.55.1 with upstream RSUSB IMU support..."
cmake -S "$SOURCE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DFORCE_RSUSB_BACKEND=ON -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_PYTHON_BINDINGS=OFF \
    -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_DDS=OFF -DBUILD_UNIT_TESTS=OFF \
    -DIMPORT_DEPTH_CAM_FW=OFF
cmake --build "$BUILD" --target realsense2 --parallel 8

# Keep the original binary for rollback. Replace atomically so a failed build
# or interrupted copy cannot leave a truncated library in the camera driver.
mkdir -p "$STATE"
if [[ ! -f "$STATE/librealsense2.so.2.55.1.original" ]]; then
    cp -p "$LIBRARY" "$STATE/librealsense2.so.2.55.1.original"
fi
install -m 755 "$BUILD/Release/librealsense2.so.2.55.1" "$LIBRARY.hyperfusion-new"
mv -f "$LIBRARY.hyperfusion-new" "$LIBRARY"
{
    echo "$PATCH_HASH"
    sha256sum "$LIBRARY"
} > "$STATE/installed.sha256"
echo "    Installed upstream RSUSB IMU fix; original SDK saved in $STATE"
CONTAINER_SCRIPT
