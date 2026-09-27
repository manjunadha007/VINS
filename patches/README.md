The librealsense patch is the upstream fix from
[PR #12857](https://github.com/realsenseai/librealsense/pull/12857), merged as
`d2467e6407a4f60f06cef2fb91493fa22e9b9135`. Trailing whitespace in added lines
has been removed. The upstream Apache 2.0 license is included alongside it.

The Isaac container uses librealsense 2.55.1 with the RSUSB backend. That release
predates this fix: its HID reader allocates a 32-byte IMU report, while D400
firmware 5.16 and newer uses a 38-byte report with 32-bit motion axes. Starting
the IMU can then stall capture. The patch handles both report formats and the
gyro sensitivity setting. It does not disable the IMU or change the ROS API.

`ensure_realsense_imu.sh` builds the pinned 2.55.1 source with this patch inside
the Isaac container, saves the original library, and atomically replaces the
same SDK ABI. It verifies the installed library's checksum on subsequent
launches and rebuilds when a fresh container needs the fix. Other SDK versions
are left alone. Run it with camera processes stopped.
