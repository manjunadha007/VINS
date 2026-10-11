# Version snapshots

Plain directory copies, not a VCS. Each is self-contained: the Python modules
have no relative-import tricks, so a snapshot can be run and tested in place.

| Version | State | Headline |
|---|---|---|
| [v1.1](v1.1/VERSION.md) | frozen | First fully hardware-validated pipeline. Correct ROS→Unreal conversion, exposure cap, pure visual odometry, snap-absorbing PoseGate. |
| [v1.2](v1.2/VERSION.md) | frozen | Stationary drift eliminated: 27.89 mm → 0.70 mm over 20 s (97.5 %), real motion 100 % preserved. |
| [v1.3](v1.3/VERSION.md) | frozen | Supervised tracking, Camera Lineup Lite, timestamp-paired depth matching. |
| [v1.4](v1.4/VERSION.md) | frozen | Marker-referenced absolute lineup, locator derived from the scan. Dot tracked a moving camera to 10 mm median over 1.13 m. |
| [v1.5](v1.5/VERSION.md) | frozen | Hardware-checked startup: no dashboard until stereo, IMU and SLAM stream; D455 IMU fixed inside the container. Validated live direct and through a 5 Gbps hub. |
| [v1.6](v1.6/VERSION.md) | frozen | Camera keeps moving on stereo/IMU after lineup (amber "Camera tracking"); VO no longer throttled to 1 Hz; scan match gate 95 % → 75 %. Not yet validated live. |
| — | **in development** — repository root | See `../VERSION.md`. |

## Verifying a snapshot

Each one carries its own tests and can be checked without ROS:

    cd versions/v1.2
    python3 -m unittest test_pose_transform test_freed_protocol

v1.2 passes 92 tests this way; v1.4 passes 123 offline plus 57 viewport tests;
v1.5 passes 131 offline plus 63 viewport/marker/channel/startup-check tests;
v1.6 passes 423 unit tests inside the Isaac container.

## Rolling back

    cp versions/v1.2/*.py versions/v1.2/*.sh versions/v1.2/*.yaml .

From v1.5 on, also restore `patches/` (read by `ensure_realsense_imu.sh`):

    cp -r versions/v1.5/patches .   # or versions/v1.6/patches

Then redeploy into the container with `./launch_isaac.sh`, which copies the
bridge modules into `~/workspaces/isaac_ros-dev/src/` on every launch.
