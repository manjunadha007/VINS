# Tuned Isaac ROS Visual SLAM launch for virtual production camera tracking.
#
# Derived from NVIDIA's isaac_ros_visual_slam_realsense.launch.py (Apache 2.0).
# The substantive change is turning OFF mapping/loop-closure so the pose never
# snaps mid-shot -- see the comment on enable_localization_n_mapping below.
#
# Resolution: 848x480x90.
#
# An earlier note here said 848x480 "does NOT resolve as a stereo Y8 pair on this
# D455 over this USB path", having failed at 848x480x60 and x30. That diagnosis
# was wrong -- almost certainly taken while the camera was negotiated at USB 2,
# where librealsense truncates the profile list severely. Queried on the same
# unit over USB 3.2, the device offers exactly what is needed:
#
#     Infrared 1  848x480  Y8   @ 90/60/30/15/5 Hz
#     Infrared 2  848x480  Y8   @ 90/60/30/15/5 Hz
#     Depth       848x480  Z16  @ 90/60/30/15/5 Hz
#
# Worth the bandwidth: focal length goes 337 -> 446 px and pixel count 1.77x, so
# stereo depth error (which scales as z^2 / f) drops by 1.32x at every range,
# and cuVSLAM gets substantially more texture to track -- which buys rotation
# accuracy, the term that dominates reprojection error in the plate.
#
# If this ever fails again, check the link is USB 3.x first:
#   rs-enumerate-devices | grep -i 'usb type'
#
# OPT-IN — launch_isaac.sh uses NVIDIA's shipped file unless you set:
#
#     VP_SLAM_LAUNCH=vp ./launch_isaac.sh

import os

import launch
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


# Depth is off by default: cuVSLAM tracks from the IR pair and does not use it,
# so it is pure USB bandwidth until something consumes it.
#
#     VP_ENABLE_DEPTH=1 ./launch_isaac.sh
#
# Cost when on is small and worth knowing: the D455 computes depth on its own D4
# ASIC, so the host pays a memcpy, not a stereo match. Two things must stay off
# or that stops being true --  align_depth spawns a resampling node (only needed
# to register depth to colour, which is disabled), and pointcloud deprojects
# every pixel on the CPU.
_ENABLE_DEPTH = os.environ.get('VP_ENABLE_DEPTH', '0').lower() not in (
    '0', '', 'false', 'no')


def generate_launch_description():
    """Visual SLAM configured for a RealSense D455 on a camera rig."""
    realsense_camera_node = Node(
        name='camera',
        # MUST be 'camera': it puts the topics at /camera/infra1/..., /camera/imu.
        # Omitting it publishes at /infra1/..., /imu instead, and every consumer
        # (this bridge, verify_tracking.py, the visual_slam remappings) silently
        # receives nothing while the camera log still says "Node Is Up!".
        namespace='camera',
        package='realsense2_camera',
        executable='realsense2_camera_node',
        parameters=[{
            'enable_infra1': True,
            'enable_infra2': True,
            'enable_color': False,
            'enable_depth': _ENABLE_DEPTH,

            # Both must stay off even with depth on. The D455 already emits its
            # depth map in the infra1 optical frame -- the same frame cuVSLAM
            # uses as camera 0 -- so it is pixel-aligned with the tracking view
            # for free and needs no resampling.
            'align_depth.enable': False,
            'pointcloud.enable': False,
            # The IR projector must stay off: its dot pattern is a static
            # texture locked to the camera, which corrupts feature tracking.
            'depth_module.emitter_enabled': 0,
            'depth_module.profile': '848x480x90',

            # Cap auto-exposure. THE most important accuracy setting measured on
            # this rig. The RealSense default limit is 200 000 us; at 90 fps that
            # smears every frame while the camera moves, cuVSLAM loses visual
            # tracking, falls back to IMU dead reckoning, and the pose diverges to
            # 138 m with impossible 1089 m/s speeds. Capping to 6 ms took peak
            # excursion to 1.53 m and *improved* image contrast and sharpness.
            #
            # Set here declaratively rather than via `ros2 param set` after
            # launch: param-service discovery in this container is unreliable
            # ("Node not found") even while topics flow normally, so a
            # post-launch param set silently does nothing.
            'depth_module.auto_exposure_limit_toggle': True,
            'depth_module.auto_exposure_limit': 6000,
            'depth_module.auto_gain_limit_toggle': True,
            'depth_module.auto_gain_limit': 200,
            'enable_gyro': True,
            'enable_accel': True,
            # 400 Hz, which the D455 offers (rs-enumerate-devices reports
            # Gyro @ 400/200). Denser gyro sampling improves the IMU
            # preintegration cuVSLAM does between camera exposures, which is
            # where rotation accuracy is won during whips and fast pans --
            # exactly the moments visual tracking is weakest. Accel stays at
            # 200; the device offers only 200/100.
            'gyro_fps': 400,
            'accel_fps': 200,
            'unite_imu_method': 2,
            'enable_sync': True,
            'initial_reset': False,
        }],
    )

    visual_slam_node = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'enable_image_denoising': False,
            'rectified_images': True,

            # Visual-inertial fusion. This is what carries tracking through
            # fast whips and low-texture moments; do not turn it off.
            'enable_imu_fusion': True,
            'gyro_noise_density': 0.000244,
            'gyro_random_walk': 0.000019393,
            'accel_noise_density': 0.001862,
            'accel_random_walk': 0.003,
            'calibration_frequency': 200.0,

            'image_jitter_threshold_ms': 22.00,

            # Pure visual odometry: no mapping, no loop closure.
            #
            # With localisation-and-mapping enabled (the default) cuVSLAM can
            # snap the pose when a loop closes. Measured on this rig: a 1254 mm
            # position change between two consecutive 90 Hz frames. Global
            # consistency is the wrong trade for camera tracking -- a shot is
            # seconds to minutes long, and a mid-take jump is unusable while a
            # slow drift is not. Ground zero is re-established from the ArUco
            # marker instead. pose_transform.PoseGate absorbs any snap that does
            # still occur, so this is belt and braces.
            'enable_localization_n_mapping': False,

            # Pose is published for camera_link, which is x-forward / y-left /
            # z-up. pose_transform.py depends on that; if you change base_frame
            # to an optical frame you must revisit the conversion.
            'base_frame': 'camera_link',
            'imu_frame': 'camera_gyro_optical_frame',

            # Visualisation costs GPU time and nothing here consumes it.
            'enable_slam_visualization': False,
            'enable_landmarks_view': False,
            'enable_observations_view': False,

            'camera_optical_frames': [
                'camera_infra1_optical_frame',
                'camera_infra2_optical_frame',
            ],
        }],
        remappings=[
            ('visual_slam/image_0', 'camera/infra1/image_rect_raw'),
            ('visual_slam/camera_info_0', 'camera/infra1/camera_info'),
            ('visual_slam/image_1', 'camera/infra2/image_rect_raw'),
            ('visual_slam/camera_info_1', 'camera/infra2/camera_info'),
            ('visual_slam/imu', 'camera/imu'),
        ],
    )

    visual_slam_launch_container = ComposableNodeContainer(
        name='visual_slam_launch_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[visual_slam_node],
        output='screen',
    )

    return launch.LaunchDescription([visual_slam_launch_container,
                                     realsense_camera_node])
