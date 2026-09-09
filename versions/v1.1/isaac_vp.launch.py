# Tuned Isaac ROS Visual SLAM launch for virtual production camera tracking.
#
# Derived from NVIDIA's isaac_ros_visual_slam_realsense.launch.py (Apache 2.0).
# The substantive change is turning OFF mapping/loop-closure so the pose never
# snaps mid-shot -- see the comment on enable_localization_n_mapping below.
#
# Resolution note, measured on this rig: 848x480 does NOT resolve as a stereo Y8
# pair on this D455 over this USB path. Both 848x480x60 and 848x480x30 fail with
#
#     Failed to resolve the request: Format: Y8, width: 848, height: 480
#
# (at that resolution the pair appears to be offered only as interleaved Y8I).
# 640x360x90 resolves and is what NVIDIA's own launch uses, so that is kept here.
# More pixels would help accuracy, so if you want to try it again: use a direct
# USB 3.x port rather than going through hubs, and re-test with
#   ros2 param describe /camera/camera depth_module.profile
#
# OPT-IN — launch_isaac.sh uses NVIDIA's shipped file unless you set:
#
#     VP_SLAM_LAUNCH=vp ./launch_isaac.sh

import launch
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


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
            'enable_depth': False,
            # The IR projector must stay off: its dot pattern is a static
            # texture locked to the camera, which corrupts feature tracking.
            'depth_module.emitter_enabled': 0,
            'depth_module.profile': '640x360x90',

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
            'gyro_fps': 200,
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
