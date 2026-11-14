#!/usr/bin/env python3
import rospy
import socket
import math
from nav_msgs.msg import Odometry
import tf.transformations

from freed import get_freed_packet, FREED_DEFAULT_IP, FREED_DEFAULT_PORT

# ==========================================
# UNREAL ENGINE LIVE LINK CONFIGURATION
# ==========================================
UDP_IP = FREED_DEFAULT_IP
UDP_PORT = FREED_DEFAULT_PORT
CAMERA_ID = 1

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def odometry_callback(msg):
    # 1. Extract Translation (X, Y, Z) in meters
    tx = msg.pose.pose.position.x
    ty = msg.pose.pose.position.y
    tz = msg.pose.pose.position.z

    # 2. Extract Rotation (Quaternion -> Euler)
    q = (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    )
    # euler_from_quaternion returns (roll, pitch, yaw) in radians
    euler = tf.transformations.euler_from_quaternion(q)
    
    # Note: Depending on how VINS-Fusion's coordinate frame aligns with Unreal's, 
    # you may need to swap axes or invert signs here!
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]

    # Convert to degrees for FreeD
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    roll_deg = math.degrees(roll)

    # 3. Format and Send Packet
    payload = get_freed_packet(CAMERA_ID, pitch_deg, yaw_deg, roll_deg, tx, ty, tz)
    sock.sendto(payload, (UDP_IP, UDP_PORT))

def main():
    # Initialize the ROS node
    rospy.init_node('vins_to_unreal_freed_bridge', anonymous=True)
    
    # The default odometry topic for VINS-Fusion is usually /vins_estimator/odometry
    # If using global fusion, it might be /global_fusion/global_odometry
    topic_name = "/vins_estimator/imu_propagate"
    
    rospy.Subscriber(topic_name, Odometry, odometry_callback)
    
    rospy.loginfo(f"Listening to VINS-Fusion on {topic_name}")
    rospy.loginfo(f"Streaming FreeD to Unreal Engine at {UDP_IP}:{UDP_PORT}")
    
    # Keep python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
