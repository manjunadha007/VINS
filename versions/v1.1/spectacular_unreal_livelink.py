import spectacularAI
import socket
import math
import time

from freed import (get_freed_packet, euler_from_quaternion,
                   FREED_DEFAULT_IP, FREED_DEFAULT_PORT)

# ==========================================
# UNREAL ENGINE LIVE LINK CONFIGURATION
# ==========================================
UDP_IP = FREED_DEFAULT_IP
UDP_PORT = FREED_DEFAULT_PORT
CAMERA_ID = 1

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def main():
    print("Starting Spectacular AI for RealSense D455...")
    
    # Spectacular AI automatically connects to the RealSense and configures the IMU/Stereo streams!
    try:
        vio_session = spectacularAI.RealsenseVio.builder().build()
    except Exception as e:
        print(f"Failed to start SpectacularAI. Error: {e}")
        return

    print(f"Streaming PERFECT 6DOF to Unreal Engine on {UDP_IP}:{UDP_PORT}...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            # Block until a new high-frequency pose is ready
            output = vio_session.waitForOutput()
            
            # 1. Extract Translation (X, Y, Z) in meters
            tx = output.pose.position.x
            ty = output.pose.position.y
            tz = output.pose.position.z
            
            # 2. Extract Rotation (Quaternion)
            qx = output.pose.orientation.x
            qy = output.pose.orientation.y
            qz = output.pose.orientation.z
            qw = output.pose.orientation.w
            
            # Convert Quaternion to Euler (Roll, Pitch, Yaw)
            roll, pitch, yaw = euler_from_quaternion(qx, qy, qz, qw)
            
            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)
            roll_deg = math.degrees(roll)
            
            # 3. Format and send to Unreal Engine
            payload = get_freed_packet(CAMERA_ID, pitch_deg, yaw_deg, roll_deg, tx, ty, tz)
            sock.sendto(payload, (UDP_IP, UDP_PORT))

    except KeyboardInterrupt:
        print("\nStopping Spectacular AI stream...")
    finally:
        vio_session.close()

if __name__ == '__main__':
    main()
