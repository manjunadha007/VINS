import pyrealsense2 as rs
import socket
import time
import math

from freed import get_freed_packet, FREED_DEFAULT_IP, FREED_DEFAULT_PORT

# ==========================================
# UNREAL ENGINE LIVE LINK CONFIGURATION
# ==========================================
UDP_IP = FREED_DEFAULT_IP   # IP of the machine running Unreal Engine
UDP_PORT = FREED_DEFAULT_PORT  # Port set in the Unreal UDP Messaging / LiveLink UI
CAMERA_ID = 1               # The camera ID for FreeD

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ==========================================
# REALSENSE INITIALIZATION
# ==========================================
pipeline = rs.pipeline()
config = rs.config()

# Enable IMU streams at 200Hz
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

print("Starting RealSense D455 Pipeline...")
try:
    profile = pipeline.start(config)
except Exception as e:
    print(f"Failed to start RealSense pipeline. Is the D455 plugged in? Error: {e}")
    exit(1)

# Variables for simple Gyroscope Integration
pitch, yaw, roll = 0.0, 0.0, 0.0
last_ts = None

try:
    print(f"Streaming FreeD data to Unreal Engine on {UDP_IP}:{UDP_PORT}...")
    print("Press Ctrl+C to stop.")
    
    while True:
        # Wait for the next set of IMU frames
        frames = pipeline.wait_for_frames()
        
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        accel_frame = frames.first_or_default(rs.stream.accel)
        
        if not gyro_frame or not accel_frame:
            continue
            
        gyro_data = gyro_frame.as_motion_frame().get_motion_data()
        ts = frames.get_timestamp()
        
        # ---------------------------------------------------------
        # 1. CALCULATE ORIENTATION (ROLL, PITCH, YAW)
        # ---------------------------------------------------------
        if last_ts is not None:
            dt = (ts - last_ts) / 1000.0 
            # Simple Gyro Integration (WARNING: Drifts over time!)
            pitch += gyro_data.x * dt
            yaw   -= gyro_data.y * dt  # Inverted for coordinate mapping
            roll  += gyro_data.z * dt
            
        last_ts = ts
        
        # ---------------------------------------------------------
        # 2. CALCULATE TRANSLATION (X, Y, Z)
        # ---------------------------------------------------------
        tx, ty, tz = 0.0, 0.0, 0.0 
        
        # ---------------------------------------------------------
        # 3. SEND TO UNREAL ENGINE VIA FREED
        # ---------------------------------------------------------
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        roll_deg = math.degrees(roll)
        
        payload = get_freed_packet(CAMERA_ID, pitch_deg, yaw_deg, roll_deg, tx, ty, tz)
        sock.sendto(payload, (UDP_IP, UDP_PORT))
        
        time.sleep(0.005)

except KeyboardInterrupt:
    print("\nStopping RealSense stream...")
finally:
    pipeline.stop()
