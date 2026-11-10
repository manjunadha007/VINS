"""Read-only rig probe: are the topics healthy and is a physical marker visible?

Subscribes to the live camera, depth and odometry, runs the real ArUco detector
for 12 s and reports what it sees. Touches nothing -- safe to run alongside the
dashboard.

    python3 verify_marker_live.py
"""
import sys, time
sys.path.insert(0, "/workspaces/isaac_ros-dev/src")
import numpy as np, cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import tracking_config

cfg = tracking_config.load_config(verbose=False)
T = cfg["topics"]; FAM = cfg["aruco"]["dictionaries"][0]
SIZE = float(cfg["aruco"]["marker_size_m"])

def dictionary():
    a = cv2.aruco; did = getattr(a, FAM)
    return (a.getPredefinedDictionary(did) if hasattr(a, "getPredefinedDictionary")
            else a.Dictionary_get(did))
def detect(img):
    a = cv2.aruco; d = dictionary()
    p = (a.DetectorParameters_create() if hasattr(a, "DetectorParameters_create")
         else a.DetectorParameters())
    if hasattr(a, "ArucoDetector"):
        return a.ArucoDetector(d, p).detectMarkers(img)
    return a.detectMarkers(img, d, parameters=p)

class Probe(Node):
    def __init__(self):
        super().__init__("marker_probe")
        self.b = CvBridge(); self.gray = None; self.K = None
        self.depth = None; self.odom = None; self.n_img = 0
        self.create_subscription(Image, T["image"], self.on_img, 10)
        self.create_subscription(CameraInfo, T["camera_info"], self.on_info, 10)
        self.create_subscription(Image, T["depth"], self.on_depth, 10)
        self.create_subscription(Odometry, T["odometry"], self.on_odom, 10)
    def on_img(self, m):
        img = self.b.imgmsg_to_cv2(m, desired_encoding="passthrough")
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = cv2.convertScaleAbs(img, alpha=255.0 / max(1, img.max()))
        self.gray = img; self.n_img += 1
    def on_info(self, m): self.K = np.array(m.k, dtype=float).reshape(3, 3)
    def on_depth(self, m): self.depth = self.b.imgmsg_to_cv2(m, "passthrough")
    def on_odom(self, m):
        p = m.pose.pose.position; self.odom = (p.x, p.y, p.z)

rclpy.init()
n = Probe()
t0 = time.time(); seen = {}
while time.time() - t0 < 12.0:
    rclpy.spin_once(n, timeout_sec=0.1)
    if n.gray is not None:
        c, ids, _ = detect(n.gray)
        if ids is not None:
            for quad, i in zip(c, ids):
                q = quad.reshape(4, 2)
                side = float(np.mean([np.linalg.norm(q[j] - q[(j+1) % 4]) for j in range(4)]))
                seen[int(i[0])] = (side, q)

print(f"image frames      : {n.n_img}  ({'ok' if n.n_img else 'NO IMAGE'})")
print(f"camera_info K     : {'ok  fx=%.1f cx=%.1f' % (n.K[0,0], n.K[0,2]) if n.K is not None else 'MISSING'}")
if n.depth is not None:
    d = np.asarray(n.depth); v = d[(d > 300) & (d < 6000)]
    print(f"depth             : ok  {d.shape} {len(v)/d.size:.0%} in 0.3-6 m, "
          f"median {np.median(v)/1000 if len(v) else 0:.2f} m")
else:
    print("depth             : MISSING")
print(f"odometry          : {'ok  pos ' + str(np.round(n.odom,3)) if n.odom else 'MISSING'}")
print(f"aruco family      : {FAM}, marker size {SIZE*100:.0f} cm")
if seen:
    for i, (side, q) in sorted(seen.items()):
        est = (n.K[0,0] * SIZE / side) if n.K is not None and side > 0 else float('nan')
        print(f"  MARKER id {i}: {side:.0f} px per side, ~{est:.2f} m away, "
              f"centre ({q[:,0].mean():.0f},{q[:,1].mean():.0f})")
else:
    print("  NO MARKER VISIBLE in 12 s of live frames")
if n.gray is not None:
    out = n.gray.copy()
    if seen:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        for i, (s, q) in seen.items():
            cv2.polylines(out, [q.astype(int)], True, (0, 255, 0), 2)
            cv2.putText(out, str(i), tuple(q[0].astype(int)), 0, 0.8, (0, 255, 0), 2)
    cv2.imwrite("/workspaces/isaac_ros-dev/src/_live_view.png", out)
    print("wrote _live_view.png")
n.destroy_node(); rclpy.shutdown()
