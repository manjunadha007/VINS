"""Does the camera dot follow the real camera through the LiDAR space?

Locks via "Find marker in scan", waits for the operator to start moving, then
samples the dot against an independent global registration of the same depth
frame. The independent fix is the point: it shares no state with the fusion, so
agreement between them is evidence rather than self-consistency. Checkpoints
whose reference scores below 0.90 are reported but excluded from the numbers.

Container-only (needs open3d, a live D455 and the scan):

    python3 verify_motion_tracking.py
"""
import os,sys,time
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
os.environ.setdefault("VP_TRACKING_CONFIG","/workspaces/isaac_ros-dev/src/tracking.yaml")
sys.path.insert(0,"/workspaces/isaac_ros-dev/src")
import numpy as np, open3d as o3d
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
app=QApplication(sys.argv)
import rclpy; rclpy.init(args=None)
import dashboard, scan_localizer
from pose_transform import T_BODY_OPTICAL, matrix_to_rpy
SCAN="/workspaces/isaac_ros-dev/Room01_Scaled_UE01.obj"; SCALE=0.01
RECORD_S, EVERY = 40.0, 4.0
w=dashboard.HyperFusionApp(); w.streaming_enabled=False; w.show()
w.marker_width.setValue(0.0904)
mesh=o3d.io.read_triangle_mesh(SCAN)
SCAN_PTS=np.asarray(mesh.sample_points_uniformly(250000).points)*SCALE
st={"p":"imp","t":time.time(),"n":0,"last":"","rec":[],"next":0.0,"prev":None,"dh":[]}
def step():
    p=st["p"]
    if p=="imp":
        w._on_map_loaded(SCAN); st["p"]="wait"; st["t"]=time.time()
    elif p=="wait":
        if w.map_viewer.has_map() and w.lidar_matcher and w.lidar_matcher.map_cloud is not None:
            print("[1] scan imported"); st["p"]="find"; st["t"]=time.time()
        elif time.time()-st["t"]>45: print("import timeout"); app.quit()
    elif p=="find":
        st["n"]+=1
        if st["n"]>12: print("could not lock after 12 tries"); app.quit(); return
        w._find_marker_in_scan(); st["p"]="lockwatch"; st["t"]=time.time()
    elif p=="lockwatch":
        m=w.lbl_marker_lineup.text()
        if m!=st["last"]:
            st["last"]=m
            if "Searching" not in m: print(f"    try {st['n']}: {m[:110]}")
        settled = w._find_thread is None and "Searching" not in m
        if settled and "Located from scan" in m:
            L=w.map_viewer.get_locator_transform()
            print(f"[2] LOCKED. locator {L[:3,3].round(3)} "
                  f"rpy {np.degrees(matrix_to_rpy(L[:3,:3])).round(1)}")
            print("[3] waiting for the fusion to validate…")
            st["p"]="validate"; st["t"]=time.time()
        elif settled and time.time()-st["t"]>2.0:
            st["p"]="find"; st["t"]=time.time()
        elif time.time()-st["t"]>75: print("lock timeout"); app.quit()
    elif p=="validate":
        if w.engine.camera_pose_ros_corrected() is not None:
            print(f"[4] dot is live after {time.time()-st['t']:.1f}s\n")
            print(">>> START MOVING THE CAMERA -- I am watching for motion <<<\n")
            st["p"]="waitmove"; st["t"]=time.time(); st["prev"]=None; st["dh"]=[]
        elif time.time()-st["t"]>30:
            print("fusion never validated:",w.engine.fusion_status()["reason"]); app.quit()
    elif p=="waitmove":
        g=w._latest_gray
        if g is not None:
            if st["prev"] is not None and g.shape==st["prev"].shape:
                st["dh"].append(float(np.mean(np.abs(g.astype(np.int16)-st["prev"].astype(np.int16)))))
            st["prev"]=g.copy()
        if len(st["dh"])>8 and np.mean(st["dh"][-8:])>2.5:
            print(f"[5] motion detected -- recording {RECORD_S:.0f}s, keep moving\n")
            st["p"]="rec"; st["t"]=time.time(); st["next"]=0.0
        elif time.time()-st["t"]>100:
            print("no motion seen in 100 s"); app.quit()
    elif p=="rec":
        el=time.time()-st["t"]
        if el>=st["next"]:
            st["next"]=el+EVERY
            s=w.engine.capture_depth()
            if s is not None:
                dep,K,_pose,_r=s
                st["rec"].append((el, w.map_viewer.cam_pose.copy(),
                                  np.array(dep,copy=True), K.copy(),
                                  bool(w.lidar_matcher.is_aligned),
                                  w.engine.fusion_status()["state"]))
        if el>RECORD_S:
            print(f"[6] recorded {len(st['rec'])} checkpoints; registering each…\n")
            st["p"]="done"
    elif p=="done":
        rows=[]
        for el,dot,dep,K,al,state in st["rec"]:
            r=scan_localizer.locate_in_scan(
                scan_localizer.depth_to_cloud(dep,K,stride=2), SCAN_PTS, attempts=8)
            optical=(dot@T_BODY_OPTICAL)[:3,3]
            rows.append((el,optical,r.get("T_map_optical",np.eye(4))[:3,3],
                         r.get("fitness",0.0),al,state))
        print(f"{'t':>5} {'dot in LiDAR space':>26} {'independent fix':>26} "
              f"{'err':>8} {'fit':>5} {'state':>10}")
        for el,o_,t_,f_,al,state in rows:
            e=f"{np.linalg.norm(o_-t_)*1000:.0f} mm" if f_>=0.90 else "--"
            print(f"{el:5.1f} {np.array2string(o_.round(3)):>26} "
                  f"{np.array2string(t_.round(3)):>26} {e:>8} {f_:5.2f} {state:>10}")
        good=[r for r in rows if r[3]>=0.90]
        print(f"\nreliable checkpoints: {len(good)}/{len(rows)}")
        if len(good)>=2:
            err=np.array([np.linalg.norm(r[1]-r[2]) for r in good])
            print(f"ABSOLUTE error: median {np.median(err)*1000:.0f} mm, "
                  f"worst {err.max()*1000:.0f} mm")
            dd=np.array([np.linalg.norm(good[i+1][1]-good[i][1]) for i in range(len(good)-1)])
            dt=np.array([np.linalg.norm(good[i+1][2]-good[i][2]) for i in range(len(good)-1)])
            mv=dt>0.05
            if mv.any():
                print(f"\nRELATIVE motion over the moving legs:")
                print(f"  real camera moved : {dt[mv].sum()*1000:.0f} mm")
                print(f"  dot moved         : {dd[mv].sum()*1000:.0f} mm")
                print(f"  scale ratio       : {np.median(dd[mv]/dt[mv]):.3f}  (1.000 = exact)")
                print(f"\n==> {'DOT TRACKS THE REAL CAMERA' if abs(np.median(dd[mv]/dt[mv])-1)<0.15 and err.max()<0.10 else 'dot does NOT track correctly'}")
            else:
                print("\ncamera did not move more than 5 cm between checkpoints")
        app.quit()
t=QTimer(); t.timeout.connect(step); t.start(350)
QTimer.singleShot(290000, app.quit); app.exec_()
