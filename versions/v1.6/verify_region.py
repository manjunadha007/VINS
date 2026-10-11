#!/usr/bin/env python3
"""Real scan: an accurate hint must solve; a lookalike far from the hint must
be refused with an actionable message, not silently accepted."""
import numpy as np, open3d as o3d, yaml
from lidar_map_matcher import LidarMapMatcher
from pose_transform import T_BODY_OPTICAL, invert_transform, so3_exp, so3_log

MESH="/workspaces/isaac_ros-dev/Room01_Scaled_UE01.obj"; S=0.01
mesh=o3d.io.read_triangle_mesh(MESH); mesh.scale(S,center=(0,0,0))
sc=o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
aabb=mesh.get_axis_aligned_bounding_box()
W,H=640,360; FX=320/np.tan(np.radians(43.5)); K=np.array([[FX,0,W/2],[0,FX,H/2],[0,0,1.]])

def look_at(eye,tgt,up=(0,0,1)):
    eye=np.array(eye,float);tgt=np.array(tgt,float);f=tgt-eye;f/=np.linalg.norm(f)
    if abs(np.dot(f,up))>0.99: up=(0,1,0)
    l=np.cross(up,f);l/=np.linalg.norm(l);u=np.cross(f,l)
    T=np.eye(4);T[:3,:3]=np.column_stack([f,l,u]);T[:3,3]=eye;return T
def render(T):
    o=T@T_BODY_OPTICAL
    r=sc.create_rays_pinhole(intrinsic_matrix=o3d.core.Tensor(K),
        extrinsic_matrix=o3d.core.Tensor(invert_transform(o)),width_px=W,height_px=H)
    z=sc.cast_rays(r)["t_hit"].numpy().copy(); z[~np.isfinite(z)]=0
    z[(z<=0.2)|(z>10)]=0
    rng=np.random.default_rng(0); m=z>0
    z[m]+=rng.normal(0,z[m]**2*0.1/(FX*0.095))
    return (z*1000).astype(np.uint16)

cfg=yaml.safe_load(open("tracking.yaml"))["lidar"]; cfg["map_file"]=MESH; cfg["map_scale"]=S
print(f"search region from config: ±{cfg['hint_radius_m']} m / ±{cfg['hint_radius_deg']}°\n")
m=LidarMapMatcher(cfg)

# Camera looking down at a marker on a surface, but with room visible beyond --
# the situation in the screenshot (3/3 axes, ~1000 inliers).
lo,hi=aabb.min_bound,aabb.max_bound
EYE=np.array([lo[0]+0.9, lo[1]+0.9, lo[2]+1.25])
TRUTH=look_at(EYE, np.array([hi[0]-0.6, hi[1]-0.6, lo[2]+0.55]))
depth=render(TRUTH)
print(f"view: {(depth>0).sum()} valid px, depth "
      f"{depth[depth>0].min()/1000:.2f}-{depth[depth>0].max()/1000:.2f} m\n")

def err(T):
    ph=np.append(TRUTH[:3,3],1.0)
    return np.linalg.norm((T@ph)[:3]-TRUTH[:3,3]), np.degrees(np.linalg.norm(so3_log(T[:3,:3])))

rng=np.random.default_rng(3)
print(f"{'hint placed':>22} {'outcome':>44}")
for dm,ddeg,label in [(0.08,2,"accurate (8 cm, 2°)"),
                      (0.30,8,"loose (30 cm, 8°)"),
                      (0.45,15,"sloppy (45 cm, 15°)"),
                      (1.10,42,"wrong place (1.1 m, 42°)")]:
    d=rng.normal(size=3); d/=np.linalg.norm(d)
    a=np.array([0,0,1.0])
    hint=np.eye(4); hint[:3,:3]=so3_exp(a*np.radians(ddeg)); hint[:3,3]=d*dm
    m.is_aligned=False; m.confidence=0.0
    meas=m.solve_from_hint(depth,K,TRUTH,hint,source="verify")
    if meas.ok:
        e,ed=err(m.T_slam_to_map)
        print(f"{label:>22}  SOLVED to {e*1000:5.0f}mm/{ed:4.2f}°  "
              f"({meas.verify_score:.0%} on scan, {meas.elapsed_ms:.0f} ms)")
    else:
        print(f"{label:>22}  REFUSED: {meas.reason[:60]}")
