"""Direct comparison: OLD buggy inline kernel vs NEW PurePDEKernel on a real
ERA5 snapshot. Reports (1) per-variable tendency magnitudes, (2) a pure-physics
Euler rollout tracking max|field| and first non-finite step.

Runs at a configurable grid via --lat-range / memmap; H,W taken from memmap.
Pure CPU. Writes JSON to OUT.
"""
import os, sys, json, math
os.environ["PYTHONNOUSERSITE"] = "1"
REPO = "/home/fa.buzaev/WeatherPredictions"
sys.path.insert(0, REPO)
import numpy as np, torch, pandas as pd
torch.manual_seed(0)

from tools.check_physics_common import (
    open_memmap, load_snapshot, split_channels_69, relhum_to_specific,
    PRESSURE_LEVELS_HPA,
)
from utils.physics import Grid, GridConfig, PurePDEKernel

MEMMAP = os.environ["MEMMAP"]
OUT    = os.environ["OUT"]
TS     = os.environ.get("TS", "2003-07-15 12:00:00")
LAT_LO = float(os.environ.get("LAT_LO", "24"))
LAT_HI = float(os.environ.get("LAT_HI", "56"))
NSTEPS = int(os.environ.get("NSTEPS", "12"))
DT     = float(os.environ.get("DT", "300"))
TAG    = os.environ.get("TAG", "usa32x64")

# ---- real ERA5 snapshot ----
handle = open_memmap(MEMMAP)
H, W = handle.shape[2], handle.shape[3]
ts = pd.Timestamp(TS)
x = load_snapshot(handle, ts, None, None)   # (1,69,H,W)
f = split_channels_69(x)
u = f["u"].float().clone(); v = f["v"].float().clone()
t = f["t"].float().clone(); z = f["z"].float().clone(); r = f["r"].float().clone()

grid = Grid(GridConfig(H=H, W=W, lat_range_deg=(LAT_LO, LAT_HI)))
p_pa = grid.pressure                          # (1,13,1,1) Pa
q = relhum_to_specific(r, t, p_pa).clamp_min(1e-8)   # specific humidity

# ================= NEW kernel (defaults = physically correct) =================
new = PurePDEKernel(grid, stencil="fd4", time_scheme="euler", coriolis="spherical",
                    block_dt=DT, t_t_formulation="adiabatic_omega", use_universal_R=False)
new.eval()

# ================= OLD kernel operators (verbatim WeatherGFT.py) ==============
class OldOps:
    def __init__(self, grid):
        self.pixel_x = grid.pixel_x
        self.pixel_y = grid.pixel_y
        self.pixel_z = grid.pixel_z            # hPa
        self.pressure_hpa = torch.tensor(PRESSURE_LEVELS_HPA, dtype=torch.float32).reshape(1,13,1,1)
        self.M_z = grid.M_z
        self.f=7.29e-5; self.L=2.5e6; self.R=8.314; self.c_p=1005; self.R_v=461.5; self.R_d=287
    def d_x(self,inp):
        B,C,Hh,Ww=inp.shape
        k=torch.zeros([1,1,1,5]); k[0,0,0,0]=1;k[0,0,0,1]=-8;k[0,0,0,3]=8;k[0,0,0,4]=-1
        p=torch.cat((inp[:,:,:,-2:],inp,inp[:,:,:,:2]),dim=3).reshape(B*C,1,Hh,Ww+4)
        return (torch.nn.functional.conv2d(p,k)/12).reshape(B,C,Hh,Ww)/self.pixel_x
    def d_y(self,inp):
        B,C,Hh,Ww=inp.shape
        k=torch.zeros([1,1,5,1]);k[0,0,0]=-1;k[0,0,1]=8;k[0,0,3]=-8;k[0,0,4]=1
        p=torch.cat((inp[:,:,:2],inp,inp[:,:,-2:]),dim=2).reshape(B*C,1,Hh+4,Ww)
        return (torch.nn.functional.conv2d(p,k)/12).reshape(B,C,Hh,Ww)/self.pixel_y
    def d_z(self,inp):
        k=torch.zeros([1,1,5,1,1]);k[0,0,0]=-1;k[0,0,1]=8;k[0,0,3]=-8;k[0,0,4]=1
        p=torch.cat((inp[:,:2],inp,inp[:,-2:]),dim=1).unsqueeze(1)
        return (torch.nn.functional.conv3d(p,k)/12).squeeze(1)/self.pixel_z
    def integral_z(self,inp):
        B,P,Hh,Ww=inp.shape
        return (self.M_z@inp.reshape(B,P,Hh*Ww)).reshape(B,P,Hh,Ww)

ops = OldOps(grid)
def old_rhs(u,v,t,z):
    u_x=ops.d_x(u); v_y=ops.d_y(v)
    w=ops.integral_z(-u_x - v_y)
    z_x=ops.d_x(z); z_y=ops.d_y(z); z_z=ops.d_z(z)
    u_y=ops.d_y(u); u_z=ops.d_z(u); v_x=ops.d_x(v); v_z=ops.d_z(v)
    u_t=-u*u_x - v*u_y - w*u_z + ops.f*v - z_x
    v_t=-u*v_x - v*v_y - w*v_z - ops.f*u - z_y
    t_x=ops.d_x(t); t_y=ops.d_y(t); t_z=ops.d_z(t)
    Q=-ops.L*z_z*w
    t_t=(Q - z_z*w)/ops.c_p - u*t_x - v*t_y - w*t_z
    z_zt=-ops.R/ops.pressure_hpa * t_t
    z_t=ops.integral_z(z_zt)
    return {"u_t":u_t,"v_t":v_t,"t_t":t_t,"z_t":z_t,"w":w}

# ---- (1) single-step tendency magnitudes ----
with torch.no_grad():
    nr = new.rhs(u.clone(), v.clone(), t.clone(), q.clone(), z.clone())
    orr = old_rhs(u.clone(), v.clone(), t.clone(), z.clone())

def mabs(x): return float(x.abs().mean())
def mx(x):   return float(x.abs().max())
res = {"meta":{"tag":TAG,"ts":TS,"H":H,"W":W,"lat":[LAT_LO,LAT_HI],"dt":DT,"nsteps":NSTEPS,
               "memmap":os.path.basename(MEMMAP)}}
res["tendencies"]={}
for key,unit in [("u_t","m/s^2"),("v_t","m/s^2"),("t_t","K/s"),("z_t","m^2/s^3")]:
    res["tendencies"][key]={
        "unit":unit,
        "OLD_mean_abs":mabs(orr[key]), "OLD_max_abs":mx(orr[key]),
        "NEW_mean_abs":mabs(nr[key]),  "NEW_max_abs":mx(nr[key]),
        "ratio_OLD_over_NEW": mabs(orr[key])/(mabs(nr[key])+1e-30),
    }
res["w_diag"]={"OLD_mean_abs":mabs(orr["w"]),"NEW_mean_abs":mabs(nr["w"])}

# ---- (2) pure-physics Euler rollout: does it blow up? ----
def rollout_old(u,v,t,z,n):
    hist=[]
    u,v,t,z=[a.clone() for a in (u,v,t,z)]
    blew=None
    for i in range(n):
        d=old_rhs(u,v,t,z)
        u=u+DT*d["u_t"]; v=v+DT*d["v_t"]; t=t+DT*d["t_t"]; z=z+DT*d["z_t"]
        mt=float(t.abs().max()); fin=bool(torch.isfinite(t).all())
        hist.append({"step":i+1,"max_abs_t":mt if math.isfinite(mt) else None,"finite":fin})
        if not fin and blew is None: blew=i+1; break
    return hist, blew
def rollout_new(u,v,t,q,z,n):
    hist=[]; blew=None
    st={"u":u.clone(),"v":v.clone(),"t":t.clone(),"q":q.clone(),"z":z.clone()}
    for i in range(n):
        with torch.no_grad():
            o=new.step(st["u"],st["v"],st["t"],st["q"],st["z"])
        st={k:o[k] for k in ("u","v","t","q","z")}
        mt=float(st["t"].abs().max()); fin=bool(torch.isfinite(st["t"]).all())
        hist.append({"step":i+1,"max_abs_t":mt if math.isfinite(mt) else None,"finite":fin})
        if not fin and blew is None: blew=i+1; break
    return hist, blew

oh, ob = rollout_old(u,v,t,z,NSTEPS)
nh, nb = rollout_new(u,v,t,q,z,NSTEPS)
res["rollout"]={
    "OLD":{"first_nonfinite_step":ob,"history":oh},
    "NEW":{"first_nonfinite_step":nb,"history":nh},
    "t_init_max_abs":float(t.abs().max()),
}
with open(OUT,"w") as fp: json.dump(res,fp,indent=2)
print("WROTE",OUT)
# compact summary
print("T_t  OLD max %.3e  NEW max %.3e" % (res["tendencies"]["t_t"]["OLD_max_abs"], res["tendencies"]["t_t"]["NEW_max_abs"]))
print("Z_t  OLD max %.3e  NEW max %.3e" % (res["tendencies"]["z_t"]["OLD_max_abs"], res["tendencies"]["z_t"]["NEW_max_abs"]))
print("rollout OLD first_nonfinite=%s  NEW first_nonfinite=%s  (t_init_max=%.1f)" % (ob, nb, res["rollout"]["t_init_max_abs"]))
print("OLD step1 max_abs_t=%s  NEW step1 max_abs_t=%s" % (oh[0]["max_abs_t"], nh[0]["max_abs_t"]))
