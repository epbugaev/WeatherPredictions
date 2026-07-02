"""Higher-resolution (global 128x256) OLD-vs-NEW comparison + time-degradation
tracking, assembled directly from fratnikov netCDF (1.40625deg).

Produces, on a real global ERA5 snapshot:
  (1) single-step tendency magnitudes OLD vs NEW per variable
  (2) multi-config pure-physics rollout (max|T| per step)
  (3) per-variable, per-step DEGRADATION curves (max & RMS) for OLD and NEW
  (4) per-level blow-up origin (which pressure level departs first)
Writes JSON to OUT.
"""
import os, sys, json, math
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions"; sys.path.insert(0,REPO)
import numpy as np, torch, h5netcdf
torch.manual_seed(0)
from tools.check_physics_common import relhum_to_specific
from utils.physics import Grid, GridConfig, PurePDEKernel

ROOT="/home/fratnikov/weather_bench/1.40625deg/"
OUT=os.environ["OUT"]
YEAR=int(os.environ.get("YEAR","2003"))
HOUR=int(os.environ.get("HOUR","4620"))   # 2003-07-15 12:00 = day 195*24+12
NSTEPS=int(os.environ.get("NSTEPS","48"))
DT=float(os.environ.get("DT","300"))
LEVELS=[50,100,150,200,250,300,400,500,600,700,850,925,1000]

def rd(var, name, levels=True):
    f=h5netcdf.File(f"{ROOT}{var}/{var}_{YEAR}_1.40625deg.nc","r")
    v=f.variables[name]
    arr = v[HOUR,0:13,:,:] if levels else v[HOUR,:,:]
    arr=np.asarray(arr, dtype=np.float32); f.close(); return arr

# assemble 69-channel state: [t2,u10,v10,tp] + z,t,r,u,v (13 each)
z = torch.from_numpy(rd("geopotential","z"))[None]          # (1,13,128,256)
t = torch.from_numpy(rd("temperature","t"))[None]
r = torch.from_numpy(rd("relative_humidity","r"))[None]
u = torch.from_numpy(rd("u_component_of_wind","u"))[None]
v = torch.from_numpy(rd("v_component_of_wind","v"))[None]
H,W = z.shape[2], z.shape[3]
grid = Grid(GridConfig(H=H, W=W, lat_range_deg=(-90.0, 90.0)))
q = relhum_to_specific(r, t, grid.pressure).clamp_min(1e-8)

VARS=["u","v","t","q","z"]
init={"u":u.clone(),"v":v.clone(),"t":t.clone(),"q":q.clone(),"z":z.clone()}
res={"meta":{"res":"global_128x256","H":H,"W":W,"year":YEAR,"hour":HOUR,"dt":DT,"nsteps":NSTEPS,
             "t_init_max":float(t.abs().max()),"levels":LEVELS,
             "umax":float(torch.sqrt(u**2+v**2).max())}}

# ---- OLD operators verbatim ----
from tools.check_physics_common import PRESSURE_LEVELS_HPA
class OldOps:
    def __init__(self,grid):
        self.pixel_x=grid.pixel_x; self.pixel_y=grid.pixel_y; self.pixel_z=grid.pixel_z
        self.pressure_hpa=torch.tensor(PRESSURE_LEVELS_HPA,dtype=torch.float32).reshape(1,13,1,1)
        self.M_z=grid.M_z
        self.f=7.29e-5; self.L=2.5e6; self.R=8.314; self.c_p=1005
    def d_x(self,i):
        B,C,Hh,Ww=i.shape; k=torch.zeros([1,1,1,5]);k[0,0,0,0]=1;k[0,0,0,1]=-8;k[0,0,0,3]=8;k[0,0,0,4]=-1
        p=torch.cat((i[:,:,:,-2:],i,i[:,:,:,:2]),3).reshape(B*C,1,Hh,Ww+4)
        return (torch.nn.functional.conv2d(p,k)/12).reshape(B,C,Hh,Ww)/self.pixel_x
    def d_y(self,i):
        B,C,Hh,Ww=i.shape; k=torch.zeros([1,1,5,1]);k[0,0,0]=-1;k[0,0,1]=8;k[0,0,3]=-8;k[0,0,4]=1
        p=torch.cat((i[:,:,:2],i,i[:,:,-2:]),2).reshape(B*C,1,Hh+4,Ww)
        return (torch.nn.functional.conv2d(p,k)/12).reshape(B,C,Hh,Ww)/self.pixel_y
    def d_z(self,i):
        k=torch.zeros([1,1,5,1,1]);k[0,0,0]=-1;k[0,0,1]=8;k[0,0,3]=-8;k[0,0,4]=1
        p=torch.cat((i[:,:2],i,i[:,-2:]),1).unsqueeze(1)
        return (torch.nn.functional.conv3d(p,k)/12).squeeze(1)/self.pixel_z
    def iz(self,i):
        B,P,Hh,Ww=i.shape; return (self.M_z@i.reshape(B,P,Hh*Ww)).reshape(B,P,Hh,Ww)
ops=OldOps(grid)
def old_rhs(u,v,t,z):
    ux=ops.d_x(u);vy=ops.d_y(v);w=ops.iz(-ux-vy)
    zx=ops.d_x(z);zy=ops.d_y(z);zz=ops.d_z(z)
    uy=ops.d_y(u);uz=ops.d_z(u);vx=ops.d_x(v);vz=ops.d_z(v)
    ut=-u*ux-v*uy-w*uz+ops.f*v-zx; vt=-u*vx-v*vy-w*vz-ops.f*u-zy
    tx=ops.d_x(t);ty=ops.d_y(t);tz=ops.d_z(t)
    Q=-ops.L*zz*w; tt=(Q-zz*w)/ops.c_p-u*tx-v*ty-w*tz
    zt=ops.iz(-ops.R/ops.pressure_hpa*tt)
    return {"u_t":ut,"v_t":vt,"t_t":tt,"z_t":zt,"w":w}

# ---- NEW kernel (correct) ----
new=PurePDEKernel(grid,stencil="fd4",time_scheme="euler",coriolis="spherical",
                  block_dt=DT,t_t_formulation="adiabatic_omega",use_universal_R=False); new.eval()

# (1) single-step tendencies
with torch.no_grad():
    nr=new.rhs(u.clone(),v.clone(),t.clone(),q.clone(),z.clone())
    orr=old_rhs(u.clone(),v.clone(),t.clone(),z.clone())
res["tendencies"]={}
for k,unit in [("u_t","m/s^2"),("v_t","m/s^2"),("t_t","K/s"),("z_t","m^2/s^3")]:
    res["tendencies"][k]={"unit":unit,"OLD_max":float(orr[k].abs().max()),"NEW_max":float(nr[k].abs().max()),
                          "OLD_mean":float(orr[k].abs().mean()),"NEW_mean":float(nr[k].abs().mean()),
                          "ratio_OLD_NEW":float(orr[k].abs().mean()/(nr[k].abs().mean()+1e-30))}

# (2)+(3) rollouts with per-variable degradation tracking
def track(state): return {v:(float(state[v].abs().max()), float((state[v]**2).mean().sqrt())) for v in VARS}
def rollout_new(kw):
    k=PurePDEKernel(grid,block_dt=DT,**kw); k.eval()
    st={v:init[v].clone() for v in VARS}
    hist={v:[] for v in VARS}; blew=None
    for i in range(NSTEPS):
        with torch.no_grad(): o=k.step(st["u"],st["v"],st["t"],st["q"],st["z"])
        st={v:o[v] for v in VARS}
        tk=track(st)
        for v in VARS: hist[v].append(tk[v] if math.isfinite(tk[v][0]) else None)
        if not all(torch.isfinite(st[v]).all() for v in VARS) and blew is None:
            blew=i+1; break
    return {"first_nonfinite":blew,"hist":hist}
def rollout_old():
    st={"u":init["u"].clone(),"v":init["v"].clone(),"t":init["t"].clone(),"z":init["z"].clone()}
    hist={v:[] for v in ("u","v","t","z")}; blew=None
    for i in range(NSTEPS):
        d=old_rhs(st["u"],st["v"],st["t"],st["z"])
        for v in ("u","v","t","z"): st[v]=st[v]+DT*d[v+"_t"]
        for v in ("u","v","t","z"):
            mx=float(st[v].abs().max()); rms=float((st[v]**2).mean().sqrt())
            hist[v].append((mx,rms) if math.isfinite(mx) else None)
        if not all(torch.isfinite(st[v]).all() for v in ("u","v","t","z")) and blew is None:
            blew=i+1; break
    return {"first_nonfinite":blew,"hist":hist}

res["rollout_OLD"]=rollout_old()
res["rollout_NEW_euler"]=rollout_new(dict(stencil="fd4",time_scheme="euler",coriolis="spherical"))
res["rollout_NEW_stab"]=rollout_new(dict(stencil="fd4",time_scheme="euler",coriolis="spherical",
                                         hyperdiffusion=True,polar_filter=True))
res["rollout_NEW_semi"]=rollout_new(dict(stencil="fd4",time_scheme="semi_implicit",coriolis="spherical",
                                         hyperdiffusion=True,polar_filter=True))

# (4) per-level origin of NEW_euler instability: max|t_t| per level at step 1
with torch.no_grad():
    tt1=new.get_t_t(u,v,new.get_w(u,v),t,new.diff.d_z(z))
res["per_level_tt_step1"]={str(LEVELS[l]):float(tt1[0,l].abs().max()) for l in range(13)}
# zonal-mean latitude profile of |NEW t_t| (to see pole amplification)
res["lat_profile_tt_NEW"]=[float(x) for x in nr["t_t"][0].abs().mean(dim=(0,2)).tolist()]
res["latitudes_deg"]=[float(x) for x in (grid.latitudes*180/math.pi).tolist()]

json.dump(res,open(OUT,"w"),indent=2)
print("WROTE",OUT)
print("t_t OLD_max %.3e NEW_max %.3e"%(res["tendencies"]["t_t"]["OLD_max"],res["tendencies"]["t_t"]["NEW_max"]))
print("OLD blew:",res["rollout_OLD"]["first_nonfinite"],"NEW_euler blew:",res["rollout_NEW_euler"]["first_nonfinite"],
      "NEW_stab blew:",res["rollout_NEW_stab"]["first_nonfinite"],"NEW_semi blew:",res["rollout_NEW_semi"]["first_nonfinite"])
print("umax %.1f"%res["meta"]["umax"])
