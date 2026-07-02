"""CFL test: does smaller dt push the NEW-kernel blow-up later?
If blow-up step scales ~1/dt -> numerical (CFL), not an equation bug."""
import os,sys,json,math
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions";sys.path.insert(0,REPO)
import numpy as np,torch,pandas as pd
from tools.check_physics_common import open_memmap,load_snapshot,split_channels_69,relhum_to_specific
from utils.physics import Grid,GridConfig,PurePDEKernel
MEMMAP=os.environ["MEMMAP"];OUT=os.environ["OUT"]
TS=os.environ.get("TS","2003-07-15 12:00:00")
LAT_LO=float(os.environ.get("LAT_LO","24"));LAT_HI=float(os.environ.get("LAT_HI","56"))
handle=open_memmap(MEMMAP);H,W=handle.shape[2],handle.shape[3]
x=load_snapshot(handle,pd.Timestamp(TS),None,None);f=split_channels_69(x)
u0,v0,t0,z0,r0=[f[k].float().clone() for k in ("u","v","t","z","r")]
grid=Grid(GridConfig(H=H,W=W,lat_range_deg=(LAT_LO,LAT_HI)))
q0=relhum_to_specific(r0,t0,grid.pressure).clamp_min(1e-8)
# report grid spacing + max wind for CFL context
umax=float(torch.sqrt(u0**2+v0**2).max())
dx_min=float(grid.pixel_x.min()); dy_min=float(grid.pixel_y.min())
res={"meta":{"H":H,"W":W,"umax":umax,"dx_min":dx_min,"dy_min":dy_min,
             "cfl_dt_adv_est":min(dx_min,dy_min)/umax},"sweep":{}}
def blowstep(dt,horizon_s=3600.0):
    n=int(horizon_s/dt)
    k=PurePDEKernel(grid,block_dt=dt,stencil="fd4",time_scheme="euler",coriolis="spherical",
                    hyperdiffusion=True,polar_filter=True);k.eval()
    st={"u":u0.clone(),"v":v0.clone(),"t":t0.clone(),"q":q0.clone(),"z":z0.clone()}
    for i in range(n):
        with torch.no_grad(): o=k.step(st["u"],st["v"],st["t"],st["q"],st["z"])
        st={kk:o[kk] for kk in ("u","v","t","q","z")}
        if not bool(torch.isfinite(st["t"]).all()): return i+1,(i+1)*dt,n
    return None,None,n
for dt in [300.0,150.0,75.0,37.5]:
    bs,bt,n=blowstep(dt)
    res["sweep"][str(dt)]={"blow_step":bs,"blow_time_s":bt,"steps_in_1h":n,"survived_1h":bs is None}
    print(f"dt={dt:7.1f}s  blow_step={bs}  blow_time_s={bt}  (1h={n} steps)")
print("umax=%.1f m/s  dx_min=%.1f m  dy_min=%.1f m  CFL dt_adv~%.1f s"%(umax,dx_min,dy_min,res["meta"]["cfl_dt_adv_est"]))
json.dump(res,open(OUT,"w"),indent=2);print("WROTE",OUT)
