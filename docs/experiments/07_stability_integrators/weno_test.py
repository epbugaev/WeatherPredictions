"""Does WENO5 + SSP-RK3 + hyperdiffusion survive a long pure-physics rollout?
This is the decisive experiment: if correct equations + dissipative numerics
stay finite, the 'pure physics blows up' conclusion is a numerics artifact."""
import os,sys,json,math
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions";sys.path.insert(0,REPO)
import numpy as np,torch,h5netcdf
from tools.check_physics_common import relhum_to_specific
from utils.physics import Grid,GridConfig,PurePDEKernel
ROOT="/home/fratnikov/weather_bench/1.40625deg/"
OUT=os.environ["OUT"]; YEAR=int(os.environ["YEAR"]); HOUR=int(os.environ["HOUR"])
NSTEPS=int(os.environ.get("NSTEPS","144")); 
def rd(var,name):
    f=h5netcdf.File(f"{ROOT}{var}/{var}_{YEAR}_1.40625deg.nc","r")
    a=np.asarray(f.variables[name][HOUR,0:13,:,:],dtype=np.float32);f.close();return a
z=torch.from_numpy(rd("geopotential","z"))[None];t=torch.from_numpy(rd("temperature","t"))[None]
r=torch.from_numpy(rd("relative_humidity","r"))[None];u=torch.from_numpy(rd("u_component_of_wind","u"))[None]
v=torch.from_numpy(rd("v_component_of_wind","v"))[None]
H,W=z.shape[2],z.shape[3];grid=Grid(GridConfig(H=H,W=W,lat_range_deg=(-90,90)))
q=relhum_to_specific(r,t,grid.pressure).clamp_min(1e-8)
VARS=["u","v","t","q","z"];init={"u":u,"v":v,"t":t,"q":q,"z":z}
def run(kw,dt,nsteps):
    k=PurePDEKernel(grid,block_dt=dt,**kw);k.eval()
    st={x:init[x].clone() for x in VARS};hist=[];blew=None
    for i in range(nsteps):
        with torch.no_grad():o=k.step(st["u"],st["v"],st["t"],st["q"],st["z"])
        st={x:o[x] for x in VARS};mt=float(st["t"].abs().max())
        fin=all(torch.isfinite(st[x]).all() for x in VARS)
        hist.append(mt if (fin and math.isfinite(mt)) else None)
        if not fin and blew is None:blew=i+1;break
    return blew,hist
res={"meta":{"res":"global_128x256","nsteps":NSTEPS,"t_init_max":float(t.abs().max())},"configs":{}}
trials={
 "fd4_euler_dt300":       (dict(stencil="fd4",time_scheme="euler",coriolis="spherical",hyperdiffusion=True,polar_filter=True),300),
 "weno5_ssprk3_dt300":    (dict(stencil="weno5",time_scheme="ssp_rk3",coriolis="spherical",hyperdiffusion=True,polar_filter=True),300),
 "weno5_ssprk3_dt75":     (dict(stencil="weno5",time_scheme="ssp_rk3",coriolis="spherical",hyperdiffusion=True,polar_filter=True),75),
 "weno5_ssprk3_dt75_flux":(dict(stencil="weno5",time_scheme="ssp_rk3",coriolis="spherical",hyperdiffusion=True,polar_filter=True,advection_form="flux",w_diagnostic="mass_consistent"),75),
}
for name,(kw,dt) in trials.items():
    n = NSTEPS if dt>=300 else int(NSTEPS*300/dt)  # same physical horizon
    n=min(n, 600)
    b,h=run(kw,dt,n)
    res["configs"][name]={"dt":dt,"nsteps":n,"first_nonfinite":b,"survived":b is None,
                          "final_maxT":h[-1] if h else None,"horizon_h":n*dt/3600}
    print(f"{name:26s} dt={dt:4g} n={n:3d} blew={b} survived={b is None} finalT={h[-1] if h and h[-1] else 'nan'}")
json.dump(res,open(OUT,"w"),indent=2);print("WROTE",OUT)
