"""Decisive test: is the pure-physics blow-up driven by initialization shock
(unbalanced IC -> fast gravity waves)? Compare raw vs geostrophic vs DFI IC."""
import os,sys,json,math
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions";sys.path.insert(0,REPO)
import numpy as np,torch,h5netcdf
from tools.check_physics_common import relhum_to_specific, balance_initial_state
from utils.physics import Grid,GridConfig,PurePDEKernel
ROOT="/home/fratnikov/weather_bench/1.40625deg/"
OUT=os.environ["OUT"];YEAR=int(os.environ["YEAR"]);HOUR=int(os.environ["HOUR"])
NSTEPS=int(os.environ.get("NSTEPS","72"));DT=float(os.environ.get("DT","150"))
def rd(var,name):
    f=h5netcdf.File(f"{ROOT}{var}/{var}_{YEAR}_1.40625deg.nc","r")
    a=np.asarray(f.variables[name][HOUR,0:13,:,:],dtype=np.float32);f.close();return a
z=torch.from_numpy(rd("geopotential","z"))[None];t=torch.from_numpy(rd("temperature","t"))[None]
r=torch.from_numpy(rd("relative_humidity","r"))[None];u=torch.from_numpy(rd("u_component_of_wind","u"))[None]
v=torch.from_numpy(rd("v_component_of_wind","v"))[None]
H,W=z.shape[2],z.shape[3];grid=Grid(GridConfig(H=H,W=W,lat_range_deg=(-90,90)))
q=relhum_to_specific(r,t,grid.pressure).clamp_min(1e-8)
VARS=["u","v","t","q","z"]
base={"u":u,"v":v,"t":t,"q":q,"z":z}
kw=dict(stencil="fd4",time_scheme="semi_implicit",coriolis="spherical",hyperdiffusion=True,polar_filter=True)
def run(state0):
    k=PurePDEKernel(grid,block_dt=DT,**kw);k.eval()
    st={x:state0[x].clone() for x in VARS};hist=[];blew=None
    for i in range(NSTEPS):
        with torch.no_grad():o=k.step(st["u"],st["v"],st["t"],st["q"],st["z"])
        st={x:o[x] for x in VARS};mt=float(st["t"].abs().max())
        fin=all(torch.isfinite(st[x]).all() for x in VARS)
        hist.append(mt if(fin and math.isfinite(mt))else None)
        if not fin and blew is None:blew=i+1;break
    return blew,hist
res={"meta":{"dt":DT,"nsteps":NSTEPS,"horizon_h":NSTEPS*DT/3600,"t_init_max":float(t.abs().max())},"modes":{}}
kbal=PurePDEKernel(grid,block_dt=DT,**kw);kbal.eval()
for mode in ["none","geostrophic","dfi"]:
    try:
        s=balance_initial_state(dict(base), kbal, mode, span_hours=1.0) if mode!="none" else dict(base)
    except Exception as e:
        res["modes"][mode]={"error":str(e)};print(mode,"ERR",e);continue
    b,h=run(s)
    res["modes"][mode]={"first_nonfinite":b,"survived":b is None,"final_maxT":h[-1] if h else None,
                        "blow_time_h":(b*DT/3600) if b else None}
    print(f"IC={mode:12s} blew@{b} ({(b*DT/3600) if b else '>'+str(NSTEPS*DT/3600)}h) survived={b is None}")
json.dump(res,open(OUT,"w"),indent=2);print("WROTE",OUT)
