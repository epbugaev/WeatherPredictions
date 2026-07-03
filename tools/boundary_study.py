"""Boundary-condition study: periodic vs replicate vs reflect on real ERA5.
Global 128x256 + USA 32x64. Single-step PDE residual + pure-physics rollout stability,
for each boundary_x mode. boundary_y kept 'replicate' (lat not cyclic).
"""
import os,sys,json,datetime
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions";sys.path.insert(0,REPO)
import numpy as np,torch,h5netcdf
from tools.check_physics_common import relhum_to_specific
from utils.physics import Grid,GridConfig,PurePDEKernel,pde_residual
OUT=os.environ["OUT"]
LEVELS=[50,100,150,200,250,300,400,500,600,700,850,925,1000]
BCS=["periodic","replicate","reflect"]

# ---- global 128x256 from netCDF ----
ROOT="/home/fratnikov/weather_bench/1.40625deg/"; YEAR=2003; HOUR=4692
def rd(var,name,hour):
    f=h5netcdf.File(f"{ROOT}{var}/{var}_{YEAR}_1.40625deg.nc","r")
    a=np.asarray(f.variables[name][hour,0:13,:,:],dtype=np.float32);f.close();return a
def gstate(hour,grid):
    z=torch.from_numpy(rd("geopotential","z",hour))[None]; t=torch.from_numpy(rd("temperature","t",hour))[None]
    r=torch.from_numpy(rd("relative_humidity","r",hour))[None]; u=torch.from_numpy(rd("u_component_of_wind","u",hour))[None]
    v=torch.from_numpy(rd("v_component_of_wind","v",hour))[None]
    q=relhum_to_specific(r,t,grid.pressure).clamp_min(1e-8)
    return {"u":u,"v":v,"t":t,"q":q,"z":z}

# ---- USA 32x64 from memmap ----
meta=json.load(open("/home/ebugaev/era5_memmap/predformer_usa_2000_2004.meta.json"))
mm=np.memmap("/home/ebugaev/era5_memmap/predformer_usa_2000_2004.dat",dtype=np.float32,mode="r",shape=tuple(meta["shape"]))
def ustate(h,grid):
    s=np.array(mm[h]); blk=lambda a,b: torch.from_numpy(s[a:b])[None].float()
    r=blk(30,43)
    q=relhum_to_specific(r,blk(17,30),grid.pressure).clamp_min(1e-8)
    return {"u":blk(43,56),"v":blk(56,69),"t":blk(17,30),"q":q,"z":blk(4,17)}

def relL1(a,b): return float((a-b).abs().sum()/(b.abs().sum()+1e-30))
DT=3600.0
res={"meta":{"snapshot":"2003-07-15 12:00->13:00","dt_s":DT,"note":"boundary_x varied; boundary_y=replicate"},
     "single_step_residual":{}, "rollout_blowup":{}}

for region,(H,W,latr) in {"global":(128,256,(-90.,90.)),"usa":(32,64,(25.,55.))}.items():
    grid=Grid(GridConfig(H=H,W=W,lat_range_deg=latr))
    if region=="global":
        s0=gstate(HOUR,grid); s1=gstate(HOUR+1,grid)
    else:
        h0=int((datetime.datetime(2003,7,15,12)-datetime.datetime(2000,1,1)).total_seconds()//3600)
        s0=ustate(h0,grid); s1=ustate(h0+1,grid)
    for bc in BCS:
        k=PurePDEKernel(grid,stencil="fd4",coriolis="spherical",boundary_x=bc,boundary_y="replicate",
                        block_dt=300,t_t_formulation="adiabatic_omega").eval()
        with torch.no_grad(): rr=pde_residual(k,s0,s1,DT)
        res["single_step_residual"][f"{region}/{bc}"]={v:relL1(rr[v],(s1[v]-s0[v])/DT) for v in ["u","v","t","z","q"]}
        # rollout
        u,v,t,q,z=(s0[x].clone() for x in ["u","v","t","q","z"]); blow=None; s1max=None
        with torch.no_grad():
            for step in range(1,49):
                o=k.step(u,v,t,q,z); u,v,t,q,z=o["u"],o["v"],o["t"],o["q"],o["z"]
                mt=float(t.abs().max())
                if step==1: s1max=mt
                if not np.isfinite(mt) or mt>1e4: blow=step;break
        res["rollout_blowup"][f"{region}/{bc}"]={"blow_step":blow,"maxT_step1":s1max}
json.dump(res,open(OUT,"w"),indent=2)
print("=== single-step PDE residual rel_L1 (t | z | u | v) ===")
for k_,v in res["single_step_residual"].items(): print(f"  {k_:18s} t={v['t']:6.1f} z={v['z']:6.1f} u={v['u']:5.1f} v={v['v']:5.1f}")
print("=== rollout blowup step ===")
for k_,v in res["rollout_blowup"].items(): print(f"  {k_:18s} blow@{v['blow_step']}")
print("WROTE",OUT)
