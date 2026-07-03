"""Deep-audit anchor experiment for the NEW PurePDEKernel on real ERA5.
Answers three improvement questions numerically:
  A) region-aware geometry (USA 25-55N) vs global default lat grid
  B) boundary_x periodic (wrong for regional lon window) vs replicate
  C) integrator intercomparison (euler/rk4/ssp_rk3/semi_implicit) stability
Metrics:
  - single-step PDE residual rel_L1 per var (geometry x boundary; integrator-independent)
  - geostrophic-balance residual (physical sanity of the state on each geometry)
  - pure-physics rollout blowup step (geometry x boundary x integrator)
"""
import os,sys,json,datetime
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions";sys.path.insert(0,REPO)
import numpy as np,torch
from tools.check_physics_common import relhum_to_specific
from utils.physics import (Grid,GridConfig,PurePDEKernel,pde_residual,
                           geostrophic_residual,cfl_number)
OUT=os.environ["OUT"]
meta=json.load(open("/home/ebugaev/era5_memmap/predformer_usa_2000_2004.meta.json"))
mm=np.memmap("/home/ebugaev/era5_memmap/predformer_usa_2000_2004.dat",dtype=np.float32,mode="r",shape=tuple(meta["shape"]))
h0=int((datetime.datetime(2003,7,15,12)-datetime.datetime(2000,1,1)).total_seconds()//3600)
H,W=32,64
def load(h):
    s=np.array(mm[h])
    def blk(a,b): return torch.from_numpy(s[a:b])[None].float()
    return dict(z=blk(4,17),t=blk(17,30),r=blk(30,43),u=blk(43,56),v=blk(56,69))
s0=load(h0); s1=load(h0+1)  # 1 hour apart
DT_DATA=3600.0

def make_state(s,grid):
    q=relhum_to_specific(s["r"],s["t"],grid.pressure).clamp_min(1e-8)
    return {"u":s["u"],"v":s["v"],"t":s["t"],"q":q,"z":s["z"]}

geoms={"global":(-90.0,90.0),"usa":(25.0,55.0)}
bcs=["periodic","replicate"]
integrators=["euler","rk4","ssp_rk3","semi_implicit"]
def relL1(a,b): return float((a-b).abs().sum()/(b.abs().sum()+1e-30))

res={"meta":{"snapshot":"2003-07-15 12:00->13:00","dt_data_s":DT_DATA,"H":H,"W":W},
     "single_step_residual":{}, "geostrophic":{}, "rollout_blowup":{}}

for gname,(lo,hi) in geoms.items():
    grid=Grid(GridConfig(H=H,W=W,lat_range_deg=(lo,hi)))
    st0=make_state(s0,grid); st1=make_state(s1,grid)
    # geostrophic residual (integrator/bc-independent; property of state+geometry)
    k_ref=PurePDEKernel(grid,stencil="fd4",coriolis="spherical",boundary_x="periodic").eval()
    with torch.no_grad(): gr=geostrophic_residual(k_ref,st0["u"],st0["v"],st0["z"])
    res["geostrophic"][gname]={
        "v_res_rel":float(gr["v_residual"].abs().mean()/(k_ref.f_field*st0["v"]).abs().mean()),
        "u_res_rel":float(gr["u_residual"].abs().mean()/(k_ref.f_field*st0["u"]).abs().mean())}
    for bc in bcs:
        # single-step PDE residual (uses RHS only -> integrator-independent)
        k=PurePDEKernel(grid,stencil="fd4",coriolis="spherical",boundary_x=bc,block_dt=300,
                        t_t_formulation="adiabatic_omega").eval()
        with torch.no_grad(): rr=pde_residual(k,st0,st1,DT_DATA)
        key=f"{gname}/{bc}"
        res["single_step_residual"][key]={v:relL1(rr[v], (st1[v]-st0[v])/DT_DATA) for v in ["u","v","t","z","q"]}
        # rollout blowup per integrator
        for it in integrators:
            ki=PurePDEKernel(grid,stencil=("weno5" if it=="ssp_rk3" else "fd4"),coriolis="spherical",
                             boundary_x=bc,block_dt=300,time_scheme=it,t_t_formulation="adiabatic_omega").eval()
            u,v,t,q,z=st0["u"].clone(),st0["v"].clone(),st0["t"].clone(),st0["q"].clone(),st0["z"].clone()
            blow=None; maxT=[]
            with torch.no_grad():
                for step in range(1,49):
                    o=ki.step(u,v,t,q,z)
                    u,v,t,q,z=o["u"],o["v"],o["t"],o["q"],o["z"]
                    mt=float(t.abs().max()); maxT.append(mt)
                    if not np.isfinite(mt) or mt>1e4:
                        blow=step; break
            res["rollout_blowup"][f"{gname}/{bc}/{it}"]={"blow_step":blow,"maxT_step1":maxT[0] if maxT else None}
json.dump(res,open(OUT,"w"),indent=2)
# print compact summary
print("=== single-step PDE residual rel_L1 (t, z) ===")
for k,v in res["single_step_residual"].items(): print(f"  {k:22s} t={v['t']:.3f} z={v['z']:.3f} u={v['u']:.3f} v={v['v']:.3f}")
print("=== geostrophic residual (rel to f*wind) ===")
for k,v in res["geostrophic"].items(): print(f"  {k:8s} v_res={v['v_res_rel']:.3f} u_res={v['u_res_rel']:.3f}")
print("=== rollout blowup step ===")
for k,v in res["rollout_blowup"].items(): print(f"  {k:30s} blow@{v['blow_step']}")
print("WROTE",OUT)
