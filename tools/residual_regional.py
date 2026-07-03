"""P1: regional decomposition of the single-step PDE residual on real ERA5.
Global 128x256, two consecutive hours (dt=1h). Reference PurePDEKernel (fixed physics).
  - residual r = (X_{t+1}-X_t)/dt - RHS(X_t), rel_L1 per variable, per region mask
  - per-TERM RHS magnitude (adv_h, adv_v, coriolis, pgf for momentum; adiabatic vs adv for T)
Region masks from constants file: ocean/land (lsm), mountain (orography>1000m),
and latitude bands tropics/mid/high.
"""
import os,sys,json
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions";sys.path.insert(0,REPO)
import numpy as np,torch,h5netcdf
from tools.check_physics_common import relhum_to_specific
from utils.physics import Grid,GridConfig,PurePDEKernel,pde_residual,geostrophic_residual
ROOT="/home/fratnikov/weather_bench/1.40625deg/"
OUT=os.environ["OUT"]; YEAR=2003; HOUR=4692  # 2003-07-15 12:00
LEVELS=[50,100,150,200,250,300,400,500,600,700,850,925,1000]
def rd(var,name,hour):
    f=h5netcdf.File(f"{ROOT}{var}/{var}_{YEAR}_1.40625deg.nc","r")
    a=np.asarray(f.variables[name][hour,0:13,:,:],dtype=np.float32);f.close();return a
def state(hour,grid):
    z=torch.from_numpy(rd("geopotential","z",hour))[None]
    t=torch.from_numpy(rd("temperature","t",hour))[None]
    r=torch.from_numpy(rd("relative_humidity","r",hour))[None]
    u=torch.from_numpy(rd("u_component_of_wind","u",hour))[None]
    v=torch.from_numpy(rd("v_component_of_wind","v",hour))[None]
    q=relhum_to_specific(r,t,grid.pressure).clamp_min(1e-8)
    return {"u":u,"v":v,"t":t,"q":q,"z":z}
H,W=128,256
grid=Grid(GridConfig(H=H,W=W,lat_range_deg=(-90.0,90.0)))
s0=state(HOUR,grid); s1=state(HOUR+1,grid); DT=3600.0
k=PurePDEKernel(grid,stencil="fd4",coriolis="spherical",boundary_x="periodic",
                block_dt=300,t_t_formulation="adiabatic_omega").eval()

# --- constants / masks ---
cf=h5netcdf.File(ROOT+"constants/constants_1.40625deg.nc","r")
orog=np.asarray(cf.variables["orography"][:],dtype=np.float32)   # (128,256) m
lsm=np.asarray(cf.variables["lsm"][:],dtype=np.float32)
lat2d=np.asarray(cf.variables["lat2d"][:],dtype=np.float32)
cf.close()
og=torch.from_numpy(orog); lm=torch.from_numpy(lsm); la=torch.from_numpy(np.abs(lat2d))
masks={
 "ocean":       (lm<0.5),
 "land":        (lm>=0.5),
 "mountain(>1km)": (og>1000.0),
 "lowland":     (og<=500.0),
 "tropics(<23)":(la<23.5),
 "midlat(23-60)":((la>=23.5)&(la<60)),
 "highlat(>60)":(la>=60),
 "global":      torch.ones_like(lm,dtype=torch.bool),
}
def m2d(mask):  # (H,W) -> broadcast to (1,13,H,W)
    return mask[None,None].expand(1,13,H,W)

with torch.no_grad():
    rr=pde_residual(k,s0,s1,DT)               # dict per var
    dd={v:(s1[v]-s0[v])/DT for v in ["u","v","t","z","q"]}
    # per-term RHS decomposition (compute pieces of the reference RHS)
    u,v,t,q,z=s0["u"],s0["v"],s0["t"],s0["q"],s0["z"]
    w=k.get_w(u,v)
    z_x=k.diff.d_x(z); z_y=k.diff.d_y(z); z_z=k.diff.d_z(z)
    f=k.f_field
    # momentum u_t terms
    adv_h_u=k._horiz_adv(u,u,v); adv_v_u=-w*k.diff.d_z(u); cor_u=f*v; pgf_u=-z_x
    # temperature terms
    omega_pa=100.0*w
    t_adia=k.consts.R_d*t*omega_pa/(k.consts.c_p*k.grid.pressure)
    t_advh=k._horiz_adv(t,u,v); t_advv=-w*k.diff.d_z(t)
    gr=geostrophic_residual(k,u,v,z)

def relL1(a,b,mask):
    m=m2d(mask)
    num=(a-b).abs()[m].sum(); den=b.abs()[m].sum()+1e-30
    return float(num/den)
def meanabs(a,mask):
    m=m2d(mask); return float(a.abs()[m].mean())

res={"meta":{"snapshot":"2003-07-15 12:00->13:00","res":"global_128x256","dt_s":DT},
     "residual_relL1":{}, "term_meanabs":{}, "geostrophic_relL1":{}, "mask_frac":{}}
for name,mask in masks.items():
    res["mask_frac"][name]=float(mask.float().mean())
    res["residual_relL1"][name]={v:relL1(rr[v],dd[v],mask) for v in ["u","v","t","z","q"]}
    res["term_meanabs"][name]={
        "u:adv_h":meanabs(adv_h_u,mask),"u:adv_v":meanabs(adv_v_u,mask),
        "u:coriolis":meanabs(cor_u,mask),"u:pgf":meanabs(pgf_u,mask),
        "t:adiabatic":meanabs(t_adia,mask),"t:adv_h":meanabs(t_advh,mask),"t:adv_v":meanabs(t_advv,mask),
        "w:mag":meanabs(w,mask),
    }
    denom=(f*v).abs()[m2d(mask)].mean()+1e-30
    res["geostrophic_relL1"][name]={
        "v_res":float(gr["v_residual"].abs()[m2d(mask)].mean()/denom),
        "u_res":float(gr["u_residual"].abs()[m2d(mask)].mean()/((f*u).abs()[m2d(mask)].mean()+1e-30))}
json.dump(res,open(OUT,"w"),indent=2)
print("=== residual rel_L1 (t | z | q) by region ===")
for n in masks: r=res["residual_relL1"][n];print(f"  {n:16s} t={r['t']:6.1f} z={r['z']:6.1f} q={r['q']:6.1f} u={r['u']:5.1f} frac={res['mask_frac'][n]:.2f}")
print("=== geostrophic v_res by region ===")
for n in masks: print(f"  {n:16s} v_res={res['geostrophic_relL1'][n]['v_res']:.2f}")
print("WROTE",OUT)
