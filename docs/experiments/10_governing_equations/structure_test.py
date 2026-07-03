"""Test C: (1) z redundancy — is observed z-tendency consistent with hydrostatic
recompute from T-tendency? corr and ratio of z_t_obs vs z_t_hydrostatic(T_t_obs).
(2) time scale — R2 of physics RHS explaining true tendency at 1h vs 6h step, for T and v.
"""
import os,sys,json
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions";sys.path.insert(0,REPO)
import numpy as np,torch,h5netcdf
from tools.check_physics_common import relhum_to_specific
from utils.physics import Grid,GridConfig,PurePDEKernel,integral_z
ROOT="/home/fratnikov/weather_bench/1.40625deg/"; YEAR=2003; OUT=os.environ["OUT"]
SNAPS=[1080,3624,4692,6540]; H,W=128,256
def rd(var,name,hour):
    f=h5netcdf.File(f"{ROOT}{var}/{var}_{YEAR}_1.40625deg.nc","r")
    a=np.asarray(f.variables[name][hour,0:13,:,:],dtype=np.float32);f.close();return a
def state(hour,grid):
    z=torch.from_numpy(rd("geopotential","z",hour))[None]; t=torch.from_numpy(rd("temperature","t",hour))[None]
    r=torch.from_numpy(rd("relative_humidity","r",hour))[None]; u=torch.from_numpy(rd("u_component_of_wind","u",hour))[None]
    v=torch.from_numpy(rd("v_component_of_wind","v",hour))[None]
    q=relhum_to_specific(r,t,grid.pressure).clamp_min(1e-8)
    return {"u":u,"v":v,"t":t,"q":q,"z":z}
grid=Grid(GridConfig(H=H,W=W,lat_range_deg=(-90.,90.)))
k=PurePDEKernel(grid,stencil="fd4",coriolis="spherical",boundary_x="periodic",
                block_dt=300,t_t_formulation="adiabatic_omega",w_diagnostic="mass_consistent").eval()
cp=k.consts.c_p; Rd=k.consts.R_d
def corr(a,b):
    a=a.reshape(-1);b=b.reshape(-1); a=a-a.mean();b=b-b.mean()
    return float((a*b).sum()/(np.sqrt((a*a).sum()*(b*b).sum())+1e-30))
def R2(pred,truth):
    pred=pred.reshape(-1).astype(np.float64);truth=truth.reshape(-1).astype(np.float64)
    ss_res=((truth-pred)**2).sum(); ss_tot=((truth-truth.mean())**2).sum()
    return float(1-ss_res/(ss_tot+1e-30))

# ---- Part 1: z redundancy ----
zt_obs_all=[];zt_hyd_all=[]
for hour in SNAPS:
    s0=state(hour,grid);s1=state(hour+1,grid)
    u,v,t=s0["u"],s0["v"],s0["t"]
    with torch.no_grad():
        w=k.get_w(u,v);omega=100.0*w
        t_p=k.diff.d_z(t);adv_h=k._horiz_adv(t,u,v)
        adia=Rd*t*omega/(cp*grid.pressure)
        T_t_obs=(s1["t"]-s0["t"])/3600.0
        z_t_obs=(s1["z"]-s0["z"])/3600.0
        # hydrostatic recompute of z-tendency from T-tendency:
        pressure_hpa=grid.pressure/100.0
        z_zt=-k.R_eff/pressure_hpa*T_t_obs
        z_t_hyd=integral_z(z_zt,grid.M_z)
    zt_obs_all.append(z_t_obs.numpy());zt_hyd_all.append(z_t_hyd.numpy())
zt_obs=np.concatenate([a.reshape(-1) for a in zt_obs_all])
zt_hyd=np.concatenate([a.reshape(-1) for a in zt_hyd_all])
z_corr=corr(zt_obs,zt_hyd)
z_ratio=float(np.abs(zt_hyd).mean()/(np.abs(zt_obs).mean()+1e-30))

# ---- Part 2: time scale 1h vs 6h ----
def phys_rhs(s0):
    u,v,t=s0["u"],s0["v"],s0["t"]
    with torch.no_grad():
        w=k.get_w(u,v);omega=100.0*w
        t_p=k.diff.d_z(t)
        t_rhs=k._horiz_adv(t,u,v)-omega*t_p+Rd*t*omega/(cp*grid.pressure)
        u_z=k.diff.d_z(u);f=k.f_field
        v_rhs=k._horiz_adv(v,u,v)-w*k.diff.d_z(v)-f*u - k.diff.d_y(s0["z"])
    return t_rhs.numpy(),v_rhs.numpy()
def r2_at(dt_h):
    t_p_list=[];t_o_list=[];v_p_list=[];v_o_list=[]
    for hour in SNAPS:
        s0=state(hour,grid);s1=state(hour+dt_h,grid)
        tr,vr=phys_rhs(s0)
        t_p_list.append(tr);t_o_list.append(((s1["t"]-s0["t"]).numpy())/(dt_h*3600.0))
        v_p_list.append(vr);v_o_list.append(((s1["v"]-s0["v"]).numpy())/(dt_h*3600.0))
    tp=np.concatenate([a.reshape(-1) for a in t_p_list]);to=np.concatenate([a.reshape(-1) for a in t_o_list])
    vp=np.concatenate([a.reshape(-1) for a in v_p_list]);vo=np.concatenate([a.reshape(-1) for a in v_o_list])
    # best-scaled physics (single gain) to isolate correlation, report corr^2 as explained-variance ceiling
    return {"t_corr2":corr(tp,to)**2,"v_corr2":corr(vp,vo)**2}
r1=r2_at(1);r6=r2_at(6)

res={"z_redundancy":{"corr_obs_vs_hydrostatic":z_corr,"ratio_absmean_hyd_over_obs":z_ratio,
      "note":"if corr high, z is diagnosable from T -> redundant as prognostic"},
     "time_scale":{"1h":r1,"6h":r6,
      "note":"corr^2 = explained-variance ceiling of physics RHS at each step"}}
json.dump(res,open(OUT,"w"),indent=2)
print("=== Part 1: z redundancy ===")
print(f"  corr(z_t_obs, z_t_hydrostatic(T_t)) = {z_corr:.3f}")
print(f"  |z_t_hyd|/|z_t_obs| = {z_ratio:.3f}")
print("=== Part 2: physics explained-variance ceiling (corr^2) ===")
print(f"  1h step: T={r1['t_corr2']:.3f}  v={r1['v_corr2']:.3f}")
print(f"  6h step: T={r6['t_corr2']:.3f}  v={r6['v_corr2']:.3f}")
print("WROTE",OUT)
