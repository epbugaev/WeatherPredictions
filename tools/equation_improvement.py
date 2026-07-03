"""Data-driven equation-improvement audit: how much of the PDE residual can be
removed by (2) recalibrating physics-term coefficients and (3) adding
data-available diabatic predictors (solar, cloud, orography, humidity)?

For each prognostic var X, true tendency dX/dt = (X_{t+1}-X_t)/dt is regressed on:
  tier1: physics terms with fixed coeffs=1  -> current RHS residual
  tier2: LSQ over physics terms             -> best linear rescale of existing physics
  tier3: LSQ over physics terms + data proxies -> what available data fields add
Reports R^2 and residual rel_L1 per tier, per variable, global + per region.
Multiple snapshots for robustness.
"""
import os,sys,json
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions";sys.path.insert(0,REPO)
import numpy as np,torch,h5netcdf
from tools.check_physics_common import relhum_to_specific
from utils.physics import Grid,GridConfig,PurePDEKernel
ROOT="/home/fratnikov/weather_bench/1.40625deg/"; YEAR=2003; OUT=os.environ["OUT"]
# 4 snapshots across seasons & times of day (hour indices in 2003 hourly)
SNAPS=[int(os.environ.get("H0","1080")),   # ~mid-Feb 00Z
       3624,  # ~end-May 00Z
       4692,  # 2003-07-15 12Z
       6540]  # ~end-Sep 12Z
H,W=128,256
def rd(var,name,hour):
    f=h5netcdf.File(f"{ROOT}{var}/{var}_{YEAR}_1.40625deg.nc","r")
    a=np.asarray(f.variables[name][hour,0:13,:,:],dtype=np.float32);f.close();return a
def rd2d(var,name,hour):
    f=h5netcdf.File(f"{ROOT}{var}/{var}_{YEAR}_1.40625deg.nc","r")
    a=np.asarray(f.variables[name][hour,:,:],dtype=np.float32);f.close();return a
def state(hour,grid):
    z=torch.from_numpy(rd("geopotential","z",hour))[None]; t=torch.from_numpy(rd("temperature","t",hour))[None]
    r=torch.from_numpy(rd("relative_humidity","r",hour))[None]; u=torch.from_numpy(rd("u_component_of_wind","u",hour))[None]
    v=torch.from_numpy(rd("v_component_of_wind","v",hour))[None]
    q=relhum_to_specific(r,t,grid.pressure).clamp_min(1e-8)
    return {"u":u,"v":v,"t":t,"q":q,"z":z}
grid=Grid(GridConfig(H=H,W=W,lat_range_deg=(-90.,90.)))
k=PurePDEKernel(grid,stencil="fd4",coriolis="spherical",boundary_x="periodic",
                block_dt=300,t_t_formulation="adiabatic_omega").eval()
DT=3600.0
# constants
cf=h5netcdf.File(ROOT+"constants/constants_1.40625deg.nc","r")
orog=np.asarray(cf.variables["orography"][:],dtype=np.float32)
lsm=np.asarray(cf.variables["lsm"][:],dtype=np.float32)
lat2d=np.abs(np.asarray(cf.variables["lat2d"][:],dtype=np.float32));cf.close()

# accumulate rows across snapshots
def phys_terms(s0):
    u,v,t,q,z=s0["u"],s0["v"],s0["t"],s0["q"],s0["z"]
    with torch.no_grad():
        w=k.get_w(u,v)
        z_x=k.diff.d_x(z); z_y=k.diff.d_y(z); z_z=k.diff.d_z(z)
        f=k.f_field
        # momentum
        advh_u=k._horiz_adv(u,u,v); advv_u=-w*k.diff.d_z(u); cor_u=f*v; pgf_u=-z_x
        advh_v=k._horiz_adv(v,u,v); advv_v=-w*k.diff.d_z(v); cor_v=-f*u; pgf_v=-z_y
        # temperature
        omega_pa=100.0*w
        t_adia=k.consts.R_d*t*omega_pa/(k.consts.c_p*k.grid.pressure)
        t_advh=k._horiz_adv(t,u,v); t_advv=-w*k.diff.d_z(t)
        # moisture (decompose get_q_dt)
        q_z=k.diff.d_z(q); rho=-1.0/k._avoid_inf(z_z); p=rho*k.consts.R*t
        q_s=torch.maximum(k._get_qs(p,t),torch.full_like(q,1e-6))
        p_t=k.get_z_t(k.get_t_t(u,v,w,t,z_z))+u*z_x+v*z_y+w*z_z
        delta=((p_t<0)&(q>=q_s)).float()
        R_moist=(1+0.608*q)*k.consts.R_d
        F=((k.consts.L*R_moist-k.consts.c_p*k.consts.R_v*t)/
           k._avoid_inf(k.consts.c_p*k.consts.R_v*t*t+k.consts.L**2*q_s)*q_s*t)
        q_advh=k._horiz_adv(q,u,v); q_advv=-w*q_z
        q_cond=p_t*delta*F/k._avoid_inf(k.consts.R*t)
    return dict(w=w,t_adia=t_adia,t_advh=t_advh,t_advv=t_advv,
                u_advh=advh_u,u_advv=advv_u,u_cor=cor_u,u_pgf=pgf_u,
                v_advh=advh_v,v_advv=advv_v,v_cor=cor_v,v_pgf=pgf_v,
                q_advh=q_advh,q_advv=q_advv,q_cond=q_cond,q=q,t=t)

def np13(x): return x.detach().numpy().reshape(13,H,W)
rows={}   # var -> list of (true, physcols dict, datacols dict, masks)
data_cols_all={}; phys_cols_all={}; true_all={}; mask_all={}
for var in ["t","q","u","v"]: true_all[var]=[]; phys_cols_all[var]=[]; data_cols_all[var]=[]
mask_lat=[]; mask_orog=[]; mask_lsm=[]
for hour in SNAPS:
    s0=state(hour,grid); s1=state(hour+1,grid)
    pt=phys_terms(s0)
    tisr=rd2d("toa_incident_solar_radiation","tisr",hour)   # (128,256)
    tcc=rd2d("total_cloud_cover","tcc",hour)
    # broadcast 2D -> (13,H,W)
    b=lambda a: np.repeat(a[None],13,axis=0)
    solar=b(tisr/1e6); cloud=b(tcc); oro=b(orog/1000.0); alat=b(lat2d/90.0)
    qv=np13(pt["q"]); tv=np13(pt["t"])
    for var,physkeys in {"t":["t_adia","t_advh","t_advv"],
                         "q":["q_advh","q_advv","q_cond"],
                         "u":["u_advh","u_advv","u_cor","u_pgf"],
                         "v":["v_advh","v_advv","v_cor","v_pgf"]}.items():
        true=((s1[var]-s0[var])/DT).detach().numpy().reshape(13,H,W)
        phys=np.stack([np13(pt[kk]) for kk in physkeys],axis=-1).reshape(-1,len(physkeys))
        # data proxies: solar, cloud, orography, |lat|, q, t (all as extra linear regressors)
        data=np.stack([solar,cloud,oro,alat,qv,tv],axis=-1).reshape(-1,6)
        true_all[var].append(true.reshape(-1))
        phys_cols_all[var].append(phys)
        data_cols_all[var].append(data)
    mask_lat.append(b(lat2d<23.5).reshape(-1))
    mask_orog.append(b(orog>1000.0).reshape(-1))
    mask_lsm.append(b(lsm<0.5).reshape(-1))

def fit_eval(y,X):
    # add intercept
    A=np.concatenate([X,np.ones((X.shape[0],1),dtype=np.float32)],axis=1)
    coef,_,_,_=np.linalg.lstsq(A,y,rcond=None)
    pred=A@coef
    return coef,pred
def relL1(y,pred,m=None):
    if m is None: return float(np.abs(y-pred).sum()/(np.abs(y).sum()+1e-30))
    return float(np.abs(y-pred)[m].sum()/(np.abs(y)[m].sum()+1e-30))
def r2(y,pred,m=None):
    if m is not None: y=y[m];pred=pred[m]
    ss=((y-y.mean())**2).sum(); return float(1-((y-pred)**2).sum()/(ss+1e-30))

res={"meta":{"snaps":SNAPS,"res":"global_128x256","dt_s":DT,
     "data_proxies":["solar","cloud","orography","|lat|","q","T"]},"by_var":{}}
physkeys_map={"t":["adia","advh","advv"],"q":["advh","advv","cond"],
              "u":["advh","advv","cor","pgf"],"v":["advh","advv","cor","pgf"]}
mlat=np.concatenate(mask_lat).astype(bool); moro=np.concatenate(mask_orog).astype(bool); mocn=np.concatenate(mask_lsm).astype(bool)
for var in ["t","q","u","v"]:
    y=np.concatenate(true_all[var]).astype(np.float64)
    Xp=np.concatenate(phys_cols_all[var]).astype(np.float64)
    Xd=np.concatenate(data_cols_all[var]).astype(np.float64)
    # tier1: physics fixed (sum of terms, coeff=1)
    pred1=Xp.sum(axis=1)
    # tier2: recalibrate physics terms
    c2,pred2=fit_eval(y,Xp)
    # tier3: physics + data proxies
    c3,pred3=fit_eval(y,np.concatenate([Xp,Xd],axis=1))
    res["by_var"][var]={
      "residual_relL1":{"tier1_physics":relL1(y,pred1),"tier2_recal":relL1(y,pred2),"tier3_data":relL1(y,pred3)},
      "R2":{"tier2_recal":r2(y,pred2),"tier3_data":r2(y,pred3)},
      "phys_coef_tier2":{physkeys_map[var][i]:float(c2[i]) for i in range(len(physkeys_map[var]))},
      "data_coef_tier3":dict(zip(["solar","cloud","orog","|lat|","q","T"],[float(x) for x in c3[len(physkeys_map[var]):-1]])),
      "residual_tier3_by_region":{
         "tropics":relL1(y,pred3,mlat),"mountain":relL1(y,pred3,moro),"ocean":relL1(y,pred3,mocn),
         "tropics_tier1":relL1(y,pred1,mlat),"mountain_tier1":relL1(y,pred1,moro),"ocean_tier1":relL1(y,pred1,mocn)},
    }
json.dump(res,open(OUT,"w"),indent=2)
print("=== residual rel_L1: tier1(physics) -> tier2(recal) -> tier3(+data)  [R2 tier3] ===")
for var in ["t","q","u","v"]:
    b=res["by_var"][var]; rr=b["residual_relL1"]
    print(f"  {var}: {rr['tier1_physics']:7.2f} -> {rr['tier2_recal']:7.2f} -> {rr['tier3_data']:7.2f}   R2d={b['R2']['tier3_data']:.3f}")
print("=== physics-term coefficients (tier2, physical value=1) ===")
for var in ["t","q","u","v"]: print(f"  {var}: {res['by_var'][var]['phys_coef_tier2']}")
print("=== top data proxy (tier3 coef) ===")
for var in ["t","q","u","v"]: print(f"  {var}: {res['by_var'][var]['data_coef_tier3']}")
print("WROTE",OUT)
