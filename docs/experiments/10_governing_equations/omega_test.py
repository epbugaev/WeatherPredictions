"""Test A: quality of kinematic omega in the code.
- omega_kin (plain): -int(div) dp, code's default
- omega_mc (mass_consistent): column-mean-removed
- omega_implied: per-column omega that best explains observed T-tendency via the
  adiabatic thermodynamic eq (residual-optimal omega, ignoring diabatic Q).
  From T_t_obs + u.Tx+v.Ty = omega*(Rd*T/(cp*p) - T_p)  => solve for omega pointwise.
Then measure: (a) correlation/magnitude kin vs implied, (b) residual of T with each omega.
Real ERA5 128x256, 4 seasonal snapshots. Regions from constants.
"""
import os,sys,json
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions";sys.path.insert(0,REPO)
import numpy as np,torch,h5netcdf
from tools.check_physics_common import relhum_to_specific
from utils.physics import Grid,GridConfig,PurePDEKernel
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
kk=PurePDEKernel(grid,stencil="fd4",coriolis="spherical",boundary_x="periodic",
                 block_dt=300,t_t_formulation="adiabatic_omega",w_diagnostic="plain").eval()
kmc=PurePDEKernel(grid,stencil="fd4",coriolis="spherical",boundary_x="periodic",
                 block_dt=300,t_t_formulation="adiabatic_omega",w_diagnostic="mass_consistent").eval()
DT=3600.0
cf=h5netcdf.File(ROOT+"constants/constants_1.40625deg.nc","r")
orog=np.asarray(cf.variables["orography"][:],dtype=np.float32)
lsm=np.asarray(cf.variables["lsm"][:],dtype=np.float32)
lat2d=np.abs(np.asarray(cf.variables["lat2d"][:],dtype=np.float32));cf.close()
pressure=grid.pressure  # (1,P,1,1) Pa

def flat(x): return x.detach().numpy().reshape(-1)
allkin=[];allmc=[];allimp=[];res_kin=[];res_mc=[];res_imp=[]
truthT=[];omega_kin_store=None;omega_imp_store=None
mlat=[];moro=[];mocn=[]
for hi,hour in enumerate(SNAPS):
    s0=state(hour,grid); s1=state(hour+1,grid)
    u,v,t,z=s0["u"],s0["v"],s0["t"],s0["z"]
    with torch.no_grad():
        w_kin=kk.get_w(u,v)          # hPa/s (kinematic)
        w_mc=kmc.get_w(u,v)
        omega_kin=100.0*w_kin        # Pa/s
        omega_mc=100.0*w_mc
        t_p=kk.diff.d_z(t)           # dT/dp
        adv_h=kk._horiz_adv(t,u,v)   # -(u tx+v ty) already advective (negative)
        T_t_obs=(s1["t"]-s0["t"])/DT
        # adiabatic thermodynamic eq:
        #   T_t_obs = adv_h - omega*T_p + Rd*T*omega/(cp*p)
        #   => T_t_obs - adv_h = omega*(Rd*T/(cp*p) - T_p)
        coef=(kk.consts.R_d*t/(kk.consts.c_p*pressure) - t_p)  # multiplies omega
        rhs_obs=T_t_obs - adv_h
        omega_imp=rhs_obs/torch.where(coef.abs()<1e-9,torch.full_like(coef,1e-9),coef)
        # residual of T tendency with each omega (adiabatic, no Q)
        def Tt(omega): return adv_h - omega*t_p + kk.consts.R_d*t*omega/(kk.consts.c_p*pressure)
        def rel(a,b): return float((a-b).abs().sum()/(b.abs().sum()+1e-30))
        res_kin.append(rel(Tt(omega_kin),T_t_obs))
        res_mc.append(rel(Tt(omega_mc),T_t_obs))
        res_imp.append(rel(Tt(omega_imp),T_t_obs))  # ~0 by construction (sanity)
    allkin.append(flat(omega_kin));allmc.append(flat(omega_mc));allimp.append(flat(omega_imp))
    mlat.append(np.repeat((lat2d<23.5)[None],13,0).reshape(-1))
    moro.append(np.repeat((orog>1000)[None],13,0).reshape(-1))
    mocn.append(np.repeat((lsm<0.5)[None],13,0).reshape(-1))
    if hi==2:  # 2003-07-15 for maps: save 500hPa level (index 9) omega
        omega_kin_store=omega_kin[0,9].detach().numpy().tolist()
        omega_imp_store=np.clip(omega_imp[0,9].detach().numpy(),-5,5).tolist()

kin=np.concatenate(allkin);mc=np.concatenate(allmc);imp=np.concatenate(allimp)
mlat=np.concatenate(mlat).astype(bool);moro=np.concatenate(moro).astype(bool);mocn=np.concatenate(mocn).astype(bool)
# clip implied omega to physical range for stats (avoid divide blowups where coef~0)
impc=np.clip(imp,-10,10)
def corr(a,b,m=None):
    if m is not None: a=a[m];b=b[m]
    a=a-a.mean();b=b-b.mean()
    return float((a*b).sum()/(np.sqrt((a*a).sum()*(b*b).sum())+1e-30))
res={"meta":{"snaps":SNAPS,"note":"omega in Pa/s; implied = residual-optimal omega from adiabatic T-budget"},
 "magnitude":{"kin_std":float(kin.std()),"mc_std":float(mc.std()),"implied_std_clipped":float(impc.std()),
   "kin_absmean":float(np.abs(kin).mean()),"mc_absmean":float(np.abs(mc).mean()),"implied_absmean_clipped":float(np.abs(impc).mean())},
 "corr_kin_vs_implied":{"global":corr(kin,impc),"ocean":corr(kin,impc,mocn),
   "mountain":corr(kin,impc,moro),"tropics":corr(kin,impc,mlat)},
 "corr_mc_vs_implied":{"global":corr(mc,impc)},
 "T_residual_adiabatic":{"kin":float(np.mean(res_kin)),"mc":float(np.mean(res_mc)),"implied_sanity":float(np.mean(res_imp))},
 "maps_500hPa":{"omega_kin":omega_kin_store,"omega_implied":omega_imp_store,"shape":[H,W]}}
json.dump(res,open(OUT,"w"),indent=2)
print("=== omega magnitude (Pa/s) ===")
print(f"  kinematic std={kin.std():.4f} absmean={np.abs(kin).mean():.4f}")
print(f"  implied(clip) std={impc.std():.4f} absmean={np.abs(impc).mean():.4f}")
print("=== correlation kinematic omega vs residual-implied omega ===")
for k_,val in res["corr_kin_vs_implied"].items(): print(f"  {k_:10s} r={val:.3f}")
print(f"  mass_consistent vs implied: r={res['corr_mc_vs_implied']['global']:.3f}")
print("=== T-tendency residual (adiabatic, no Q) with each omega ===")
print(f"  kin={np.mean(res_kin):.2f}  mc={np.mean(res_mc):.2f}  implied(sanity~0)={np.mean(res_imp):.3f}")
print("WROTE",OUT)
