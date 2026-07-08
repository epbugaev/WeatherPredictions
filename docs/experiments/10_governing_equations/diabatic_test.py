"""Test B: diabatic heating budget.
Q/cp = T_t_obs + (u.Tx+v.Ty) + omega.T_p - Rd.T.omega/(cp.p)   [residual of adiabatic budget]
using the code's own kinematic omega. Compare |Q/cp| vs |adiabatic term| per region.
Correlate column-integrated |Q| with total_precipitation (latent heat) and TISR (radiation).
Real ERA5 128x256, 4 seasonal snapshots.
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
                block_dt=300,t_t_formulation="adiabatic_omega",w_diagnostic="mass_consistent").eval()
DT=3600.0; cp=k.consts.c_p; Rd=k.consts.R_d
cf=h5netcdf.File(ROOT+"constants/constants_1.40625deg.nc","r")
orog=np.asarray(cf.variables["orography"][:],dtype=np.float32)
lsm=np.asarray(cf.variables["lsm"][:],dtype=np.float32)
lat2d=np.abs(np.asarray(cf.variables["lat2d"][:],dtype=np.float32));cf.close()
pressure=grid.pressure

def flat(x): return x.detach().numpy().reshape(-1)
Qcp_all=[];adia_all=[];Qcol_all=[];tp_all=[];tisr_all=[]
mlat=[];moro=[];mocn=[];Qmap=None;tpmap=None
for hi,hour in enumerate(SNAPS):
    s0=state(hour,grid); s1=state(hour+1,grid)
    u,v,t=s0["u"],s0["v"],s0["t"]
    with torch.no_grad():
        w=k.get_w(u,v); omega=100.0*w
        t_p=k.diff.d_z(t)
        adv_h=k._horiz_adv(t,u,v)             # = -(u tx+v ty)
        adia=Rd*t*omega/(cp*pressure)          # adiabatic term
        T_t_obs=(s1["t"]-s0["t"])/DT
        # Q/cp = T_t_obs - adv_h + omega*t_p - adia
        Qcp = T_t_obs - adv_h + omega*t_p - adia
    Qcp_all.append(flat(Qcp)); adia_all.append(flat(adia))
    # column integral of |Q|*cp weighted by dp -> proxy heating (W/m2-ish, arbitrary units)
    pz=grid.pixel_z  # hPa
    Qcol=(Qcp.abs()*pz).sum(dim=1,keepdim=False)[0]  # (H,W)
    Qcol_all.append(Qcol.detach().numpy().reshape(-1))
    tp_all.append(rd2d("total_precipitation","tp",hour).reshape(-1))
    tisr_all.append((rd2d("toa_incident_solar_radiation","tisr",hour)/1e6).reshape(-1))
    mlat.append(np.repeat((lat2d<23.5)[None],13,0).reshape(-1))
    moro.append(np.repeat((orog>1000)[None],13,0).reshape(-1))
    mocn.append(np.repeat((lsm<0.5)[None],13,0).reshape(-1))
    if hi==2:
        Qmap=Qcol.detach().numpy().tolist(); tpmap=rd2d("total_precipitation","tp",hour).tolist()

Qcp=np.concatenate(Qcp_all);adia=np.concatenate(adia_all)
mlat=np.concatenate(mlat).astype(bool);moro=np.concatenate(moro).astype(bool);mocn=np.concatenate(mocn).astype(bool)
Qcol=np.concatenate(Qcol_all);tp=np.concatenate(tp_all);tisr=np.concatenate(tisr_all)
def ratio(m=None):
    a=np.abs(Qcp);b=np.abs(adia)
    if m is not None:a=a[m];b=b[m]
    return float(a.mean()/(b.mean()+1e-30)),float(a.mean()),float(b.mean())
def corr(a,b):
    a=a-a.mean();b=b-b.mean();return float((a*b).sum()/(np.sqrt((a*a).sum()*(b*b).sum())+1e-30))
res={"meta":{"snaps":SNAPS,"note":"Q = residual of adiabatic thermodynamic budget using kinematic omega"},
 "Q_vs_adiabatic_ratio":{}, "corr_Qcol":{"precip":corr(Qcol,tp),"solar":corr(Qcol,tisr)}}
for name,m in {"global":None,"ocean":mocn,"mountain":moro,"tropics":mlat}.items():
    r_,qa,aa=ratio(m); res["Q_vs_adiabatic_ratio"][name]={"ratio_absmean":r_,"Q_absmean":qa,"adia_absmean":aa}
res["maps"]={"Qcol":Qmap,"tp":tpmap,"shape":[H,W]}
json.dump(res,open(OUT,"w"),indent=2)
print("=== |Q/cp| vs |adiabatic term| (mean abs, K/s) ===")
for name,d_ in res["Q_vs_adiabatic_ratio"].items():
    print(f"  {name:9s} |Q|={d_['Q_absmean']:.2e} |adia|={d_['adia_absmean']:.2e} ratio={d_['ratio_absmean']:.2f}")
print("=== correlation column |Q| with data fields ===")
print(f"  precip r={res['corr_Qcol']['precip']:.3f}  solar r={res['corr_Qcol']['solar']:.3f}")
print("WROTE",OUT)
