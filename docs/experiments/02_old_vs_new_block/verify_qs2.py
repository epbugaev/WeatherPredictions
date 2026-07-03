"""Corrected bug-5 check: (1) compute p=rho*R*t for FIXED, OLD AND the reference
kernel in the same code path; (2) controlled geometry test — run the reference
PurePDEKernel on the SAME global lat grid the inline latent kernel uses, to test
whether the residual q_t/z_t gap is geometry (lat range) or an equation error."""
import os,sys,json,importlib.util,datetime
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions";sys.path.insert(0,REPO)
import numpy as np,torch
from tools.check_physics_common import relhum_to_specific
from utils.physics import Grid,GridConfig,PurePDEKernel
OUT=os.environ["OUT"]; STAGE=os.path.dirname(OUT)
meta=json.load(open("/home/ebugaev/era5_memmap/predformer_usa_2000_2004.meta.json"))
mm=np.memmap("/home/ebugaev/era5_memmap/predformer_usa_2000_2004.dat",dtype=np.float32,mode="r",shape=tuple(meta["shape"]))
h=int((datetime.datetime(2003,7,15,12)-datetime.datetime(2000,1,1)).total_seconds()//3600)
snap=np.array(mm[h])
def blk(s,e): return torch.from_numpy(snap[s:e])[None].float()
z,t,r,u,v=blk(4,17),blk(17,30),blk(30,43),blk(43,56),blk(56,69)
H,W=32,64

from Models.WeatherGFT import PDE_kernel as PDEk
pk=PDEk(in_dim=65,physics_part_coef=0.0,variable_dim=13,block_dt=300,latents_size=(H,W)).eval()
oldp=os.path.join(STAGE,"WeatherGFT_old.py")
spec=importlib.util.spec_from_file_location("wg_old",oldp); mo=importlib.util.module_from_spec(spec)
sys.modules["wg_old"]=mo; spec.loader.exec_module(mo)
pk_old=mo.PDE_kernel(in_dim=65,physics_part_coef=0.0,variable_dim=13,block_dt=300,latents_size=(H,W)).eval()

def stat(x): return dict(mean=float(x.abs().mean()),max=float(x.abs().max()))

def inline_qbranch(k, grid_pressure):
    q=relhum_to_specific(r,t,grid_pressure).clamp_min(1e-8)
    with torch.no_grad():
        w=k.get_w(u,v); k.share_z_dxyz(z)
        k.get_uv_dt(u,v,w); k.get_t_t(u,v,w,t); k.get_z_t()
        rho=-1/k.avoid_inf(k.z_z); p=rho*k.R*t
        q_t=k.get_q_dt(u,v,t,w,q)
    return dict(p_max=float(p.abs().max()),p_mean=float(p.abs().mean()),q_t=q_t)

def ref_qbranch(kernel):
    q=relhum_to_specific(r,t,kernel.grid.pressure).clamp_min(1e-8)
    with torch.no_grad():
        w_n=kernel.get_w(u,v); dz=kernel.diff.d_z(z)
        zx=kernel.diff.d_x(z); zy=kernel.diff.d_y(z)
        tt=kernel.get_t_t(u,v,w_n,t,dz); zt=kernel.get_z_t(tt)
        # replicate the p=rho*R*t used INSIDE reference get_q_dt (avoid_inf on z_z, consts.R)
        rho=-1.0/kernel._avoid_inf(dz); p=rho*kernel.consts.R*t
        q_t=kernel.get_q_dt(u,v,t,w_n,q,zx,zy,dz,zt)
    return dict(p_max=float(p.abs().max()),p_mean=float(p.abs().mean()),q_t=q_t,z_t=zt)

# reference on TWO geometries
g_usa=Grid(GridConfig(H=H,W=W,lat_range_deg=(25.0,55.0)))
g_glob=Grid(GridConfig(H=H,W=W,lat_range_deg=(-90.0,90.0)))  # matches inline latent global grid
new_usa=PurePDEKernel(g_usa,stencil="fd4",time_scheme="euler",coriolis="spherical",block_dt=300,
                      t_t_formulation="adiabatic_omega",use_universal_R=False).eval()
new_glob=PurePDEKernel(g_glob,stencil="fd4",time_scheme="euler",coriolis="spherical",block_dt=300,
                       t_t_formulation="adiabatic_omega",use_universal_R=False).eval()

# inline uses its OWN pressure buffer (global-agnostic; pressure levels are lat-independent)
fx=inline_qbranch(pk, pk.pressure); od=inline_qbranch(pk_old, pk_old.pressure)
rf_usa=ref_qbranch(new_usa); rf_glob=ref_qbranch(new_glob)

res={
 "p_used_Pa":{"OLD_max":od["p_max"],"FIXED_max":fx["p_max"],
              "REF_usa_max":rf_usa["p_max"],"REF_glob_max":rf_glob["p_max"]},
 "q_t":{"OLD":stat(od["q_t"]),"FIXED":stat(fx["q_t"]),
        "REF_usa":stat(rf_usa["q_t"]),"REF_glob":stat(rf_glob["q_t"])},
 "z_t_ref":{"REF_usa":stat(rf_usa["z_t"]),"REF_glob":stat(rf_glob["z_t"])},
 "geometry_test":{
   "ratio_FIXED_vs_REFusa_mean":float(fx["q_t"].abs().mean()/(rf_usa["q_t"].abs().mean()+1e-30)),
   "ratio_FIXED_vs_REFglob_mean":float(fx["q_t"].abs().mean()/(rf_glob["q_t"].abs().mean()+1e-30)),
   "rel_L1_FIXED_REFusa":float((fx["q_t"]-rf_usa["q_t"]).abs().sum()/(rf_usa["q_t"].abs().sum()+1e-30)),
   "rel_L1_FIXED_REFglob":float((fx["q_t"]-rf_glob["q_t"]).abs().sum()/(rf_glob["q_t"].abs().sum()+1e-30)),
 }}
json.dump(res,open(OUT,"w"),indent=2)
print("p_used Pa: OLD %.1f FIXED %.1f REF_usa %.1f REF_glob %.1f"%(
    od["p_max"],fx["p_max"],rf_usa["p_max"],rf_glob["p_max"]))
print("q_t FIXED/REF_usa ratio %.3f rel_L1 %.3f | FIXED/REF_glob ratio %.3f rel_L1 %.3f"%(
    res["geometry_test"]["ratio_FIXED_vs_REFusa_mean"],res["geometry_test"]["rel_L1_FIXED_REFusa"],
    res["geometry_test"]["ratio_FIXED_vs_REFglob_mean"],res["geometry_test"]["rel_L1_FIXED_REFglob"]))
print("WROTE",OUT)
