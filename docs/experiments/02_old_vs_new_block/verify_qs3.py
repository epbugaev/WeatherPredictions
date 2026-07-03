"""Clean bug-5 + geometry check: ONE consistent q input fed to inline, REF_usa,
REF_glob. Compare inline (global latent geometry) vs REF_glob (same geometry)."""
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
g_usa=Grid(GridConfig(H=H,W=W,lat_range_deg=(25.0,55.0)))
g_glob=Grid(GridConfig(H=H,W=W,lat_range_deg=(-90.0,90.0)))
# SINGLE consistent q input (pressure levels are lat-independent; use g_usa.pressure)
q=relhum_to_specific(r,t,g_usa.pressure).clamp_min(1e-8)

from Models.WeatherGFT import PDE_kernel as PDEk
pk=PDEk(in_dim=65,physics_part_coef=0.0,variable_dim=13,block_dt=300,latents_size=(H,W)).eval()
oldp=os.path.join(STAGE,"WeatherGFT_old.py")
spec=importlib.util.spec_from_file_location("wg_old",oldp); mo=importlib.util.module_from_spec(spec)
sys.modules["wg_old"]=mo; spec.loader.exec_module(mo)
pk_old=mo.PDE_kernel(in_dim=65,physics_part_coef=0.0,variable_dim=13,block_dt=300,latents_size=(H,W)).eval()

def stat(x): return dict(mean=float(x.abs().mean()),max=float(x.abs().max()))
def inline_q(k):
    with torch.no_grad():
        w=k.get_w(u,v); k.share_z_dxyz(z); k.get_uv_dt(u,v,w); k.get_t_t(u,v,w,t); k.get_z_t()
        return k.get_q_dt(u,v,t,w,q)
def ref_q(kernel):
    with torch.no_grad():
        w_n=kernel.get_w(u,v); dz=kernel.diff.d_z(z)
        zx=kernel.diff.d_x(z); zy=kernel.diff.d_y(z)
        tt=kernel.get_t_t(u,v,w_n,t,dz); zt=kernel.get_z_t(tt)
        return kernel.get_q_dt(u,v,t,w_n,q,zx,zy,dz,zt)
new_usa=PurePDEKernel(g_usa,stencil="fd4",time_scheme="euler",coriolis="spherical",block_dt=300,t_t_formulation="adiabatic_omega",use_universal_R=False).eval()
new_glob=PurePDEKernel(g_glob,stencil="fd4",time_scheme="euler",coriolis="spherical",block_dt=300,t_t_formulation="adiabatic_omega",use_universal_R=False).eval()
qo=inline_q(pk_old); qf=inline_q(pk); qru=ref_q(new_usa); qrg=ref_q(new_glob)
def rel(a,b): return float((a-b).abs().sum()/(b.abs().sum()+1e-30))
res={"q_t":{"OLD":stat(qo),"FIXED":stat(qf),"REF_usa":stat(qru),"REF_glob":stat(qrg)},
     "FIXED_vs_REFglob":{"ratio_mean":float(qf.abs().mean()/(qrg.abs().mean()+1e-30)),"rel_L1":rel(qf,qrg)},
     "FIXED_vs_REFusa":{"ratio_mean":float(qf.abs().mean()/(qru.abs().mean()+1e-30)),"rel_L1":rel(qf,qru)}}
json.dump(res,open(OUT,"w"),indent=2)
print("q_t OLD max %.4g | FIXED max %.4g | REF_glob max %.4g | REF_usa max %.4g"%(
    res["q_t"]["OLD"]["max"],res["q_t"]["FIXED"]["max"],res["q_t"]["REF_glob"]["max"],res["q_t"]["REF_usa"]["max"]))
print("FIXED vs REF_glob (same geom): ratio %.3f rel_L1 %.3f"%(res["FIXED_vs_REFglob"]["ratio_mean"],res["FIXED_vs_REFglob"]["rel_L1"]))
print("FIXED vs REF_usa (diff geom):  ratio %.3f rel_L1 %.3f"%(res["FIXED_vs_REFusa"]["ratio_mean"],res["FIXED_vs_REFusa"]["rel_L1"]))
print("WROTE",OUT)
