"""Focused check of bug-5 (Magnus q_s / q_t) in the ACTUAL inline code path:
compare fixed-inline get_q_dt/get_qs vs reference PurePDEKernel on real ERA5.
The audit's 1391x was for the STANDALONE magnus formula (pressure in hPa vs Pa);
here we test how the inline actually computes q_s (p = rho*R*t) vs how the
reference does it in the same code path."""
import os,sys,json,importlib.util,datetime
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions";sys.path.insert(0,REPO)
import numpy as np,torch
from tools.check_physics_common import relhum_to_specific, magnus_qs
from utils.physics import Grid,GridConfig,PurePDEKernel
OUT=os.environ["OUT"]; STAGE=os.path.dirname(OUT)
meta=json.load(open("/home/ebugaev/era5_memmap/predformer_usa_2000_2004.meta.json"))
mm=np.memmap("/home/ebugaev/era5_memmap/predformer_usa_2000_2004.dat",dtype=np.float32,mode="r",shape=tuple(meta["shape"]))
h=int((datetime.datetime(2003,7,15,12)-datetime.datetime(2000,1,1)).total_seconds()//3600)
snap=np.array(mm[h])
def blk(s,e): return torch.from_numpy(snap[s:e])[None].float()
z,t,r,u,v=blk(4,17),blk(17,30),blk(30,43),blk(43,56),blk(56,69)
H,W=32,64
grid=Grid(GridConfig(H=H,W=W,lat_range_deg=(25.0,55.0)))
q=relhum_to_specific(r,t,grid.pressure).clamp_min(1e-8)

def load_pkg_class():
    from Models.WeatherGFT import PDE_kernel
    return PDE_kernel
PDEk=load_pkg_class()
pk=PDEk(in_dim=65,physics_part_coef=0.0,variable_dim=13,block_dt=300,latents_size=(H,W)).eval()
# OLD from staged file
oldp=os.path.join(STAGE,"WeatherGFT_old.py")
spec=importlib.util.spec_from_file_location("wg_old",oldp); mo=importlib.util.module_from_spec(spec)
sys.modules["wg_old"]=mo; spec.loader.exec_module(mo)
pk_old=mo.PDE_kernel(in_dim=65,physics_part_coef=0.0,variable_dim=13,block_dt=300,latents_size=(H,W)).eval()

# drive q-branch: need w and shared z-derivatives + z_t populated
def q_branch(k):
    with torch.no_grad():
        w=k.get_w(u,v); k.share_z_dxyz(z)
        k.get_uv_dt(u,v,w); k.get_t_t(u,v,w,t); k.get_z_t()  # populates k.z_t etc.
        # extract q_s exactly as inline does: p = rho*R*t
        rho=-1/k.avoid_inf(k.z_z); p=rho*k.R*t
        # call the nested get_qs via get_q_dt path indirectly:
        q_t=k.get_q_dt(u,v,t,w,q)
    return dict(rho_max=float(rho.abs().max()),p_max=float(p.abs().max()),q_t=q_t)
fx=q_branch(pk); od=q_branch(pk_old)

# reference q_dt
new=PurePDEKernel(grid,stencil="fd4",time_scheme="euler",coriolis="spherical",block_dt=300,
                  t_t_formulation="adiabatic_omega",use_universal_R=False).eval()
with torch.no_grad():
    w_n=new.get_w(u,v); dz=new.diff.d_z(z)
    zx=new.diff.d_x(z); zy=new.diff.d_y(z)
    tt=new.get_t_t(u,v,w_n,t,dz); zt=new.get_z_t(tt)
    q_t_ref=new.get_q_dt(u,v,t,w_n,q,zx,zy,dz,zt)

def stat(x): return dict(mean=float(x.abs().mean()),max=float(x.abs().max()))
res={"q_t":{"OLD":stat(od["q_t"]),"FIXED":stat(fx["q_t"]),"REF":stat(q_t_ref)},
     "p_used":{"OLD_max":od["p_max"],"FIXED_max":fx["p_max"]},
     "ratio_FIXED_vs_REF_mean":float(fx["q_t"].abs().mean()/(q_t_ref.abs().mean()+1e-30)),
     "rel_L1_FIXED_REF":float((fx["q_t"]-q_t_ref).abs().sum()/(q_t_ref.abs().sum()+1e-30))}
json.dump(res,open(OUT,"w"),indent=2)
print("q_t OLD max %.4g | FIXED max %.4g | REF max %.4g"%(res["q_t"]["OLD"]["max"],res["q_t"]["FIXED"]["max"],res["q_t"]["REF"]["max"]))
print("q_t ratio FIXED/REF (mean) %.4g ; rel_L1 %.4g"%(res["ratio_FIXED_vs_REF_mean"],res["rel_L1_FIXED_REF"]))
print("WROTE",OUT)
