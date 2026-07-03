"""Unit-check: does the PATCHED inline PDE_kernel now produce physical
tendencies matching the PurePDEKernel reference on a real ERA5 snapshot?
Imports the ACTUAL patched class (not a re-implementation) and drives its
raw operators, bypassing conv layers / scale_diff / detach."""
import os,sys,json
os.environ["PYTHONNOUSERSITE"]="1"
REPO="/home/fa.buzaev/WeatherPredictions";sys.path.insert(0,REPO)
import numpy as np,torch,h5netcdf,importlib.util
from tools.check_physics_common import relhum_to_specific
from utils.physics import Grid,GridConfig,PurePDEKernel
OUT=os.environ["OUT"]
# --- load real ERA5 USA-crop snapshot from memmap (32x64, lat 25-55N) ---
import json as _j
meta=_j.load(open("/home/ebugaev/era5_memmap/predformer_usa_2000_2004.meta.json"))
vl=meta["variables_list"]; cut=meta["cut"]  # [75,107,164,228]
mm=np.memmap("/home/ebugaev/era5_memmap/predformer_usa_2000_2004.dat",dtype=np.float32,mode="r",
             shape=tuple(meta["shape"]))
# 2003-07-15 12:00 -> hour index within 2000-2004 memmap
import datetime
h=int((datetime.datetime(2003,7,15,12)-datetime.datetime(2000,1,1)).total_seconds()//3600)
snap=np.array(mm[h])  # (69,32,64)
def block(a,s,e): return torch.from_numpy(a[s:e])[None].float()
z=block(snap,4,17); t=block(snap,17,30); r=block(snap,30,43); u=block(snap,43,56); v=block(snap,56,69)
H,W=32,64
grid=Grid(GridConfig(H=H,W=W,lat_range_deg=(25.0,55.0)))
q=relhum_to_specific(r,t,grid.pressure).clamp_min(1e-8)

# --- import PATCHED inline PDE_kernel directly from Models/WeatherGFT.py ---
spec=importlib.util.spec_from_file_location("wgft_patched",f"{REPO}/Models/WeatherGFT.py")
wg=importlib.util.module_from_spec(spec)
sys.modules["wgft_patched"]=wg; spec.loader.exec_module(wg)
PDEk=wg.PDE_kernel
# instantiate on 32x64 latent grid; physics_part_coef=0 (not used for raw ops)
pk=PDEk(in_dim=65, physics_part_coef=0.0, variable_dim=13, block_dt=300,
        latents_size=(H,W)); pk.eval()

# --- OLD (buggy) operators from staged pre-fix file ---
bakpath=os.path.join(os.path.dirname(OUT),"WeatherGFT_old.py")
spec2=importlib.util.spec_from_file_location("wgft_old",bakpath)
wgo=importlib.util.module_from_spec(spec2); sys.modules["wgft_old"]=wgo; spec2.loader.exec_module(wgo)
pk_old=wgo.PDE_kernel(in_dim=65,physics_part_coef=0.0,variable_dim=13,block_dt=300,latents_size=(H,W)); pk_old.eval()

def run_inline(k):
    with torch.no_grad():
        w=k.get_w(u,v); k.share_z_dxyz(z)
        ut,vt=k.get_uv_dt(u,v,w)
        tt=k.get_t_t(u,v,w,t)
        zt=k.get_z_t()
    return dict(w=w,u_t=ut,v_t=vt,t_t=tt,z_t=zt)
fx=run_inline(pk); od=run_inline(pk_old)

# --- reference PurePDEKernel (correct) on same fields ---
new=PurePDEKernel(grid,stencil="fd4",time_scheme="euler",coriolis="spherical",
                  block_dt=300,t_t_formulation="adiabatic_omega",use_universal_R=False); new.eval()
with torch.no_grad():
    nr=new.rhs(u.clone(),v.clone(),t.clone(),q.clone(),z.clone())

def stat(x): return dict(mean=float(x.abs().mean()),max=float(x.abs().max()))
res={"meta":{"snapshot":"2003-07-15 12:00","H":H,"W":W,"lat_range":[25.0,55.0],
             "hour_idx":h,"t_init_max":float(t.abs().max())},
     "T_t":{"OLD":stat(od["t_t"]),"FIXED":stat(fx["t_t"]),"REF_PurePDE":stat(nr["t_t"])},
     "z_t":{"OLD":stat(od["z_t"]),"FIXED":stat(fx["z_t"]),"REF_PurePDE":stat(nr["z_t"])},
     "u_t":{"OLD":stat(od["u_t"]),"FIXED":stat(fx["u_t"]),"REF_PurePDE":stat(nr["u_t"])},
     "v_t":{"OLD":stat(od["v_t"]),"FIXED":stat(fx["v_t"]),"REF_PurePDE":stat(nr["v_t"])}}
# coriolis f-field check
res["coriolis"]={"OLD_f_scalar":float(pk_old.f),
                 "FIXED_f_field_min":float(pk.f_field.min()),"FIXED_f_field_max":float(pk.f_field.max()),
                 "REF_f_field_min":float(new.grid.f_spherical.min()) if hasattr(new.grid,'f_spherical') else None}
# avoid_inf unit test
tin=torch.tensor([0.0,0.05,-0.05,0.5,-2.0,3.0])
res["avoid_inf"]={"input":tin.tolist(),"OLD":pk_old.avoid_inf(tin.clone()).tolist(),
                  "FIXED":pk.avoid_inf(tin.clone()).tolist()}
json.dump(res,open(OUT,"w"),indent=2)
print("T_t OLD mean %.4g max %.4g | FIXED mean %.4g max %.4g | REF mean %.4g max %.4g"%(
    res["T_t"]["OLD"]["mean"],res["T_t"]["OLD"]["max"],
    res["T_t"]["FIXED"]["mean"],res["T_t"]["FIXED"]["max"],
    res["T_t"]["REF_PurePDE"]["mean"],res["T_t"]["REF_PurePDE"]["max"]))
print("z_t FIXED max %.4g REF max %.4g"%(res["z_t"]["FIXED"]["max"],res["z_t"]["REF_PurePDE"]["max"]))
print("avoid_inf FIXED:",res["avoid_inf"]["FIXED"])
print("WROTE",OUT)
