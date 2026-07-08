"""Numerical audit: buggy inline PDE_kernel (WeatherGFT.py) vs correct reference
(tools/check_physics_common.py) on a REAL ERA5 snapshot from the USA memmap.

Runs 4 tests:
  T1 Coriolis     : f=7.29e-5 (=Omega, no 2 sin phi) vs 2*Omega*sin(phi)
  T2 Temp tendency: (Q - z_z w)/c_p with Q=-L z_z w  vs  R_d*T*omega/(c_p*p)
  T3 Hydrostatic  : z_zt=-R_univ/p*t_t (R=8.314) vs R_d=287 ; state eqn p=rho*R*t
  T4 Magnus+avoid_inf: batch-minmax scale_tensor exp + broken avoid_inf vs clean Magnus

Outputs JSON to results path. Pure CPU.
"""
import os, sys, json, math
os.environ["PYTHONNOUSERSITE"] = "1"  # ignore ~/.local torchvision (register_fake shadow)
REPO = "/home/fa.buzaev/WeatherPredictions"
sys.path.insert(0, REPO)
import numpy as np
import torch
torch.manual_seed(0)

from tools.check_physics_common import (
    GeometryCPU, open_memmap, load_snapshot, split_channels_69,
    magnus_qs, relhum_to_specific, adiabatic_temperature_tendency,
    coriolis_spherical, coriolis_constant, coriolis_beta_plane,
    PRESSURE_LEVELS_HPA,
)

MEMMAP = os.environ.get("MEMMAP", "/home/ebugaev/era5_memmap/predformer_usa_2000_2004.dat")
OUT    = os.environ.get("OUT", "/home/fa.buzaev/wp_scratch/audit/results.json")
import pandas as pd

# ---- real ERA5 snapshot (USA crop 32x64) ----
handle = open_memmap(MEMMAP)
H, W = handle.shape[2], handle.shape[3]
ts = pd.Timestamp("2003-07-15 12:00:00")   # mid-summer noon, mid-record
x = load_snapshot(handle, ts, None, None)  # (1,69,H,W) physical units
fields = split_channels_69(x)
z = fields["z"].float(); t = fields["t"].float()
u = fields["u"].float(); v = fields["v"].float(); r = fields["r"].float()

# USA crop latitude band 24..56 N (per check_physics_weathergft.py default)
geom = GeometryCPU(H=H, W=W, lat_range_deg=(24.0, 56.0))
lat_deg = (geom.latitudes * 180.0 / math.pi)           # (H,)
p_pa = geom.pressure_pa_t                               # (1,13,1,1) Pa
p_hpa = torch.tensor(PRESSURE_LEVELS_HPA, dtype=torch.float32).reshape(1,13,1,1)

results = {"meta": {"timestamp": str(ts), "H": H, "W": W,
                    "lat_min": float(lat_deg.min()), "lat_max": float(lat_deg.max()),
                    "memmap": MEMMAP}}

def stats(name, buggy, correct, unit=""):
    buggy = buggy.detach().float(); correct = correct.detach().float()
    err = buggy - correct
    denom = correct.abs().mean().item() + 1e-30
    d = {
        "unit": unit,
        "buggy_mean_abs": float(buggy.abs().mean()),
        "correct_mean_abs": float(correct.abs().mean()),
        "mean_abs_err": float(err.abs().mean()),
        "max_abs_err": float(err.abs().max()),
        "rel_L1": float(err.abs().mean().item()/denom),
        "ratio_buggy_over_correct": float(buggy.abs().mean().item()/(correct.abs().mean().item()+1e-30)),
    }
    results[name] = d
    return d

# =========================================================================
# T1 Coriolis
# =========================================================================
f_buggy = torch.tensor(7.29e-5)                        # scalar used everywhere
f_correct_field = coriolis_spherical(geom)             # (1,1,H,W-broadcast) = 2 Omega sin phi
# effect on u-tendency term f*v: compare f*v fields
term_buggy   = f_buggy * v                             # (1,13,H,W)
term_correct = f_correct_field * v
stats("T1_coriolis_term_fv", term_buggy, term_correct, "m/s^2")
# also raw f comparison per latitude
f_corr_lat = (2*7.2921e-5*torch.sin(geom.latitudes)).reshape(-1)
results["T1_coriolis_f"] = {
    "f_buggy_const": 7.29e-5,
    "f_correct_at_lat_min": float(f_corr_lat[0]),
    "f_correct_at_lat_max": float(f_corr_lat[-1]),
    "f_correct_at_45N": float(2*7.2921e-5*math.sin(math.radians(45))),
    "ratio_buggy_over_correct_45N": float(7.29e-5/(2*7.2921e-5*math.sin(math.radians(45)))),
    "sign_flips_in_S_hemisphere": bool((lat_deg.min()<0)),
}

# =========================================================================
# Need vertical velocity w and z-derivatives for T2/T3.
# Reproduce buggy operators minimally (exact code from WeatherGFT.py).
# =========================================================================
class BuggyOps:
    def __init__(self, geom):
        self.pixel_x = geom.pixel_x
        self.pixel_y = geom.pixel_y
        self.pixel_z = geom.pixel_z          # hPa
        self.pressure = torch.tensor(PRESSURE_LEVELS_HPA, dtype=torch.float32).reshape(1,13,1,1) # hPa!
        self.M_z = geom.M_z
        self.L=2.5e6; self.R=8.314; self.c_p=1005; self.R_v=461.5; self.R_d=287; self.diff_ratio=0.05
    def d_x(self,inp):
        B,C,Hh,Ww=inp.shape
        k=torch.zeros([1,1,1,5]); k[0,0,0,0]=1;k[0,0,0,1]=-8;k[0,0,0,3]=8;k[0,0,0,4]=-1
        p=torch.cat((inp[:,:,:,-2:],inp,inp[:,:,:,:2]),dim=3)
        p=p.reshape(B*C,1,p.shape[2],p.shape[3])
        o=torch.nn.functional.conv2d(p,k)/12
        o=o.reshape(B,C,Hh,Ww)/self.pixel_x
        return o
    def d_y(self,inp):
        B,C,Hh,Ww=inp.shape
        k=torch.zeros([1,1,5,1]);k[0,0,0]=-1;k[0,0,1]=8;k[0,0,3]=-8;k[0,0,4]=1
        p=torch.cat((inp[:,:,:2],inp,inp[:,:,-2:]),dim=2)
        p=p.reshape(B*C,1,p.shape[2],p.shape[3])
        o=torch.nn.functional.conv2d(p,k)/12
        o=o.reshape(B,C,Hh,Ww)/self.pixel_y
        return o
    def d_z(self,inp):
        k=torch.zeros([1,1,5,1,1]);k[0,0,0]=-1;k[0,0,1]=8;k[0,0,3]=-8;k[0,0,4]=1
        p=torch.cat((inp[:,:2],inp,inp[:,-2:]),dim=1).unsqueeze(1)
        o=torch.nn.functional.conv3d(p,k)/12
        o=o.squeeze(1)/self.pixel_z
        return o
    def integral_z(self,inp):
        B,P,Hh,Ww=inp.shape
        return (self.M_z@inp.reshape(B,P,Hh*Ww)).reshape(B,P,Hh,Ww)
    def avoid_inf(self, tensor, threshold=1.0):
        tensor = torch.where(torch.abs(tensor)==0.0, torch.ones_like(tensor)*0.1, tensor)
        tensor = torch.where(torch.abs(tensor)<threshold, torch.sign(tensor)*threshold, tensor)
        return tensor
    def scale_tensor(self,tensor,a,b):
        mn=tensor.min().detach();mx=tensor.max().detach()
        return (tensor-mn)/(mx-mn)*(b-a)+a

ops = BuggyOps(geom)
u_x = ops.d_x(u); v_y = ops.d_y(v)
w = ops.integral_z(-u_x - v_y)          # buggy w (hPa/s), .detach() in orig irrelevant here
z_z = ops.d_z(z)                        # dz/dp

# =========================================================================
# T2 Temperature tendency
# =========================================================================
t_x=ops.d_x(t); t_y=ops.d_y(t); t_z=ops.d_z(t)
adv = -u*t_x - v*t_y - w*t_z
Q = -ops.L * z_z * w
t_t_buggy = (Q - z_z*w)/ops.c_p + adv          # buggy diabatic+adiabatic
# correct: adiabatic dT/dt = R_d T omega/(c_p p) with omega=100*w(hPa/s); advection same
t_t_adia_correct = adiabatic_temperature_tendency(t, w, p_pa)   # K/s
t_t_correct = t_t_adia_correct + adv
stats("T2_temp_tendency_full", t_t_buggy, t_t_correct, "K/s")
# isolate the adiabatic/diabatic core (no advection) — this is where the bug lives
core_buggy = (Q - z_z*w)/ops.c_p
stats("T2_temp_core_no_adv", core_buggy, t_t_adia_correct, "K/s")
results["T2_temp_tendency_full"]["buggy_coef_on_zz_w"] = float(-(ops.L+1)/ops.c_p)

# =========================================================================
# T3 Hydrostatic z_zt + equation of state
# =========================================================================
# buggy: z_zt = -R_univ/pressure_hPa * t_t   (R=8.314, pressure in hPa)
z_zt_buggy = -ops.R / ops.pressure * t_t_buggy
# correct: hydrostatic in pressure coords dz/dt tendency uses R_d and SI pressure (Pa)
z_zt_correct = -287.0 / p_pa * t_t_correct
stats("T3_hydrostatic_z_zt", z_zt_buggy, z_zt_correct, "m^2/s^2/s per level")
results["T3_hydrostatic_z_zt"]["R_used_buggy"]=8.314
results["T3_hydrostatic_z_zt"]["R_d_correct"]=287.0
results["T3_hydrostatic_z_zt"]["R_ratio"]=287.0/8.314
# equation of state p=rho*R*t with rho=-1/z_z : buggy uses R=8.314
rho = -1.0/ops.avoid_inf(z_z)
p_buggy = rho * ops.R * t
p_correct = rho * 287.0 * t
stats("T3_state_eqn_p", p_buggy, p_correct, "Pa")

# =========================================================================
# T4 Magnus q_s + avoid_inf
# =========================================================================
def buggy_qs(p_hpa_field, T):
    tc = T - 273.15
    e_s = 6.112*torch.exp(ops.scale_tensor(17.67*tc/ops.avoid_inf(tc+243.5), -3.47, 3.01))*100
    q_s = 0.622*e_s/ops.avoid_inf(p_hpa_field - 0.378*e_s)
    return q_s, e_s
qs_buggy, es_buggy = buggy_qs(ops.pressure, t)          # pressure in hPa (bug: should be Pa)
qs_correct = magnus_qs(t, p_pa)                          # Pa, clean
es_correct = 611.2*torch.exp(17.67*(t-273.15)/((t-273.15)+243.5))
stats("T4_magnus_qs", qs_buggy, qs_correct, "kg/kg")
stats("T4_magnus_es", es_buggy, es_correct, "Pa")
# avoid_inf unit test: zeros should map to 0.1 (intended) but get overwritten to 1.0
probe = torch.tensor([0.0, 0.05, -0.05, 0.5, -2.0, 3.0])
ai = ops.avoid_inf(probe.clone())
results["T4_avoid_inf_unittest"] = {
    "input": probe.tolist(),
    "output": ai.tolist(),
    "zero_maps_to": float(ai[0]),           # BUG: 1.0, intended 0.1
    "bug_zero_overwritten": bool(abs(ai[0].item()-1.0)<1e-9),
}

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print("WROTE", OUT)
print(json.dumps({k:(v if not isinstance(v,dict) else {kk:vv for kk,vv in v.items() if kk in
    ("ratio_buggy_over_correct","rel_L1","f_buggy_const","f_correct_at_45N","R_ratio","bug_zero_overwritten","zero_maps_to")})
    for k,v in results.items()}, indent=1))
