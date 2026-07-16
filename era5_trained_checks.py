"""Items 1 & 2 on the TRAINED B1 forecast (the correct object), plus a no-physics
baseline for spectrum. Reuses the repo model + WeatherNormalize from checkpoints.

Item 1: zonal power spectrum of trained forecast vs ERA5 (blurring diagnostic).
Item 2: column-water-path drift of trained forecast vs ERA5 (water conservation).
"""
import os, sys, json
import numpy as np
import torch

REPO = os.environ.get("REPO_ROOT", "/home/fa.buzaev/WeatherPredictions")
sys.path.insert(0, REPO); os.chdir(REPO)
import Models  # registers PI-IAM4VP
from utils.registry import get_model
from utils.normalize import WeatherNormalize
from utils.physics_hybrid import relative_to_specific_humidity

torch.set_default_dtype(torch.float32)
G = 9.80665
CKPTS = {
  "B1_stable_physics": "/home/fa.buzaev/checkpoints/PI-IAM4VP-ResidualStablePhysics-USA-v4-T12/2026-07-08-02:247dabp/best.pt",
  "no_physics":        None,  # filled below if found
}
# find a no-physics T12 checkpoint
import glob
npg = sorted(glob.glob("/home/fa.buzaev/checkpoints/PI-IAM4VP-ResidualNoPhysics-USA-v4-T12/*/best.pt"))
CKPTS["no_physics"] = npg[-1] if npg else None

mm = "/home/ebugaev/era5_memmap/predformer_usa_2000_2004.dat"
meta = json.load(open(mm.replace(".dat", ".meta.json")))
Ntot, C, Hd, Wd = meta["shape"]
arr = np.memmap(mm, dtype="float32", mode="r", shape=tuple(meta["shape"]))
plev_hpa = torch.tensor([50,100,150,200,250,300,400,500,600,700,850,925,1000.])
dp = (plev_hpa[1:]-plev_hpa[:-1])*100.0
wint = torch.zeros(13)
for i in range(13):
    b = dp[i-1] if i>0 else None; a = dp[i] if i<12 else None
    wint[i] = (a/2 if b is None else b/2 if a is None else (a+b)/2)
wint = wint.reshape(1,13,1,1)
lat0 = -89.296875 + meta["cut"][0]*1.40625
lats = torch.tensor([lat0 + i*1.40625 for i in range(Hd)])
w_lat = torch.cos(torch.deg2rad(lats)).reshape(1,1,Hd,1)
pres_pa = (plev_hpa*100.0).reshape(1,13,1,1)
i500 = 7

def build(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    p = dict(ck["config"]["model"].get("params", {}))
    p.setdefault("C_data", Hd and C); p["C_data"]=C; p["H_data"]=Hd; p["W_data"]=Wd
    # geometry (B1 values for the 1.40625deg USA cut, factor-4 patch)
    p["physics_lat_start_deg"]=18.28125; p["physics_dlat_deg"]=5.625; p["physics_dlon_deg"]=5.625
    m = get_model(ck["config"]["model"]["type"])(**p)
    res = m.load_state_dict(ck["model"], strict=False); m.eval()
    miss=[k for k in res.missing_keys if not any(s in k for s in ("latitudes","pixel_x","pixel_y"))]
    assert not miss and not res.unexpected_keys, f"unexpected load mismatch: miss={miss} unexp={res.unexpected_keys}"
    nm = ck["normalize"]
    norm = WeatherNormalize(nm["mean"].reshape(-1), nm["std"].reshape(-1))
    m.set_physics_normalization(nm["mean"].reshape(-1), nm["std"].reshape(-1))
    return m, norm, p

def rollout(model, norm, x_raw_phys, T_pred):
    # x_raw_phys: (1, T_in, C, H, W) physical; normalize, iterate, denorm preds
    x = norm(x_raw_phys)
    preds=[]
    with torch.no_grad():
        for idx in range(T_pred):
            t = torch.tensor((idx+1)*100).repeat(x.shape[0])
            pr = model(x, preds, t)          # normalized (1,C,H,W)
            preds.append(pr.detach())
    y_norm = torch.stack(preds, dim=1)       # (1,T_pred,C,H,W)
    return norm.denormalize(y_norm)

def zonal_hi_power(f2d):
    sp = torch.fft.rfft(f2d, dim=-1); p=(sp.abs()**2).mean(dim=0)
    return float(p[len(p)//2:].sum())
def amean(x): return float((x*w_lat).sum()/(torch.ones_like(x)*w_lat).sum())
def cwp(r_phys, T_phys):
    q = relative_to_specific_humidity(r_phys, T_phys, pres_pa)
    return (q*wint).sum(dim=1,keepdim=True)/G

Tin = 12; Tpred = 10; j0 = 12000
win = torch.from_numpy(np.asarray(arr[j0-Tin:j0+Tpred]).astype(np.float32))  # (Tin+Tpred,C,H,W)
x_in = win[:Tin].unsqueeze(0)
truth = win[Tin:Tin+Tpred].unsqueeze(0)  # (1,Tpred,C,H,W)

print("="*78)
for name, path in CKPTS.items():
    if not path: print(f"[{name}] checkpoint not found, skip"); continue
    try:
        m, norm, p = build(path)
    except Exception as e:
        print(f"[{name}] build FAILED: {type(e).__name__}: {e}"); continue
    y = rollout(m, norm, x_in, Tpred)  # (1,Tpred,C,H,W) physical
    print(f"\n[{name}]  hybrid_mode={p.get('physics_residual_hybrid_mode')} w_diag={p.get('physics_w_diagnostic','?')}")
    # ITEM 1 spectrum at hour 10 (last predicted frame), T@500 and z@500
    for var,ofs in (("T",17),("z",4)):
        ch = ofs + i500
        rp = zonal_hi_power(y[0,-1,ch]); ep = zonal_hi_power(truth[0,-1,ch])
        print(f"   ITEM1 {var}500 hi-k ratio (pred/ERA5) = {rp/ep:.3f}  ({'blur' if rp<ep else 'rough'})")
    # ITEM 2 water: dW pred vs ERA5 over Tpred frames
    r0,T0 = x_in[0,-1,30:43].unsqueeze(0), x_in[0,-1,17:30].unsqueeze(0)
    rP,TP = y[0,-1,30:43].unsqueeze(0), y[0,-1,17:30].unsqueeze(0)
    rE,TE = truth[0,-1,30:43].unsqueeze(0), truth[0,-1,17:30].unsqueeze(0)
    W0 = amean(cwp(r0,T0)); dW_pred = amean(cwp(rP,TP))-W0; dW_era = amean(cwp(rE,TE))-W0
    print(f"   ITEM2 water  ΔW_pred={dW_pred:+.4f}  ΔW_ERA5={dW_era:+.4f}  imbalance={dW_pred-dW_era:+.4f} kg/m^2")
    # forecast skill sanity: RMSE z500 at h10
    err = float((((y[0,-1,4+i500]-truth[0,-1,4+i500])**2*w_lat[0,0]).sum()/w_lat[0,0].sum()).sqrt())
    print(f"   sanity z500 RMSE @h10 = {err:.2f} (geopotential units)")
print("\n[done]")
