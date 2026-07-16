"""Four cheap diagnostics on a real ERA5 slice (one CPU job).

Scope note: no trained model checkpoint is loaded here, so items (1) spectrum and
(3) rollout-A/B run on the PHYSICS CORE itself (physics-only multi-step rollout),
not on a trained forecast. They characterize the kernel's own spectral/stability
behavior and the omega-lever differences on a multi-step basis. Item (4) proves the
online-training gradient path is viable through the torch kernel. Item (2) measures
column-water conservation of the physics core.
"""
import os, sys, json
import numpy as np
import torch

REPO = os.environ.get("REPO_ROOT", "/home/fa.buzaev/WeatherPredictions")
sys.path.insert(0, REPO)
from utils.physics import GridConfig, Grid, PurePDEKernel, pde_residual  # noqa: E402
from utils.physics_hybrid import relative_to_specific_humidity  # noqa: E402

torch.set_default_dtype(torch.float64)
G = 9.80665
mm = "/home/ebugaev/era5_memmap/predformer_usa_2000_2004.dat"
meta = json.load(open(mm.replace(".dat", ".meta.json")))
shape = tuple(meta["shape"]); Ntot, C, Hd, Wd = shape
arr = np.memmap(mm, dtype="float32", mode="r", shape=shape)
lat0 = -89.296875 + meta["cut"][0] * 1.40625
lats = tuple(float(lat0 + i * 1.40625) for i in range(Hd))
plev_hpa = torch.tensor([50,100,150,200,250,300,400,500,600,700,850,925,1000.])
i500 = 7  # index of 500 hPa

def unpack(t4):
    return t4[:,4:17], t4[:,17:30], t4[:,30:43], t4[:,43:56], t4[:,56:69]  # z,T,r,u,v

def make_kernel(**kw):
    cfg = GridConfig(H=Hd, W=Wd, latitudes_deg=lats, lon_step_deg=1.40625)
    grid = Grid(cfg)
    base = dict(stencil="fd4", coriolis="spherical", rows_south_to_north=True,
                advection_form="flux", block_dt=400.0)
    base.update(kw)
    return PurePDEKernel(grid, **base), grid

_, g0 = make_kernel(w_diagnostic="plain")
w_lat = torch.cos(g0.latitudes).reshape(1,1,Hd,1)
pres_pa = g0.pressure  # (1,P,1,1) Pa
# trapezoid column-integration weights in Pa (centered half-thickness)
dp = (plev_hpa[1:]-plev_hpa[:-1])*100.0  # Pa
wint = torch.zeros(13)
for i in range(13):
    below = dp[i-1] if i>0 else None; above = dp[i] if i<12 else None
    if below is None: wint[i]=above/2
    elif above is None: wint[i]=below/2
    else: wint[i]=(below+above)/2
wint = wint.reshape(1,13,1,1)

def wrms(a):
    return float(((a**2)*w_lat).sum()/(torch.ones_like(a)*w_lat).sum()).__pow__(0.5)
def cwp(q):  # column water path kg/m^2
    return (q*wint).sum(dim=1,keepdim=True)/G

i0 = 12000; Khours = 10
frames = torch.from_numpy(np.asarray(arr[i0:i0+Khours+3]).astype(np.float32)).double()
def state(fi):
    z,T,r,u,v = unpack(frames[fi:fi+1])
    q = relative_to_specific_humidity(r, T, pres_pa)
    return {"u":u,"v":v,"t":T,"q":q,"z":z}
truth = [state(k) for k in range(Khours+1)]

levers = {
    "mass_consistent": dict(w_diagnostic="mass_consistent"),
    "obrien":          dict(w_diagnostic="obrien"),
    "mc+omega_free":   dict(w_diagnostic="mass_consistent", omega_free=("t","q")),
    "plain":           dict(w_diagnostic="plain"),
}
substeps = 9  # 9 * 400s = 3600s = 1 hour

# ================= ITEM 3: multi-step physics-only rollout A/B =================
print("="*78)
print("ITEM 3 — multi-step physics-only rollout, lat-weighted RMS error vs ERA5")
print("(physics core only, no trained corrector; free-run from ERA5 frame 0)")
print("-"*78)
rollout_states = {}
for name, kw in levers.items():
    k,_ = make_kernel(**kw)
    s = {kk:vv.clone() for kk,vv in truth[0].items()}
    errs = {}
    diverged = False
    for hr in range(1, Khours+1):
        for _ in range(substeps):
            out = k.step(s["u"],s["v"],s["t"],s["q"],s["z"])
            s = {kk:out[kk] for kk in ("u","v","t","q","z")}
            if not torch.isfinite(s["t"]).all():
                diverged=True; break
        if diverged: break
        if hr in (1,3,6,10):
            errs[hr] = {vv: wrms(s[vv]-truth[hr][vv]) for vv in ("z","t","q","u","v")}
    rollout_states[name] = s if not diverged else None
    tag = "DIVERGED@hr%d"%hr if diverged else "ok"
    print(f"\n[{name}] {tag}")
    for hr in sorted(errs):
        e=errs[hr]; print(f"  h{hr:>2}: z={e['z']:.3e} t={e['t']:.3e} q={e['q']:.3e} u={e['u']:.3e} v={e['v']:.3e}")

# ================= ITEM 1: zonal power spectrum (blurring/roughening) ==========
print("\n"+"="*78)
print("ITEM 1 — zonal power spectrum of physics rollout vs ERA5 @ hour 10")
print("(high-wavenumber energy ratio: <1 blurs, >1 roughens; T & z @500hPa)")
print("-"*78)
def zonal_hi_power(field2d):  # field (H,W) -> mean power in upper half of wavenumbers
    sp = torch.fft.rfft(field2d, dim=-1)
    p = (sp.abs()**2).mean(dim=0)  # avg over lat, per wavenumber
    hi = p[len(p)//2:].sum()
    return float(hi)
era = truth[Khours]
for name, s in rollout_states.items():
    if s is None:
        print(f"[{name}] diverged, skip"); continue
    row=[]
    for var,lev in (("t",i500),("z",i500)):
        rp = zonal_hi_power(s[var][0,lev]); ep = zonal_hi_power(era[var][0,lev])
        row.append(f"{var}500 hi-k ratio={rp/ep:.3f}")
    print(f"[{name}] " + "  ".join(row))

# ================= ITEM 2: column-water conservation ==========================
print("\n"+"="*78)
print("ITEM 2 — column water path drift (physics core) vs ERA5, kg/m^2 over 10h")
print("(imbalance = physics d/dt∫q dp − ERA5 d/dt∫q dp; 'water from nothing')")
print("-"*78)
era_cwp0 = cwp(truth[0]["q"]); era_cwp10 = cwp(truth[Khours]["q"])
era_dW = float(((era_cwp10-era_cwp0)*w_lat).sum()/w_lat.sum()/torch.ones_like(era_cwp0).sum()*era_cwp0.numel())
# simpler area-mean:
def amean(x): return float((x*w_lat).sum()/(torch.ones_like(x)*w_lat).sum())
era_dW = amean(era_cwp10)-amean(era_cwp0)
print(f"  ERA5 area-mean ΔW over 10h = {era_dW:+.4f} kg/m^2")
for name,s in rollout_states.items():
    if s is None: print(f"[{name}] diverged, skip"); continue
    phys_dW = amean(cwp(s["q"]))-amean(cwp(truth[0]["q"]))
    print(f"[{name}] physics ΔW = {phys_dW:+.4f}  imbalance(phys-ERA5) = {phys_dW-era_dW:+.4f} kg/m^2")

# ================= ITEM 4: online-loss gradient PoC ===========================
print("\n"+"="*78)
print("ITEM 4 — online-loss gradient PoC: does grad flow through K=2 kernel steps?")
print("-"*78)
k,_ = make_kernel(w_diagnostic="mass_consistent")
theta = torch.tensor(1.0, requires_grad=True)  # learnable scale on t-tendency
s = {kk:vv.clone() for kk,vv in truth[0].items()}
for _ in range(2):  # K=2 substeps with grad
    rhs = k.rhs(s["u"],s["v"],s["t"],s["q"],s["z"])
    dt = k.block_dt
    s = {"u":s["u"]+dt*rhs["u_t"], "v":s["v"]+dt*rhs["v_t"],
         "t":s["t"]+dt*theta*rhs["t_t"], "q":s["q"]+dt*rhs["q_t"], "z":s["z"]+dt*rhs["z_t"]}
loss = ((s["t"]-truth[0]["t"])**2).mean()  # dummy target; just checking grad
loss.backward()
g = theta.grad
print(f"  loss={float(loss):.3e}  dloss/dtheta={float(g):.3e}  finite={bool(torch.isfinite(g))}  nonzero={float(g)!=0.0}")
print("  => online training through the differentiable kernel is viable" if torch.isfinite(g) and float(g)!=0 else "  => grad path FAILED")
print("\n[done]")
