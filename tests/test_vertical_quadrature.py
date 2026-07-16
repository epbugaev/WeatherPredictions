"""Unit tests for vertical_quadrature flag in Grid (physics.py priority 1)."""
import os, sys
import torch

REPO = os.environ.get("REPO_ROOT", "/Users/buzaev-fa/WeatherPredictions")
sys.path.insert(0, REPO)
from utils.physics import GridConfig, Grid, integral_z, PurePDEKernel  # noqa: E402

torch.set_default_dtype(torch.float64)
H, W = 8, 16
plevs = torch.tensor([50,100,150,200,250,300,400,500,600,700,850,925,1000.])
p_top, p_s = 50.0, 1000.0

def make(quad):
    cfg = GridConfig(H=H, W=W, vertical_quadrature=quad,
                     latitudes_deg=tuple(float(24+ i*1.40625) for i in range(H)),
                     lon_step_deg=1.40625)
    return Grid(cfg)

g_rect = make("rectangle")
g_trap = make("trapezoid")
fails = []

# --- 1. rectangle is bit-identical to legacy construction ---
P = 13
M_legacy = torch.zeros(P, P)
pz = torch.tensor([50,50,50,50,50,75,100,100,100,125,112,75,75.])
for i in range(P):
    for j in range(P):
        if i <= j:
            M_legacy[i, j] = pz[j]
ok = torch.equal(g_rect.M_z, M_legacy)
print(f"[1] rectangle M_z bit-identical to legacy: {ok}")
if not ok: fails.append("rectangle M_z changed")
ok = torch.equal(g_rect.w_int.reshape(-1), pz)
print(f"[1] rectangle w_int == pixel_z: {ok}")
if not ok: fails.append("rectangle w_int changed")

# --- 2. trapezoid: surface anchor integral_z(ones)[surface] == 0 ---
ones = torch.ones(1, P, H, W)
iz_trap = integral_z(ones, g_trap.M_z)
surf = float(iz_trap[0, -1, 0, 0].abs())
print(f"[2] trapezoid integral_z(ones)[surface] = {surf:.3e} (must be 0)")
if surf > 1e-9: fails.append(f"trap surface anchor {surf}")

# --- 3. trapezoid: total column depth == p_s - p_top == 950 ---
top = float(iz_trap[0, 0, 0, 0])
print(f"[3] trapezoid column depth integral_z(ones)[top] = {top:.3f} (must be {p_s-p_top})")
if abs(top - (p_s - p_top)) > 1e-6: fails.append(f"trap depth {top}")

# --- 4. rectangle still has the +6.5% bias (documents the bug) ---
iz_rect = integral_z(ones, g_rect.M_z)
print(f"[4] rectangle depth[top] = {float(iz_rect[0,0,0,0]):.1f} (biased 1012), "
      f"surface = {float(iz_rect[0,-1,0,0]):.1f} (spurious 75)")

# --- 5. trapezoid w_int sums to true column depth ---
s = float(g_trap.w_int.sum())
print(f"[5] trapezoid w_int sum = {s:.3f} (must be {p_s-p_top})")
if abs(s - (p_s - p_top)) > 1e-6: fails.append(f"trap w_int sum {s}")

# --- 6. d_z divisor unchanged (still local pixel_z, both configs) ---
ok = torch.equal(g_rect.pixel_z, g_trap.pixel_z)
print(f"[6] pixel_z (d_z divisor) identical rect vs trap: {ok}")
if not ok: fails.append("pixel_z divisor changed under trapezoid")

# --- 7. invalid value raises ---
try:
    make("simpson"); fails.append("invalid quad did not raise")
except ValueError:
    print("[7] invalid vertical_quadrature raises ValueError: True")

# --- 8. kernel builds and hydrostatic surface anchor holds under trapezoid ---
k = PurePDEKernel(g_trap, stencil="fd4", coriolis="spherical",
                  rows_south_to_north=True, w_diagnostic="mass_consistent", block_dt=400.)
T = 250 + 5*torch.randn(2, P, H, W)
q = torch.zeros(2, P, H, W); u = torch.randn(2,P,H,W); v = torch.randn(2,P,H,W)
z = 5e4 + torch.randn(2,P,H,W)*1e3
rhs = k.rhs(u, v, T, q, z)
zt_surf = float((rhs["z_t"][:, -1]**2).mean().sqrt())
print(f"[8] trapezoid kernel z_t at surface rms = {zt_surf:.3e} (must be ~0)")
if zt_surf > 1e-9: fails.append(f"kernel surface anchor {zt_surf}")

# ============ physics_hybrid (deployed path) ============
import utils.physics_hybrid as ph  # noqa: E402
print("\n--- physics_hybrid.py (deployed path) ---")

# H1. build_vertical_quadrature('rectangle') bit-identical to module globals
mz_rect, wint_rect = ph.build_vertical_quadrature("rectangle")
ok = torch.equal(mz_rect, ph.M_z)
print(f"[H1] rectangle M_z == module global M_z: {ok}")
if not ok: fails.append("hybrid rectangle M_z changed")
ok = torch.equal(wint_rect, ph.pixel_z)
print(f"[H1] rectangle w_int == module pixel_z: {ok}")
if not ok: fails.append("hybrid rectangle w_int changed")

# H2. trapezoid anchor + depth via module integral_z with passed matrix
mz_trap, wint_trap = ph.build_vertical_quadrature("trapezoid")
iz = ph.integral_z(torch.ones(1, 13, 4, 4), mz_trap)
surf = float(iz[0, -1, 0, 0].abs()); top = float(iz[0, 0, 0, 0])
print(f"[H2] trapezoid surface anchor={surf:.3e} (0), depth={top:.3f} (950)")
if surf > 1e-9 or abs(top - 950) > 1e-6: fails.append("hybrid trap anchor/depth")

# H3. legacy integral_z(x) with no matrix arg == module-global (bit-for-bit back-compat)
x = torch.randn(2, 13, 4, 4)
ok = torch.equal(ph.integral_z(x), ph.integral_z(x, ph.M_z))
print(f"[H3] integral_z(x) back-compat (defaults to global M_z): {ok}")
if not ok: fails.append("hybrid integral_z back-compat")

# H4. PDE_kernel default vertical_quadrature == 'rectangle', buffers match global
k_def = ph.PDE_kernel(in_dim=69, physics_part_coef=1.0, grid_h=8,
                      lat_start_deg=18.28125, dlat_deg=5.625, dlon_deg=5.625)
print(f"[H4] PDE_kernel default vertical_quadrature = {k_def.vertical_quadrature!r}")
if k_def.vertical_quadrature != "rectangle": fails.append("hybrid default not rectangle")
ok = torch.equal(k_def.M_z_quad, ph.M_z)
print(f"[H4] default kernel M_z_quad == module global: {ok}")
if not ok: fails.append("hybrid default kernel M_z changed")

# H5. PDE_kernel trapezoid kernel builds and has corrected buffers
k_trap = ph.PDE_kernel(in_dim=69, physics_part_coef=1.0, grid_h=8,
                       lat_start_deg=18.28125, dlat_deg=5.625, dlon_deg=5.625,
                       vertical_quadrature="trapezoid")
ok = abs(float(k_trap.w_int.sum()) - 950.0) < 1e-6
print(f"[H5] trapezoid kernel w_int sum = {float(k_trap.w_int.sum()):.3f} (950): {ok}")
if not ok: fails.append("hybrid trap kernel w_int sum")

# H6. invalid raises
try:
    ph.build_vertical_quadrature("simpson"); fails.append("hybrid invalid did not raise")
except ValueError:
    print("[H6] invalid vertical_quadrature raises: True")

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
