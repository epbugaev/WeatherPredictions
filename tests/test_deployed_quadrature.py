"""End-to-end: physics_vertical_quadrature flows YAML->model->kernel buffers."""
import os, sys
import torch
REPO = os.environ.get("REPO_ROOT", "/Users/buzaev-fa/WeatherPredictions")
sys.path.insert(0, REPO)
import torch.nn as nn  # noqa: E402
import utils.physics_hybrid as ph  # noqa: E402
from utils.physics_residual import PhysicsResidualMixin  # noqa: E402

torch.set_default_dtype(torch.float32)
fails = []

class TinyModel(nn.Module, PhysicsResidualMixin):
    """Minimal host for init_physics_residual (deployed B1 config values)."""
    def __init__(self, **kw):
        super().__init__()
        self.init_physics_residual(
            C_data=69, H_data=32, W_data=64, downscaling_factor_all=4,
            use_physics_residual_corrector=True,
            physics_residual_apply_to="upper_air_only",
            physics_residual_hybrid_mode="stable_physical_v2",
            physics_residual_input_space="physical",
            physics_feature_mode="tendency",
            physics_lat_start_deg=18.28125, physics_dlat_deg=5.625, physics_dlon_deg=5.625,
            physics_w_diagnostic="mass_consistent",
            **kw,
        )

def find_kernel(model):
    for m in model.modules():
        if isinstance(m, ph.PDE_kernel):
            return m
    return None

# default (no key) -> rectangle, bit-for-bit
m_def = TinyModel()
k = find_kernel(m_def)
q = getattr(k, "vertical_quadrature", "MISSING")
print(f"[E1] default deployed kernel vertical_quadrature = {q!r}")
if q != "rectangle": fails.append(f"default not rectangle: {q}")
if not torch.equal(k.M_z_quad, ph.M_z): fails.append("default M_z != global")

# explicit trapezoid via kwarg (as a YAML key would arrive)
m_trap = TinyModel(physics_vertical_quadrature="trapezoid")
kt = find_kernel(m_trap)
print(f"[E2] trapezoid deployed kernel vertical_quadrature = {kt.vertical_quadrature!r}")
if kt.vertical_quadrature != "trapezoid": fails.append("trap not set")
depth = float(kt.w_int.sum())
print(f"[E3] trapezoid kernel w_int sum = {depth:.3f} (950)")
if abs(depth - 950.0) > 1e-3: fails.append(f"trap w_int sum {depth}")
iz = ph.integral_z(torch.ones(1,13,4,4), kt.M_z_quad)
print(f"[E4] trapezoid kernel surface anchor = {float(iz[0,-1,0,0].abs()):.3e} (0)")
if float(iz[0,-1,0,0].abs()) > 1e-5: fails.append("trap anchor")

# invalid rejected
try:
    TinyModel(physics_vertical_quadrature="simpson"); fails.append("invalid not raised")
except ValueError:
    print("[E5] invalid physics_vertical_quadrature raises: True")

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
