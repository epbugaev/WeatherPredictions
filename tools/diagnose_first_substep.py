"""Диагностика первого Euler-substep’а: что улетает первым на реальной ERA5.

Один IC (2005-01-01 00 UTC), один substep `block_dt=300 s` через FD-4 +
spherical Coriolis. Печатает абсолютные min/max/mean каждой промежуточной
величины, чтобы найти конкретный fact выражение, дающий Inf/NaN.

Структура отчёта:
    [STAGE name] tensor.shape  min  max  mean  abs_max  has_nan  has_inf

Stages:
  1. raw ERA5 state (u, v, t, r, z)
  2. q via Magnus
  3. spatial derivatives (u_x, u_y, ..., z_z)
  4. diagnostic w = integral_z(-u_x - v_y)
  5. Q = -L · z_z · w  (this is the suspected one)
  6. tendencies u_t, v_t, t_t, q_t, z_t
  7. updated state u_new = u + dt · u_t, ...
  8. check NaN/Inf in u_new

CPU, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_physics_common import (
    GeometryCPU,
    load_snapshot,
    open_memmap,
    relhum_to_specific,
    split_channels_69,
)
from utils.old_physics import make_weathergft_ops
from utils.paths import globe_memmap_path

L = 2.5e6
R = 8.314
c_p = 1005.0
BLOCK_DT = 300.0


def dump(name: str, t: torch.Tensor | float) -> None:
    if isinstance(t, (int, float)):
        print(f"  {name:<35s}  scalar  val={t:+.4e}")
        return
    arr = t.detach()
    has_nan = bool(torch.isnan(arr).any())
    has_inf = bool(torch.isinf(arr).any())
    finite = arr[torch.isfinite(arr)]
    if finite.numel() == 0:
        print(
            f"  {name:<35s}  shape={tuple(arr.shape)}  ALL_NONFINITE  NaN={has_nan} Inf={has_inf}"
        )
        return
    print(
        f"  {name:<35s}  shape={tuple(arr.shape)}  "
        f"min={finite.min().item():+.3e}  "
        f"max={finite.max().item():+.3e}  "
        f"mean={finite.mean().item():+.3e}  "
        f"|max|={finite.abs().max().item():.3e}  "
        f"NaN={has_nan} Inf={has_inf}"
    )


def main() -> None:
    memmap_path = globe_memmap_path()
    ts = pd.Timestamp("2005-01-01 00:00:00")
    print("=== Diagnostic first-substep dump ===")
    print(f"memmap: {memmap_path}")
    print(f"IC: {ts}")

    handle = open_memmap(memmap_path)
    x0 = load_snapshot(handle, ts, mean=None, std=None)  # (1, 69, 32, 64) raw
    print(f"\nloaded ERA5 snapshot shape={tuple(x0.shape)}")
    geom = GeometryCPU(H=32, W=64)
    ops = make_weathergft_ops(latents_size=(32, 64))

    print("\n[STAGE 1] raw ERA5 channels (physical units):")
    parts = split_channels_69(x0)
    for k in ("t2m", "u10", "v10", "tp", "z", "t", "r", "u", "v"):
        dump(f"raw[{k}]", parts[k])

    print("\n[STAGE 1b] grid buffers:")
    dump("pixel_x (m, lat-dep)", geom.pixel_x)
    dump("pixel_y (m, const)", geom.pixel_y)
    dump("pixel_z (hPa, levels)", geom.pixel_z)
    dump("pressure_pa (Pa)", geom.pressure_pa_t)

    print("\n[STAGE 2] q = (r/100) · q_s(t, p_pa) via Magnus (SI: Pa):")
    q = relhum_to_specific(parts["r"], parts["t"], geom.pressure_pa_t)
    dump("q (kg/kg)", q)

    # Build state for rollout
    u = parts["u"]
    v = parts["v"]
    t = parts["t"]
    z = parts["z"]

    print("\n[STAGE 3] spatial FD-4 derivatives:")
    u_x = ops.d_x(u)
    dump("u_x = ∂u/∂x (1/s)", u_x)
    u_y = ops.d_y(u)
    dump("u_y = ∂u/∂y (1/s)", u_y)
    u_z = ops.d_z(u)
    dump("u_z = ∂u/∂p (m/(s·hPa))", u_z)
    v_x = ops.d_x(v)
    dump("v_x", v_x)
    v_y = ops.d_y(v)
    dump("v_y", v_y)
    v_z = ops.d_z(v)
    dump("v_z", v_z)
    t_x = ops.d_x(t)
    dump("t_x (K/m)", t_x)
    t_y = ops.d_y(t)
    dump("t_y", t_y)
    t_z = ops.d_z(t)
    dump("t_z (K/hPa)", t_z)
    z_x = ops.d_x(z)
    dump("z_x (m²/s²/m)", z_x)
    z_y = ops.d_y(z)
    dump("z_y", z_y)
    z_z = ops.d_z(z)
    dump("z_z (m²/s²/hPa)", z_z)
    q_x = ops.d_x(q)
    dump("q_x", q_x)
    q_y = ops.d_y(q)
    dump("q_y", q_y)
    q_z = ops.d_z(q)
    dump("q_z", q_z)

    print("\n[STAGE 4] diagnostic w = integral_z(-(u_x + v_y)):")
    w_z = -(u_x + v_y)
    dump("w_z = -(u_x+v_y) (1/s)", w_z)
    w = ops.integral_z(w_z)
    dump("w = integral_z(w_z)  (hPa/s?)", w)
    print(
        "  NB: w units here are hPa·(1/s) = hPa/s, не m/s! "
        "Это omega (pressure velocity), а не вертикальная скорость в физ.метрах."
    )

    print("\n[STAGE 5] heating Q = -L · z_z · w  (the suspect):")
    Q = -L * z_z * w
    dump("Q = -L·z_z·w", Q)
    print(f"  L = {L:.3e},  |z_z|·|w|·L gives massive values if units mismatch.")
    Q_minus_zzw_over_cp = (Q - z_z * w) / c_p
    dump("(Q - z_z·w)/c_p (K/s)", Q_minus_zzw_over_cp)

    print("\n[STAGE 6] tendencies (WeatherGFT semantics, f=const=7.29e-5):")
    f_const = 7.29e-5
    u_t = -u * u_x - v * u_y - w * u_z + f_const * v - z_x
    v_t = -u * v_x - v * v_y - w * v_z - f_const * u - z_y
    t_t = Q_minus_zzw_over_cp - u * t_x - v * t_y - w * t_z
    z_zt = -R / geom.pressure_pa_t * t_t
    z_t = ops.integral_z(z_zt)
    q_t = -(u * q_x + v * q_y + w * q_z)
    dump("u_t (m/s²)", u_t)
    dump("v_t (m/s²)", v_t)
    dump("t_t (K/s)", t_t)
    dump("z_t (m²/s³)", z_t)
    dump("q_t (1/s)", q_t)

    print("\n[STAGE 6b] tendency components separately for u_t:")
    dump("  -u·u_x", -u * u_x)
    dump("  -v·u_y", -v * u_y)
    dump("  -w·u_z", -w * u_z)
    dump("  +f·v", f_const * v)
    dump("  -z_x", -z_x)
    print("\n[STAGE 6c] tendency components for t_t:")
    dump("  (Q - z_z·w)/c_p", Q_minus_zzw_over_cp)
    dump("  -u·t_x", -u * t_x)
    dump("  -v·t_y", -v * t_y)
    dump("  -w·t_z", -w * t_z)

    print("\n[STAGE 7] Euler substep (block_dt=300 s):")
    dt = BLOCK_DT
    u_new = u + dt * u_t
    dump("u_new", u_new)
    v_new = v + dt * v_t
    dump("v_new", v_new)
    t_new = t + dt * t_t
    dump("t_new", t_new)
    z_new = z + dt * z_t
    dump("z_new", z_new)
    q_new = q + dt * q_t
    dump("q_new", q_new)

    print("\n[STAGE 7b] |Δ| per variable:")
    dump("Δu = dt·u_t", dt * u_t)
    dump("Δv = dt·v_t", dt * v_t)
    dump("Δt = dt·t_t", dt * t_t)
    dump("Δz = dt·z_t", dt * z_t)
    dump("Δq = dt·q_t", dt * q_t)
    print("\n  Reference physical scales:")
    print("    |u|  ≈ 50 m/s          → realistic Δu/hour ≈ 2 m/s, per 300s ≈ 0.2 m/s")
    print("    |t|  ≈ 250 K           → realistic ΔT/hour ≈ 0.5 K,  per 300s ≈ 0.04 K")
    print("    |z|  ≈ 5e4 m²/s²       → realistic Δz/hour ≈ 1e2 m²/s², per 300s ≈ 8 m²/s²")
    print("    |q|  ≈ 1e-2 kg/kg      → realistic Δq/hour ≈ 1e-4,    per 300s ≈ 1e-5")

    # Diagnose which variable poisons first.
    has_any_nan = lambda x: bool(torch.isnan(x).any() or torch.isinf(x).any())
    print("\n[STAGE 8] which new state has NaN/Inf?")
    for name, x in (
        ("u_new", u_new),
        ("v_new", v_new),
        ("t_new", t_new),
        ("z_new", z_new),
        ("q_new", q_new),
    ):
        print(f"  {name}: NaN/Inf = {has_any_nan(x)}")


if __name__ == "__main__":
    main()
