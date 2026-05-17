"""Sanity-чек физического pipeline: lat-weights, t_t formula, single substep.

Не использует ERA5 — синтетика. Цель: убедиться что:
  1. `Grid.latitudes` и `GeometryCPU.lat_weights` совпадают по форме.
  2. `t_t_formulation='adiabatic_omega'` даёт реалистичные |t_t| ~ 1e-5..1e-3 K/s.
  3. `t_t_formulation='legacy_paper'` даёт огромные |t_t| ~ 1e3..1e7 K/s — это
     подтверждает blow-up из tools/diagnose_first_substep.py.
  4. `use_universal_R=False` (default) даёт R_eff=287; True → 8.314.
  5. One Euler substep на синтетическом state не выдаёт NaN/Inf.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_physics_common import GeometryCPU
from utils.physics import Grid, GridConfig, PurePDEKernel


def _make_synthetic_state(H: int, W: int) -> dict[str, torch.Tensor]:
    """Физически разумный (B=1) state с (B, P=13, H, W) для u/v/t/q/z."""
    P = 13
    rng = torch.Generator().manual_seed(0)
    u = 15.0 * torch.randn(1, P, H, W, generator=rng)
    v = 10.0 * torch.randn(1, P, H, W, generator=rng)
    t = 240.0 + 30.0 * torch.rand(1, P, H, W, generator=rng)  # 240..270 K
    q = 0.005 * torch.rand(1, P, H, W, generator=rng)  # ~0..5 g/kg
    z = (
        5e4
        - 4e4 * torch.linspace(0, 1, P).reshape(1, P, 1, 1)
        + 100.0 * torch.randn(1, P, H, W, generator=rng)
    )
    return {"u": u, "v": v, "t": t, "q": q, "z": z}


def check_lat_weights(H: int = 32, W: int = 64) -> None:
    """`Grid.latitudes` vs `GeometryCPU.lat_weights` — должны быть согласованы."""
    grid = Grid(GridConfig(H=H, W=W))
    geom = GeometryCPU(H=H, W=W)
    cos_grid = torch.cos(grid.latitudes)
    lat_w_grid = H * cos_grid / cos_grid.sum()
    diff = (lat_w_grid - geom.lat_weights).abs().max().item()
    assert diff < 1e-5, f"lat_weights mismatch grid vs geom: {diff:.2e}"
    print(f"[lat_weights] H={H} OK, max|grid - geom|={diff:.2e}")


def check_t_t_formulations(H: int = 32, W: int = 64) -> None:
    """adiabatic_omega → реалистичные |t_t|; legacy_paper → blow-up scale."""
    grid = Grid(GridConfig(H=H, W=W))
    state = _make_synthetic_state(H, W)
    for formulation in ("adiabatic_omega", "legacy_paper"):
        kernel = PurePDEKernel(
            grid,
            stencil="fd4",
            coriolis="spherical",
            time_scheme="euler",
            boundary_x="periodic",
            boundary_y="replicate",
            boundary_z="replicate",
            t_t_formulation=formulation,
        )
        rhs = kernel.rhs(state["u"], state["v"], state["t"], state["q"], state["z"])
        t_t_max = rhs["t_t"].abs().max().item()
        print(f"[t_t/{formulation:>14s}] |t_t|max = {t_t_max:.3e} K/s")


def check_R_eff(H: int = 32, W: int = 64) -> None:
    """use_universal_R=False → R_eff=287; True → 8.314."""
    grid = Grid(GridConfig(H=H, W=W))
    k_default = PurePDEKernel(grid)
    k_universal = PurePDEKernel(grid, use_universal_R=True)
    assert k_default.R_eff == 287.0, f"default R_eff != 287: {k_default.R_eff}"
    assert k_universal.R_eff == 8.314, f"universal R_eff != 8.314: {k_universal.R_eff}"
    print(f"[R_eff] default={k_default.R_eff}, universal={k_universal.R_eff} OK")


def check_one_substep_no_nan(H: int = 32, W: int = 64) -> None:
    """One Euler substep не должен дать NaN/Inf."""
    grid = Grid(GridConfig(H=H, W=W))
    kernel = PurePDEKernel(
        grid,
        stencil="fd4",
        coriolis="spherical",
        time_scheme="euler",
        boundary_x="periodic",
        boundary_y="replicate",
        boundary_z="replicate",
    )
    state = _make_synthetic_state(H, W)
    out = kernel.step(state["u"], state["v"], state["t"], state["q"], state["z"])
    for k in ("u", "v", "t", "q", "z"):
        v = out[k]
        assert not torch.isnan(v).any() and not torch.isinf(v).any(), (
            f"{k}_new contains NaN/Inf: any_nan={torch.isnan(v).any().item()}"
        )
    print("[one_substep] no NaN/Inf in u_new/v_new/t_new/q_new/z_new OK")


def check_z_t_units(H: int = 32, W: int = 64) -> None:
    """БАГ-1 регрессия: get_z_t не должен быть ×100 занижен (Pa/hPa mismatch).

    Гидростатика: integral_z интегрирует с pixel_z в hPa, поэтому делитель
    давления тоже обязан быть в hPa. Эталон — SI (Pa-делитель, Pa-интеграл
    через M_z·100). Отношение correct/get_z_t должно быть ≈ 1, а |Δz| за 1ч
    при t_t=1e-3 K/s — порядка десятков м²/с², не O(1).
    """
    from utils.physics import integral_z

    grid = Grid(GridConfig(H=H, W=W))
    kernel = PurePDEKernel(grid, stencil="fd4", coriolis="spherical")
    P = grid.pressure.shape[1]
    t_t = torch.full((1, P, H, W), 1e-3)
    z_t_code = kernel.get_z_t(t_t)
    z_zt_si = -kernel.R_eff / grid.pressure * t_t  # Pa-делитель
    z_t_correct = integral_z(z_zt_si, grid.M_z * 100.0)  # Pa-интеграл
    ratio = (z_t_correct.abs().mean() / z_t_code.abs().mean()).item()
    assert abs(ratio - 1.0) < 1e-3, f"get_z_t off by {ratio:.4f}× (Pa/hPa mismatch?)"
    dz_1h = z_t_code.abs().max().item() * 3600.0
    assert 1.0 < dz_1h < 1e4, f"|Δz|@1h={dz_1h:.2f} m²/s² вне реалистичного диапазона"
    print(f"[z_t_units] correct/get_z_t={ratio:.4f}, |Δz|@1h={dz_1h:.1f} m²/s² OK")


def main() -> None:
    print("=== Sanity check: physics pipeline ===\n")
    check_lat_weights()
    check_R_eff()
    check_t_t_formulations()
    check_z_t_units()
    check_one_substep_no_nan()
    print("\n[all OK]")


if __name__ == "__main__":
    main()
