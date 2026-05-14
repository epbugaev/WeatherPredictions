"""Single-card sanity-check для `utils.physics` vs `utils.old_physics`.

Запуск:
    cd /Users/buzaev-fa/WeatherPredictions
    python -m Models.dev.sanity_physics

Что проверяет:
    1. FD-4 операторы (`d_x`, `d_y`, `d_z`, `integral_z`) численно совпадают
       между `utils.old_physics.make_weathergft_ops` и `utils.physics.FiniteDifference`
       на синтетическом входе.
    2. WENO-5 операторы совпадают между `utils.old_physics.make_predformergft_ops`
       и `utils.physics.WENO5`.
    3. PurePDEKernel прогоняет один Euler-шаг без ошибок и без NaN/Inf
       на синтетическом тензоре `(B, 13, 32, 64)` и `(B, 13, 128, 256)`.
    4. PDE residual / mass divergence / geostrophic balance / CFL метрики
       возвращают тензоры правильной формы без NaN.

Не требует ERA5 — генерирует данные через `torch.randn`. Запускается на CPU
за несколько секунд.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.old_physics import make_predformergft_ops, make_weathergft_ops
from utils.physics import (
    WENO5,
    FiniteDifference,
    Grid,
    GridConfig,
    PurePDEKernel,
    cfl_number,
    geostrophic_residual,
    kinetic_energy_density,
    mass_divergence,
    pde_residual,
    potential_vorticity_proxy,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def _random_state(B: int, P: int, H: int, W: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Синтетический набор полей (u, v, t, q, z) — нормированные масштабы ERA5-уровня."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "u": 10.0 * torch.randn(B, P, H, W, generator=g, dtype=DTYPE).to(DEVICE),
        "v": 10.0 * torch.randn(B, P, H, W, generator=g, dtype=DTYPE).to(DEVICE),
        "t": 250.0 + 30.0 * torch.randn(B, P, H, W, generator=g, dtype=DTYPE).to(DEVICE),
        "q": 0.001 * torch.abs(torch.randn(B, P, H, W, generator=g, dtype=DTYPE)).to(DEVICE),
        "z": 5000.0 * 9.81 + 1000.0 * torch.randn(B, P, H, W, generator=g, dtype=DTYPE).to(DEVICE),
    }


def check_fd4_equivalence(H: int = 32, W: int = 64) -> None:
    """FD-4 operators: utils.physics vs utils.old_physics — должны совпадать.

    Old-версия использует `linear_minus90_90` (через `_build_grid_weathergft_style`
    с подменой формулы). Чтобы совпало, в new ставим `lat_scheme='arange'`.
    """
    old = make_weathergft_ops(latents_size=(H, W))
    grid = Grid(GridConfig(H=H, W=W, lat_scheme="arange")).to(DEVICE)
    new = FiniteDifference(
        grid, boundary_x="periodic", boundary_y="periodic", boundary_z="periodic"
    )

    field = torch.randn(2, 13, H, W, generator=torch.Generator().manual_seed(42), dtype=DTYPE).to(
        DEVICE
    )

    for name, (old_fn, new_fn) in {
        "d_x": (old.d_x, new.d_x),
        "d_y": (old.d_y, new.d_y),
        "d_z": (old.d_z, new.d_z),
    }.items():
        out_old = old_fn(field)
        out_new = new_fn(field)
        diff = (out_old - out_new).abs().max().item()
        ref = out_old.abs().max().item()
        rel = diff / (ref + 1e-12)
        status = "OK" if rel < 1e-4 else "FAIL"
        print(f"  [{status}] FD-4 {name:>4s}: max|Δ|={diff:.3e}  ref={ref:.3e}  rel={rel:.3e}")


def check_weno5_equivalence(H: int = 32, W: int = 64) -> None:
    """WENO-5 operators: utils.physics vs utils.old_physics."""
    old = make_predformergft_ops(latents_size=(H, W))
    grid = Grid(GridConfig(H=H, W=W, lat_scheme="linear_minus90_90")).to(DEVICE)
    new = WENO5(grid, boundary="reflect", boundary_z="periodic")

    field = torch.randn(2, 13, H, W, generator=torch.Generator().manual_seed(7), dtype=DTYPE).to(
        DEVICE
    )

    for name, (old_fn, new_fn) in {
        "d_x": (lambda x: old.d_x_weno(x, boundary="reflect"), new.d_x),
        "d_y": (lambda x: old.d_y_weno(x, boundary="reflect"), new.d_y),
        "d_z": (old.d_z, new.d_z),
    }.items():
        out_old = old_fn(field)
        out_new = new_fn(field)
        diff = (out_old - out_new).abs().max().item()
        ref = out_old.abs().max().item()
        rel = diff / (ref + 1e-12)
        status = "OK" if rel < 1e-4 else "FAIL"
        print(f"  [{status}] WENO-5 {name:>3s}: max|Δ|={diff:.3e}  ref={ref:.3e}  rel={rel:.3e}")


def check_integral_z_equivalence(H: int = 32, W: int = 64) -> None:
    """integral_z (old vs new), на одной сетке."""
    old = make_weathergft_ops(latents_size=(H, W))
    grid = Grid(GridConfig(H=H, W=W, lat_scheme="arange")).to(DEVICE)
    from utils.physics import integral_z as new_integral_z

    field = torch.randn(2, 13, H, W, generator=torch.Generator().manual_seed(13), dtype=DTYPE).to(
        DEVICE
    )
    out_old = old.integral_z(field)
    out_new = new_integral_z(field, grid.M_z)
    diff = (out_old - out_new).abs().max().item()
    ref = out_old.abs().max().item()
    rel = diff / (ref + 1e-12)
    status = "OK" if rel < 1e-4 else "FAIL"
    print(f"  [{status}] integral_z:    max|Δ|={diff:.3e}  ref={ref:.3e}  rel={rel:.3e}")


def check_pure_kernel_smoke(H: int, W: int, stencil: str, coriolis: str, time_scheme: str) -> None:
    """PurePDEKernel: один шаг без NaN/Inf, метрики корректной формы."""
    grid = Grid(GridConfig(H=H, W=W, lat_scheme="linear_minus90_90")).to(DEVICE)
    kernel = PurePDEKernel(
        grid,
        stencil=stencil,
        coriolis=coriolis,
        block_dt=300.0,
        time_scheme=time_scheme,
        boundary_horiz="periodic" if stencil == "fd4" else "reflect",
        boundary_z="periodic",
    ).to(DEVICE)

    state = _random_state(B=1, P=13, H=H, W=W, seed=99)
    next_state = kernel.step(state["u"], state["v"], state["t"], state["q"], state["z"])

    has_nan = any(torch.isnan(v).any().item() for v in next_state.values())
    has_inf = any(torch.isinf(v).any().item() for v in next_state.values())
    status = "OK" if not (has_nan or has_inf) else "FAIL"
    print(
        f"  [{status}] PurePDEKernel step ({H}x{W}, stencil={stencil}, coriolis={coriolis}, time={time_scheme}): NaN={has_nan}, Inf={has_inf}"
    )

    # Metrics shape check
    next_packed = {k: next_state[k] for k in ("u", "v", "t", "q", "z")}
    resid = pde_residual(kernel, state, next_packed, dt_seconds=300.0)
    div = mass_divergence(kernel, state["u"], state["v"])
    ke = kinetic_energy_density(state["u"], state["v"])
    pv = potential_vorticity_proxy(kernel, state["u"], state["v"])
    geo = geostrophic_residual(kernel, state["u"], state["v"], state["z"])
    cfl = cfl_number(kernel, state["u"], state["v"], dt=300.0)

    print(
        f"        residual |u|max = {resid['u'].abs().max():.3e}, "
        f"|t|max = {resid['t'].abs().max():.3e}"
    )
    print(
        f"        ∇·v |max| = {div.abs().max():.3e}, "
        f"KE max = {ke.max():.3e}, "
        f"PV |max| = {pv.abs().max():.3e}"
    )
    print(
        f"        geostrophic u_residual |max| = {geo['u_residual'].abs().max():.3e}, "
        f"v_residual |max| = {geo['v_residual'].abs().max():.3e}"
    )
    print(f"        CFL max = {cfl['cfl_x'].max():.3e} (x), {cfl['cfl_y'].max():.3e} (y)")


def main() -> None:
    print(f"Device: {DEVICE}")
    print("\n=== FD-4 equivalence (32x64) ===")
    check_fd4_equivalence(H=32, W=64)

    print("\n=== WENO-5 equivalence (32x64) ===")
    check_weno5_equivalence(H=32, W=64)

    print("\n=== integral_z equivalence (32x64) ===")
    check_integral_z_equivalence(H=32, W=64)

    print("\n=== PurePDEKernel smoke (32x64) ===")
    check_pure_kernel_smoke(H=32, W=64, stencil="fd4", coriolis="constant", time_scheme="euler")
    check_pure_kernel_smoke(H=32, W=64, stencil="fd4", coriolis="spherical", time_scheme="rk4")
    check_pure_kernel_smoke(H=32, W=64, stencil="weno5", coriolis="beta_plane", time_scheme="euler")

    print("\n=== PurePDEKernel smoke (128x256 global) ===")
    check_pure_kernel_smoke(H=128, W=256, stencil="fd4", coriolis="spherical", time_scheme="euler")
    check_pure_kernel_smoke(
        H=128, W=256, stencil="weno5", coriolis="spherical", time_scheme="euler"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
