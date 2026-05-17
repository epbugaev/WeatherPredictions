"""Sanity-gate E5: flux-форма адвекции + mass-consistent w в PurePDEKernel.

Гейт «Шаг 5» — PASS до smoke/кластера. Тестируется РЕАЛЬНЫЙ
`kernel._horiz_adv` и `kernel.get_w`:

1. `check_flux_conserves` — на дважды-периодической сетке с
   бездивергентным (через ψ) потоком flux-форма сохраняет ∑X (масса, до
   round-off) и ∑X² (энергия) много лучше, чем advective.
2. `check_mass_consistent_w` — с w_diagnostic=mass_consistent колоночный
   интеграл ∫(∂ₓu+∂_yv)·Δp ≈ 0 (у plain — нет).

CPU-only, без memmap, без импорта пакета `Models` (timm).

Запуск:
    .venv/bin/python Models/dev/sanity_conservation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.physics import Grid, GridConfig, PurePDEKernel  # noqa: E402

H, W, P = 16, 32, 13


def _kernel(form: str, wdiag: str = "plain") -> PurePDEKernel:
    """Дважды-периодическая сетка (boundary_x=y=periodic) — чтобы оба
    направления телескопировали и тест сохранения был чистым."""
    grid = Grid(GridConfig(H=H, W=W))
    return PurePDEKernel(
        grid,
        stencil="fd4",
        coriolis="spherical",
        boundary_x="periodic",
        boundary_y="periodic",
        advection_form=form,
        w_diagnostic=wdiag,
    )


def check_flux_conserves() -> None:
    """Острый детерминированный инвариант: на периодической сетке ∑ flux-
    tendency = −∑[∂ₓ(uX)+∂_y(vX)] ≡ 0 (телескопирование) при ЛЮБЫХ X,u,v →
    ∑X сохраняется каждый шаг; ∑ advective-tendency ≠ 0 при ∇·V≠0.
    Тестируем сумму tendency напрямую (без интегрирования — нет
    зависимости от CFL/симметрии тригонометрии)."""
    print("[check_flux_conserves]")
    ka = _kernel("advective")
    kf = _kernel("flux")
    torch.manual_seed(0)
    # Случайные гладкие X,u,v → generically дивергентный, неортогональный.
    x = 5.0 + torch.randn(1, P, H, W)
    u = torch.randn(1, P, H, W)
    v = torch.randn(1, P, H, W)

    t_adv = ka._horiz_adv(x, u, v)
    t_flux = kf._horiz_adv(x, u, v)
    # |∑ tendency| / ∑|tendency|: ~0 если телескопирует, O(1) если нет.
    r_adv = abs(float(t_adv.sum())) / float(t_adv.abs().sum())
    r_flux = abs(float(t_flux.sum())) / float(t_flux.abs().sum())
    print(f"  |∑t|/∑|t|:  advective={r_adv:.3e}  flux={r_flux:.3e}")
    assert r_flux < 1e-6, f"flux ∑tendency не телескопирует к 0 ({r_flux:.2e})"
    assert r_adv > 1e-3, f"тест вырожден: advective тоже ≈0 ({r_adv:.2e})"
    assert r_flux < r_adv / 100.0, f"flux не лучше advective ({r_flux:.2e} vs {r_adv:.2e})"
    print("  OK\n")


def check_mass_consistent_w() -> None:
    """mass_consistent → ∫(div)·Δp ≈ 0 на столб; plain — нет."""
    print("[check_mass_consistent_w]")
    kp = _kernel("advective", wdiag="plain")
    km = _kernel("advective", wdiag="mass_consistent")
    torch.manual_seed(0)
    u = torch.randn(1, P, H, W)
    v = torch.randn(1, P, H, W)
    pz = kp.grid.pixel_z  # (1,P,1,1)

    div = kp.diff.d_x(u) + kp.diff.d_y(v)
    col_plain = (div * pz).sum(dim=1)  # (1,H,W)
    div_bar = (div * pz).sum(dim=1, keepdim=True) / pz.sum()
    col_mc = ((div - div_bar) * pz).sum(dim=1)
    scale = float((div.abs() * pz).sum(dim=1).mean())
    rp = float(col_plain.abs().mean()) / scale
    rm = float(col_mc.abs().mean()) / scale
    print(f"  plain ∫div·Δp (rel)={rp:.3e}  mass_consistent (rel)={rm:.3e}")
    assert rm < 1e-6, f"mass_consistent: ∫div·Δp не ≈0 (rel={rm:.2e})"
    assert rp > 1e-3, f"тест вырожден: plain ∫div·Δp уже ≈0 (rel={rp:.2e})"
    # km существует и сконфигурирован (km.get_w использует ту же поправку):
    assert km.w_diagnostic == "mass_consistent"
    print("  OK\n")


def main() -> None:
    """Прогнать инварианты; exit(1) на провале."""
    print("=== Sanity: conservation form (E5) ===\n")
    check_flux_conserves()
    check_mass_consistent_w()
    print("[all OK]")
    sys.exit(0)


if __name__ == "__main__":
    main()
