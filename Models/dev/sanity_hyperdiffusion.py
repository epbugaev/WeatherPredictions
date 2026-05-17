"""Sanity-gate E2: НЕЯВНАЯ зональная ∇⁴-гипердиффузия (B1-fixed).

Прежняя версия тестировала явный `_biharmonic`/`hyperdiff_k4` и
ПРОПУСТИЛА B1: явный −K4·∇⁴ с единым K4 на полюсе усиливал 2Δx ×172/шаг
(explicit-нестабильность). Теперь гипердиффузия — безусловно устойчивый
неявный спектральный фильтр `kernel._apply_hyperdiffusion`. Гейт ловит B1:

1. `check_unconditional_contraction` — фильтр НИКОГДА не усиливает (output
   max-abs ≤ input) при ЛЮБЫХ τ/dt, на ВСЕХ широтах включая полюс
   (именно это нарушал старый явный ∇⁴). Δx-независим (одинаков по широте).
2. `check_scale_selective` — 2Δx-мода гаснет (e-folding≈τ за шаг), m=0 и
   крупный масштаб (kx→0) сохранены точно.

CPU-only, без memmap/timm. Запуск:
    .venv/bin/python Models/dev/sanity_hyperdiffusion.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.physics import Grid, GridConfig, PurePDEKernel  # noqa: E402

H, W, P = 32, 64, 13


def _kernel(tau_hours: float, dt: float = 300.0) -> PurePDEKernel:
    grid = Grid(GridConfig(H=H, W=W))
    return PurePDEKernel(
        grid,
        stencil="fd4",
        coriolis="spherical",
        block_dt=dt,
        time_scheme="ssp_rk3",
        hyperdiffusion=True,
        hyperdiff_tau_hours=tau_hours,
    )


def check_unconditional_contraction() -> None:
    """Фильтр не усиливает НИКОГДА (любой τ/dt, все широты) — это и есть
    отсутствие B1 (старый явный ∇⁴ давал ×172 на полюсе)."""
    print("[check_unconditional_contraction]")
    torch.manual_seed(0)
    # Жёсткий режим: крошечный τ + огромный dt (старый явный ∇⁴ тут взорвался
    # бы катастрофически). Шумное поле со всеми масштабами.
    worst = 0.0
    for tau_h, dt in ((1e-3, 3600.0), (0.5, 300.0), (6.0, 300.0)):
        k = _kernel(tau_h, dt)
        x = torch.randn(1, P, H, W)
        y = k._apply_hyperdiffusion(x)
        ratio = float(y.abs().amax() / x.abs().amax())
        worst = max(worst, ratio)
        assert bool(torch.isfinite(y).all()), f"NaN/Inf при τ={tau_h}h dt={dt}"
        assert ratio <= 1.0 + 1e-5, (
            f"УСИЛЕНИЕ (B1!) τ={tau_h}h dt={dt}: |y|max/|x|max={ratio:.3e} > 1"
        )
    print(f"  max(|y|max/|x|max) over harsh regimes = {worst:.4f} (≤1 ✓)")
    # Δx-независимость: лат-однородный зональный вход → лат-однородный выход
    # (фильтр нормирован на Найквист, символ без широтной зависимости).
    k = _kernel(6.0, 300.0)
    jx = torch.arange(W).float()
    row = torch.cos(5.0 * 2 * math.pi * jx / W) + 0.4 * torch.cos(math.pi * jx)
    xu = row.reshape(1, 1, 1, W).expand(1, P, H, W).contiguous()
    yu = k._apply_hyperdiffusion(xu)
    spread = float((yu[0, 0] - yu[0, 0, 0:1, :]).abs().max())  # ряд-в-ряд разброс
    print(f"  lat-uniform in → lat row-to-row spread = {spread:.2e} (Δx-независим ✓)")
    assert spread < 1e-5, f"фильтр зависит от широты ({spread:.2e})"
    print("  OK\n")


def check_scale_selective() -> None:
    """2Δx гаснет (≈e-folding τ за шаг); m=0 и k=1 сохранены."""
    print("[check_scale_selective]")
    tau_h, dt = 6.0, 300.0
    k = _kernel(tau_h, dt)
    j = torch.arange(W).reshape(1, 1, 1, W).float()
    nyq = torch.cos(math.pi * j).expand(1, P, H, W).contiguous()  # 2Δx (±1)
    k1 = torch.cos(2.0 * math.pi * j / W).expand(1, P, H, W).contiguous()  # крупн.
    mean0 = torch.ones(1, P, H, W)  # m=0

    a_nyq = float(k._apply_hyperdiffusion(nyq).abs().amax() / nyq.abs().amax())
    a_k1 = float(k._apply_hyperdiffusion(k1).abs().amax() / k1.abs().amax())
    a_m0 = float(k._apply_hyperdiffusion(mean0).abs().amax() / mean0.abs().amax())
    expect_nyq = 1.0 / (1.0 + dt / (tau_h * 3600.0))  # теор. множитель Найквиста
    print(f"  2Δx={a_nyq:.4f} (теор {expect_nyq:.4f})  k=1={a_k1:.5f}  m=0={a_m0:.6f}")
    assert abs(a_nyq - expect_nyq) < 1e-3, f"2Δx-затухание ≠ теории ({a_nyq:.4f})"
    assert a_k1 > 0.999, f"крупный масштаб тронут ({a_k1:.4f})"
    assert abs(a_m0 - 1.0) < 1e-5, f"m=0 (среднее) не сохранено ({a_m0:.6f})"
    assert a_nyq < a_k1, "не scale-selective"
    print("  OK\n")


def main() -> None:
    """Прогнать инварианты; exit(1) на провале."""
    print("=== Sanity: implicit hyperdiffusion (E2, B1-fixed) ===\n")
    check_unconditional_contraction()
    check_scale_selective()
    print("[all OK]")
    sys.exit(0)


if __name__ == "__main__":
    main()
