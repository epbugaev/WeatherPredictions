"""Sanity-gate E3: корректность полярного Фурье-фильтра в PurePDEKernel.

Гейт «Шаг 3» — PASS до smoke/кластера. Тестируется РЕАЛЬНЫЙ
`kernel._apply_polar_filter`:

1. `check_zonal_mean_preserved` — m=0 не трогается → зональное среднее
   каждой широты и глобальное среднее сохраняются точно (фильтр не
   вносит/не убирает массу).
2. `check_pole_vs_equator` — высокая зональная мода m=W/4 у полюса
   (|φ|>φ0) убирается, та же мода у экватора (|φ|≤φ0) сохраняется.
3. `check_finite_real` — выход конечный, вещественный, той же формы.

CPU-only, без memmap и без импорта пакета `Models` (timm).

Запуск:
    .venv/bin/python Models/dev/sanity_polar_filter.py
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
PHI0_DEG = 60.0


def _kernel() -> PurePDEKernel:
    grid = Grid(GridConfig(H=H, W=W))
    return PurePDEKernel(
        grid,
        stencil="fd4",
        coriolis="spherical",
        time_scheme="ssp_rk3",
        polar_filter=True,
        polar_filter_lat_deg=PHI0_DEG,
    )


def _rms_lon(field_row: torch.Tensor) -> float:
    """RMS по долготе для одной широты (амплитуда зональной волны)."""
    return float(torch.sqrt(torch.mean(field_row**2)))


def check_zonal_mean_preserved() -> None:
    """m=0 нетронут → зональное и глобальное среднее сохраняются точно."""
    print("[check_zonal_mean_preserved]")
    k = _kernel()
    torch.manual_seed(0)
    x = torch.randn(1, P, H, W) + 3.0  # ненулевое m=0
    y = k._apply_polar_filter(x)
    dz = float((y.mean(dim=-1) - x.mean(dim=-1)).abs().max())  # per-lat zonal mean
    dg = abs(float(y.mean()) - float(x.mean()))  # глобальное среднее
    print(f"  max|Δ zonal_mean|={dz:.3e}  |Δ global_mean|={dg:.3e}")
    assert dz < 1e-4, f"зональное среднее не сохранено ({dz:.3e}) — m=0 тронут"
    assert dg < 1e-4, f"глобальное среднее не сохранено ({dg:.3e})"
    print("  OK\n")


def check_pole_vs_equator() -> None:
    """m=W/4 у полюса убрана, у экватора сохранена."""
    print("[check_pole_vs_equator]")
    k = _kernel()
    m0 = W // 4  # высокая зональная мода (16 при W=64)
    j = torch.arange(W).float()
    wave = torch.cos(2.0 * math.pi * m0 * j / W)  # (W,)
    x = wave.reshape(1, 1, 1, W).expand(1, P, H, W).contiguous()
    y = k._apply_polar_filter(x)

    lat_deg = k.grid.latitudes * 180.0 / math.pi  # (H,)
    eq_row = int(torch.argmin(lat_deg.abs()))  # |φ|→0, внутри φ0 → keep
    pole_row = int(torch.argmax(lat_deg.abs()))  # |φ| макс, > φ0 → filtered
    a0 = _rms_lon(x[0, 0, eq_row])
    a_eq = _rms_lon(y[0, 0, eq_row])
    a_pole = _rms_lon(y[0, 0, pole_row])
    print(
        f"  m={m0}  amp0={a0:.3f}  eq(φ={lat_deg[eq_row]:.1f}°)={a_eq:.3f}  "
        f"pole(φ={lat_deg[pole_row]:.1f}°)={a_pole:.3e}"
    )
    assert a_eq > 0.9 * a0, f"экв. мода затронута ({a_eq:.3f} ≤ 0.9·{a0:.3f})"
    assert a_pole < 0.1 * a0, f"полюсная мода НЕ убрана ({a_pole:.3e} ≥ 0.1·{a0:.3f})"
    print("  OK\n")


def check_finite_real() -> None:
    """Выход конечный, вещественный, той же формы."""
    print("[check_finite_real]")
    k = _kernel()
    torch.manual_seed(1)
    x = torch.randn(1, P, H, W)
    y = k._apply_polar_filter(x)
    assert y.shape == x.shape, f"форма изменилась: {y.shape} ≠ {x.shape}"
    assert not y.is_complex(), "выход комплексный — нет irfft"
    assert bool(torch.isfinite(y).all()), "NaN/Inf в выходе фильтра"
    print("  OK\n")


def main() -> None:
    """Прогнать инварианты; exit(1) на провале."""
    print("=== Sanity: polar Fourier filter (E3) ===\n")
    check_zonal_mean_preserved()
    check_pole_vs_equator()
    check_finite_real()
    print("[all OK]")
    sys.exit(0)


if __name__ == "__main__":
    main()
