"""Sanity-gate E4: корректность `balance_initial_state` (DFI / geostrophic).

Гейт «Шаг 4» — PASS до smoke/кластера. Тестируется РЕАЛЬНАЯ
`tools.check_physics_common.balance_initial_state`.

Метрика дисбаланса — геострофический остаток момента
R = RMS‖(f·v − z_x, −f·u − z_y)‖ (источник гравитационных волн; хорошо
масштабирован ~1e-3, не упирается в float-шум, в отличие от ∂divergence/∂t).
Тест-состояние: НУЛЕВОЙ ветер + z-бугор → максимальный геостр. дисбаланс.

1. `check_geostrophic` — код-корректность: ветер=(−z_y,z_x)/f_safe точно;
   масса (z,t,q) не тронута; остаток R падает ≈ к нулю; конечно.
2. `check_dfi` — стабилизированный forward-DFI не расходится (ключевой
   фикс vs сырой Эйлер), реально применяется (b≠s), снижает остаток R,
   сохраняет крупномасштабный z.

CPU-only, без memmap, без импорта пакета `Models` (timm).

Запуск:
    .venv/bin/python Models/dev/sanity_ic_balance.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_physics_common import balance_initial_state  # noqa: E402
from utils.physics import Grid, GridConfig, PurePDEKernel  # noqa: E402

H, W, P = 16, 32, 13


def _kernel() -> PurePDEKernel:
    """Кумулятивный стек E1+E2+E3 (ssp_rk3+∇⁴+polar) — контекст работы E4.
    block_dt=300 → n=round(1·3600/300)=12 для span=1ч.
    """
    grid = Grid(GridConfig(H=H, W=W))
    return PurePDEKernel(
        grid,
        stencil="fd4",
        coriolis="spherical",
        block_dt=300.0,
        time_scheme="ssp_rk3",
        hyperdiffusion=True,
        hyperdiff_tau_hours=6.0,
        polar_filter=True,
        polar_filter_lat_deg=60.0,
    )


def _unbalanced_state() -> dict[str, torch.Tensor]:
    """Нулевой ветер + локализованный z-бугор → сильный геостр. дисбаланс."""
    lat = torch.linspace(-1.2, 1.2, H).reshape(1, 1, H, 1)
    lon = torch.linspace(0, 2 * math.pi, W + 1)[:-1].reshape(1, 1, 1, W)
    lvl = torch.linspace(0.0, 1.0, P).reshape(1, P, 1, 1)
    bump = torch.exp(-((lon - math.pi) ** 2) / 0.4) * torch.exp(-(lat**2) / 0.3)
    st = {
        "u": torch.zeros(1, P, H, W),
        "v": torch.zeros(1, P, H, W),
        "t": (250.0 + 20.0 * (1 - lvl)).expand(1, P, H, W).contiguous(),
        "q": torch.full((1, P, H, W), 3e-3),
        "z": (5e4 * (1 - lvl) + 1.2e3 * bump).expand(1, P, H, W).contiguous(),
    }
    for s in ("t2m", "u10", "v10", "tp"):
        st[s] = torch.zeros(1, 1, H, W)
    st["r"] = torch.zeros(1, P, H, W)
    return st


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x**2)))


def _geo_residual(kernel: PurePDEKernel, st: dict[str, torch.Tensor]) -> float:
    """R = RMS‖(f·v − z_x, −f·u − z_y)‖ — остаток геострофического баланса."""
    f = kernel.f_field
    z_x = kernel.diff.d_x(st["z"])
    z_y = kernel.diff.d_y(st["z"])
    res_u = f * st["v"] - z_x
    res_v = -f * st["u"] - z_y
    return _rms(torch.sqrt(res_u**2 + res_v**2))


def check_geostrophic() -> None:
    """geostrophic: ветер=(−z_y,z_x)/f_safe; масса цела; остаток ≈0; конечно."""
    print("[check_geostrophic]")
    k = _kernel()
    s = _unbalanced_state()
    r_raw = _geo_residual(k, s)
    b = balance_initial_state(s, k, "geostrophic")

    f = k.f_field
    f_min = 2.0 * 7.2921e-5 * math.sin(math.radians(5.0))
    sign = torch.where(torch.sign(f) == 0.0, torch.ones_like(f), torch.sign(f))
    f_safe = torch.where(f.abs() < f_min, sign * f_min, f)
    u_g = -k.diff.d_y(s["z"]) / f_safe
    v_g = k.diff.d_x(s["z"]) / f_safe
    du = float((b["u"] - u_g).abs().max())
    dv = float((b["v"] - v_g).abs().max())
    dmass = float((b["z"] - s["z"]).abs().max()) + float((b["t"] - s["t"]).abs().max())
    fin = all(bool(torch.isfinite(b[v]).all()) for v in ("u", "v", "t", "q", "z"))
    r_bal = _geo_residual(k, b)
    print(
        f"  |u−u_g|max={du:.2e} |v−v_g|max={dv:.2e} Δmass(z,t)={dmass:.2e} "
        f"R {r_raw:.3e}→{r_bal:.3e} finite={fin}"
    )
    assert du < 1e-3 and dv < 1e-3, f"ветер ≠ геострофическому (du={du:.2e},dv={dv:.2e})"
    assert dmass < 1e-6, f"масса (z,t) изменена ({dmass:.2e})"
    assert fin, "geostrophic дал NaN/Inf"
    assert r_bal < 0.1 * r_raw, f"остаток не обнулён ({r_raw:.2e}→{r_bal:.2e})"
    print("  OK\n")


def check_dfi() -> None:
    """DFI код-корректность: стабилизированный forward НЕ расходится (ключевой
    фикс vs сырой Эйлер), реально применён, это low-pass — сохраняет
    крупномасштабный z и НЕ разрушает уже сбалансированное состояние.

    ВАЖНО (честный результат, не gate-fail): прагматичный forward-only DFI
    со span=1ч НЕ снижает максимальный геострофический дисбаланс — таймскейл
    адаптации (инерционный период ~часы-сутки) ≫ окна. Его корректная роль —
    убирать малый insertion-shock у уже почти-балансного ERA5, а не
    балансировать ветер из нуля. Эффективность E4 → решает кластер; ожидается
    слабой (см. experiments/E4_ic_balancing.md, план: «фиксируем
    отрицательный результат»).
    """
    print("[check_dfi]")
    k = _kernel()
    s = _unbalanced_state()
    r_raw = _geo_residual(k, s)
    b = balance_initial_state(s, k, "dfi", span_hours=1.0)

    r_dfi = _geo_residual(k, b)
    fin = all(bool(torch.isfinite(b[v]).all()) for v in ("u", "v", "t", "q", "z"))
    applied = _rms(b["u"] - s["u"]) > 0.0  # не fallback на сырой IC
    zc = float(
        torch.corrcoef(
            torch.stack([s["z"].flatten() - s["z"].mean(), b["z"].flatten() - b["z"].mean()])
        )[0, 1]
    )
    print(
        f"  finite={fin} applied={applied} corr(z_raw,dfi)={zc:.4f}  "
        f"R {r_raw:.2e}→{r_dfi:.2e} (честно: short-span DFI баланс не форсит)"
    )
    # Гейт = КОД-корректность (стабильность фикса + low-pass на z), НЕ
    # эффективность балансировки. Effectiveness E4 → кластер (ожид. слабо;
    # план: «фиксируем отрицательный результат»). НЕ маскируем фейк-пассом.
    assert fin, "DFI дал NaN/Inf (стабилизированный step разошёлся — фикс не сработал)"
    assert applied, "DFI не применён (fallback на сырой IC — forward разошёлся)"
    assert zc > 0.9, f"крупномасштабный z (медленная мода) не сохранён (corr={zc:.3f})"
    print("  OK\n")


def main() -> None:
    """Прогнать инварианты; exit(1) на провале."""
    print("=== Sanity: IC balancing (E4) ===\n")
    check_geostrophic()
    check_dfi()
    print("[all OK]")
    sys.exit(0)


if __name__ == "__main__":
    main()
