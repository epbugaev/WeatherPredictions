"""Sanity-gate E1: корректность SSP-RK3 в `utils.physics.PurePDEKernel`.

Гейт «Шаг 2» из плана абляции — должен пройти PASS до smoke/кластера.
Тестируется РЕАЛЬНЫЙ путь `kernel.step(time_scheme="ssp_rk3")`, не отдельная
реализация схемы:

1. `check_temporal_order` — наблюдаемый порядок по времени через Richardson
   (одна и та же схема, сетка dt → dt/2). SSP-RK3 должен дать slope ≈ 3,
   Forward Euler — slope ≈ 1. Если перепутан коэффициент (¾/¼, ⅓/⅔),
   порядок упадёт — тест острый.
2. `check_ssp_bounded_vs_euler` — на гладком осцилляторном режиме, где
   Forward Euler расходится, SSP-RK3 остаётся ограниченным (суть E1).

CPU-only, без memmap и без импорта пакета `Models` (timm). Гладкое
маломодовое состояние → rhs хорошо обусловлен на коротком окне, виден
именно truncation-error.

Запуск:
    .venv/bin/python Models/dev/sanity_ssp_rk3.py
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

H, W, P = 16, 32, 13


def _smooth_state(seed: int = 0) -> dict[str, torch.Tensor]:
    """Гладкое маломодовое физически-правдоподобное состояние (B=1,P,H,W).

    Только низкие волновые числа (k≤2) и малые амплитуды — производные
    гладкие, rhs не стиф на коротком интегрировании.
    """
    torch.manual_seed(seed)
    lat = torch.linspace(-1.4, 1.4, H).reshape(1, 1, H, 1)
    lon = torch.linspace(0, 2 * math.pi, W + 1)[:-1].reshape(1, 1, 1, W)
    lvl = torch.linspace(0.0, 1.0, P).reshape(1, P, 1, 1)
    base = torch.cos(lat) * torch.cos(lon)
    u = 8.0 * torch.sin(2 * lat) + 1.5 * base
    v = 1.2 * torch.cos(lon) * torch.cos(lat)
    t = 250.0 + 20.0 * (1.0 - lvl) + 1.0 * base
    q = 4e-3 * (0.5 + 0.5 * torch.cos(lon)) * torch.cos(lat).clamp_min(0.0) + 1e-4
    z = 5e4 * (1.0 - lvl) + 200.0 * base
    expand = lambda a: a.expand(1, P, H, W).contiguous()
    return {
        "u": expand(u),
        "v": expand(v),
        "t": expand(t),
        "q": expand(q.expand(1, P, H, W)),
        "z": expand(z),
    }


def _integrate(
    scheme: str, dt: float, n_steps: int, double: bool = False
) -> dict[str, torch.Tensor]:
    """n_steps шагов `kernel.step` со схемой scheme, шаг dt. Возвращает поля.

    double=True → float64 (для order-теста: убирает round-off floor, иначе
    SSP-RK3 сходится ниже float32-epsilon и наклон не наблюдаем).
    """
    grid = Grid(GridConfig(H=H, W=W))
    kernel = PurePDEKernel(
        grid, stencil="fd4", coriolis="spherical", block_dt=dt, time_scheme=scheme
    )
    s = _smooth_state()
    if double:
        kernel = kernel.double()
        s = {k: v.double() for k, v in s.items()}
    with torch.no_grad():
        for _ in range(n_steps):
            out = kernel.step(s["u"], s["v"], s["t"], s["q"], s["z"])
            s = {k: out[k] for k in ("u", "v", "t", "q", "z")}
    return s


def _err_vs_ref(scheme: str, total_time: float, dt: float, ref: dict[str, torch.Tensor]) -> float:
    """L2-ошибка поля u относительно эталона ref после total_time секунд (float64)."""
    n = round(total_time / dt)
    s = _integrate(scheme, dt, n, double=True)
    return float(torch.sqrt(torch.mean((s["u"] - ref["u"]) ** 2)))


def _observed_order(scheme: str) -> float:
    """Наблюдаемый порядок по времени: log-log наклон err(dt) ~ dt^p (float64)."""
    total_time = 120.0  # короткое окно: гладко, конечно, доминирует truncation
    dt_ref = 1.875
    ref = _integrate(scheme, dt_ref, round(total_time / dt_ref), double=True)
    dts = [60.0, 30.0, 15.0, 7.5]
    errs = [_err_vs_ref(scheme, total_time, dt, ref) for dt in dts]
    logs_dt = [math.log(d) for d in dts]
    logs_e = [math.log(max(e, 1e-300)) for e in errs]
    m = len(dts)
    mean_x = sum(logs_dt) / m
    mean_y = sum(logs_e) / m
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(logs_dt, logs_e, strict=True))
    var = sum((x - mean_x) ** 2 for x in logs_dt)
    slope = cov / var
    errs_fmt = [f"{e:.3e}" for e in errs]
    print(f"  [{scheme}] err(dt)={errs_fmt} → slope={slope:.2f}")
    return slope


def check_temporal_order() -> None:
    """SSP-RK3 порядок ≈3, Euler ≈1, и rk3 строго точнее euler."""
    print("[check_temporal_order]")
    p_rk3 = _observed_order("ssp_rk3")
    p_eul = _observed_order("euler")
    assert p_rk3 >= 2.6, f"SSP-RK3 наблюдаемый порядок {p_rk3:.2f} < 2.6 (ошибка в коэффициентах?)"
    assert p_eul <= 1.5, f"Euler наблюдаемый порядок {p_eul:.2f} > 1.5 (неожиданно)"
    assert p_rk3 > p_eul + 1.0, f"SSP-RK3 ({p_rk3:.2f}) не выше Euler ({p_eul:.2f})"
    print("  OK\n")


def check_ssp_bounded_vs_euler() -> None:
    """В осцилляторном режиме SSP-RK3 ограничен там, где Euler расходится."""
    print("[check_ssp_bounded_vs_euler]")
    dt, n = 300.0, 24  # 2 «часа» при block_dt=300 — Euler уже взрывается
    s_eul = _integrate("euler", dt, n)
    s_rk3 = _integrate("ssp_rk3", dt, n)
    m_eul = float(torch.nan_to_num(s_eul["u"].abs().amax(), nan=float("inf")))
    m_rk3 = float(s_rk3["u"].abs().amax())
    print(f"  |u|max euler={m_eul:.3e}  ssp_rk3={m_rk3:.3e}")
    assert math.isfinite(m_rk3), f"SSP-RK3 сам разошёлся (|u|max={m_rk3})"
    assert m_rk3 < m_eul, f"SSP-RK3 не лучше Euler ({m_rk3:.2e} ≥ {m_eul:.2e})"
    print("  OK\n")


def main() -> None:
    """Прогнать оба инварианта; exit(1) на провале."""
    print("=== Sanity: SSP-RK3 (E1) ===\n")
    check_temporal_order()
    check_ssp_bounded_vs_euler()
    print("[all OK]")
    sys.exit(0)


if __name__ == "__main__":
    main()
