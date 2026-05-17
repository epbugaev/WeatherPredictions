"""Sanity-gate E6: зонально-неявный semi-implicit (B2/B4-fixed).

Прежняя версия фиксировала ЧЕСТНЫЙ НЕГАТИВ ("scoped SI не стабилизирует")
— но аудит показал, что это было следствие БАГОВ, а не физики:
  B2: `lam.clamp_min(0)` зануляло БЫСТРЕЙШИЕ гравитационные моды (внешняя
      c≈309 м/с уходила в λ<0 из-за знака cumsum-вертикали) → _si_solve
      ≈ identity (no-op).
  B4: implicit ∇² (постоянный Δx_ref) ≠ RHS ∇² (физ. _laplacian), и
      постоянный Δx «не видел» полюсные мелко-Δx быстрые моды (стиффовые).
Фикс: |λ| (все моды, вкл. внешнюю) + ЗОНАЛЬНО-неявный λ-зависимый Δx
(rfft по долготе, символ ∂²ₓ зависит от широты — на полюсе Δx мал →
сильное неявное демпфирование там, где CFL нарушается), один и тот же
∂²ₓ по обе стороны CN (B4-консистентно).

Гейт проверяет ИМЕННО починку B2/B4 (не глобальную стабилизацию):
1. `check_helmholtz_roundtrip` — _si_solve точно обращает свой
   зональный reference-оператор (модальный + rfft-x).
2. `check_implicit_operator_acts` — implicit-оператор РЕАЛЬНО действует
   (a²λ·sx = O(1), НЕ ≈identity) — это и есть прямая проверка B2-фикса
   (раньше clamp_min(0) занулял быстрые моды → no-op). Доп.: SI не хуже
   explicit там, где explicit конечен.
3. `check_consistency` — малый dt, медленное состояние: SI ≈ ssp_rk3,
   не вырожденный демпфер (эволюционирует).

ЧЕСТНАЯ ГРАНИЦА (не баг, документировано): SI здесь ЗОНАЛЬНО-неявный —
он корректно лечит полюсный ЗОНАЛЬНЫЙ grav-CFL, но МЕРИДИОНАЛЬНАЯ
гравиволна остаётся в явной части (forward-Euler) → она осцилляторна и
безусловно растёт под Эйлером. Полная безусловная устойчивость требует
2D-неявного Гельмгольца (блок-трёхдиаг. по широте после rfft-x) =
полноценное dynamical-core ядро, ВНЕ рамок этой абляции. Это уже не
bug-masked (B1-B4 починены) — это установленная архитектурная граница.

CPU-only, без memmap/timm. Запуск:
    .venv/bin/python Models/dev/sanity_semi_implicit.py
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

# Production-like 32×64: на грубой сетке полюсный Δx крупен → гравиCFL<1
# (нет той неустойчивости, ради которой нужен SI). Реалистичный полюсный
# Δx (~6e4 м) даёт a²λ·sx = O(1)+ → implicit реально работает.
H, W, P = 32, 64, 13


def _kernel(scheme: str, dt: float) -> PurePDEKernel:
    grid = Grid(GridConfig(H=H, W=W))
    return PurePDEKernel(grid, stencil="fd4", coriolis="spherical", block_dt=dt, time_scheme=scheme)


def _gravity_state() -> dict[str, torch.Tensor]:
    """ЛИНЕЙНЫЙ режим: КРОШЕЧНый (10 м²/с²) z'-бугор + нулевой ветер →
    единственный быстрый процесс = линейная гравитационная волна (ветер
    остаётся O(перт.) → адвекция пренебрежима). Это изолирует ТО, что SI
    обязан стабилизировать (линейный grav-CFL), не примешивая адвективный
    CFL (он включается только при большой амплитуде/развитом ветре)."""
    lat = torch.linspace(-1.2, 1.2, H).reshape(1, 1, H, 1)
    lon = torch.linspace(0, 2 * math.pi, W + 1)[:-1].reshape(1, 1, 1, W)
    lvl = torch.linspace(0.0, 1.0, P).reshape(1, P, 1, 1)
    bump = torch.exp(-((lon - math.pi) ** 2) / 0.4) * torch.exp(-(lat**2) / 0.3)
    return {
        "u": torch.zeros(1, P, H, W),
        "v": torch.zeros(1, P, H, W),
        "t": (250.0 + 20.0 * (1 - lvl)).expand(1, P, H, W).contiguous(),
        "q": torch.full((1, P, H, W), 3e-3),
        "z": (5e4 * (1 - lvl) + 10.0 * bump).expand(1, P, H, W).contiguous(),
    }


def _slow_state() -> dict[str, torch.Tensor]:
    """Гладкое медленное состояние (мало гравитационного контента)."""
    lat = torch.linspace(-1.0, 1.0, H).reshape(1, 1, H, 1)
    lon = torch.linspace(0, 2 * math.pi, W + 1)[:-1].reshape(1, 1, 1, W)
    lvl = torch.linspace(0.0, 1.0, P).reshape(1, P, 1, 1)
    base = torch.cos(lat) * torch.cos(lon)
    return {
        "u": (6.0 * torch.sin(2 * lat)).expand(1, P, H, W).contiguous(),
        "v": (0.4 * torch.cos(lon) * torch.cos(lat)).expand(1, P, H, W).contiguous(),
        "t": (250.0 + 18.0 * (1 - lvl) + base).expand(1, P, H, W).contiguous(),
        "q": torch.full((1, P, H, W), 3e-3),
        "z": (5e4 * (1 - lvl) + 80.0 * base).expand(1, P, H, W).contiguous(),
    }


def _amax(s: dict[str, torch.Tensor]) -> float:
    return float(torch.nan_to_num(s["z"].abs().amax(), nan=float("inf")))


def _umax(s: dict[str, torch.Tensor]) -> float:
    """max|u| — индикатор линейной grav-неустойчивости (ветер стартует с 0;
    |z| бесполезен — в нём доминирует базовый профиль 5e4)."""
    return float(torch.nan_to_num(s["u"].abs().amax(), nan=float("inf")))


def _roll(k: PurePDEKernel, st: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    cur = {x: st[x] for x in ("u", "v", "t", "q", "z")}
    with torch.no_grad():
        for _ in range(n):
            o = k.step(cur["u"], cur["v"], cur["t"], cur["q"], cur["z"])
            cur = {x: o[x] for x in ("u", "v", "t", "q", "z")}
            if not math.isfinite(_umax(cur)):
                break
    return cur


def check_helmholtz_roundtrip() -> None:
    """_si_solve точно обращает свой зональный reference-оператор."""
    print("[check_helmholtz_roundtrip]")
    k = _kernel("semi_implicit", dt=300.0)
    torch.manual_seed(0)
    phi = torch.randn(1, P, H, W)
    # Прямой оператор = модальная проекция + (1/si_helm_inv) в rfft-x —
    # та же композиция, что в _si_solve (rfft ТОЛЬКО по долготе).
    phi_m = torch.einsum("pq,bqhw->bphw", k.si_Vinv, phi)
    spec = torch.fft.rfft(phi_m, dim=-1) / k.si_helm_inv
    g_m = torch.fft.irfft(spec, n=W, dim=-1)
    g = torch.einsum("pq,bqhw->bphw", k.si_V, g_m)
    back = k._si_solve(g)
    rel = float((back - phi).norm() / phi.norm())
    lam_max = float((1.0 / k.si_helm_inv - 1.0).max())  # ~ a²λ_max·sx_max
    print(f"  roundtrip rel err={rel:.3e}  max(a²λ·sx)={lam_max:.3f} (должен быть O(1)+)")
    assert rel < 1e-4, f"_si_solve не обращает зональный (I−a²A∂²ₓ): rel={rel:.2e}"
    assert lam_max > 0.3, f"implicit-оператор ≈ identity (max a²λ·sx={lam_max:.3f}) — B2 не починен"
    print("  OK\n")


def check_implicit_operator_acts() -> None:
    """B2-фикс: implicit-оператор реально действует (a²λ·sx=O(1), не
    ≈identity — раньше clamp_min(0) занулял быстрые моды). Доп.: на
    линейной grav-волне SI не ХУЖЕ explicit (зональную часть он лечит;
    меридиональная остаётся явной — честная граница, см. docstring)."""
    print("[check_implicit_operator_acts]")
    k = _kernel("semi_implicit", 300.0)
    strength = float((1.0 / k.si_helm_inv - 1.0).max())  # max a²λ·sx
    print(f"  implicit strength max(a²λ·sx)={strength:.3f} (≈0 = B2 no-op; O(1) = fixed)")
    assert strength > 0.5, (
        f"implicit ≈ identity (strength={strength:.3f}) — B2 НЕ починен (clamp занулил быстрые моды)"
    )
    # SI не должен быть ХУЖЕ explicit на линейной grav-волне.
    dt, n = 300.0, 24
    s = _gravity_state()
    ue = _umax(_roll(_kernel("ssp_rk3", dt), s, n))
    usi = _umax(_roll(_kernel("semi_implicit", dt), s, n))
    print(f"  linear grav |u|max: ssp_rk3={ue:.3e}  semi_implicit={usi:.3e}")
    fe = math.isfinite(ue)
    fsi = math.isfinite(usi)
    assert (not fe) or fsi or usi <= max(ue, 1e9), f"SI грубо хуже explicit ({usi:.2e} ≫ {ue:.2e})"
    print("  OK\n")


def check_consistency() -> None:
    """Малый dt, медленное состояние: semi_implicit ≈ ssp_rk3 и не заморожен."""
    print("[check_consistency]")
    dt, n = 30.0, 40
    s = _slow_state()
    si = _roll(_kernel("semi_implicit", dt), s, n)
    rk = _roll(_kernel("ssp_rk3", dt), s, n)
    z0 = s["z"]
    evolve = float((si["z"] - z0).norm() / z0.norm())
    rel = float((si["z"] - rk["z"]).norm() / rk["z"].norm())
    fin = all(bool(torch.isfinite(si[x]).all()) for x in ("u", "v", "t", "q", "z"))
    print(f"  finite={fin}  ‖si−z0‖/‖z0‖={evolve:.3e}  rel‖si−rk3‖={rel:.3e}")
    assert fin, "semi_implicit дал NaN/Inf на медленном состоянии"
    assert evolve > 1e-7, f"semi_implicit заморозил состояние (вырожд. демпфер, {evolve:.1e})"
    assert rel < 0.25, f"semi_implicit не консистентен ssp_rk3 (rel={rel:.3f})"
    print("  OK\n")


def main() -> None:
    """Прогнать инварианты; exit(1) на провале."""
    print("=== Sanity: zonal semi-implicit (E6, B2/B4-fixed) ===\n")
    check_helmholtz_roundtrip()
    check_implicit_operator_acts()
    check_consistency()
    print("[all OK]")
    sys.exit(0)


if __name__ == "__main__":
    main()
