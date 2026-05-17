"""Локальный re-run абляции на ГЛАДКОМ сбалансированном synthetic
(когда кластер недоступен). Для каждого Ek (кумулятивные флаги, как в
sh_files/check_physics_ablation.sh) строит kernel, опц. балансирует IC
(DFI/geostrophic), катит до 144 substep'ов (12 ч при dt=300) и печатает:
первый substep с NaN/взрывом (|u|max>1e7) и |u|max на момент выживания.

Цель: показать, что ПОСЛЕ фикса B1-B4 методы РАЗЛИЧАЮТСЯ (baseline E0
взрывается рано, стабилизаторы продлевают), т.е. прежний «все 1в1
взрыв@h1» был артефактом багов, а не только физикой.

Запуск: .venv/bin/python Models/dev/rerun_ablation_local.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from tools.check_physics_common import balance_initial_state, split_channels_69  # noqa: E402
from utils.physics import Grid, GridConfig, PurePDEKernel  # noqa: E402

SMOOTH = Path("/tmp/syn_smooth.dat")
H, W, P = 32, 64, 13
DT, NSUB = 300.0, 144  # 12 ч

# (label, kernel-kwargs, balance-mode)  — кумулятивно, как ABL[E*].
CONFIGS = [
    ("E0 euler", {"time_scheme": "euler"}, "none"),
    ("E1 +ssp_rk3", {"time_scheme": "ssp_rk3"}, "none"),
    ("E2 +hyperdiff", {"time_scheme": "ssp_rk3", "hyperdiffusion": True}, "none"),
    ("E3 +polar", {"time_scheme": "ssp_rk3", "hyperdiffusion": True, "polar_filter": True}, "none"),
    ("E4 +dfi", {"time_scheme": "ssp_rk3", "hyperdiffusion": True, "polar_filter": True}, "dfi"),
    (
        "E5 +flux+mc",
        {
            "time_scheme": "ssp_rk3",
            "hyperdiffusion": True,
            "polar_filter": True,
            "advection_form": "flux",
            "w_diagnostic": "mass_consistent",
        },
        "dfi",
    ),
    (
        "E6 semi_impl",
        {
            "time_scheme": "semi_implicit",
            "hyperdiffusion": True,
            "polar_filter": True,
            "advection_form": "flux",
            "w_diagnostic": "mass_consistent",
        },
        "dfi",
    ),
]


def _state_from_smooth() -> dict[str, torch.Tensor]:
    if not SMOOTH.exists():
        runpy.run_path(
            str(REPO_ROOT / "Models/dev/make_synthetic_era5_smooth.py"), run_name="__main__"
        )
    arr = np.memmap(SMOOTH, dtype=np.float32, mode="r", shape=(240, 69, H, W))
    x = torch.from_numpy(np.array(arr[0], dtype=np.float32)).reshape(1, 69, H, W)
    parts = split_channels_69(x)
    parts["q"] = (parts["r"] / 100.0 * 6e-3).clamp_min(1e-6)  # грубое r→q, гладкое
    return parts


def _blowup_step(label: str, kw: dict, bal: str) -> tuple[int, float]:
    grid = Grid(GridConfig(H=H, W=W))
    kernel = PurePDEKernel(grid, stencil="fd4", coriolis="spherical", block_dt=DT, **kw)
    st = _state_from_smooth()
    if bal != "none":
        st = balance_initial_state(st, kernel, bal, span_hours=1.0)
    cur = {k: st[k] for k in ("u", "v", "t", "q", "z")}
    last_finite_umax = 0.0
    with torch.no_grad():
        for step in range(1, NSUB + 1):
            o = kernel.step(cur["u"], cur["v"], cur["t"], cur["q"], cur["z"])
            cur = {k: o[k] for k in ("u", "v", "t", "q", "z")}
            um = float(torch.nan_to_num(cur["u"].abs().amax(), nan=float("inf")))
            if not np.isfinite(um) or um > 1e7:
                return step, last_finite_umax
            last_finite_umax = um
    return NSUB + 1, last_finite_umax  # выжил все NSUB


def main() -> None:
    print(
        f"=== Local ablation re-run (smooth balanced IC, dt={DT:.0f}s, "
        f"{NSUB} substeps = {NSUB * DT / 3600:.0f} h) ===\n"
    )
    print(f"{'method':<16s}{'blow@substep':>14s}{'(= hours)':>12s}{'|u|max@survival':>18s}")
    print("-" * 60)
    for label, kw, bal in CONFIGS:
        step, umax = _blowup_step(label, kw, bal)
        tag = "SURVIVED" if step > NSUB else f"{step}"
        hrs = "all" if step > NSUB else f"{step * DT / 3600:.2f}"
        print(f"{label:<16s}{tag:>14s}{hrs:>12s}{umax:>18.3e}")
    print(
        "\n(Различие blow@substep между методами = фиксы B1-B4 работают;"
        "\n одинаковый ранний взрыв у всех = всё ещё bug/genuine.)"
    )


if __name__ == "__main__":
    main()
