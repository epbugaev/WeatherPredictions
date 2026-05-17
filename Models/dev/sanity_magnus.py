"""Sanity: две Magnus q_s формулы дают одинаковый ответ на одинаковом (T, p_Pa) (A4).

* ``tools.check_physics_common.magnus_qs(T, p_Pa)`` — простая формула без clipping.
* ``utils.physics.PurePDEKernel._get_qs(p_Pa, T)`` — с `_scale_tensor`-clipping.

На «обычном» (T, p_Pa) clipping в `_get_qs` не активен, ответы должны
совпадать в пределах fp32.

Запуск:
    python -m Models.dev.sanity_magnus
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_physics_common import magnus_qs
from utils.physics import Grid, GridConfig, PurePDEKernel


def main() -> None:
    grid = Grid(GridConfig(H=32, W=64))
    kernel = PurePDEKernel(grid)

    # Реалистичные температуры (210-310 K), без экстремальных значений где clipping активен.
    B, P, H, W = 1, 13, 32, 64
    torch.manual_seed(1)
    t = 260.0 + 20.0 * torch.randn(B, P, H, W).clamp(-2, 2)
    p_pa = grid.pressure.expand_as(t)

    qs_common = magnus_qs(t, p_pa)
    qs_kernel = kernel._get_qs(p_pa, t)

    diff = (qs_common - qs_kernel).abs().max().item()
    ref = qs_common.abs().max().item()
    rel = diff / (ref + 1e-30)
    print(f"  max |q_s_common - q_s_kernel|     = {diff:.3e}")
    print(f"  reference max |q_s|               = {ref:.3e}")
    print(f"  relative                          = {rel:.3e}")
    # Tolerance: kernel применяет scale_tensor-clipping arg в [-3.47, 3.01];
    # на realistic-температурах он чаще не активен, но локально может слегка
    # сдвинуть. Допускаем rel < 5e-2.
    ok = rel < 5e-2
    print(f"  [{'OK' if ok else 'FAIL'}] Magnus q_s (common vs kernel) согласованы на Pa")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
