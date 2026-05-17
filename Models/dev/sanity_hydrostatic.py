"""Sanity: PurePDEKernel.get_z_t использует R_d=287 по умолчанию (A1).

Тест: пропустим однородный `t_t` и сравним z_t против ожидаемого
``-R_d/p * t_t``-интеграла. С `use_universal_R=True` ответ должен быть
в ~287/8.314 ≈ 34.5× меньше.

Запуск:
    python -m Models.dev.sanity_hydrostatic
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.physics import Grid, GridConfig, PurePDEKernel

R_D = 287.0
R_UNIV = 8.314


def main() -> None:
    grid = Grid(GridConfig(H=32, W=64))
    kernel_rd = PurePDEKernel(grid)
    kernel_univ = PurePDEKernel(grid, use_universal_R=True)

    B, P, H, W = 1, 13, 32, 64
    torch.manual_seed(0)
    # Однородная по (H, W) t_t — упрощает проверку соотношения по R.
    t_t = torch.ones(B, P, H, W) * 1e-3

    z_t_rd = kernel_rd.get_z_t(t_t)
    z_t_univ = kernel_univ.get_z_t(t_t)

    ratio = (z_t_rd.abs().mean() / z_t_univ.abs().mean()).item()
    expected = R_D / R_UNIV
    print(f"  |z_t|_Rd / |z_t|_R_univ = {ratio:.3f}  (ожидаем ≈ {expected:.3f})")

    ok = abs(ratio - expected) / expected < 0.01
    print(f"  [{'OK' if ok else 'FAIL'}] hydrostatic default = R_d=287 (per-mass)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
