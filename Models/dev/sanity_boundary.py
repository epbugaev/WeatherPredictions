"""Sanity: 'periodic' и 'replicate' boundary в utils.physics дают разные значения
на краях (B1). Цель — убедиться, что 'periodic' — это настоящая периодика
(roll), а 'replicate' — это cat-pad первых/последних строк (старое поведение
с именем 'periodic').

Запуск:
    python -m Models.dev.sanity_boundary
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.physics import FiniteDifference, Grid, GridConfig


def main() -> None:
    H, W = 32, 64
    grid = Grid(GridConfig(H=H, W=W))
    # Поле с асимметрией по lon-краям: field[..., 0] = 0, field[..., -1] = 100.
    field = torch.zeros(1, 13, H, W)
    field[..., 0] = 0.0
    field[..., -1] = 100.0
    field[..., 1:-1] = torch.linspace(0, 100, W - 2).view(1, 1, 1, -1)

    # boundary_x='periodic' и 'replicate' должны давать разные производные на
    # лево-краях (где разница между «обходом через правый край» и «replicate»
    # максимальна).
    diff_periodic = FiniteDifference(grid, boundary_x="periodic")
    diff_replicate = FiniteDifference(grid, boundary_x="replicate")

    d_per = diff_periodic.d_x(field)
    d_rep = diff_replicate.d_x(field)

    edge_diff = (d_per[..., 0] - d_rep[..., 0]).abs().max().item()
    interior_diff = (d_per[..., 5:-5] - d_rep[..., 5:-5]).abs().max().item()

    print(f"  max|d_per - d_rep| на левом краю (col 0): {edge_diff:.3e}")
    print(f"  max|d_per - d_rep| внутри (col 5..-5):    {interior_diff:.3e}")

    edge_ok = edge_diff > 1e-6  # ожидаем расхождение на краю
    interior_ok = interior_diff < 1e-10  # внутри идентично (далеко от границы)

    # boundary_z='periodic' должен зарезать __init__:
    raised = False
    try:
        FiniteDifference(grid, boundary_z="periodic")
    except ValueError:
        raised = True
    print(f"  boundary_z='periodic' raises ValueError: {raised}")

    all_ok = edge_ok and interior_ok and raised
    for name, ok in [
        ("periodic vs replicate расходятся на границе", edge_ok),
        ("periodic vs replicate совпадают внутри", interior_ok),
        ("boundary_z='periodic' запрещён", raised),
    ]:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
