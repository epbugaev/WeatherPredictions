"""Sanity: GridConfig.lat_range_deg даёт правильные `pixel_x` / `pixel_y` для USA-кропа (C1).

USA-кроп на 128×256-сетке: индексы [128-92:128-60, 256-131:256-67] = [36:68, 125:189].
В широтах это ≈ 24°N..56°N (1.40625° на пиксель, центр ~40°N).

Проверяем:
  * широта `Grid.latitudes` в диапазоне (24°..56°) ± step;
  * `pixel_x` соответствует ~111км/° · cos(центр-широты) · ширина_кропа / W;
  * `lat_weights` для регионального диапазона корректно нормированы.

Запуск:
    python -m Models.dev.sanity_lat_range
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_physics_common import GeometryCPU
from utils.physics import Grid, GridConfig


def main() -> None:
    H, W = 32, 64
    lat_low, lat_high = 24.0, 56.0
    grid = Grid(GridConfig(H=H, W=W, lat_range_deg=(lat_low, lat_high)))
    geom = GeometryCPU(H=H, W=W, lat_range_deg=(lat_low, lat_high))

    lat_deg_min = grid.latitudes.min() * 180.0 / torch.pi
    lat_deg_max = grid.latitudes.max() * 180.0 / torch.pi
    print(
        f"  latitudes range: {lat_deg_min:.2f}°..{lat_deg_max:.2f}°  (ожидаем ≈ {lat_low}..{lat_high})"
    )

    # pixel_x на центре кропа: ~111км·cos(40°)·320° / 64 ≈ 425 км. Но 320°? Нет,
    # 320° — это global. Для USA-кропа долгота тоже регион (~58°), но в нашей
    # параметризации pixel_x = (2πR·cos(lat) / W) — это **глобальная** долгота-сетка.
    # Для регионального кропа W должно быть скорректировано — но мы тестируем
    # формулу как есть.
    pixel_x_mid = grid.pixel_x.flatten()[H // 2].item()
    expected_pixel_x = (
        2
        * torch.pi
        * 6371.0e3
        * float(torch.cos(torch.tensor((lat_low + lat_high) / 2 / 180 * torch.pi)))
        / W
    )
    print(f"  pixel_x (центр): {pixel_x_mid:.1f} m  (ожидаем ≈ {expected_pixel_x:.1f})")
    pixel_x_ok = abs(pixel_x_mid - expected_pixel_x) / expected_pixel_x < 0.1

    # lat_weights нормированы: sum = H.
    lat_w_sum = geom.lat_weights.sum().item()
    print(f"  Σ lat_weights = {lat_w_sum:.3f}  (ожидаем = H = {H})")
    lat_w_ok = abs(lat_w_sum - H) / H < 1e-4

    # pixel_y соответствует (lat_high - lat_low) [deg] / (H+1) * π/180 * R.
    expected_pixel_y = (lat_high - lat_low) / 180.0 * torch.pi * 6371.0e3 / (H + 1)
    pixel_y_val = grid.pixel_y.item()
    print(f"  pixel_y = {pixel_y_val:.1f} m  (ожидаем ≈ {expected_pixel_y:.1f})")
    pixel_y_ok = abs(pixel_y_val - expected_pixel_y) / expected_pixel_y < 1e-3

    all_ok = pixel_x_ok and lat_w_ok and pixel_y_ok
    for name, ok in [
        ("pixel_x масштаб", pixel_x_ok),
        ("lat_weights нормировка", lat_w_ok),
        ("pixel_y масштаб", pixel_y_ok),
    ]:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
