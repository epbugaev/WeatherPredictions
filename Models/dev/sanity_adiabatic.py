"""Sanity: PurePDEKernel.get_t_t выдаёт R_d·T·ω/(c_p·p) на static-flow (A2).

Если u=v=0, t_x=t_y=t_z=0, то t_t = R_d·T·ω/(c_p·p), где ω=100·w (hPa/s → Pa/s).
Сверим численно.

Запуск:
    python -m Models.dev.sanity_adiabatic
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
C_P = 1005.0


def main() -> None:
    grid = Grid(GridConfig(H=32, W=64))
    kernel = PurePDEKernel(grid)

    B, P, H, W = 1, 13, 32, 64
    # Все производные нулевые → нет адвекции; t однородна, w задаём вручную.
    u = torch.zeros(B, P, H, W)
    v = torch.zeros(B, P, H, W)
    t = torch.full((B, P, H, W), 250.0)
    w = torch.full((B, P, H, W), 0.01)  # hPa/s
    z_z = torch.zeros(B, P, H, W)  # не используется в adiabatic_omega

    t_t = kernel.get_t_t(u, v, w, t, z_z)

    omega_pa = 100.0 * w
    expected = R_D * t * omega_pa / (C_P * grid.pressure)

    rel = (t_t - expected).abs().max().item() / expected.abs().max().item()
    print(f"  max relative diff = {rel:.3e}")
    ok = rel < 1e-5
    print(f"  [{'OK' if ok else 'FAIL'}] get_t_t == R_d·T·ω/(c_p·p) на static flow")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
