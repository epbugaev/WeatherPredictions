"""Sanity: все три варианта Coriolis в utils.physics.Grid дают 2Ω·sin(...).

Регрессия A5, A6, A7 (см. plan): после фиксов constant/beta_plane/spherical
имеют одинаковый масштаб 2Ω·sin(...) ≈ 1.03e-4 на mid-latitude.

Запуск:
    python -m Models.dev.sanity_coriolis
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.physics import Grid, GridConfig

OMEGA = 7.2921e-5
F_MID = 2 * OMEGA * float(torch.sin(torch.tensor(torch.pi / 4)))  # ≈ 1.03e-4


def main() -> None:
    grid = Grid(GridConfig(H=32, W=64))

    f_const = grid.f_constant.item()
    f_beta_mid = grid.f_beta_plane.flatten()[grid.f_beta_plane.numel() // 2].item()
    # f_spherical усреднён по mid-latitude (±45° индексам около H/2):
    half = grid.f_spherical.shape[-2] // 2
    f_sph_mid = grid.f_spherical[..., half, 0].abs().item()

    print(f"  reference 2Ω·sin(45°) = {F_MID:+.3e} rad/s")
    print(f"  f_constant            = {f_const:+.3e}")
    print(f"  f_beta_plane (центр)  = {f_beta_mid:+.3e}")
    print(f"  |f_spherical| (центр) = {f_sph_mid:+.3e}")

    tol = 0.05
    checks = [
        ("f_constant ≈ 2Ω·sin(45°)", abs(f_const - F_MID) / F_MID < tol),
        ("f_beta_plane(центр) ≈ f0 = 2Ω·sin(45°)", abs(f_beta_mid - F_MID) / F_MID < tol),
        ("|f_spherical(центр)| > Ω (не легаси 7.29e-5)", f_sph_mid > 1.5 * OMEGA),
    ]
    all_ok = True
    for name, ok in checks:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}")
        all_ok = all_ok and ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
