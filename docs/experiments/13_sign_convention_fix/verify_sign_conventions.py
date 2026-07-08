"""Численная верификация знаковых конвенций физядер (аудит/фикс 13).

Прогоняет шесть решающих проб на живом коде (без реимплементации) и пишет
JSON с ключевыми величинами. До фикса (a827d32) пробы 3-6 давали неверные
знаки/величины (см. README.md, колонка «до»); после фикса все пробы обязаны
давать физичные значения — это же покрыто юнит-тестами
``tests/test_physics_sign_conventions.py``.

Запуск из корня репозитория:
    .venv/bin/python docs/experiments/13_sign_convention_fix/verify_sign_conventions.py
"""

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.physics import Grid, GridConfig, PurePDEKernel  # noqa: E402
from utils.physics_hybrid import PDE_kernel, d_z, integral_z, pressure  # noqa: E402

H, W, P = 8, 16, 13
PHI_STD = torch.tensor(
    [199300.0, 157400, 134000, 116600, 102300, 90000, 69700, 54100, 40700, 28600, 13500, 7300, 800]
).reshape(1, P, 1, 1)


def make_kernel() -> PDE_kernel:
    """PDE_kernel на геометрии латента USA-v4 (8x16, 5.625 град, block_dt=400 c)."""
    return PDE_kernel(
        in_dim=65,
        physics_part_coef=0.5,
        block_dt=400.0,
        lat_start_deg=18.28125,
        dlat_deg=5.625,
        dlon_deg=5.625,
        grid_h=H,
    )


def probe_d_z_sign() -> float:
    """d_z на поле f=p: −1 означает d_z = −∂/∂p (конвенция ζ=−p)."""
    p_field = pressure.float().expand(1, P, H, W).clone()
    return float(d_z(p_field)[0, 4:9].mean())


def probe_integral_orientation() -> dict[str, float]:
    """integral_z(1): полная колонка на верхнем уровне => интеграл уровень->поверхность."""
    iz = integral_z(torch.ones(1, P, 1, 1))
    return {"top_50hpa": float(iz[0, 0, 0, 0]), "bottom_1000hpa": float(iz[0, -1, 0, 0])}


def probe_w_and_adiabatic() -> dict[str, float]:
    """Однородная дивергенция (субсиденция, ω>0): w<0 (=-ω), t_t>0 (нагрев)."""
    kernel = make_kernel()
    x_idx = torch.arange(W, dtype=torch.float32).reshape(1, 1, 1, W)
    u = (1e-5 * kernel.pixel_x * x_idx).expand(1, P, H, W).clone()
    v = torch.zeros_like(u)
    w = kernel.get_w(u, v)
    t = torch.full_like(u, 250.0)
    kernel.share_z_dxyz(torch.zeros_like(u))
    t_t = kernel.get_t_t(u, v, w, t)
    return {
        "w_400hpa_interior": float(w[0, 6, 2:-2, 4:-4].mean()),
        "t_t_400hpa_interior": float(t_t[0, 6, 2:-2, 4:-4].mean()),
    }


def probe_hydrostatic() -> dict[str, float]:
    """Однородный прогрев t_t=1e-4 K/с: z_t>0 c ростом вверх (толщина растёт)."""
    kernel = make_kernel()
    kernel.t_t = torch.full((1, P, H, W), 1e-4)
    z_t = kernel.get_z_t()
    return {
        "z_t_50hpa": float(z_t[0, 0, 0, 0]),
        "z_t_500hpa": float(z_t[0, 7, 0, 0]),
        "z_t_1000hpa": float(z_t[0, -1, 0, 0]),
    }


def probe_condensation() -> dict[str, float]:
    """Насыщенный придонный подъём сушит (~-1e-8), опускание/ненасыщение - ноль."""
    results: dict[str, float] = {}
    for name, w_value, q_value in (
        ("saturated_ascent_q_t_1000hpa", 1e-3, 6.5e-3),
        ("saturated_descent_q_t_max_abs", -1e-3, 6.5e-3),
        ("subsaturated_ascent_q_t_max_abs", 1e-3, 3e-3),
    ):
        kernel = make_kernel()
        t = torch.full((1, P, H, W), 280.0)
        zeros = torch.zeros_like(t)
        kernel.share_z_dxyz(PHI_STD.expand(1, P, H, W).clone())
        kernel.t_t = torch.zeros_like(t)
        kernel.get_z_t()
        q_t = kernel.get_q_dt(
            zeros, zeros, t, torch.full_like(t, w_value), torch.full_like(t, q_value)
        )
        if name.startswith("saturated_ascent"):
            results[name] = float(q_t[0, 12, 2:-2, 4:-4].mean())
        else:
            results[name] = float(q_t.abs().max())
    return results


def probe_pure_kernel_qs() -> dict[str, float]:
    """PurePDEKernel._get_qs: физичность и batch-независимость Магнуса."""
    grid = Grid(GridConfig(H=H, W=W, lat_range_deg=(16.17, 59.77)))
    kernel = PurePDEKernel(grid, boundary_x="replicate")
    q_s = kernel._get_qs(torch.full((1, 1, 1, 1), 5e4), torch.full((1, 1, 1, 1), 253.0))
    p2 = torch.full((2, 1, 1, 1), 5e4)
    qs_a = kernel._get_qs(p2, torch.tensor([280.0, 250.0]).reshape(2, 1, 1, 1))[0]
    qs_b = kernel._get_qs(p2, torch.tensor([280.0, 320.0]).reshape(2, 1, 1, 1))[0]
    return {
        "qs_500hpa_253K": float(q_s),
        "qs_batch_dependence_abs": float((qs_a - qs_b).abs().max()),
    }


def main() -> None:
    """Собирает все пробы и пишет sign_convention_results.json рядом со скриптом."""
    results = {
        "d_z_of_p_interior_mean": probe_d_z_sign(),
        "integral_z_orientation": probe_integral_orientation(),
        "divergence_probe": probe_w_and_adiabatic(),
        "hydrostatic_probe": probe_hydrostatic(),
        "condensation_probe": probe_condensation(),
        "pure_kernel_qs_probe": probe_pure_kernel_qs(),
    }
    out_path = Path(__file__).with_name("sign_convention_results.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
