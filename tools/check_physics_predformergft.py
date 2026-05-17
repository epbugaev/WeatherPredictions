"""72-h rollout проверка метода PredFormerGFT: WENO-5 + Forward Euler + beta-plane Coriolis.

Геометрия и операторы — через :mod:`utils.old_physics.make_predformergft_ops`,
семантика физики обновлена (см. CHANGELOG): сломанная `Q = -L·z_z·w` заменена
на правильную адиабатику `dT/dt|_adia = R_d·T·ω/(c_p·p)`, гидростатика
использует `R_d` вместо универсальной `R`.

    Momentum (conservative form with AMR pseudo-refinement):
        adv_u = ∂(u·u)/∂x + ∂(u·v)/∂y + ∂(u·w)/∂z
        adv_v = ∂(u·v)/∂x + ∂(v·v)/∂y + ∂(v·w)/∂z
        u_t = -adv_u + f_field·v - z_x   (+ eddy_viscosity·∇²u, если включено)
        v_t = -adv_v - f_field·u - z_y   (+ eddy_viscosity·∇²v)
    Temperature: t_t = R_d·T·ω/(c_p·p) - u·t_x - v·t_y - w·t_z, ω = 100·w (hPa/s → Pa/s)
    Geopotent.:  z_t = integral_z(-R_d / p · t_t)
    Continuity:  w   = integral_z(-(u_x + v_y))

Время: Forward Euler, block_dt=300 s, 12 substep’ов/час, 72 часа.
Сетка: 128×256. Coriolis: beta-plane (f0=7.29e-5 + β·R·φ, β=1.6e-11).
Boundary: reflect для WENO по horizontal, periodic по pressure.

CPU-only. Запуск:
    python tools/check_physics_predformergft.py \
        --memmap-path /home/fa.buzaev/era5_memmap/predformer_globe_2000_2018.dat \
        --mean-std-path /home/epbugaev/weather_bench/1.40625deg/mean_std.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_physics_common import (
    GeometryCPU,
    adiabatic_temperature_tendency,
    coriolis_beta_plane,
    default_initial_conditions,
    run_72h_rollout,
)
from utils.old_physics import make_predformergft_ops

# Физические константы — R_d (per-mass) в гидростатике + адиабатика.
R_D = 287.0  # сухой воздух, J/(kg·K)
C_P = 1005.0  # теплоёмкость, J/(kg·K)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--memmap-path", default="/home/fa.buzaev/era5_memmap/predformer_globe_2000_2018.dat"
    )
    p.add_argument("--memmap-meta-path", default=None)
    p.add_argument(
        "--mean-std-path",
        default="",
        help="Empty: assume memmap holds raw physical units (v3/v4 default).",
    )
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--W", type=int, default=64)
    p.add_argument(
        "--lat-range-deg",
        nargs=2,
        type=float,
        default=[-90.0, 90.0],
        metavar=("LOW", "HIGH"),
        help="Диапазон широт. Дефолт global. USA-кроп: 24 56.",
    )
    p.add_argument("--year", type=int, default=2005)
    p.add_argument("--horizon-hours", type=int, default=48)
    p.add_argument("--block-dt-seconds", type=float, default=300.0)
    p.add_argument("--f0", type=float, default=7.29e-5)
    p.add_argument("--beta", type=float, default=1.6e-11)
    p.add_argument("--eddy-viscosity", type=float, default=0.0)
    p.add_argument(
        "--use-amr",
        action="store_true",
        help="Включить adaptive_mesh_refinement (default off — оригинал PredFormerGFT включает его, но он медленный)",
    )
    p.add_argument("--offline", action="store_true")
    p.add_argument("--project-name", default="WeatherPredictions")
    args = p.parse_args()

    device = torch.device("cpu")
    print(f"[init] device={device}, threads={torch.get_num_threads()}")

    geom = GeometryCPU(H=args.H, W=args.W, lat_range_deg=tuple(args.lat_range_deg))
    ops = make_predformergft_ops(latents_size=(args.H, args.W))
    d_x = ops.d_x_weno
    d_y = ops.d_y_weno
    d_z = ops.d_z
    integral_z = ops.integral_z
    laplacian = ops.laplacian_tensor
    compute_with_amr = ops.compute_derivative_with_amr

    f_field = coriolis_beta_plane(geom, f0=args.f0, beta=args.beta).to(device)
    pressure_pa = geom.pressure_pa_t.to(device)

    def rollout_step(
        state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Один Euler-substep `block_dt` секунд по PredFormerGFT-семантике."""
        u, v, t, q, z = state["u"], state["v"], state["t"], state["q"], state["z"]

        # diagnostic w из u_x + v_y
        u_x = d_x(u)
        v_y = d_y(v)
        w = integral_z(-(u_x + v_y))

        # Spatial derivatives для z (используются в momentum-форсинге).
        z_x = d_x(z)
        z_y = d_y(z)
        # z_z не нужен: правильная адиабатика использует ω (=100·w), не z_z·w.

        # Conservative-form advection (matches PredFormerGFT.py:277-283).
        if args.use_amr:
            adv_u = compute_with_amr(u * u, d_x) + compute_with_amr(u * v, d_y) + d_z(u * w)
            adv_v = compute_with_amr(u * v, d_x) + compute_with_amr(v * v, d_y) + d_z(v * w)
        else:
            adv_u = d_x(u * u) + d_y(u * v) + d_z(u * w)
            adv_v = d_x(u * v) + d_y(v * v) + d_z(v * w)

        u_t = -adv_u + f_field * v - z_x
        v_t = -adv_v - f_field * u - z_y

        if args.eddy_viscosity > 0:
            u_t = u_t + args.eddy_viscosity * laplacian(u)
            v_t = v_t + args.eddy_viscosity * laplacian(v)

        # Temperature: адиабатика R_d·T·ω/(c_p·p) + advection.
        t_x = d_x(t)
        t_y = d_y(t)
        t_z = d_z(t)
        t_t_adia = adiabatic_temperature_tendency(t, w, pressure_pa, r_d=R_D, c_p=C_P)
        t_t = t_t_adia - u * t_x - v * t_y - w * t_z

        # Hydrostatic z evolution (R_d). integral_z в hPa → делитель в hPa
        # (pressure_pa/100), иначе z_t ×100 меньше корректного.
        z_zt = -R_D / (pressure_pa / 100.0) * t_t
        z_t = integral_z(z_zt)

        # Humidity (advection-only, без condensation switch)
        q_x = d_x(q)
        q_y = d_y(q)
        q_z = d_z(q)
        q_t = -(u * q_x + v * q_y + w * q_z)

        dt = args.block_dt_seconds
        new_state = {
            "t2m": state["t2m"],
            "u10": state["u10"],
            "v10": state["v10"],
            "tp": state["tp"],
            "z": z + dt * z_t,
            "t": t + dt * t_t,
            "u": u + dt * u_t,
            "v": v + dt * v_t,
            "q": q + dt * q_t,
            "r": state["r"],
        }

        rhs = {"u_t": u_t, "v_t": v_t, "t_t": t_t, "q_t": q_t, "z_t": z_t}
        return new_state, rhs

    run_72h_rollout(
        method_name="weno5_euler_betaPlane_AMR",
        rollout_step_fn=rollout_step,
        d_x_fn=d_x,
        d_y_fn=d_y,
        f_field=f_field,
        geom=geom,
        initial_conditions=default_initial_conditions(year=args.year),
        memmap_path=args.memmap_path,
        memmap_meta_path=args.memmap_meta_path,
        mean_std_path=args.mean_std_path,
        horizon_hours=args.horizon_hours,
        block_dt_seconds=args.block_dt_seconds,
        project_name=args.project_name,
        tags=[
            "weno5",
            "euler",
            "coriolis_beta_plane",
            f"eddy_viscosity_{args.eddy_viscosity}",
            f"amr_{args.use_amr}",
            "method_predformergft",
        ],
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
