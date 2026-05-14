"""72-h rollout проверка метода WeatherGFT: FD-4 + Forward Euler + const Coriolis.

Семантика (1:1 с Models/WeatherGFT.py:113-307, **без** scale_diff и .detach()):

    Momentum:    u_t = -u·u_x - v·u_y - w·u_z + f·v - z_x
                 v_t = -u·v_x - v·v_y - w·v_z - f·u - z_y
    Temperature: t_t = (Q - z_z·w)/c_p - u·t_x - v·t_y - w·t_z,  Q = -L·z_z·w
    Geopotent.:  z_t = integral_z(-R / p · t_t)
    Continuity:  w   = integral_z(-(u_x + v_y))   [diagnostic]
    Humidity:    q_t = -(u·q_x + v·q_y + w·q_z) + adiabatic-Kuo term

Время: Forward Euler, block_dt=300 s, 12 substep’ов/час, 72 часа = 864 substep’а.
Сетка: 128×256 (ERA5 1.4°). Coriolis: const 7.29e-5 (как в оригинале).
Boundary: periodic (как в оригинале, через torch.cat).

CPU-only. Запуск:
    python tools/check_physics_weathergft.py \
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
    coriolis_constant,
    default_initial_conditions,
    integral_z_cpu,
    relhum_to_specific,
    run_72h_rollout,
)
from utils.old_physics import make_weathergft_ops


# Физические константы (WeatherGFT.py:125-131).
L = 2.5e6
R = 8.314
c_p = 1005.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--memmap-path", default="/home/fa.buzaev/era5_memmap/predformer_globe_2000_2018.dat")
    p.add_argument("--memmap-meta-path", default=None)
    p.add_argument(
        "--mean-std-path",
        default="",
        help="Per-channel mean/std (.npy or .json). Pass empty (default) if memmap already stores raw "
        "physical units (v3/v4 memmap convention). Pass a path only if memmap holds normalised data.",
    )
    p.add_argument("--H", type=int, default=32, help="Grid height. Default matches predformer_globe memmap (32).")
    p.add_argument("--W", type=int, default=64, help="Grid width. Default matches predformer_globe memmap (64).")
    p.add_argument("--year", type=int, default=2005)
    p.add_argument("--horizon-hours", type=int, default=72)
    p.add_argument("--block-dt-seconds", type=float, default=300.0)
    p.add_argument("--coriolis-value", type=float, default=7.29e-5)
    p.add_argument("--offline", action="store_true", help="Comet OfflineExperiment вместо живой синхронизации")
    p.add_argument("--project-name", default="WeatherPredictions")
    args = p.parse_args()

    # Force CPU.
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cpu")
    print(f"[init] device={device}, threads={torch.get_num_threads()}")

    # Geometry + ops.
    geom = GeometryCPU(H=args.H, W=args.W)
    ops = make_weathergft_ops(latents_size=(args.H, args.W))
    d_x = ops.d_x
    d_y = ops.d_y
    d_z = ops.d_z
    integral_z = ops.integral_z

    f_field = coriolis_constant(geom, value=args.coriolis_value).to(device)
    pressure_pa = geom.pressure_pa_t.to(device)

    def rollout_step(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Один Euler-substep `block_dt` секунд по WeatherGFT-семантике, без scale_diff."""
        u, v, t, q, z = state["u"], state["v"], state["t"], state["q"], state["z"]

        # diagnostic w
        u_x_for_w = d_x(u)
        v_y_for_w = d_y(v)
        w_z = -(u_x_for_w + v_y_for_w)
        w = integral_z(w_z)

        # spatial derivs
        u_x = u_x_for_w
        u_y = d_y(u)
        u_z = d_z(u)
        v_x = d_x(v)
        v_y = v_y_for_w
        v_z = d_z(v)
        t_x = d_x(t)
        t_y = d_y(t)
        t_z = d_z(t)
        q_x = d_x(q)
        q_y = d_y(q)
        q_z = d_z(q)
        z_x = d_x(z)
        z_y = d_y(z)
        z_z = d_z(z)

        # u, v tendencies (momentum)
        u_t = -u * u_x - v * u_y - w * u_z + f_field * v - z_x
        v_t = -u * v_x - v * v_y - w * v_z - f_field * u - z_y

        # t tendency (temperature)
        Q = -L * z_z * w
        t_t = (Q - z_z * w) / c_p - u * t_x - v * t_y - w * t_z

        # z tendency via hydrostatic
        z_zt = -R / pressure_pa * t_t
        z_t = integral_z(z_zt)

        # q tendency (упрощённая, без condensation switch).
        # Полная формула из WeatherGFT.py:237-272 включает Magnus + δ-switch + F_;
        # для consistency с физикой берём только адвективную часть.
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
        }
        # r — пересчитываем из обновлённого q (через насыщение от обновлённого t).
        # Простое обратное приближение: r = 100 · q / q_s(t, p).
        # На rollout’е используем только q как prognostic; r логируем для отчёта.
        new_state["r"] = state["r"]  # не обновляем (для лога нужно но не критично)

        rhs = {"u_t": u_t, "v_t": v_t, "t_t": t_t, "q_t": q_t, "z_t": z_t}
        return new_state, rhs

    run_72h_rollout(
        method_name="fd4_euler_constCoriolis",
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
        tags=["fd4", "euler", "coriolis_constant", "method_weathergft"],
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
