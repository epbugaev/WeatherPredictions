"""72-h CPU rollout проверка рефакторенной :class:`utils.physics.PurePDEKernel`.

Это параметризованный entry-point для НОВОЙ семьи интеграторов (через
:mod:`utils.physics`, не :mod:`utils.old_physics`). Все буферы регистрируются
через `register_buffer`, нет module-globals; stencil/coriolis/time-scheme/
boundary — параметры. Никаких обучаемых слоёв, никаких Conv2d/BatchNorm.

CLI задаёт конкретный интегратор:
    --stencil       fd4 | weno5
    --time-scheme   euler | rk4
    --coriolis      constant | beta_plane | spherical
    --boundary-h    periodic | reflect      (по горизонтали)
    --boundary-z    periodic | reflect      (по давлению)
    --use-R-d       (флаг: использовать R_d=287 в гидростатике, fix из аудита)

Имя метода в Comet строится автоматически как
``<stencil>_<time-scheme>_<coriolis>[_Rd]`` — соответствует «типу интегратора»,
не названию файла-источника.

Запуск (cluster):
    python tools/check_physics_new_kernel.py \
        --stencil fd4 --time-scheme euler --coriolis spherical
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
    default_initial_conditions,
    run_72h_rollout,
)
from utils.physics import Grid, GridConfig, PurePDEKernel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--memmap-path", default="/home/fa.buzaev/era5_memmap/predformer_globe_2000_2018.dat"
    )
    p.add_argument("--memmap-meta-path", default=None)
    p.add_argument(
        "--mean-std-path", default="", help="Empty: assume memmap holds raw physical units."
    )
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--year", type=int, default=2005)
    p.add_argument("--horizon-hours", type=int, default=72)
    p.add_argument("--block-dt-seconds", type=float, default=300.0)
    p.add_argument("--stencil", choices=["fd4", "weno5"], required=True)
    p.add_argument("--time-scheme", choices=["euler", "rk4"], required=True)
    p.add_argument(
        "--coriolis", choices=["constant", "beta_plane", "spherical"], default="spherical"
    )
    p.add_argument("--boundary-h", choices=["periodic", "reflect"], default="periodic")
    p.add_argument("--boundary-z", choices=["periodic", "reflect"], default="periodic")
    p.add_argument(
        "--use-R-d", action="store_true", help="Use R_d=287 in hydrostatic eq instead of R=8.314."
    )
    p.add_argument("--offline", action="store_true")
    p.add_argument("--project-name", default="WeatherPredictions")
    args = p.parse_args()

    device = torch.device("cpu")
    print(f"[init] device={device}, threads={torch.get_num_threads()}")

    # Method name == integrator type signature (matches user request).
    method_parts = [args.stencil, args.time_scheme, args.coriolis]
    if args.use_R_d:
        method_parts.append("Rd")
    method_name = "_".join(method_parts)
    print(f"[init] method_name={method_name}")

    # Build grid + kernel via utils.physics (NEW family — register_buffer-based).
    grid = Grid(GridConfig(H=args.H, W=args.W, lat_scheme="linear_minus90_90")).to(device)
    kernel = PurePDEKernel(
        grid,
        stencil=args.stencil,
        coriolis=args.coriolis,
        block_dt=args.block_dt_seconds,
        time_scheme=args.time_scheme,
        boundary_horiz=args.boundary_h,
        boundary_z=args.boundary_z,
        use_R_d_in_hydrostatic=args.use_R_d,
    ).to(device)

    # f_field для метрик physics consistency.
    f_field = kernel.f_field

    # d_x/d_y для метрик. PurePDEKernel экспортирует через .diff.
    d_x_fn = kernel.diff.d_x
    d_y_fn = kernel.diff.d_y

    # GeometryCPU нужен для lat_weights/pixel_x/pixel_y в forecast метриках.
    # Reuse параметров новой Grid через CPU shadow (значения совпадают).
    geom = GeometryCPU(H=args.H, W=args.W)

    def rollout_step(state):
        """Один substep через PurePDEKernel.step()."""
        u, v, t, q, z = state["u"], state["v"], state["t"], state["q"], state["z"]
        out = kernel.step(u, v, t, q, z)
        new_state = {
            "t2m": state["t2m"],
            "u10": state["u10"],
            "v10": state["v10"],
            "tp": state["tp"],
            "r": state["r"],
            "z": out["z"],
            "t": out["t"],
            "u": out["u"],
            "v": out["v"],
            "q": out["q"],
        }
        rhs = {
            "u_t": out["u_t"],
            "v_t": out["v_t"],
            "t_t": out["t_t"],
            "q_t": out["q_t"],
            "z_t": out["z_t"],
        }
        return new_state, rhs

    tags = [
        f"stencil_{args.stencil}",
        f"time_{args.time_scheme}",
        f"coriolis_{args.coriolis}",
        f"bdry_h_{args.boundary_h}",
        f"bdry_z_{args.boundary_z}",
        "new_kernel",  # семья utils.physics
    ]
    if args.use_R_d:
        tags.append("R_d_hydrostatic")

    run_72h_rollout(
        method_name=method_name,
        rollout_step_fn=rollout_step,
        d_x_fn=d_x_fn,
        d_y_fn=d_y_fn,
        f_field=f_field,
        geom=geom,
        initial_conditions=default_initial_conditions(year=args.year),
        memmap_path=args.memmap_path,
        memmap_meta_path=args.memmap_meta_path,
        mean_std_path=args.mean_std_path,
        horizon_hours=args.horizon_hours,
        block_dt_seconds=args.block_dt_seconds,
        project_name=args.project_name,
        tags=tags,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
