"""72-h rollout проверка метода WeatherGFT: FD-4 + Forward Euler + const Coriolis.

Геометрия и операторы воспроизводят Models/WeatherGFT.py через
:mod:`utils.old_physics.make_weathergft_ops`. Семантика **физики** — обновлённая
(см. CHANGELOG): сломанная формула `Q = -L·z_z·w` заменена на правильную
адиабатику `dT/dt|_adia = R_d·T·ω/(c_p·p)`, гидростатика использует `R_d`
вместо универсальной `R`.

    Momentum:    u_t = -u·u_x - v·u_y - w·u_z + f·v - z_x
                 v_t = -u·v_x - v·v_y - w·v_z - f·u - z_y
    Temperature: t_t = R_d·T·ω/(c_p·p) - u·t_x - v·t_y - w·t_z,
                 где ω = 100·w (w в hPa/s → ω в Pa/s).
    Geopotent.:  z_t = integral_z(-R_d / p · t_t)
    Continuity:  w   = integral_z(-(u_x + v_y))   [diagnostic]
    Humidity:    q_t = -(u·q_x + v·q_y + w·q_z)   [advection-only]

Время: Forward Euler, block_dt=300 s, 12 substep’ов/час.
Сетка: 32×64 (ERA5 1.4°). Coriolis: --coriolis-value override (default
тот же легаси 7.29e-5 = Ω, чтобы конфигурация воспроизводила старые
численные эксперименты; для физически правильного 2Ω·sin(45°) передайте
`--coriolis-value 1.03e-4` или используйте tools/check_physics_new_kernel.py).
Boundary: replicate-через-cat (как в оригинале старой физики).

CPU-only. Запуск:
    python tools/check_physics_weathergft.py \
        --memmap-path "$WEATHERPRED_GLOBE_MEMMAP"
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
    coriolis_constant,
    default_initial_conditions,
    run_72h_rollout,
)
from utils.old_physics import make_weathergft_ops
from utils.paths import globe_memmap_path

# Физические константы — R_d (per-mass) в гидростатике + адиабатика.
R_D = 287.0  # сухой воздух, J/(kg·K)
C_P = 1005.0  # теплоёмкость, J/(kg·K)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--memmap-path", default=globe_memmap_path())
    p.add_argument("--memmap-meta-path", default=None)
    p.add_argument(
        "--mean-std-path",
        default="",
        help="Per-channel mean/std (.npy or .json). Pass empty (default) if memmap already stores raw "
        "physical units (v3/v4 memmap convention). Pass a path only if memmap holds normalised data.",
    )
    p.add_argument(
        "--H",
        type=int,
        default=32,
        help="Grid height. Default matches predformer_globe memmap (32).",
    )
    p.add_argument(
        "--W",
        type=int,
        default=64,
        help="Grid width. Default matches predformer_globe memmap (64).",
    )
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
    p.add_argument("--coriolis-value", type=float, default=7.29e-5)
    p.add_argument(
        "--offline", action="store_true", help="Comet OfflineExperiment вместо живой синхронизации"
    )
    p.add_argument("--project-name", default="WeatherPredictions")
    args = p.parse_args()

    # Force CPU.
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cpu")
    print(f"[init] device={device}, threads={torch.get_num_threads()}")

    # Geometry + ops.
    geom = GeometryCPU(H=args.H, W=args.W, lat_range_deg=tuple(args.lat_range_deg))
    ops = make_weathergft_ops(latents_size=(args.H, args.W))
    d_x = ops.d_x
    d_y = ops.d_y
    d_z = ops.d_z
    integral_z = ops.integral_z

    f_field = coriolis_constant(geom, value=args.coriolis_value).to(device)
    pressure_pa = geom.pressure_pa_t.to(device)

    def rollout_step(
        state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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
        # z_z не нужен: правильная адиабатика использует ω (=100·w), не z_z·w.

        # u, v tendencies (momentum)
        u_t = -u * u_x - v * u_y - w * u_z + f_field * v - z_x
        v_t = -u * v_x - v * v_y - w * v_z - f_field * u - z_y

        # t tendency: адиабатика R_d·T·ω/(c_p·p) + горизонтальная адвекция
        t_t_adia = adiabatic_temperature_tendency(t, w, pressure_pa, r_d=R_D, c_p=C_P)
        t_t = t_t_adia - u * t_x - v * t_y - w * t_z

        # z tendency via hydrostatic (R_d, не универсальная).
        # integral_z интегрирует с pixel_z в hPa, поэтому делитель давления
        # тоже в hPa (pressure_pa/100). Иначе z_t был бы ×100 меньше.
        z_zt = -R_D / (pressure_pa / 100.0) * t_t
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

    # БАГ-4 fix: имя метода и тег самодокументируют реальное значение Coriolis,
    # чтобы Comet-эксперимент не вводил в заблуждение (legacy Ω=7.29e-5 vs
    # каноничное 2Ω·sin45°≈1.03e-4). 7.29e-5 — это Ω (paper), не f.
    cval = args.coriolis_value
    is_legacy_omega = abs(cval - 7.29e-5) < 1e-7
    coriolis_tag = "constOmega_legacy" if is_legacy_omega else f"const_{cval:.3e}"
    method_name = f"fd4_euler_{coriolis_tag}"
    print(f"[init] method_name={method_name} (coriolis f={cval:.3e})")

    run_72h_rollout(
        method_name=method_name,
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
        tags=["fd4", "euler", coriolis_tag, f"coriolis_value_{cval:.3e}", "method_weathergft"],
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
