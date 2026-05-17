"""72-h rollout проверка метода WeatherGFT_3 (dev): FD-4 + RK4 + spherical Coriolis +
turbulent mixing + radiative cooling + convective precipitation.

Семантика (1:1 с Models/dev/WeatherGFT_3.py:114-541, **без** scale_diff и .detach()):

    Momentum (non-conservative form, как в FD-варианте):
        u_t = -u·u_x - v·u_y - w·u_z + f_sph·v - z_x + apply_turbulent_mixing(u)
        v_t = -u·v_x - v·v_y - w·v_z - f_sph·u - z_y + apply_turbulent_mixing(v)
    Temperature:
        t_t = R_d·T·ω/(c_p·p) - u·t_x - v·t_y - w·t_z + turb_mix(t) + radiative_cooling + latent_heating,
        где ω = 100·w (hPa/s → Pa/s).
    Geopotent.:  z_t = integral_z(-R_d / p · t_t)
    Continuity:  w   = integral_z(-(u_x + v_y))
    Humidity:    q_t = advection + Kuo + turb_mix(q) - precip_rate
    Turbulent mixing: K_h·(∂²/∂x² + ∂²/∂y²) + K_v·∂²/∂z²
    Radiative cooling: -ε·σ·T⁴ / (c_p·ρ)  (Stefan-Boltzmann)
    Convective precipitation: triggers when q > 0.8·q_s, releases L·precip/c_p

Время: RK4, block_dt=300 s, 12 substep’ов/час, 72 часа = 864 substep’а.
RK4 в 4 раза дороже Euler, ожидаемое время ~ 60 мин на IC на CPU.
Сетка: 128×256. Coriolis: spherical (f = 2Ω sin φ).
Boundary: periodic (как в FD).

CPU-only. Запуск:
    python tools/check_physics_weathergft_3.py \
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
    coriolis_spherical,
    default_initial_conditions,
    magnus_qs,
    run_72h_rollout,
)
from utils.old_physics import make_weathergft_ops

# Константы из Models/dev/WeatherGFT_3.py:127-148.
L = 2.5e6  # Дж/кг (для latent_heating)
R_d = 287.0  # сухой воздух (per-mass)
c_p = 1005.0  # Дж/(кг·К)
sigma_sb = 5.67e-8  # Stefan-Boltzmann
emissivity = 0.7
K_H = 15.0  # горизонтальная диффузия, м²/с
K_V = 0.1  # вертикальная диффузия, м²/с
PRECIP_THRESHOLD = 0.8  # порог rel humidity для convective precip


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
    p.add_argument("--offline", action="store_true")
    p.add_argument("--project-name", default="WeatherPredictions")
    args = p.parse_args()

    device = torch.device("cpu")
    print(f"[init] device={device}, threads={torch.get_num_threads()}")

    geom = GeometryCPU(H=args.H, W=args.W, lat_range_deg=tuple(args.lat_range_deg))
    ops = make_weathergft_ops(latents_size=(args.H, args.W))
    d_x = ops.d_x
    d_y = ops.d_y
    d_z = ops.d_z
    integral_z = ops.integral_z

    f_field = coriolis_spherical(geom).to(device)
    pressure_pa = geom.pressure_pa_t.to(device)

    def apply_turbulent_mixing(field: torch.Tensor) -> torch.Tensor:
        """K_h·(∂²/∂x² + ∂²/∂y²) + K_v·∂²/∂z² (dev/WeatherGFT_3.py:190-199)."""
        return K_H * (d_x(d_x(field)) + d_y(d_y(field))) + K_V * d_z(d_z(field))

    def radiative_cooling(t_kelvin: torch.Tensor) -> torch.Tensor:
        """Stefan-Boltzmann cooling rate в К/с (dev/WeatherGFT_3.py:201-215).

        В оригинале: dt_rad = -ε·σ·T⁴ / (c_p · p·100 / (R_d·T))
                    = -ε·σ·T⁴·R_d·T / (c_p·p_Pa)
        Переписываем безопаснее: p в Па, T в К, итог в К/с.
        """
        cooling_W_m2 = emissivity * sigma_sb * t_kelvin**4
        # rho = p / (R_d · T) → масса воздуха на m². dT/dt = -Q / (c_p · rho · 1m).
        # В оригинале: dt_rad = -Q / (c_p · pressure_Pa / (R_d · T_abs))
        return -cooling_W_m2 / (c_p * pressure_pa / (R_d * t_kelvin))

    def convective_precipitation(
        q: torch.Tensor, t_kelvin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Простая Kuo-схема (dev/WeatherGFT_3.py:217-229).

        Returns:
            precip_rate: кг/(м²·с) — отнимается от q.
            latent_heating: К/с — добавляется к t.
        """
        q_s = torch.maximum(magnus_qs(t_kelvin, pressure_pa), torch.full_like(q, 1e-6))
        rel_hum = q / q_s
        precip_mask = (rel_hum > PRECIP_THRESHOLD).float()
        precip_rate = precip_mask * (q - PRECIP_THRESHOLD * q_s) / args.block_dt_seconds
        latent_heating = precip_rate * L / c_p
        return precip_rate, latent_heating

    def rhs(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Полный RHS dev/WeatherGFT_3.py для всех 5 prognostic vars."""
        u, v, t, q, z = state["u"], state["v"], state["t"], state["q"], state["z"]

        u_x = d_x(u)
        v_y = d_y(v)
        w = integral_z(-(u_x + v_y))

        u_y = d_y(u)
        u_z = d_z(u)
        v_x = d_x(v)
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

        u_mix = apply_turbulent_mixing(u)
        v_mix = apply_turbulent_mixing(v)
        t_mix = apply_turbulent_mixing(t)
        q_mix = apply_turbulent_mixing(q)

        u_t = -u * u_x - v * u_y - w * u_z + f_field * v - z_x + u_mix
        v_t = -u * v_x - v * v_y - w * v_z - f_field * u - z_y + v_mix

        # Temperature: adiabatic R_d·T·ω/(c_p·p) + advection + radiation +
        # latent heating + turb-mixing.
        t_t_adia = adiabatic_temperature_tendency(t, w, pressure_pa, r_d=R_d, c_p=c_p)
        rad_cool = radiative_cooling(t)
        precip_rate, latent_heat = convective_precipitation(q, t)
        t_t = t_t_adia - u * t_x - v * t_y - w * t_z + t_mix + rad_cool + latent_heat

        # Hydrostatic z evolution (R_d). integral_z в hPa → делитель в hPa
        # (pressure_pa/100), иначе z_t ×100 меньше корректного.
        z_zt = -R_d / (pressure_pa / 100.0) * t_t
        z_t = integral_z(z_zt)

        q_t = -(u * q_x + v * q_y + w * q_z) + q_mix - precip_rate

        return {"u_t": u_t, "v_t": v_t, "t_t": t_t, "q_t": q_t, "z_t": z_t}

    def add_dt_to_state(
        state: dict[str, torch.Tensor], rhs_dict: dict[str, torch.Tensor], dt: float
    ) -> dict[str, torch.Tensor]:
        return {
            "t2m": state["t2m"],
            "u10": state["u10"],
            "v10": state["v10"],
            "tp": state["tp"],
            "r": state["r"],
            "u": state["u"] + dt * rhs_dict["u_t"],
            "v": state["v"] + dt * rhs_dict["v_t"],
            "t": state["t"] + dt * rhs_dict["t_t"],
            "q": state["q"] + dt * rhs_dict["q_t"],
            "z": state["z"] + dt * rhs_dict["z_t"],
        }

    def rollout_step(
        state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """RK4 substep (dev/WeatherGFT_3.py:255-285, без scale_diff)."""
        dt = args.block_dt_seconds

        k1 = rhs(state)
        s1 = add_dt_to_state(state, k1, 0.5 * dt)
        k2 = rhs(s1)
        s2 = add_dt_to_state(state, k2, 0.5 * dt)
        k3 = rhs(s2)
        s3 = add_dt_to_state(state, k3, dt)
        k4 = rhs(s3)

        comb = {key: (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key]) / 6.0 for key in k1}
        new_state = add_dt_to_state(state, comb, dt)
        return new_state, comb

    run_72h_rollout(
        method_name="fd4_RK4_sphericalCoriolis_radiative_turbMixing",
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
            "fd4",
            "rk4",
            "coriolis_spherical",
            "turbulent_mixing",
            "radiative_cooling",
            "convective_precip",
            "method_weathergft_3",
        ],
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
