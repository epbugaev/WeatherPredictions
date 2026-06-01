"""72-h rollout с тремя кандидатами фиксов оригинальной PDE-семантики.

Базовый интегратор фиксирован: FD-4 + Forward Euler + spherical Coriolis +
periodic boundaries (минимальная конфигурация, чтобы выделить вклад фикса).

Три fix-режима (по результатам ``tools/diagnose_first_substep.py``):

  --fix-mode=A  «alpha-omega heating»
        Заменяем broken `Q = -L·z_z·w` на правильную адиабатику
        `dT/dt|_adia = α·ω/(c_p) = R_d·T·ω/(c_p·p)`, где `ω` приведён к Pa/s.
        Также гидростатика: `z_zt = -R_d/p_hPa · t_t` (R_d вместо универсальной;
        p в hPa под hPa-интеграл integral_z — см. БАГ-1 fix).
        Это устраняет 7-порядковый overflow `t_t` за substep.

  --fix-mode=B  «Pa units + R_d»
        Оригинальная формула `Q = -L·z_z·w` сохраняется (т.е. бага не
        исправляется), но pressure-units переведены в Pa, hydrostatic ↔ R_d.
        Изолирует вклад «корректных размерностей» БЕЗ замены формулы.
        Прогноз: одной этой смены недостаточно, blow-up сохранится — потому
        что `z_z` (1/100×) и `w` (100×) при смене hPa→Pa умножаются взаимно
        и `Q` не меняется по абсолютной величине.

  --fix-mode=C  «production scale_diff + detach»
        Воспроизводит исходную production-семантику: после каждого Euler-
        шага tendency прогоняется через `scale_diff` (зажим в
        `(x_max-x_mean)·0.05` диапазон) и `.detach()`. Реальная физика
        не возникает (амплитуды клипуются), но NaN’ов не будет.

Логирование: Comet ML, имя эксперимента ``fd4_euler_spherical_fix<X>_<tag>``.

CPU-only.
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
    coriolis_spherical,
    default_initial_conditions,
    run_72h_rollout,
)
from utils.old_physics import make_weathergft_ops
from utils.paths import globe_memmap_path

# Physical constants
L = 2.5e6  # Latent heat of vaporisation, J/kg
R_UNIVERSAL = 8.314  # Universal gas constant J/(mol·K) — wrong-units choice in WeatherGFT.py
R_D = 287.0  # Dry-air gas constant, J/(kg·K) — correct one
C_P = 1005.0  # Specific heat of air at const pressure, J/(kg·K)
DIFF_RATIO = 0.05  # scale_diff coefficient (matches WeatherGFT.py:131)


def scale_tensor(tensor: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Min-max rescale tensor into [a, b]. Matches WeatherGFT.py:146-151."""
    mn = tensor.min().detach()
    mx = tensor.max().detach()
    span = mx - mn
    safe = torch.where(span.abs() > 1e-12, span, torch.full_like(span, 1.0))
    scaled = (tensor - mn) / safe
    return scaled * (b - a) + a


def scale_diff(diff: torch.Tensor, x: torch.Tensor, diff_ratio: float = DIFF_RATIO) -> torch.Tensor:
    """Original WeatherGFT.py:154-159 clipping of tendency increments."""
    x_min = x.min().detach()
    x_mean = x.mean().detach()
    x_max = x.max().detach()
    diff_min = (x_min - x_mean) * diff_ratio
    diff_max = (x_max - x_mean) * diff_ratio
    return scale_tensor(diff, diff_min.item(), diff_max.item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--memmap-path", default=globe_memmap_path())
    p.add_argument("--memmap-meta-path", default=None)
    p.add_argument("--mean-std-path", default="")
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
    p.add_argument("--fix-mode", choices=["A", "B", "C"], required=True)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--project-name", default="WeatherPredictions")
    args = p.parse_args()

    device = torch.device("cpu")
    print(f"[init] device={device}, threads={torch.get_num_threads()}, fix_mode={args.fix_mode}")

    geom = GeometryCPU(H=args.H, W=args.W, lat_range_deg=tuple(args.lat_range_deg))
    ops = make_weathergft_ops(latents_size=(args.H, args.W))
    d_x = ops.d_x
    d_y = ops.d_y
    d_z = ops.d_z
    integral_z = ops.integral_z

    f_field = coriolis_spherical(geom).to(device)
    pressure_pa = geom.pressure_pa_t.to(device)  # (1, 13, 1, 1) — Pa

    fix = args.fix_mode

    # ---------------- Common derivative computation ----------------

    def shared_derivs(u, v, t, q, z):
        u_x_ = d_x(u)
        u_y_ = d_y(u)
        u_z_ = d_z(u)
        v_x_ = d_x(v)
        v_y_ = d_y(v)
        v_z_ = d_z(v)
        t_x_ = d_x(t)
        t_y_ = d_y(t)
        t_z_ = d_z(t)
        q_x_ = d_x(q)
        q_y_ = d_y(q)
        q_z_ = d_z(q)
        z_x_ = d_x(z)
        z_y_ = d_y(z)
        z_z_ = d_z(z)
        w_z_ = -(u_x_ + v_y_)
        w_ = integral_z(w_z_)  # hPa/s (legacy units)
        return {
            "u_x": u_x_,
            "u_y": u_y_,
            "u_z": u_z_,
            "v_x": v_x_,
            "v_y": v_y_,
            "v_z": v_z_,
            "t_x": t_x_,
            "t_y": t_y_,
            "t_z": t_z_,
            "q_x": q_x_,
            "q_y": q_y_,
            "q_z": q_z_,
            "z_x": z_x_,
            "z_y": z_y_,
            "z_z": z_z_,
            "w": w_,
        }

    # ---------------- RHS variants ----------------

    def rhs_fixA(u, v, t, q, z):
        """Replace broken Q with α·ω/c_p. R_d in hydrostatic for consistency."""
        D = shared_derivs(u, v, t, q, z)
        w = D["w"]
        # Convert legacy w (hPa/s) → ω (Pa/s) for correct adiabatic.
        omega_Pa = 100.0 * w
        t_t_adia = R_D * t * omega_Pa / (C_P * pressure_pa)
        t_t = t_t_adia - u * D["t_x"] - v * D["t_y"] - w * D["t_z"]
        # Hydrostatic with R_d. integral_z в hPa → делитель в hPa
        # (pressure_pa/100), иначе z_t ×100 меньше корректного.
        z_zt = -R_D / (pressure_pa / 100.0) * t_t
        z_t = integral_z(z_zt)
        u_t = -u * D["u_x"] - v * D["u_y"] - w * D["u_z"] + f_field * v - D["z_x"]
        v_t = -u * D["v_x"] - v * D["v_y"] - w * D["v_z"] - f_field * u - D["z_y"]
        q_t = -(u * D["q_x"] + v * D["q_y"] + w * D["q_z"])
        return u_t, v_t, t_t, q_t, z_t

    def rhs_fixB(u, v, t, q, z):
        """Keep broken Q formula; only fix units (pressure→Pa, R→R_d)."""
        D = shared_derivs(u, v, t, q, z)
        w = D["w"]
        Q = -L * D["z_z"] * w  # ← still dimensionally broken
        t_t = (Q - D["z_z"] * w) / C_P - u * D["t_x"] - v * D["t_y"] - w * D["t_z"]
        # R_d, делитель в hPa (pressure_pa/100) под hPa-интеграл integral_z.
        z_zt = -R_D / (pressure_pa / 100.0) * t_t
        z_t = integral_z(z_zt)
        u_t = -u * D["u_x"] - v * D["u_y"] - w * D["u_z"] + f_field * v - D["z_x"]
        v_t = -u * D["v_x"] - v * D["v_y"] - w * D["v_z"] - f_field * u - D["z_y"]
        q_t = -(u * D["q_x"] + v * D["q_y"] + w * D["q_z"])
        return u_t, v_t, t_t, q_t, z_t

    def rhs_fixC(u, v, t, q, z):
        """Original semantics (broken Q + universal R). scale_diff applied in step()."""
        D = shared_derivs(u, v, t, q, z)
        w = D["w"]
        Q = -L * D["z_z"] * w
        t_t = (Q - D["z_z"] * w) / C_P - u * D["t_x"] - v * D["t_y"] - w * D["t_z"]
        # Универсальная R (легаси-некорректность fixC), делитель в hPa под integral_z.
        z_zt = -R_UNIVERSAL / (pressure_pa / 100.0) * t_t
        z_t = integral_z(z_zt)
        u_t = -u * D["u_x"] - v * D["u_y"] - w * D["u_z"] + f_field * v - D["z_x"]
        v_t = -u * D["v_x"] - v * D["v_y"] - w * D["v_z"] - f_field * u - D["z_y"]
        q_t = -(u * D["q_x"] + v * D["q_y"] + w * D["q_z"])
        return u_t, v_t, t_t, q_t, z_t

    RHS_TABLE = {"A": rhs_fixA, "B": rhs_fixB, "C": rhs_fixC}
    rhs_fn = RHS_TABLE[fix]

    def rollout_step(state):
        u, v, t, q, z = state["u"], state["v"], state["t"], state["q"], state["z"]
        u_t, v_t, t_t, q_t, z_t = rhs_fn(u, v, t, q, z)
        dt = args.block_dt_seconds

        if fix == "C":
            # Production-semantic clipping: scale_diff(tendency·dt, state).detach().
            du = scale_diff(u_t * dt, u).detach()
            dv = scale_diff(v_t * dt, v).detach()
            dT = scale_diff(t_t * dt, t).detach()
            dq = scale_diff(q_t * dt, q).detach()
            dz = scale_diff(z_t * dt, z).detach()
            u_new = u + du
            v_new = v + dv
            t_new = t + dT
            q_new = q + dq
            z_new = z + dz
        else:
            u_new = u + dt * u_t
            v_new = v + dt * v_t
            t_new = t + dt * t_t
            q_new = q + dt * q_t
            z_new = z + dt * z_t

        new_state = {
            "t2m": state["t2m"],
            "u10": state["u10"],
            "v10": state["v10"],
            "tp": state["tp"],
            "r": state["r"],
            "u": u_new,
            "v": v_new,
            "t": t_new,
            "q": q_new,
            "z": z_new,
        }
        rhs_dict = {"u_t": u_t, "v_t": v_t, "t_t": t_t, "q_t": q_t, "z_t": z_t}
        return new_state, rhs_dict

    # method_name reflects integrator + applied fix
    method_name = f"fd4_euler_spherical_fix{fix}"
    method_name += {"A": "_alphaOmegaQ", "B": "_PaUnits_R_d", "C": "_scaleDiffDetach"}[fix]
    print(f"[init] method_name={method_name}")

    tags = [
        "stencil_fd4",
        "time_euler",
        "coriolis_spherical",
        f"fix_mode_{fix}",
        "diagnostic_followup",
    ]

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
        tags=tags,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
