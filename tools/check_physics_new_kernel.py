"""72-h CPU rollout проверка рефакторенной :class:`utils.physics.PurePDEKernel`.

Это параметризованный entry-point для НОВОЙ семьи интеграторов (через
:mod:`utils.physics`, не :mod:`utils.old_physics`). Все буферы регистрируются
через `register_buffer`, нет module-globals; stencil/coriolis/time-scheme/
boundary — параметры. Никаких обучаемых слоёв, никаких Conv2d/BatchNorm.

CLI задаёт конкретный интегратор:
    --stencil       fd4 | weno5
    --time-scheme   euler | rk4
    --coriolis      constant | beta_plane | spherical
    --boundary-x    periodic | reflect | replicate  (по lon; дефолт periodic)
    --boundary-y    periodic | reflect | replicate  (по lat; дефолт replicate)
    --boundary-z    reflect | replicate              (по давлению; дефолт replicate)
    --use-universal-R   (флаг: использовать R=8.314 Дж/(моль·К) вместо
                         дефолтного R_d=287 Дж/(кг·К); opt-in для регрессии)

Имя метода в Comet строится автоматически как
``<stencil>_<time-scheme>_<coriolis>[_Runiv]`` — соответствует «типу интегратора»,
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
    balance_initial_state,
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
    p.add_argument(
        "--lat-range-deg",
        nargs=2,
        type=float,
        default=[-90.0, 90.0],
        metavar=("LOW", "HIGH"),
        help="Диапазон широт. Дефолт global (-90 90). USA-кроп: 24 56.",
    )
    p.add_argument("--year", type=int, default=2005)
    p.add_argument("--horizon-hours", type=int, default=48)
    p.add_argument("--block-dt-seconds", type=float, default=300.0)
    p.add_argument("--stencil", choices=["fd4", "weno5"], required=True)
    p.add_argument(
        "--time-scheme", choices=["euler", "rk4", "ssp_rk3", "semi_implicit"], required=True
    )
    p.add_argument(
        "--coriolis", choices=["constant", "beta_plane", "spherical"], default="spherical"
    )
    p.add_argument(
        "--boundary-x",
        choices=["periodic", "reflect", "replicate"],
        default="periodic",
        help="Граничное условие по оси W (lon). Дефолт 'periodic' — долгота циклична.",
    )
    p.add_argument(
        "--boundary-y",
        choices=["periodic", "reflect", "replicate"],
        default="replicate",
        help="Граничное условие по оси H (lat). Дефолт 'replicate' — широта не циклична.",
    )
    p.add_argument(
        "--boundary-z",
        choices=["reflect", "replicate"],
        default="replicate",
        help="Граничное условие по оси P (pressure). 'periodic' запрещён (давление не цикличное).",
    )
    p.add_argument(
        "--use-universal-R",
        action="store_true",
        help="Использовать универсальную R=8.314 Дж/(моль·К) в гидростатике вместо "
        "дефолтного R_d=287 Дж/(кг·К). Opt-in для регрессии старой физики.",
    )
    p.add_argument(
        "--abl-label",
        default="",
        help="Метка абляции (E0..E5). Добавляется суффиксом в method_name и тегом "
        "abl_<label>, чтобы каждый шаг абляции был отдельным Comet-экспериментом.",
    )
    p.add_argument(
        "--hyperdiffusion",
        action="store_true",
        help="E2: включить ∇⁴-гипердиффузию (scale-selective анти-алиасинг, "
        "принципиальная замена scale_diff).",
    )
    p.add_argument(
        "--hyperdiff-tau-hours",
        type=float,
        default=6.0,
        help="E2: e-folding моды 2Δx, часы (default 6). Меньше = сильнее демпфирование.",
    )
    p.add_argument(
        "--polar-filter",
        action="store_true",
        help="E3: зональный Фурье-фильтр у полюса (снимает полюсную CFL lat-lon).",
    )
    p.add_argument(
        "--polar-filter-lat-deg",
        type=float,
        default=60.0,
        help="E3: широта (°), полярнее которой обрезаются высокие зональные m.",
    )
    p.add_argument(
        "--balance-ic",
        choices=["none", "dfi", "geostrophic"],
        default="none",
        help="E4: балансировка IC до rollout (убрать initialization-shock). "
        "dfi=digital filter init (Lynch–Huang), geostrophic=ветер из массы.",
    )
    p.add_argument(
        "--balance-span-hours",
        type=float,
        default=1.0,
        help="E4: размах DFI-окна (часы); n=round(span·3600/dt) шагов forward. "
        "Default 1ч (=12 шагов при dt=300) — не 6ч (=72 шага, B3-фикс: длинное "
        "окно расходилось → silent fallback на сырой IC).",
    )
    p.add_argument(
        "--advection-form",
        choices=["advective", "flux"],
        default="advective",
        help="E5: форма горизонтальной адвекции. flux=консервативная "
        "(∂ₓ(uX)+∂_y(vX)−X·div) — сохраняет моменты на периодич. сетке.",
    )
    p.add_argument(
        "--w-diagnostic",
        choices=["plain", "mass_consistent"],
        default="plain",
        help="E5: диагностический w. mass_consistent вычитает p-взвеш. "
        "колоночное среднее дивергенции (∫div dp≈0).",
    )
    p.add_argument("--offline", action="store_true")
    p.add_argument("--project-name", default="WeatherPredictions")
    args = p.parse_args()

    device = torch.device("cpu")
    print(f"[init] device={device}, threads={torch.get_num_threads()}")

    # Method name == integrator type signature (matches user request).
    method_parts = [args.stencil, args.time_scheme, args.coriolis]
    if args.use_universal_R:
        method_parts.append("Runiv")
    if args.abl_label:
        method_parts.append(args.abl_label)
    method_name = "_".join(method_parts)
    # Суффикс активных стабилизаторов абляции (накопительно по E2..E5).
    stab: list[str] = []
    if args.hyperdiffusion:
        stab.append("hyperdiff")
    if args.polar_filter:
        stab.append("polar")
    if args.balance_ic != "none":
        stab.append("dfi" if args.balance_ic == "dfi" else "geo")
    if args.advection_form == "flux":
        stab.append("flux")
    if args.w_diagnostic == "mass_consistent":
        stab.append("masscons")
    if stab:
        method_name = method_name + "+" + "+".join(stab)
    print(f"[init] method_name={method_name}")

    # Build grid + kernel via utils.physics (NEW family — register_buffer-based).
    grid = Grid(GridConfig(H=args.H, W=args.W, lat_range_deg=tuple(args.lat_range_deg))).to(device)
    kernel = PurePDEKernel(
        grid,
        stencil=args.stencil,
        coriolis=args.coriolis,
        block_dt=args.block_dt_seconds,
        time_scheme=args.time_scheme,
        boundary_x=args.boundary_x,
        boundary_y=args.boundary_y,
        boundary_z=args.boundary_z,
        use_universal_R=args.use_universal_R,
        hyperdiffusion=args.hyperdiffusion,
        hyperdiff_tau_hours=args.hyperdiff_tau_hours,
        polar_filter=args.polar_filter,
        polar_filter_lat_deg=args.polar_filter_lat_deg,
        advection_form=args.advection_form,
        w_diagnostic=args.w_diagnostic,
    ).to(device)

    # f_field для метрик physics consistency.
    f_field = kernel.f_field

    # d_x/d_y для метрик. PurePDEKernel экспортирует через .diff.
    d_x_fn = kernel.diff.d_x
    d_y_fn = kernel.diff.d_y

    # GeometryCPU нужен для lat_weights/pixel_x/pixel_y в forecast метриках.
    # Reuse параметров новой Grid через CPU shadow (значения совпадают).
    geom = GeometryCPU(H=args.H, W=args.W, lat_range_deg=tuple(args.lat_range_deg))

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
        f"bdry_x_{args.boundary_x}",
        f"bdry_y_{args.boundary_y}",
        f"bdry_z_{args.boundary_z}",
        "new_kernel",  # семья utils.physics
    ]
    if args.use_universal_R:
        tags.append("R_universal_hydrostatic")
    if args.abl_label:
        tags.append(f"abl_{args.abl_label}")
    if args.hyperdiffusion:
        tags.append("hyperdiff")
        tags.append(f"hyperdiff_tau_{args.hyperdiff_tau_hours:g}h")
    if args.polar_filter:
        tags.append("polar_filter")
        tags.append(f"polar_lat_{args.polar_filter_lat_deg:g}")
    if args.balance_ic != "none":
        tags.append(f"balance_ic_{args.balance_ic}")
        if args.balance_ic == "dfi":
            tags.append(f"dfi_span_{args.balance_span_hours:g}h")
    if args.advection_form == "flux":
        tags.append("advection_flux")
    if args.w_diagnostic == "mass_consistent":
        tags.append("w_mass_consistent")

    def _balance_hook(s: dict) -> dict:
        return balance_initial_state(
            s,
            kernel,
            args.balance_ic,
            span_hours=args.balance_span_hours,
        )

    prepare_hook = _balance_hook if args.balance_ic != "none" else None

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
        prepare_hook=prepare_hook,
    )


if __name__ == "__main__":
    main()
