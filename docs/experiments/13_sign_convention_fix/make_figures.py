"""Рендер фигур exp 13 из results-JSON и NPZ карт (см. PHYSICS_STATS_2000.md).

Вход: physics_stats_2000_results.json (рядом) и NPZ с картами (аргумент 1 или
$MAPS_NPZ). Выход: fig1..fig5 PNG в каталоге эксперимента.

Правила оформления — skill dataviz: diverging (RdBu, нейтральный центр) только
для знаковых полей (ω, невязка), sequential одно-хьюные для величин, стадии
фикса — порядковая одно-хьюная шкала, идентичности линий — Okabe-Ito
(валидировано: CVD ΔE 17.9, у розового contrast-WARN → прямые подписи),
log-шкала — точками, а не барами; суша — контур lsm на каждой карте.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
RESULTS = json.loads((HERE / "physics_stats_2000_results.json").read_text())
NPZ_PATH = sys.argv[1] if len(sys.argv) > 1 else os.environ["MAPS_NPZ"]
MAPS = np.load(NPZ_PATH)

STAGE_COLORS = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]  # порядковая одно-хьюная
STAGE_LABELS = ["до фикса", "фикс, plain ω", "только адвекция", "фикс + mc ω"]
VAR_COLORS = {"u": "#0072B2", "v": "#D55E00", "t": "#009E73", "z": "#CC79A7"}
INK, MUTED = "#1a1a1a", "#8a8a8a"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": "#e3e3e0",
        "grid.linewidth": 0.6,
    }
)


def despine(ax) -> None:
    """Убирает верхнюю/правую рамки (recessive axes)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def land_contour(ax, dom: str) -> None:
    """Контур суши (lsm=0.5) — «изображение земли» на каждой карте."""
    ax.contour(
        MAPS[f"{dom}_lon"],
        MAPS[f"{dom}_lat"],
        MAPS[f"{dom}_lsm"],
        levels=[0.5],
        colors="#1a1a1a",
        linewidths=0.5,
    )


def map_panel(ax, dom: str, field: np.ndarray, cmap: str, norm, title: str):
    """Одна карта: pcolormesh + суша + заголовок; возвращает mesh для colorbar."""
    mesh = ax.pcolormesh(
        MAPS[f"{dom}_lon"], MAPS[f"{dom}_lat"], field, cmap=cmap, norm=norm, shading="auto"
    )
    land_contour(ax, dom)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("долгота, °E")
    ax.set_ylabel("широта, °")
    ax.grid(False)
    return mesh


def fig1_residual_stages() -> None:
    """Точечный график стадий фикса (log-x): T и z, оба домена."""
    rows = [
        ("T · USA", "usa", "t"),
        ("T · глобус", "globe", "t"),
        ("z · USA", "usa", "z"),
        ("z · глобус", "globe", "z"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for yi, (label, dom, var) in enumerate(rows):
        d = RESULTS[dom]
        stages = [
            d["residual_rel"]["prefix_emulated"][var],
            d["residual_rel"]["fixed_plain"][var],
            d["residual_rel_adv_only"][var],
            d["residual_rel"]["fixed_mc"][var],
        ]
        ax.plot(stages, [yi] * 4, color="#cfcfcc", lw=1.2, zorder=1)
        for stage_value, color in zip(stages, STAGE_COLORS):
            ax.scatter(stage_value, yi, s=52, color=color, zorder=3, edgecolor="white", lw=0.6)
        for stage_value in stages:
            ax.annotate(
                f"{stage_value:.1f}",
                (stage_value, yi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7.5,
                color=INK,
            )
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("относительная невязка Σ|obs−model| / Σ|obs| (log)")
    ax.set_title("PDE-невязка T и z по стадиям фикса (ERA5-2000, 96 пар)", fontsize=10)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=s)
        for c, s in zip(STAGE_COLORS, STAGE_LABELS)
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
    )
    despine(ax)
    fig.tight_layout()
    fig.savefig(HERE / "fig1_residual_stages.png", bbox_inches="tight")
    plt.close(fig)


def fig2_omega_maps() -> None:
    """Глобус, 500 гПа, 2000-07-15: ω plain / mass-consistent / implied."""
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.6, vmax=0.6)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4), sharey=True)
    panels = [
        ("omega_plain500", "кинематическая ω (plain)"),
        ("omega_mc500", "кинематическая ω (mass-consistent)"),
        ("omega_implied500", "implied ω из T-бюджета"),
    ]
    mesh = None
    for ax, (key, title) in zip(axes, panels):
        field = MAPS[f"globe_jul15_{key}"]
        mesh = map_panel(
            ax, "globe", field, "RdBu_r", norm, f"{title}\n⟨|ω|⟩ = {np.abs(field).mean():.2f} Па/с"
        )
        ax.label_outer()
    cbar = fig.colorbar(
        mesh, ax=axes, orientation="horizontal", fraction=0.055, pad=0.16, extend="both"
    )
    cbar.set_label("ω на 500 гПа, Па/с (красное — опускание); 2000-07-15 12z")
    fig.suptitle(
        "Кинематическая ω шумит на порядок сильнее требуемой; mc давит амплитуду, "
        "но поточечной корреляции нет (r ≈ 0.02)",
        fontsize=10,
        y=1.04,
    )
    fig.savefig(HERE / "fig2_omega_maps_globe.png", bbox_inches="tight")
    plt.close(fig)


def fig3_usa_maps() -> None:
    """USA-кроп, 2000-07-15: фон T850+z500, T-невязка, конденсация, ω."""
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2))
    (ax_bg, ax_res), (ax_cond, ax_omega) = axes

    t850 = MAPS["usa_jul15_t850"]
    mesh = ax_bg.pcolormesh(
        MAPS["usa_lon"], MAPS["usa_lat"], t850, cmap="Oranges", shading="auto"
    )
    cs = ax_bg.contour(
        MAPS["usa_lon"],
        MAPS["usa_lat"],
        MAPS["usa_jul15_z500"] / 9.80665,
        levels=8,
        colors="#4d4d4d",
        linewidths=0.7,
    )
    ax_bg.clabel(cs, fontsize=6, fmt="%.0f")
    land_contour(ax_bg, "usa")
    ax_bg.set_title("Состояние: T на 850 гПа + изогипсы z500 (гпм)", fontsize=9)
    ax_bg.grid(False)
    fig.colorbar(mesh, ax=ax_bg, fraction=0.04, pad=0.02).set_label("T₈₅₀, K")

    res = MAPS["usa_jul15_t_res_mc500"] * 3600
    vmax = float(np.quantile(np.abs(res), 0.98))
    mesh = map_panel(
        ax_res,
        "usa",
        res,
        "RdBu_r",
        TwoSlopeNorm(0.0, -vmax, vmax),
        "T-невязка (фикс + mc ω), 500 гПа",
    )
    fig.colorbar(mesh, ax=ax_res, fraction=0.04, pad=0.02, extend="both").set_label("K/ч")

    # Январский снимок: конденсация активнее (55 точек против 16 в июле).
    # vmax по 90-му перцентилю активных точек: один экстремум не съедает шкалу.
    drying = np.clip(-MAPS["usa_jan15_cond_source925"] * 3.6e6, 0, None)  # г/кг/ч
    active = drying[drying > 1e-4]
    vmax = float(np.quantile(active, 0.90)) if active.size else 1.0
    mesh = ax_cond.pcolormesh(
        MAPS["usa_lon"], MAPS["usa_lat"], drying, cmap="Blues", vmin=0.0, vmax=vmax,
        shading="auto",
    )
    land_contour(ax_cond, "usa")
    ax_cond.set_title(
        "Конденсационное осушение, 925 гПа, 2000-01-15\n(после B3; до фикса ≡ 0)",
        fontsize=9,
    )
    ax_cond.grid(False)
    fig.colorbar(mesh, ax=ax_cond, fraction=0.04, pad=0.02, extend="max").set_label(
        "−dq/dt, г/кг·ч"
    )

    mesh = map_panel(
        ax_omega,
        "usa",
        MAPS["usa_jul15_omega_mc500"],
        "RdBu_r",
        TwoSlopeNorm(0.0, -0.6, 0.6),
        "ω mass-consistent, 500 гПа",
    )
    fig.colorbar(mesh, ax=ax_omega, fraction=0.04, pad=0.02, extend="both").set_label("Па/с")

    for ax in axes.flat:
        ax.set_xlabel("долгота, °E")
        ax.set_ylabel("широта, °N")
    fig.suptitle("USA-кроп 32×64, 2000-07-15 12z", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "fig3_usa_maps.png", bbox_inches="tight")
    plt.close(fig)


def fig4_lat_profiles() -> None:
    """Широтные профили невязки: v растёт к полюсам (u,v — plain; t,z — mc)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for ax, dom, title in zip(axes, ("globe", "usa"), ("Глобус", "USA-кроп")):
        prof = RESULTS[dom]["residual_rel_by_lat"]
        lat = prof["lat_deg"]
        for var in ("u", "v", "t", "z"):
            ax.plot(lat, prof[var], color=VAR_COLORS[var], lw=1.8)
            ax.annotate(
                var,
                (lat[-1], prof[var][-1]),
                textcoords="offset points",
                xytext=(5, 0),
                color=VAR_COLORS[var],
                fontsize=9,
                fontweight="bold",
                va="center",
            )
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("широта, °")
        despine(ax)
    axes[0].set_ylabel("относительная невязка (log)")
    handles = [
        plt.Line2D([], [], color=VAR_COLORS[v], lw=1.8, label={"u": "u (plain ω)", "v": "v (plain ω)", "t": "T (mc ω)", "z": "z (mc ω)"}[v])
        for v in ("u", "v", "t", "z")
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="upper center", ncols=2)
    fig.suptitle("Невязка по широте: v ≫ u растёт к полюсам (баланс f·u ↔ z_y)", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "fig4_lat_profiles.png", bbox_inches="tight")
    plt.close(fig)


def fig5_cfl_map() -> None:
    """CFL на 250 гПа при dt=300 c: полярные ряды глобуса против USA-кропа."""
    jan = MAPS["globe_jan15_cfl250"]
    jul = MAPS["globe_jul15_cfl250"]
    field = jan if jan.max() >= jul.max() else jul
    label = "2000-01-15" if jan.max() >= jul.max() else "2000-07-15"
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    mesh = ax.pcolormesh(
        MAPS["globe_lon"],
        MAPS["globe_lat"],
        np.clip(field, 1e-3, None),
        cmap="Blues",
        norm=LogNorm(vmin=1e-3, vmax=max(field.max(), 1.0)),
        shading="auto",
    )
    land_contour(ax, "globe")
    if field.max() >= 1.0:
        ax.contour(
            MAPS["globe_lon"], MAPS["globe_lat"], field, levels=[1.0],
            colors="#C4260E", linewidths=1.4,
        )
    band_max = RESULTS["globe"]["cfl_dt300"]["band_max"]
    ax.set_title(
        f"CFL = |u|·Δt/Δx на 250 гПа, dt=300 c ({label}; snapshot max = {field.max():.2f}).\n"
        f"Max по 96 парам: полюса {band_max['polar']:.2f}, средние {band_max['midlat']:.2f}, "
        f"тропики {band_max['tropics']:.2f}; USA-кроп {RESULTS['usa']['cfl_dt300']['adv_max']:.3f}",
        fontsize=9,
    )
    ax.set_xlabel("долгота, °E")
    ax.set_ylabel("широта, °")
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02).set_label("CFL (log)")
    fig.tight_layout()
    fig.savefig(HERE / "fig5_cfl_globe.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Рендерит все фигуры в каталог эксперимента."""
    fig1_residual_stages()
    fig2_omega_maps()
    fig3_usa_maps()
    fig4_lat_profiles()
    fig5_cfl_map()
    print("figures written to", HERE)


if __name__ == "__main__":
    main()
