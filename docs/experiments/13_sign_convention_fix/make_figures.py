"""Рендер фигур эксперимента 13 из results-JSON и NPZ карт.

Вход: physics_stats_2000_results.json (рядом) и NPZ с полями карт (аргумент 1
или переменная окружения MAPS_NPZ). Выход: пять PNG в каталоге эксперимента.
Каждая фигура самодостаточна: заголовок на простом русском плюс врезка-пояснение
внизу («Что показано / Как читать / Цвета»), чтобы график читался без текста
отчёта. Смысл величин и выводы разобраны в PHYSICS_STATS_2000.md.

Оформление следует навыку dataviz: расходящаяся палитра (RdBu, нейтральный
центр) — только для знаковых полей (вертикальная скорость, невязка); одноцветные
шкалы — для величин одного знака; стадии исправления — порядковая одноцветная
шкала; цвета линий — Okabe-Ito (проверено валидатором); логарифмическая ось —
точками, а не столбцами; на каждой карте — контур береговой линии.
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

STAGE_COLORS = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]  # порядковая одноцветная
STAGE_LABELS = [
    "старое ядро (знаки перевёрнуты)",
    "исправлены знаки",
    "без адиабатического члена",
    "исправлено + массо-согл. ω",
]
VAR_COLORS = {"u": "#0072B2", "v": "#D55E00", "t": "#009E73", "z": "#CC79A7"}
INK, MUTED = "#1a1a1a", "#8a8a8a"
CAPTION_BG = "#f4f4f2"

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
    """Убирает верхнюю и правую рамки — оси не должны спорить с данными."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_caption(fig, text: str, y: float = -0.02) -> None:
    """Печатает врезку-пояснение под фигурой (что показано, как читать, цвета).

    Текст кладётся ниже осей и попадает в сохранённый PNG за счёт
    ``bbox_inches='tight'``. Выравнивание по левому краю для читаемости.
    Параметр ``y`` (в координатах фигуры) опускает врезку ниже, когда под
    осями уже есть легенда.
    """
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="top",
        ma="left",
        fontsize=8,
        color=INK,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": CAPTION_BG, "edgecolor": "#d5d5d2"},
    )


def land_contour(ax, dom: str) -> None:
    """Контур береговой линии (маска суша/море = 0.5) на карте."""
    ax.contour(
        MAPS[f"{dom}_lon"],
        MAPS[f"{dom}_lat"],
        MAPS[f"{dom}_lsm"],
        levels=[0.5],
        colors="#1a1a1a",
        linewidths=0.5,
    )


def map_panel(ax, dom: str, field: np.ndarray, cmap: str, norm, title: str):
    """Одна карта: заливка поля, береговая линия, заголовок. Возвращает mesh."""
    mesh = ax.pcolormesh(
        MAPS[f"{dom}_lon"], MAPS[f"{dom}_lat"], field, cmap=cmap, norm=norm, shading="auto"
    )
    land_contour(ax, dom)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("долгота, °в.д.")
    ax.set_ylabel("широта, °")
    ax.grid(False)
    return mesh


def fig1_residual_stages() -> None:
    """Точечный график стадий исправления (лог-ось): T и z, обе области."""
    rows = [
        ("T · США", "usa", "t"),
        ("T · весь мир", "globe", "t"),
        ("z · США", "usa", "z"),
        ("z · весь мир", "globe", "z"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for yi, (_label, dom, var) in enumerate(rows):
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
    ax.set_xlabel("во сколько раз ошибка физики больше реального изменения за час (лог-ось)")
    ax.set_title(
        "Ошибка температуры и геопотенциала падает после исправления знаков", fontsize=10
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=s)
        for c, s in zip(STAGE_COLORS, STAGE_LABELS)
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7.5,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
    )
    despine(ax)
    add_caption(
        fig,
        "Что показано: относительная ошибка физики для температуры (T) и геопотенциала (z) —\n"
        "во сколько раз она больше реального часового изменения поля. Одно деление оси = ×10.\n"
        "Как читать: чем левее точка, тем точнее. Цвета точек — стадии, от светлой к тёмной:\n"
        "старое ядро → исправлены знаки → без адиабаты → исправлено с массо-согласованной ω.\n"
        "Вывод: тёмные точки левее светлых в 10–30 раз.",
        y=-0.52,
    )
    fig.savefig(HERE / "fig1_residual_stages.png", bbox_inches="tight")
    plt.close(fig)


def fig2_omega_maps() -> None:
    """Весь мир, 500 гПа, 15.07.2000: три способа получить вертикальную скорость."""
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.6, vmax=0.6)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), sharey=True)
    panels = [
        ("omega_plain500", "обычная (из расхождения ветра)"),
        ("omega_mc500", "массо-согласованная"),
        ("omega_implied500", "требуемая (из баланса тепла)"),
    ]
    mesh = None
    for ax, (key, title) in zip(axes, panels):
        field = MAPS[f"globe_jul15_{key}"]
        mesh = map_panel(
            ax, "globe", field, "RdBu_r", norm, f"{title}\nсредний модуль = {np.abs(field).mean():.2f} Па/с"
        )
        ax.label_outer()
    cbar = fig.colorbar(
        mesh, ax=axes, orientation="horizontal", fraction=0.05, pad=0.18, extend="both"
    )
    cbar.set_label("вертикальная скорость ω на 500 гПа, Па/с")
    fig.suptitle(
        "Вертикальную скорость нельзя брать напрямую из ветра: она шумит в разы сильнее нужной",
        fontsize=10,
        y=1.02,
    )
    add_caption(
        fig,
        "Что показано: вертикальная скорость воздуха ω на высоте 500 гПа, весь мир, 15 июля 2000.\n"
        "Три способа её получить: из расхождения ветра (обычная), с вычтенным средним по столбу\n"
        "(массо-согласованная) и та, что нужна для точного объяснения нагрева (требуемая).\n"
        "Цвета: синее — подъём воздуха, красное — опускание, белое — нет вертикального движения.\n"
        "Чёрные линии — берега. Как читать: левая карта яркая и пёстрая — это шум; правая\n"
        "(требуемая) выглядит совсем иначе, узоры не совпадают.",
    )
    fig.savefig(HERE / "fig2_omega_maps_globe.png", bbox_inches="tight")
    plt.close(fig)


def fig3_usa_maps() -> None:
    """Окно над США: погода-фон, ошибка T, конденсация, вертикальная скорость."""
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.6))
    (ax_bg, ax_res), (ax_cond, ax_omega) = axes

    t850 = MAPS["usa_jul15_t850"]
    mesh = ax_bg.pcolormesh(MAPS["usa_lon"], MAPS["usa_lat"], t850, cmap="Oranges", shading="auto")
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
    ax_bg.set_title("Погода для контекста: температура 850 гПа + высота 500 гПа", fontsize=9)
    ax_bg.grid(False)
    fig.colorbar(mesh, ax=ax_bg, fraction=0.04, pad=0.02).set_label("T на 850 гПа, K (темнее = теплее)")

    res = MAPS["usa_jul15_t_res_mc500"] * 3600
    vmax = float(np.quantile(np.abs(res), 0.98))
    mesh = map_panel(
        ax_res,
        "usa",
        res,
        "RdBu_r",
        TwoSlopeNorm(0.0, -vmax, vmax),
        "Ошибка температуры исправленного ядра, 500 гПа",
    )
    fig.colorbar(mesh, ax=ax_res, fraction=0.04, pad=0.02, extend="both").set_label(
        "K/ч (красное — недооценка нагрева)"
    )

    # Январский снимок: конденсация активнее (55 точек против 16 в июле).
    # Верхний предел шкалы по 90-му перцентилю активных точек: один экстремум
    # не должен съедать шкалу.
    drying = np.clip(-MAPS["usa_jan15_cond_source925"] * 3.6e6, 0, None)  # г/кг/ч
    active = drying[drying > 1e-4]
    vmax = float(np.quantile(active, 0.90)) if active.size else 1.0
    mesh = ax_cond.pcolormesh(
        MAPS["usa_lon"], MAPS["usa_lat"], drying, cmap="Blues", vmin=0.0, vmax=vmax, shading="auto"
    )
    land_contour(ax_cond, "usa")
    ax_cond.set_title(
        "Где идёт конденсация, 925 гПа, 15.01.2000\n(до исправления это поле было всюду нулём)",
        fontsize=9,
    )
    ax_cond.grid(False)
    fig.colorbar(mesh, ax=ax_cond, fraction=0.04, pad=0.02, extend="max").set_label(
        "осушение воздуха, г/кг·ч (синее — идёт)"
    )

    mesh = map_panel(
        ax_omega,
        "usa",
        MAPS["usa_jul15_omega_mc500"],
        "RdBu_r",
        TwoSlopeNorm(0.0, -0.6, 0.6),
        "Вертикальная скорость (массо-согл.), 500 гПа",
    )
    fig.colorbar(mesh, ax=ax_omega, fraction=0.04, pad=0.02, extend="both").set_label(
        "ω, Па/с (синее — подъём)"
    )

    for ax in axes.flat:
        ax.set_xlabel("долгота, °в.д.")
        ax.set_ylabel("широта, °с.ш.")
    fig.suptitle("Окно над США, 15 июля 2000: физика после исправления знаков", fontsize=11)
    fig.tight_layout()
    add_caption(
        fig,
        "Четыре карты окна над США; чёрная линия везде — берега.\n"
        "• Слева вверху — погода для контекста: цвет = температура на 850 гПа, серые линии = высота 500 гПа.\n"
        "• Справа вверху — ошибка температуры: красное — физика недооценила нагрев, синее — переоценила, белое — точно.\n"
        "• Слева внизу — где идёт конденсация: синее — идёт, белое — нет (до исправления везде был ноль).\n"
        "• Справа внизу — вертикальная скорость: синее — подъём, красное — опускание.",
    )
    fig.savefig(HERE / "fig3_usa_maps.png", bbox_inches="tight")
    plt.close(fig)


def fig4_lat_profiles() -> None:
    """Невязка по широте: ветер v больше u и растёт к полюсам."""
    labels = {
        "u": "u — зональный ветер",
        "v": "v — меридиональный ветер",
        "t": "T — температура",
        "z": "z — геопотенциал",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), sharey=True)
    for ax, dom, title in zip(axes, ("globe", "usa"), ("Весь мир", "Окно над США")):
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
        ax.set_xlabel("широта, ° (0 — экватор, ±90 — полюса)")
        despine(ax)
    axes[0].set_ylabel("ошибка / реальное изменение (лог-ось)")
    handles = [plt.Line2D([], [], color=VAR_COLORS[v], lw=1.8, label=labels[v]) for v in labels]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="upper center", ncols=2)
    fig.suptitle("Ошибка ветра максимальна у полюсов, и v всюду хуже u", fontsize=10)
    fig.tight_layout()
    add_caption(
        fig,
        "Что показано: относительная ошибка физики в зависимости от широты. Слева весь мир,\n"
        "справа окно над США. Ось высоты логарифмическая. Цвет линии — переменная (подписана справа):\n"
        "синяя u — зональный ветер, оранжевая v — меридиональный, зелёная T — температура, розовая z — геопотенциал.\n"
        "Как читать: оранжевая линия v выше синей u, и обе растут к краям (полюсам) — ошибка ветра там максимальна.",
    )
    fig.savefig(HERE / "fig4_lat_profiles.png", bbox_inches="tight")
    plt.close(fig)


def fig5_cfl_map() -> None:
    """Число Куранта на 250 гПа: предел устойчивости достигается лишь у полюсов."""
    jan = MAPS["globe_jan15_cfl250"]
    jul = MAPS["globe_jul15_cfl250"]
    field = jan if jan.max() >= jul.max() else jul
    label = "15.01.2000" if jan.max() >= jul.max() else "15.07.2000"
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
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
            MAPS["globe_lon"], MAPS["globe_lat"], field, levels=[1.0], colors="#C4260E", linewidths=1.4
        )
    band_max = RESULTS["globe"]["cfl_dt300"]["band_max"]
    ax.set_title(
        "Предел устойчивости счёта достигается только у полюсов", fontsize=10
    )
    ax.set_xlabel("долгота, °в.д.")
    ax.set_ylabel("широта, °")
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02).set_label("число Куранта (лог-шкала)")
    add_caption(
        fig,
        "Что показано: число Куранта на 250 гПа — путь ветра за шаг счёта (300 с), делённый на размер\n"
        "ячейки сетки; при значении больше 1 явная схема разваливается. Снимок за " + label + ".\n"
        "Цвета: светлое — малые значения (запас большой), тёмно-синее — близко к пределу; шкала логарифмическая.\n"
        "Красная линия — уровень 1, чёрные линии — берега.\n"
        "Как читать: красная зона есть только у самой кромки (полюса). Максимум по 96 снимкам: "
        f"полюса {band_max['polar']:.1f}, средние широты {band_max['midlat']:.2f},\n"
        f"тропики {band_max['tropics']:.2f}; над США — {RESULTS['usa']['cfl_dt300']['adv_max']:.3f}. В остальном мире и над США счёт устойчив.",
    )
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
