"""Рендер фигур эксперимента 15 из results-JSON, NPZ карт и климатологий.

Вход: results/eq15_{domain}_{year}.json (6 файлов, раунд 2 с клим-вариантами),
results/eq15_maps_{domain}_{year}.npz, results/eq15_clim_{domain}_2000.npz.
Выход: PNG в каталоге эксперимента. Каждая фигура самодостаточна: заголовок
плюс врезка-пояснение внизу («Что показано / Как читать»).

Оформление: расходящаяся палитра только для знаковых полей; линии Okabe-Ito;
логарифмические оси для отношений; контур береговой линии на картах.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
YEARS = (2000, 2001, 2002)
DOMS = ("usa", "globe")
DOM_TITLES = {"usa": "окно над США", "globe": "весь мир"}
ALL_VARS = ("u", "v", "t", "q", "z")
VAR_TITLES = {
    "u": "u — зональный ветер",
    "v": "v — меридиональный ветер",
    "t": "T — температура",
    "q": "q — влажность",
    "z": "z — геопотенциал",
}
BEFORE = "base13"
BASE = "C_best"
AFTER = "C15_full"  # лучший принятый вариант (см. отчёт)
COLORS = {
    "base13": "#8a8a8a",
    "C_best": "#0072B2",
    "C15_phys": "#CC79A7",
    "S12": "#009E73",
    "C15_full": "#D55E00",
}
INK, MUTED = "#1a1a1a", "#8a8a8a"
CAPTION_BG = "#f4f4f2"

RESULTS = {
    (dom, year): json.loads((HERE / "results" / f"eq15_{dom}_{year}.json").read_text())
    for dom in DOMS
    for year in YEARS
}
MAPS = {dom: np.load(HERE / "results" / f"eq15_maps_{dom}_2002.npz") for dom in DOMS}
CLIM = {dom: np.load(HERE / "results" / f"eq15_clim_{dom}_2000.npz") for dom in DOMS}
P_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

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
    """Врезка-пояснение под фигурой (попадает в PNG через bbox_inches='tight')."""
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
        MAPS[dom]["lon"],
        MAPS[dom]["lat"],
        MAPS[dom]["lsm"],
        levels=[0.5],
        colors="#1a1a1a",
        linewidths=0.5,
    )


def fig1_residual_maps() -> None:
    """Для каждого из 5 уравнений: карты log10-невязки «до» и «после», оба домена."""
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.3)
    for var in ALL_VARS:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4), height_ratios=[1, 1.15])
        for col, variant, vtitle in (
            (0, BEFORE, f"до ({BEFORE}, exp 13)"),
            (1, AFTER, f"после ({AFTER})"),
        ):
            for row, dom in ((0, "usa"), (1, "globe")):
                ax = axes[row, col]
                ratio = MAPS[dom][f"resmap_{variant}_{var}"]
                mesh = ax.pcolormesh(
                    MAPS[dom]["lon"],
                    MAPS[dom]["lat"],
                    np.log10(np.clip(ratio, 1e-2, None)),
                    cmap="RdBu_r",
                    norm=norm,
                    shading="auto",
                )
                land_contour(ax, dom)
                med = float(np.median(ratio))
                ax.set_title(f"{DOM_TITLES[dom]} · {vtitle} · медиана {med:.2f}", fontsize=9)
                ax.grid(False)
                if row == 1:
                    ax.set_xlabel("долгота, °в.д.")
                if col == 0:
                    ax.set_ylabel("широта, °")
        cbar = fig.colorbar(
            mesh, ax=axes, orientation="vertical", fraction=0.03, pad=0.02, extend="both"
        )
        cbar.set_label("log₁₀(невязка/наблюдённая тенденция)")
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(["0.1", "0.3", "1", "3", "10"])
        fig.suptitle(
            f"{VAR_TITLES[var]}: накопленная относительная невязка, ERA5-2002 "
            "(вневыборочный год, все часы)",
            fontsize=11,
            y=0.98,
        )
        add_caption(
            fig,
            "Что показано: в каждой ячейке — сумма |ошибка тенденции| за все часы 2002 года\n"
            "(вне выборки: климатологии построены на 2000), делённая на сумму |наблюдённая\n"
            f"тенденция|. Слева {BEFORE} (конфигурация эксперимента 13), справа {AFTER}\n"
            "(геометрия exp 14 + диабатика, скрытое тепло, баротропный якорь z и климатологии\n"
            "Q₁/Q₂ эксперимента 15). Красное — уравнение хуже нулевого прогноза (отношение > 1),\n"
            "синее — лучше, белое — 1. Чёрные линии — берега. В заголовках — медиана по ячейкам.",
        )
        fig.savefig(HERE / f"fig1_maps_{var}.png", bbox_inches="tight")
        plt.close(fig)


def fig2_lat_profiles() -> None:
    """Широтные профили невязки всех 5 уравнений: до/база/после (2000)."""
    fig, axes = plt.subplots(5, 2, figsize=(10.4, 13.0), sharex="col")
    for col, dom in enumerate(DOMS):
        prof = RESULTS[(dom, 2000)]["lat_profile"]
        lat = prof["lat_deg"]
        for row, var in enumerate(ALL_VARS):
            ax = axes[row, col]
            for name in (BEFORE, BASE, "C15_phys"):
                ax.plot(lat, prof[name][var], color=COLORS[name], lw=1.6, label=name)
            ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
            ax.set_yscale("log")
            ax.set_ylabel(f"{var}")
            despine(ax)
            if row == 0:
                ax.set_title(DOM_TITLES[dom], fontsize=10)
            if row == 4:
                ax.set_xlabel("широта, ° (отрицательная — южное полушарие)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper center")
    fig.suptitle(
        "Невязка всех пяти уравнений по широте (ERA5-2000, все часы года)", fontsize=11, y=0.995
    )
    fig.tight_layout()
    add_caption(
        fig,
        "Что показано: относительная невязка каждого уравнения по строкам широты; слева США,\n"
        "справа глобус. Ось Y логарифмическая; пунктир на 1 — граница «хуже нулевого прогноза».\n"
        f"Серый — {BEFORE} (exp 13), синий — {BASE} (геометрия exp 14), розовый — C15_phys\n"
        "(+ диабатика Хелда–Суареса, скрытое тепло конденсации, баротропный якорь z).",
        y=-0.02,
    )
    fig.savefig(HERE / "fig2_lat_profiles.png", bbox_inches="tight")
    plt.close(fig)


def fig3_term_decomposition() -> None:
    """Кумулятивная сборка каждого уравнения по членам: до и после."""
    for var in ALL_VARS:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharey=True)
        for col, dom in enumerate(DOMS):
            td = RESULTS[(dom, 2000)]["term_decomposition"]
            ax = axes[col]
            names = [BEFORE, BASE, "C15_phys"]
            n_groups = max(len(td[n][var]["cumulative"]) for n in names)
            width = 0.8 / len(names)
            for k, name in enumerate(names):
                stages = td[name][var]["cumulative"]
                keys = list(stages.keys())
                vals = [stages[s] for s in keys]
                xs = np.arange(len(keys)) + (k - 1) * width
                ax.bar(xs, vals, width=width, color=COLORS[name], label=name)
                for x, val in zip(xs, vals):
                    ax.annotate(
                        f"{val:.2f}",
                        (x, val),
                        textcoords="offset points",
                        xytext=(0, 2),
                        ha="center",
                        fontsize=6,
                        rotation=90,
                    )
            longest = max(names, key=lambda n: len(td[n][var]["cumulative"]))
            keys = list(td[longest][var]["cumulative"].keys())
            short = [k.split("+")[-1] if "+" in k else k for k in keys]
            ax.set_yscale("log")
            ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
            ax.set_xticks(np.arange(n_groups), short[:n_groups], fontsize=7, rotation=20)
            ax.set_title(DOM_TITLES[dom], fontsize=10)
            despine(ax)
        axes[0].set_ylabel("невязка/тенденция (лог)")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(
            f"{VAR_TITLES[var]}: сборка уравнения по членам (ERA5-2000)", fontsize=11
        )
        fig.tight_layout()
        add_caption(
            fig,
            "Что показано: невязка модели, собранной из членов уравнения по нарастающей\n"
            "(подпись столбца — последний добавленный член). Цвет — вариант ядра. Ось Y\n"
            "логарифмическая, пунктир — «хуже нулевого прогноза». Как читать: рост столбца\n"
            "после добавления члена означает, что член в такой формулировке вносит шум.",
            y=-0.08,
        )
        fig.savefig(HERE / f"fig3_terms_{var}.png", bbox_inches="tight")
        plt.close(fig)


def fig4_q1_q2_structure() -> None:
    """Оценённые из невязки Q₁ и Q₂ (климатология 2000): широта × давление."""
    c_p, L = 1005.0, 2.5e6
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    lat = CLIM["globe"]["lat_deg"]
    q1 = CLIM["globe"]["C_best__annual_t"] * 86400.0  # Q₁/c_p, К/сут
    q2 = -L * CLIM["globe"]["C_best__annual_q"] * 86400.0 / c_p  # Q₂/c_p, К/сут
    for ax, field, title, label in (
        (axes[0], q1, "Q₁/c_p — кажущийся источник тепла", "К/сут"),
        (axes[1], q2, "Q₂/c_p — кажущийся сток влаги", "К/сут"),
    ):
        vmax = float(np.percentile(np.abs(field), 99))
        mesh = ax.pcolormesh(
            lat,
            P_LEVELS,
            field,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
            shading="auto",
        )
        ax.invert_yaxis()
        ax.set_xlabel("широта, °")
        ax.set_ylabel("давление, гПа")
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.02).set_label(label)
    fig.suptitle(
        "Структура недостающих источников из невязки C_best (глобус, 2000, зональное среднее)",
        fontsize=11,
    )
    fig.tight_layout()
    add_caption(
        fig,
        "Что показано: годовое зонально-среднее невязки T-уравнения (слева, в К/сут — это\n"
        "оценка Q₁/c_p по Yanai et al. 1973) и q-уравнения, переведённой в эквивалентный\n"
        "нагрев −L·(невязка q)/c_p (справа, оценка Q₂/c_p). Ожидаемая климатология для\n"
        "сравнения: узкий максимум нагрева в ВЗК (тропики, средняя тропосфера), максимумы\n"
        "штормтреков в средних широтах, радиационное выхолаживание 1–2 К/сут в свободной\n"
        "тропосфере; Q₂ сосредоточен ниже Q₁ (конденсация в нижней тропосфере).",
        y=-0.10,
    )
    fig.savefig(HERE / "fig4_q1_q2_structure.png", bbox_inches="tight")
    plt.close(fig)


def fig5_variants_matrix() -> None:
    """Сводка: невязка всех вариантов × переменные × домены × годы."""
    order = list(RESULTS[("usa", 2002)]["residual_rel"].keys())
    year_colors = {2000: "#0072B2", 2001: "#D55E00", 2002: "#009E73"}
    fig, axes = plt.subplots(2, 5, figsize=(15.5, 8.2), sharey=True)
    for r, dom in enumerate(DOMS):
        for c, var in enumerate(ALL_VARS):
            ax = axes[r, c]
            for yi, name in enumerate(order):
                vals = [
                    RESULTS[(dom, yr)]["residual_rel"].get(name, {}).get(var) for yr in YEARS
                ]
                vals_ok = [(yr, v) for yr, v in zip(YEARS, vals) if v is not None]
                if len(vals_ok) > 1:
                    vv = [v for _, v in vals_ok]
                    ax.plot([min(vv), max(vv)], [yi, yi], color="#cfcfcc", lw=1.0, zorder=1)
                for yr, val in vals_ok:
                    ax.scatter(
                        val, yi, s=22, color=year_colors[yr], zorder=3,
                        edgecolor="white", lw=0.4,
                    )
            ax.set_xscale("log")
            ax.axvline(1.0, color=MUTED, lw=0.8, ls="--")
            ax.set_title(f"{DOM_TITLES[dom]} · {var}", fontsize=9)
            if r == 1:
                ax.set_xlabel("невязка/тенденция (лог)")
            despine(ax)
    axes[0, 0].set_yticks(range(len(order)), order, fontsize=8)
    axes[0, 0].invert_yaxis()
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=str(y))
        for y, c in year_colors.items()
    ]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(
        "Матрица эксперимента 15: невязка всех уравнений по вариантам, доменам и годам",
        fontsize=12,
    )
    fig.tight_layout()
    add_caption(
        fig,
        "Что показано: относительная невязка каждого уравнения (столбцы панелей) для каждого\n"
        "варианта ядра (строки) на двух доменах (ряды панелей) за 2000–2002 (цвет точки; для\n"
        "клим-вариантов S1/S2/S12/C15_full 2000 год — in-sample, 2001–2002 — вне выборки).\n"
        "Ось X логарифмическая; пунктир на 1 — «хуже нулевого прогноза»; левее — лучше.",
        y=-0.04,
    )
    fig.savefig(HERE / "fig5_variants_matrix.png", bbox_inches="tight")
    plt.close(fig)


def fig6_temporal_and_levels() -> None:
    """Контроль временной схемы + профили по уровням для T и z."""
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 4.0))
    for ax, dom in zip(axes[:2], DOMS):
        tp = RESULTS[(dom, 2000)]["temporal_scheme"][BASE]
        schemes = ("fwd", "centered", "trapezoid")
        xs = np.arange(len(schemes))
        width = 0.16
        for k, var in enumerate(ALL_VARS):
            vals = [tp[var][s] for s in schemes]
            ax.bar(xs + (k - 2) * width, vals, width=width, label=var)
        ax.set_xticks(xs, ["fwd 1ч", "центр. 2ч", "трапеция"], fontsize=8)
        ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
        ax.set_title(f"временная схема · {DOM_TITLES[dom]}", fontsize=9)
        despine(ax)
    axes[0].set_ylabel("невязка/тенденция")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    for ax, dom in zip(axes[2:], DOMS):
        bl = RESULTS[(dom, 2000)]["by_level"]
        for var, ls in (("t", "-"), ("z", "--"), ("q", ":")):
            for name in (BASE, "C15_phys"):
                ax.plot(
                    bl[name][var],
                    P_LEVELS,
                    ls=ls,
                    color=COLORS[name],
                    lw=1.6,
                    label=f"{name} · {var}",
                )
        ax.set_xscale("log")
        ax.axvline(1.0, color=MUTED, lw=0.8, ls="--")
        ax.invert_yaxis()
        ax.set_xlabel("невязка/тенденция (лог)")
        ax.set_title(f"по уровням · {DOM_TITLES[dom]}", fontsize=9)
        despine(ax)
    axes[2].set_ylabel("давление, гПа")
    axes[2].legend(frameon=False, fontsize=6.5)
    fig.suptitle(
        "Контроль временной схемы (слева) и распределение невязки T/q/z по уровням (справа), 2000",
        fontsize=11,
    )
    fig.tight_layout()
    add_caption(
        fig,
        "Слева: одна и та же физика (C_best), три способа сопоставления с данными — forward-\n"
        "разность за час против мгновенного RHS, центрированная разность за 2 ч, трапеция.\n"
        "Если бы рассогласование «мгновенная тенденция vs часовая разность» доминировало,\n"
        "центрированные столбцы были бы кратно ниже. Справа: невязка T (сплошная), z (штрих),\n"
        "q (точки) по уровням давления для C_best (синий) и C15_phys (розовый).",
        y=-0.12,
    )
    fig.savefig(HERE / "fig6_temporal_levels.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Рендерит все фигуры в каталог эксперимента."""
    fig1_residual_maps()
    fig2_lat_profiles()
    fig3_term_decomposition()
    fig4_q1_q2_structure()
    fig5_variants_matrix()
    fig6_temporal_and_levels()
    print("figures written to", HERE)  # noqa: T201


if __name__ == "__main__":
    main()
