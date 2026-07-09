"""Рендер фигур эксперимента 14 из results-JSON и NPZ карт невязки.

Вход: results/physics_stats_uv_{domain}_{year}.json (6 файлов) и
results/physics_maps_uv_{domain}_2000.npz (2 файла). Выход: PNG в каталоге
эксперимента. Каждая фигура самодостаточна: заголовок плюс врезка-пояснение
внизу («Что показано / Как читать»), чтобы график читался без текста отчёта.

Оформление по навыку dataviz: расходящаяся палитра только для знаковых полей
(log10 отношения невязок с нейтральным центром на 1); линии — Okabe-Ito;
логарифмические оси для отношений; на картах — контур береговой линии.
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
RESULTS = {
    (dom, year): json.loads(
        (HERE / "results" / f"physics_stats_uv_{dom}_{year}.json").read_text()
    )
    for dom in DOMS
    for year in YEARS
}
MAPS = {dom: np.load(HERE / "results" / f"physics_maps_uv_{dom}_2000.npz") for dom in DOMS}

VAR_TITLES = {
    "u": "u — зональный ветер",
    "v": "v — меридиональный ветер",
    "t": "T — температура",
    "q": "q — влажность",
    "z": "z — геопотенциал",
}
OKABE = {"base": "#8a8a8a", "V12_geom": "#0072B2", "C_best": "#D55E00", "base_mc": "#009E73"}
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
    """Для каждого из 5 уравнений: карты log10-невязки base и C_best, оба домена."""
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.3)
    for var in ("u", "v", "t", "q", "z"):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4), height_ratios=[1, 1.15])
        for col, variant, vtitle in ((0, "base", "до (base)"), (1, "C_best", "после (C_best)")):
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
                ax.set_title(
                    f"{DOM_TITLES[dom]} · {vtitle} · медиана {med:.2f}", fontsize=9
                )
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
            f"{VAR_TITLES[var]}: накопленная относительная невязка, ERA5-2000 (все часы года)",
            fontsize=11,
            y=0.98,
        )
        add_caption(
            fig,
            "Что показано: в каждой ячейке — сумма |ошибка тенденции| за все часы 2000 года,\n"
            "делённая на сумму |наблюдённая тенденция|. Слева base (конфигурация эксперимента 13),\n"
            "справа C_best (исправленная ориентация d_y, точные широты, кривизна сферы,\n"
            "сферическая дивергенция, трение Хелда–Суареса, массо-согласованная ω).\n"
            "Цвета: красное — уравнение хуже нулевого прогноза (отношение > 1), синее — лучше,\n"
            "белое — отношение 1. Чёрные линии — берега. Числа в заголовках — медиана по ячейкам.",
        )
        fig.savefig(HERE / f"fig1_maps_{var}.png", bbox_inches="tight")
        plt.close(fig)


def fig2_lat_profiles() -> None:
    """Широтные профили невязки u и v: base → V12_geom → C_best (2000)."""
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2), sharex="col")
    for col, dom in enumerate(DOMS):
        prof = RESULTS[(dom, 2000)]["lat_profile"]
        lat = prof["lat_deg"]
        for row, var in enumerate(("u", "v")):
            ax = axes[row, col]
            for name, label in (
                ("base", "base (exp 13)"),
                ("V12_geom", "V12: d_y + широты"),
                ("C_best", "C_best (итог)"),
            ):
                ax.plot(lat, prof[name][var], color=OKABE[name], lw=1.8, label=label)
            ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
            ax.set_yscale("log")
            ax.set_ylabel(f"{var}: невязка/тенденция")
            despine(ax)
            if row == 0:
                ax.set_title(DOM_TITLES[dom], fontsize=10)
            if row == 1:
                ax.set_xlabel("широта, ° (отрицательная — южное полушарие)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper center")
    fig.suptitle(
        "Невязка ветра по широте: знак d_y снимает основную ошибку на всех широтах", fontsize=11
    )
    fig.tight_layout()
    add_caption(
        fig,
        "Что показано: относительная невязка уравнений движения по строкам широты, ERA5-2000\n"
        "(все часы года). Верхний ряд — u, нижний — v; слева США, справа глобус. Ось Y —\n"
        "логарифмическая; пунктир на 1 — граница «хуже нулевого прогноза». Цвета: серый —\n"
        "base, синий — исправленные ориентация d_y и широты сетки, оранжевый — итоговая\n"
        "комбинация C_best (плюс кривизна, сферическая дивергенция, трение, массо-согл. ω).",
        y=-0.06,
    )
    fig.savefig(HERE / "fig2_lat_profiles.png", bbox_inches="tight")
    plt.close(fig)


def fig3_term_decomposition() -> None:
    """Кумулятивная сборка уравнения по членам: невязка после каждого шага."""
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.4), sharey="row")
    for col, dom in enumerate(DOMS):
        td = RESULTS[(dom, 2000)]["term_decomposition"]
        for row, var in enumerate(("u", "v")):
            ax = axes[row, col]
            width = 0.38
            for k, (name, label) in enumerate((("base", "base"), ("C_best", "C_best"))):
                stages = td[name][var]["cumulative"]
                keys = list(stages.keys())
                vals = [stages[s] for s in keys]
                xs = np.arange(len(keys)) + (k - 0.5) * width
                color = OKABE[name]
                ax.bar(xs, vals, width=width, color=color, label=label)
                for x, val in zip(xs, vals):
                    ax.annotate(
                        f"{val:.2f}",
                        (x, val),
                        textcoords="offset points",
                        xytext=(0, 2),
                        ha="center",
                        fontsize=6.5,
                    )
            ax.set_yscale("log")
            ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
            ax.set_xticks(np.arange(len(keys)), keys, fontsize=7.5)
            ax.set_ylabel(f"{var}: невязка/тенденция")
            despine(ax)
            if row == 0:
                ax.set_title(DOM_TITLES[dom], fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Сборка уравнения по членам: где невязка падает, а где растёт (ERA5-2000)", fontsize=11
    )
    fig.tight_layout()
    add_caption(
        fig,
        "Что показано: невязка модели, собранной из членов уравнения по нарастающей:\n"
        "только PGF+Кориолис → +горизонтальная адвекция → +вертикальная адвекция →\n"
        "+кривизна сферы → +трение. Серые столбцы — на геометрии base (перевёрнутый d_y),\n"
        "оранжевые — C_best. Ось Y логарифмическая, пунктир — «хуже нулевого прогноза».\n"
        "Как читать: если столбец после добавления члена выше предыдущего — член вносит шум\n"
        "(так ведёт себя вертикальная адвекция с шумной ω из континуити в base).",
        y=-0.05,
    )
    fig.savefig(HERE / "fig3_term_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def fig4_variants_matrix() -> None:
    """Сводка матрицы прогонов: невязка u и v всех вариантов × годы × домены."""
    order = list(RESULTS[("usa", 2000)]["residual_rel"].keys())
    year_colors = {2000: "#0072B2", 2001: "#D55E00", 2002: "#009E73"}
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 5.4), sharey=True)
    panels = [(dom, var) for dom in DOMS for var in ("u", "v")]
    for ax, (dom, var) in zip(axes, panels):
        for yi, name in enumerate(order):
            vals = [RESULTS[(dom, yr)]["residual_rel"][name][var] for yr in YEARS]
            ax.plot(
                [min(vals), max(vals)], [yi, yi], color="#cfcfcc", lw=1.0, zorder=1
            )
            for yr, val in zip(YEARS, vals):
                ax.scatter(
                    val, yi, s=26, color=year_colors[yr], zorder=3, edgecolor="white", lw=0.4
                )
        ax.set_xscale("log")
        ax.axvline(1.0, color=MUTED, lw=0.8, ls="--")
        ax.set_title(f"{DOM_TITLES[dom]} · {var}", fontsize=9)
        ax.set_xlabel("невязка/тенденция (лог)")
        despine(ax)
    axes[0].set_yticks(range(len(order)), order, fontsize=8)
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=str(y))
        for y, c in year_colors.items()
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "Матрица вариантов: невязка уравнений движения по вариантам, доменам и годам",
        fontsize=11,
    )
    fig.tight_layout()
    add_caption(
        fig,
        "Что показано: относительная невязка u и v для каждого варианта уравнений (строки),\n"
        "каждого домена (панели) и каждого года 2000–2002 (цвет точки). Ось X логарифмическая,\n"
        "пунктир на 1 — «хуже нулевого прогноза», серая линия соединяет min и max по годам.\n"
        "Как читать: чем левее точки, тем точнее вариант; согласие трёх точек — стабильность по годам.",
        y=-0.10,
    )
    fig.savefig(HERE / "fig4_variants_matrix.png", bbox_inches="tight")
    plt.close(fig)


def fig5_levels() -> None:
    """Профили невязки по уровням давления: где живёт остаточная ошибка."""
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), sharey=True, sharex="col")
    for col, dom in enumerate(DOMS):
        bl = RESULTS[(dom, 2000)]["by_level"]
        p_levels = bl["pressure_hpa"]
        for row, var in enumerate(("u", "v")):
            ax = axes[row, col]
            for name, label in (
                ("base", "base"),
                ("V12_geom", "V12"),
                ("C_best", "C_best"),
            ):
                ax.plot(bl[name][var], p_levels, color=OKABE[name], lw=1.8, marker="o", ms=3, label=label)
            ax.set_xscale("log")
            ax.axvline(1.0, color=MUTED, lw=0.8, ls="--")
            ax.set_ylabel("давление, гПа")
            ax.set_xlabel(f"{var}: невязка/тенденция (лог)")
            despine(ax)
            if row == 0:
                ax.set_title(DOM_TITLES[dom], fontsize=10)
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Невязка ветра по уровням давления (ERA5-2000): максимум — стратосфера и приземный слой",
        fontsize=11,
    )
    fig.tight_layout()
    add_caption(
        fig,
        "Что показано: относительная невязка u (верх) и v (низ) на каждом из 13 уровней давления;\n"
        "1000 гПа — у земли (внизу оси), 50 гПа — стратосфера (вверху). Цвета как на фиг. 2.\n"
        "Как читать: по положению оранжевой кривой видно, какие уровни остаются хуже нулевого\n"
        "прогноза (правее пунктира) после всех правок — там нужны отсутствующие параметризации.",
        y=-0.04,
    )
    fig.savefig(HERE / "fig5_levels.png", bbox_inches="tight")
    plt.close(fig)


def fig6_temporal() -> None:
    """Контроль временной схемы: forward против centered и trapezoid."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    schemes = ("fwd", "centered", "trapezoid")
    scheme_labels = {
        "fwd": "разность за 1 ч\nпротив RHS(t)",
        "centered": "центрированная за 2 ч\nпротив RHS(t+1)",
        "trapezoid": "разность за 1 ч против\n½·(RHS(t)+RHS(t+1))",
    }
    for ax, dom in zip(axes, DOMS):
        tp = RESULTS[(dom, 2000)]["temporal_scheme"]
        width = 0.2
        xs = np.arange(len(schemes))
        for k, (name, var) in enumerate((("V12_geom", "u"), ("V12_geom", "v"), ("C_best", "u"), ("C_best", "v"))):
            vals = [tp[name][var][s] for s in schemes]
            color = OKABE[name]
            hatch = "" if var == "u" else "//"
            ax.bar(
                xs + (k - 1.5) * width,
                vals,
                width=width,
                color=color,
                hatch=hatch,
                edgecolor="white",
                lw=0.4,
                label=f"{name} · {var}",
            )
        ax.set_xticks(xs, [scheme_labels[s] for s in schemes], fontsize=7.5)
        ax.set_title(DOM_TITLES[dom], fontsize=10)
        despine(ax)
    axes[0].set_ylabel("невязка/тенденция")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(
        "Временная схема сравнения объясняет малую долю невязки (ERA5-2000)", fontsize=11
    )
    fig.tight_layout()
    add_caption(
        fig,
        "Что показано: одна и та же физика, три способа сопоставить её с данными: forward-разность\n"
        "за час против мгновенного RHS; центрированная разность за 2 часа; трапецеидальное среднее\n"
        "RHS двух моментов. Цвет — вариант физики, штриховка — переменная (сплошное u, штрих v).\n"
        "Как читать: если бы рассогласование «мгновенная тенденция против часовой разности» было\n"
        "главной причиной невязки, центрированные/трапецеидальные столбцы были бы кратно ниже.",
        y=-0.14,
    )
    fig.savefig(HERE / "fig6_temporal.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Рендерит все фигуры в каталог эксперимента."""
    fig1_residual_maps()
    fig2_lat_profiles()
    fig3_term_decomposition()
    fig4_variants_matrix()
    fig5_levels()
    fig6_temporal()
    print("figures written to", HERE)


if __name__ == "__main__":
    main()
