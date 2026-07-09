"""Фигуры и таблица эксперимента 16 (модельная лестница) из abl16_metrics.json.

Вход: ``results/abl16_metrics.json`` (создаётся ``collect_metrics.py``).
Выход: ``fig_val_rmse_curves.png`` (per-variable кривые val-RMSE по эпохам, линия
на арм) и ``results/abl16_final_table.md`` (RMSE финальной и лучшей эпохи).

Ось X — эпоха: трейнер логирует метрики с ``step=global_step`` (не epoch), а
валидация идёт каждые ``val_every_n_epochs``; поэтому i-я val-точка (по возрастанию
step) соответствует эпохе ``val_every_n_epochs·(i+1)``.

Оформление зеркалит exp13-15: палитра Okabe-Ito, срезанные рамки, врезка-пояснение.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

# Порядок ступеней лестницы + человекочитаемые подписи и цвета (Okabe-Ito).
ARM_ORDER = (
    "abl16-r0-no-physics-s0",
    "abl16-r1-legacy-hybrid-s0",
    "abl16-r2a-a1-pre13-s0",
    "abl16-r2-a2-pre13-s0",
    "abl16-r3-a2-exp13-s0",
    "abl16-r4-exp14-s0",
    "abl16-r5-exp15-s0",
)
ARM_LABELS = {
    "abl16-r0-no-physics-s0": "R0 · без физики",
    "abl16-r1-legacy-hybrid-s0": "R1 · легаси-hybrid",
    "abl16-r2a-a1-pre13-s0": "R2a · A1 (до exp13)",
    "abl16-r2-a2-pre13-s0": "R2 · A2 (до exp13)",
    "abl16-r3-a2-exp13-s0": "R3 · A2 (exp13)",
    "abl16-r4-exp14-s0": "R4 · +exp14",
    "abl16-r5-exp15-s0": "R5 · +exp15",
}
ARM_COLORS = {
    "abl16-r0-no-physics-s0": "#8a8a8a",
    "abl16-r1-legacy-hybrid-s0": "#E69F00",
    "abl16-r2a-a1-pre13-s0": "#56B4E9",
    "abl16-r2-a2-pre13-s0": "#009E73",
    "abl16-r3-a2-exp13-s0": "#0072B2",
    "abl16-r4-exp14-s0": "#D55E00",
    "abl16-r5-exp15-s0": "#CC79A7",
}
STAT_PREFERENCE = ("mean", "last", "first")
INK, MUTED = "#1a1a1a", "#8a8a8a"
CAPTION_BG = "#f4f4f2"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": "#e3e3e0",
        "grid.linewidth": 0.6,
    }
)


def parse_var_stat(metric_name: str) -> tuple[str, str] | None:
    """Разобрать имя метрики ``RMSE_<var>_<stat>`` в ``(var, stat)``.

    Args:
        metric_name: имя метрики Comet.

    Returns:
        ``(var, stat)`` для RMSE-метрик, иначе ``None`` (напр. для ``val_loss``).
    """
    if not metric_name.startswith("RMSE_"):
        return None
    body = metric_name.removeprefix("RMSE_")
    var, _, stat = body.rpartition("_")
    if stat in ("first", "last", "mean") and var:
        return var, stat
    return body, ""


def map_points_to_epochs(
    points: list[list[float]], val_every_n_epochs: int
) -> list[tuple[int, float]]:
    """Сопоставить отсортированным по step val-точкам номера эпох.

    Args:
        points: список ``[step, value]`` (порядок любой — сортируется по step).
        val_every_n_epochs: период валидации в эпохах.

    Returns:
        Список ``(epoch, value)``: i-я точка → эпоха ``val_every_n_epochs·(i+1)``.
    """
    ordered = sorted(points, key=lambda sv: sv[0])
    return [(val_every_n_epochs * (i + 1), value) for i, (_step, value) in enumerate(ordered)]


def select_rmse_series(
    run_metrics: dict[str, list[list[float]]], val_every_n_epochs: int
) -> dict[str, list[tuple[int, float]]]:
    """Для каждой переменной выбрать один RMSE-стат (mean ≻ last ≻ first).

    Args:
        run_metrics: ``{metric_name: [[step, value], ...]}`` одного рана.
        val_every_n_epochs: период валидации.

    Returns:
        ``{var: [(epoch, value), ...]}`` — по одной кривой на переменную.
    """
    by_var_stat: dict[str, dict[str, list[list[float]]]] = {}
    for name, points in run_metrics.items():
        parsed = parse_var_stat(name)
        if parsed is None:
            continue
        var, stat = parsed
        by_var_stat.setdefault(var, {})[stat or "mean"] = points
    series: dict[str, list[tuple[int, float]]] = {}
    for var, stats in by_var_stat.items():
        chosen = next((s for s in STAT_PREFERENCE if s in stats), None)
        if chosen is None:
            chosen = next(iter(stats))
        series[var] = map_points_to_epochs(stats[chosen], val_every_n_epochs)
    return series


def build_final_table(
    parsed_by_run: dict[str, dict[str, list[tuple[int, float]]]],
    variables: list[str],
) -> str:
    """Markdown-таблица: RMSE финальной эпохи по переменным, min в столбце — жирным.

    Args:
        parsed_by_run: ``{run: {var: [(epoch, value), ...]}}``.
        variables: порядок столбцов-переменных.

    Returns:
        Строка Markdown-таблицы (арм × переменная), плюс колонка «финальная эпоха».
    """
    # финальные значения на арм×переменную
    final: dict[str, dict[str, float]] = {}
    last_epoch: dict[str, int] = {}
    for run, series in parsed_by_run.items():
        final[run] = {}
        run_last = 0
        for var in variables:
            pts = series.get(var, [])
            if pts:
                epoch, value = pts[-1]
                final[run][var] = value
                run_last = max(run_last, epoch)
        last_epoch[run] = run_last
    # минимум по столбцу — для выделения
    col_min = {
        var: min((final[r][var] for r in final if var in final[r]), default=float("nan"))
        for var in variables
    }
    header = "| Арм | фин. эпоха | " + " | ".join(variables) + " |"
    sep = "|" + "---|" * (len(variables) + 2)
    rows = [header, sep]
    ordered_runs = [r for r in ARM_ORDER if r in parsed_by_run]
    ordered_runs += [r for r in parsed_by_run if r not in ARM_ORDER]
    for run in ordered_runs:
        cells = [ARM_LABELS.get(run, run), str(last_epoch.get(run, 0) or "—")]
        for var in variables:
            if var not in final[run]:
                cells.append("—")
                continue
            value = final[run][var]
            text = f"{value:.4g}"
            if value == col_min[var]:
                text = f"**{text}**"
            cells.append(text)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


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


def render_curves(
    parsed_by_run: dict[str, dict[str, list[tuple[int, float]]]],
    variables: list[str],
    dst: Path,
) -> None:
    """Сетка панелей: одна на переменную, val-RMSE по эпохам, линия на арм."""
    n = len(variables)
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow), squeeze=False)
    for idx, var in enumerate(variables):
        ax = axes[idx // ncol][idx % ncol]
        for run in ARM_ORDER:
            series = parsed_by_run.get(run, {}).get(var, [])
            if not series:
                continue
            epochs = [e for e, _ in series]
            values = [v for _, v in series]
            ax.plot(
                epochs,
                values,
                color=ARM_COLORS.get(run, INK),
                lw=1.7,
                marker="o",
                ms=3,
                label=ARM_LABELS.get(run, run),
            )
        ax.set_title(f"val-RMSE · {var}", fontsize=10)
        ax.set_xlabel("эпоха")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for idx in range(n, nrow * ncol):
        axes[idx // ncol][idx % ncol].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper center", ncol=4)
    fig.suptitle(
        "Лестница exp 16: val-RMSE по эпохам (сид 0, одинаковые данные/эпохи)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    add_caption(
        fig,
        "Что показано: RMSE на валидации (2004, вне обучения) по переменным; линия — ступень\n"
        "лестницы, точки — эпохи валидации (каждые 3). Ниже — лучше. Как читать: расхождение\n"
        "линий = вклад соответствующей ступени эволюции физического ядра при равных эпохах.",
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Собрать фигуру и таблицу из results/abl16_metrics.json."""
    payload = json.loads((HERE / "results" / "abl16_metrics.json").read_text())
    val_every = payload["meta"]["val_every_n_epochs"]
    runs = payload["runs"]
    parsed = {run: select_rmse_series(m, val_every) for run, m in runs.items()}
    variables = sorted({var for series in parsed.values() for var in series})
    render_curves(parsed, variables, HERE / "fig_val_rmse_curves.png")
    table = build_final_table(parsed, variables)
    (HERE / "results" / "abl16_final_table.md").write_text(table)
    print(f"[figures] {len(parsed)} runs, vars={variables}")  # noqa: T201
    print(f"[figures] written {HERE / 'fig_val_rmse_curves.png'}")  # noqa: T201


if __name__ == "__main__":
    main()
