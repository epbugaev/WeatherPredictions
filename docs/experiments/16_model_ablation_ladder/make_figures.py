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
VAR_ORDER = ("z", "t", "u", "v", "r", "tp")
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


def base_variable(channel: str) -> str:
    """Свести имя канала к базовой переменной, отбросив уровень/высоту.

    ``z500``/``z850`` → ``z``; ``t2`` → ``t``; ``u10`` → ``u``; ``tp`` → ``tp``.
    Так по-уровневые каналы WeatherBench-69 сворачиваются в 5-6 переменных.

    Args:
        channel: имя канала (буквенный префикс + опциональный числовой уровень).

    Returns:
        Базовая переменная (буквенный префикс; для чисто-буквенных — сам канал).
    """
    return channel.rstrip("0123456789") or channel


def select_rmse_series(
    run_metrics: dict[str, list[list[float]]], val_every_n_epochs: int
) -> dict[str, list[tuple[int, float]]]:
    """Свести per-level RMSE к кривой на базовую переменную (среднее по уровням).

    Для каждого канала берётся один стат (mean ≻ last ≻ first), точки
    привязываются к эпохам, затем каналы одной базовой переменной усредняются
    поэпохно.

    Args:
        run_metrics: ``{metric_name: [[step, value], ...]}`` одного рана.
        val_every_n_epochs: период валидации.

    Returns:
        ``{base_var: [(epoch, mean_over_levels), ...]}`` — по кривой на переменную.
    """
    stats_by_channel: dict[str, dict[str, list[list[float]]]] = {}
    for name, points in run_metrics.items():
        parsed = parse_var_stat(name)
        if parsed is None:
            continue
        channel, stat = parsed
        stats_by_channel.setdefault(channel, {})[stat or "mean"] = points
    # выбранный стат каждого канала → кривая (epoch, value)
    per_channel: dict[str, list[tuple[int, float]]] = {}
    for channel, stats in stats_by_channel.items():
        chosen = next((s for s in STAT_PREFERENCE if s in stats), None)
        if chosen is None:
            chosen = next(iter(stats))
        per_channel[channel] = map_points_to_epochs(stats[chosen], val_every_n_epochs)
    # усреднить каналы одной базовой переменной поэпохно
    grouped: dict[str, dict[int, list[float]]] = {}
    for channel, series in per_channel.items():
        by_epoch = grouped.setdefault(base_variable(channel), {})
        for epoch, value in series:
            by_epoch.setdefault(epoch, []).append(value)
    return {
        var: [(epoch, sum(vals) / len(vals)) for epoch, vals in sorted(by_epoch.items())]
        for var, by_epoch in grouped.items()
    }


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
    fig.suptitle(
        "Лестница exp 16: val-RMSE по эпохам (сид 0, вал-2004)",
        fontsize=12,
        y=1.11,
    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=7,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    add_caption(
        fig,
        "Что показано: RMSE на валидации (2004, вне обучения) по переменным; линия — ступень\n"
        "лестницы, точки — эпохи валидации (каждые 3). Ниже — лучше. Как читать: расхождение\n"
        "линий = вклад соответствующей ступени эволюции физического ядра при равных эпохах.",
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def order_variables(variables: list[str]) -> list[str]:
    """Упорядочить переменные по ``VAR_ORDER`` (неизвестные — в конец по алфавиту)."""
    return sorted(
        variables, key=lambda v: (VAR_ORDER.index(v) if v in VAR_ORDER else len(VAR_ORDER), v)
    )


def common_epoch(parsed_by_run: dict[str, dict[str, list[tuple[int, float]]]]) -> int:
    """Наибольшая эпоха, достигнутая ВСЕМИ армами (для честного сравнения усечённых).

    Args:
        parsed_by_run: ``{run: {var: [(epoch, value), ...]}}``.

    Returns:
        ``min`` по армам от их максимальной эпохи (0, если данных нет).
    """
    per_run_max = [
        max((pts[-1][0] for pts in series.values() if pts), default=0)
        for series in parsed_by_run.values()
        if series
    ]
    return min(per_run_max) if per_run_max else 0


def value_at_epoch(
    series: dict[str, list[tuple[int, float]]], var: str, epoch: int
) -> float | None:
    """Значение переменной на конкретной эпохе (``None``, если её нет)."""
    return next((v for e, v in series.get(var, []) if e == epoch), None)


def _delta_table(
    parsed_by_run: dict[str, dict[str, list[tuple[int, float]]]],
    baseline_run: str,
    variables: list[str],
    epochs: tuple[int, ...],
    title: str,
) -> str:
    """Общий корень дельта-таблиц: средняя по окну эпох Δ% к baseline на ячейку."""
    ordered_vars = order_variables(variables)
    base = parsed_by_run.get(baseline_run)
    header = "| ступень | " + " | ".join(ordered_vars) + " |"
    sep = "|" + "---|" * (len(ordered_vars) + 1)
    lines = [title, "", header, sep]
    runs = [r for r in ARM_ORDER if r in parsed_by_run and r != baseline_run]
    runs += [r for r in parsed_by_run if r not in ARM_ORDER and r != baseline_run]
    for run in runs:
        cells = [ARM_LABELS.get(run, run)]
        for var in ordered_vars:
            deltas = []
            for epoch in epochs:
                a = value_at_epoch(base, var, epoch) if base else None
                b = value_at_epoch(parsed_by_run[run], var, epoch)
                if a and b:
                    deltas.append(100 * (b - a) / a)
            cells.append(f"{sum(deltas) / len(deltas):+.1f}%" if deltas else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_delta_table(
    parsed_by_run: dict[str, dict[str, list[tuple[int, float]]]],
    baseline_run: str,
    variables: list[str],
    epoch: int,
) -> str:
    """Markdown: Δ% RMSE к baseline на общей эпохе (``−`` = лучше baseline).

    Снимок одной эпохи: для медленных полей (t/u/v/r) устойчив, для z шумит
    (±5–8 п.п. между соседними val-точками) — робастная версия
    ``build_window_delta_table``.

    Args:
        parsed_by_run: ``{run: {var: [(epoch, value), ...]}}``.
        baseline_run: ключ рана-эталона (напр. R0 «без физики»).
        variables: столбцы-переменные (упорядочиваются ``order_variables``).
        epoch: общая эпоха сравнения.

    Returns:
        Строка Markdown-таблицы Δ% (арм × переменная) относительно baseline.
    """
    title = f"Δ% RMSE к {ARM_LABELS.get(baseline_run, baseline_run)} @ эпоха {epoch} (− = лучше):"
    return _delta_table(parsed_by_run, baseline_run, variables, (epoch,), title)


def build_window_delta_table(
    parsed_by_run: dict[str, dict[str, list[tuple[int, float]]]],
    baseline_run: str,
    variables: list[str],
    epochs: tuple[int, ...],
) -> str:
    """Markdown: Δ% RMSE к baseline, усреднённая по окну эпох (``−`` = лучше).

    Ячейка — среднее по-эпоховых Δ%; эпохи, отсутствующие у арма или baseline,
    пропускаются (усечённые армы сравниваются по доступной части окна).

    Args:
        parsed_by_run: ``{run: {var: [(epoch, value), ...]}}``.
        baseline_run: ключ рана-эталона.
        variables: столбцы-переменные (упорядочиваются ``order_variables``).
        epochs: окно эпох усреднения (напр. последние 4 общих val-эпохи).

    Returns:
        Строка Markdown-таблицы средних Δ% (арм × переменная).
    """
    window = ", ".join(str(e) for e in epochs)
    title = (
        f"Δ% RMSE к {ARM_LABELS.get(baseline_run, baseline_run)}, "
        f"среднее по эпохам {{{window}}} (− = лучше):"
    )
    return _delta_table(parsed_by_run, baseline_run, variables, epochs, title)


def main() -> None:
    """Собрать фигуру и таблицы из results/abl16_metrics.json."""
    payload = json.loads((HERE / "results" / "abl16_metrics.json").read_text())
    val_every = payload["meta"]["val_every_n_epochs"]
    runs = payload["runs"]
    parsed = {run: select_rmse_series(m, val_every) for run, m in runs.items()}
    variables = order_variables(list({var for series in parsed.values() for var in series}))
    render_curves(parsed, variables, HERE / "fig_val_rmse_curves.png")
    (HERE / "results" / "abl16_final_table.md").write_text(build_final_table(parsed, variables))
    epoch = common_epoch(parsed)
    delta = build_delta_table(parsed, ARM_ORDER[0], variables, epoch)
    (HERE / "results" / "abl16_delta_table.md").write_text(delta)
    # последние 4 общих val-эпохи: устойчивая к шуму одной точки версия дельт
    window = tuple(range(val_every, epoch + 1, val_every))[-4:]
    window_delta = build_window_delta_table(parsed, ARM_ORDER[0], variables, window)
    (HERE / "results" / "abl16_window_delta_table.md").write_text(window_delta)
    print(f"[figures] {len(parsed)} runs, vars={variables}, common_epoch={epoch}")  # noqa: T201
    print(delta)  # noqa: T201
    print(window_delta)  # noqa: T201
    print(f"[figures] written {HERE / 'fig_val_rmse_curves.png'}")  # noqa: T201


if __name__ == "__main__":
    main()
