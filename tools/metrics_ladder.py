"""Фигуры и таблицы полного набора метрик физической «лестницы» (общее ядро).

Раньше эта логика жила целиком в
``docs/experiments/16_model_ablation_ladder/metrics/metrics_figures.py``.
Эксперимент 20 (PI-PredRNNv2) строит те же артефакты из тех же npz — отличаются
только имена армов, подписи, эпоха и пути, — поэтому ядро вынесено сюда, а скрипты
экспериментов стали тонкими обёртками (тот же приём, что у :mod:`tools.ladder_figures`).

Вход (каталог ``results/`` эксперимента):

* ``metrics_<arm>.npz`` — сводки армов (:mod:`metrics_eval` эксперимента);
* ``paired_deltas.npz`` — парные дельты к контрольному арму с бутстрап-CI
  (``paired_deltas.py``).

Выход (:func:`write_metric_outputs`):

* ``fig_ci_forest.png`` — главная: дельта каждого арма к контролю по всем метрикам
  с 95% CI (значимо ⇔ интервал не накрывает ноль);
* ``fig_levels_<метрика>.png`` — сводный хитмап [уровень × шаг], строка на арм,
  столбец на переменную; **незначимые ячейки заштрихованы** — их читать нельзя;
* ``<heatmaps>/<метрика>/<арм>.png`` — пер-армовый хитмап [уровень × шаг] с числами;
* ``fig_psd.png`` — спектр по зональному волновому числу m: прогноз против истины;
* ``results/metrics_table.md`` — те же числа текстом.

Ключ арма всюду канонический (:func:`canonical_arm`): ``abl16L-r3-a2-exp13-t12-s0``
и ``exp20-p3-a2-exp13-s0`` → ``r3-a2-exp13`` и ``exp20-p3-a2-exp13``. Тем же ключом
``paired_deltas.py`` именует поля npz, поэтому правило канонизации одно на весь
харнесс и живёт здесь.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UPPER_VARS = ("z", "t", "r", "u", "v")
VAR_BASE = {"z": 4, "t": 17, "r": 30, "u": 43, "v": 56}
PRESSURE_HPA = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
SURFACE_CHANNELS = ("t2", "u10", "v10", "tp")

# Подпись, направление («+1» = больше лучше) и единицы дельты. Единицы разные не для
# красоты: у ограниченных оценок (ACC/CSI/FSS) относительная дельта не определена
# (у контроля CSI бывает ровно 0), а у знакопеременного bias не интерпретируется.
METRICS = {
    "rmse": ("RMSE, Δ%", -1),
    "acc": ("ACC, Δ пункта", +1),
    "mcsi": ("mCSI (p90/95/99), Δ пункта", +1),
    "fss3": ("FSS (окно 3), Δ пункта", +1),
    "w1": ("W1 распределения, Δ%", -1),
    "bias": ("|bias|, Δ в % от σ", -1),
    "std_pred": ("дисперсия поля, Δ%", +1),
}
# Шкала цвета на метрику: у них разные единицы и разный масштаб эффекта.
METRIC_CLIP = {
    "rmse": 8.0,
    "acc": 2.0,
    "mcsi": 4.0,
    "fss3": 4.0,
    "w1": 12.0,
    "bias": 3.0,
    "std_pred": 6.0,
}
INK, MUTED = "#1a1a1a", "#8a8a8a"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.edgecolor": MUTED})


def canonical_arm(name: str) -> str:
    """Каноническое имя арма: ``abl16L-r3-a2-exp13-t12-s0`` → ``r3-a2-exp13``.

    Срезаются два стиля суффикса сида с ЛЮБЫМ номером: exp16/exp20 (``-s0``,
    ``-t12-s0``, а для повторных волн ``-s1`` и т.д.) и exp21 (``-seed0``,
    ``-t12-seed0``, ``-seed1`` …). Префикс ``abl16``/``abl16L`` срезается (там
    арм самодостаточен); ``exp20``/``exp21`` — нет: он несёт эксперимент, а не
    арм, и без него ключи разных лестниц столкнулись бы (у всех есть «A2 exp13»).

    Args:
        name: имя рана (``experiment.name`` из конфига) или основа имени файла.

    Returns:
        Ключ арма, которым именуются поля ``paired_deltas.npz`` и подписи фигур.
    """
    stem = name.removeprefix("abl16L-").removeprefix("abl16-")
    return re.sub(r"-(t12-)?(s|seed)\d+$", "", stem)


@dataclass(frozen=True)
class LadderSpec:
    """Специфика лестницы: имена армов, подписи, эпоха. Всё остальное — общее.

    Attributes:
        baseline: канонический ключ контрольного арма (без физики).
        baseline_label: короткая подпись контроля в осях и заголовках («R0», «P0»).
        arm_order: армы сверху вниз на forest-фигуре и в таблице (без контроля).
        labels: подпись на арм; ДОЛЖНА содержать и ключ ``baseline`` (легенда PSD).
        level_arms: армы сводного хитмапа ``fig_levels_<метрика>.png`` (строки).
        psd_arms: армы на спектральной фигуре (обычно контроль + 2 характерных).
        psd_colors: цвет линии на арм для ``fig_psd.png``.
        epoch: общая эпоха чекпоинтов всех армов (подписи фигур).
        table_title: заголовок ``results/metrics_table.md``.
        arm_filename_prefix: префикс, срезаемый из ИМЕНИ пер-армового PNG
            (``<метрика>/<арм>.png``), чтобы имена были чистыми как у exp16
            (``r3-a2-exp13.png``). Ключ арма в npz и подписях не меняется — режется
            только имя файла. Пусто → имя = ключ (exp16/exp20).
    """

    baseline: str
    baseline_label: str
    arm_order: tuple[str, ...]
    labels: dict[str, str]
    level_arms: tuple[str, ...]
    psd_arms: tuple[str, ...]
    psd_colors: dict[str, str]
    epoch: int
    table_title: str
    arm_filename_prefix: str = ""


def load(
    results_dir: Path, spec: LadderSpec
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Прочитать сводки армов, парные дельты и имена каналов.

    Args:
        results_dir: каталог с ``metrics_<arm>.npz`` и ``paired_deltas.npz``.
        spec: спецификация лестницы (нужен ключ контрольного арма).

    Returns:
        ``(summaries {арм: npz}, deltas npz, channels [69])``.
    """
    summaries = {}
    for path in sorted(results_dir.glob("metrics_*.npz")):
        data = np.load(path, allow_pickle=False)
        summaries[canonical_arm(str(data["arm"]))] = data
    deltas = np.load(results_dir / "paired_deltas.npz", allow_pickle=False)
    channels = [str(c) for c in summaries[spec.baseline]["channels"]]
    return summaries, deltas, channels


def delta_grid(deltas: np.ndarray, arm: str, metric: str, key: str) -> np.ndarray:
    """Дельта метрики → ``(12, 69)``.

    Агрегаты по порогам (``mcsi``) и окнам (``fss3``) — отдельные ключи в npz: они
    усреднены ДО бутстрапа, поэтому их CI настоящий. Усреднять готовые интервалы
    нельзя — среднее интервалов не является интервалом среднего.
    """
    return deltas[f"{arm}__{metric}__{key}"]


def upper_channels(channels: list[str]) -> list[int]:
    """Индексы каналов уровней давления (приземные исключены)."""
    return [i for i, name in enumerate(channels) if name not in SURFACE_CHANNELS]


def oriented_delta_and_mask(
    deltas: np.ndarray, arm: str, metric: str, var: str
) -> tuple[np.ndarray, np.ndarray]:
    """Сетка ``[13 уровней × 12 шагов]``: выигрыш к контролю и маска НЕзначимых ячеек.

    Знак разворачивается так, чтобы «синее = физика лучше» для любой метрики:
    у RMSE/W1/bias лучше меньше, у ACC/CSI/FSS — больше.

    Args:
        deltas: npz парных дельт.
        arm: канонический ключ арма.
        metric: ключ метрики из `METRICS`.
        var: переменная из `UPPER_VARS`.

    Returns:
        ``(gain (13, 12), insignificant (13, 12))`` — во второй True там, где CI
        накрывает ноль (такую ячейку читать нельзя).
    """
    _, better = METRICS[metric]
    base = VAR_BASE[var]
    columns = slice(base, base + 13)
    gain = delta_grid(deltas, arm, metric, "delta")[:, columns].T * better
    low = delta_grid(deltas, arm, metric, "ci_low")[:, columns].T
    high = delta_grid(deltas, arm, metric, "ci_high")[:, columns].T
    return gain, ~((low > 0) | (high < 0))


def render_forest(deltas: np.ndarray, channels: list[str], spec: LadderSpec, dst: Path) -> None:
    """Главная фигура: дельта к контролю по каждой метрике с 95% CI (значимо ⇔ CI мимо нуля)."""
    upper = upper_channels(channels)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.1 * len(METRICS), 4.6), sharey=True)
    positions = np.arange(len(spec.arm_order))[::-1]
    for ax, (metric, (label, better)) in zip(axes, METRICS.items(), strict=True):
        for pos, arm in zip(positions, spec.arm_order, strict=True):
            mean = delta_grid(deltas, arm, metric, "delta")[:, upper].mean()
            low = delta_grid(deltas, arm, metric, "ci_low")[:, upper].mean()
            high = delta_grid(deltas, arm, metric, "ci_high")[:, upper].mean()
            significant = low > 0 or high < 0
            good = significant and (mean * better > 0)
            color = "#0072B2" if good else ("#D55E00" if significant else MUTED)
            ax.plot([low, high], [pos, pos], color=color, lw=2.4, solid_capstyle="round")
            ax.plot(
                mean,
                pos,
                "o" if significant else "x",
                color=color,
                ms=6,
                mfc="white" if not significant else color,
            )
        ax.axvline(0, color=INK, lw=1.0, ls="--")
        ax.set_title(label, fontsize=10)
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_xlabel(f"Δ к {spec.baseline_label}")  # единицы — в заголовке панели
    axes[0].set_yticks(positions, [spec.labels[arm] for arm in spec.arm_order], fontsize=9)
    base = spec.baseline_label
    fig.suptitle(
        f"Вклад физики относительно {base} — все армы на ОБЩЕЙ эпохе {spec.epoch} "
        f"(парный бутстрап, 95% CI)\n"
        f"синее = значимо лучше {base} · красное = значимо хуже · "
        f"серое ×  = неотличимо от {base}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def render_levels(
    deltas: np.ndarray,
    metric: str,
    spec: LadderSpec,
    dst: Path,
    title_prefix: str = "вклад физики",
) -> None:
    """Хитмап [уровень × шаг] к контролю: строка = арм, столбец = переменная.

    Незначимые ячейки (CI накрывает ноль) заштрихованы — по ним нельзя судить.

    Args:
        title_prefix: ведущая фраза заголовка (по умолчанию «вклад физики» —
            подходит для армов физ-лестницы; для нефизических сравнений
            (например, модель против модели) передайте свою фразу).
    """
    label, _ = METRICS[metric]
    clip = METRIC_CLIP[metric]
    level_arms = spec.level_arms
    fig, axes = plt.subplots(
        len(level_arms), len(UPPER_VARS), figsize=(3.0 * len(UPPER_VARS), 2.6 * len(level_arms))
    )
    image = None
    for row, arm in enumerate(level_arms):
        for col, var in enumerate(UPPER_VARS):
            block, insignificant = oriented_delta_and_mask(deltas, arm, metric, var)
            ax = axes[row, col]
            image = ax.imshow(
                block, cmap="RdBu", vmin=-clip, vmax=clip, aspect="auto", interpolation="nearest"
            )
            ax.contourf(
                insignificant.astype(float),
                levels=[0.5, 1.5],
                colors="none",
                hatches=["////"],
                extend="neither",
            )
            if row == 0:
                ax.set_title(var, fontsize=11)
            if col == 0:
                ax.set_ylabel(spec.labels[arm], fontsize=8)
                ax.set_yticks(range(13), [str(p) for p in PRESSURE_HPA], fontsize=6)
            else:
                ax.set_yticks([])
            xticks = range(0, block.shape[1], max(1, block.shape[1] // 4))
            ax.set_xticks(list(xticks), [str(s + 1) for s in xticks], fontsize=6)
            if row == len(level_arms) - 1:
                ax.set_xlabel("шаг", fontsize=8)
            ax.grid(False)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.015, pad=0.01)
    colorbar.set_label(f"выигрыш к {spec.baseline_label}, % ({label})")
    fig.suptitle(
        f"{label}: {title_prefix} по [уровень × шаг] относительно {spec.baseline_label} "
        f"(эпоха {spec.epoch})\n"
        "синее = физика лучше · штриховка = CI накрывает ноль (незначимо)",
        fontsize=12,
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def render_arm_metric_heatmap(
    deltas: np.ndarray, arm: str, metric: str, spec: LadderSpec, dst: Path
) -> None:
    """Один арм × одна метрика: хитмап [уровень × шаг], панель на переменную.

    Числа — в ячейках; **штриховка = CI накрывает ноль** (ячейку читать нельзя).
    Синее всегда означает «физика лучше контроля», независимо от направления метрики.
    """
    label, _ = METRICS[metric]
    clip = METRIC_CLIP[metric]
    base = spec.baseline_label
    fig, axes = plt.subplots(1, len(UPPER_VARS), figsize=(4.0 * len(UPPER_VARS), 4.4), sharey=True)
    image = None
    for ax, var in zip(axes, UPPER_VARS, strict=True):
        gain, insignificant = oriented_delta_and_mask(deltas, arm, metric, var)
        image = ax.imshow(
            gain, cmap="RdBu", vmin=-clip, vmax=clip, aspect="auto", interpolation="nearest"
        )
        ax.contourf(
            insignificant.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["////"],
            extend="neither",
        )
        for row in range(gain.shape[0]):
            for col in range(gain.shape[1]):
                value = gain[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:+.0f}" if abs(value) >= 0.5 else "·",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if abs(value) > 0.62 * clip else INK,
                )
        ax.set_title(var, fontsize=11)
        ax.set_xticks(
            range(gain.shape[1]), [str(step + 1) for step in range(gain.shape[1])], fontsize=6
        )
        ax.set_xlabel("шаг прогноза")
        ax.grid(False)
    axes[0].set_yticks(range(13), [str(p) for p in PRESSURE_HPA], fontsize=7)
    axes[0].set_ylabel("уровень давления, гПа")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.012, pad=0.01)
    colorbar.set_label(f"выигрыш к {base} ({label})")
    fig.suptitle(
        f"{spec.labels[arm]} против {base} — {label} по [уровень × шаг] "
        f"(эпоха {spec.epoch}, парный бутстрап)",
        fontsize=12,
        y=1.02,
    )
    add_caption(
        fig,
        f"Синее = физика лучше {base}, красное = хуже "
        "(знак развёрнут под направление метрики).\n"
        "ШТРИХОВКА = 95 % CI накрывает ноль: эффект от нуля не отличим, "
        "ячейку читать нельзя.\n"
        "Точка вместо числа = |эффект| < 0.5 в единицах метрики.",
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def add_caption(fig, text: str) -> None:
    """Врезка-пояснение под фигурой (попадает в PNG через bbox_inches='tight')."""
    fig.text(
        0.5,
        -0.06,
        text,
        ha="center",
        va="top",
        ma="left",
        fontsize=8,
        color=INK,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f4f4f2", "edgecolor": "#d5d5d2"},
    )


def render_psd(
    summaries: dict[str, np.ndarray], channels: list[str], spec: LadderSpec, dst: Path
) -> None:
    """Спектр по зональному волновому числу m: истина против прогнозов (шаг 1 и 12)."""
    fig, axes = plt.subplots(2, len(UPPER_VARS), figsize=(3.1 * len(UPPER_VARS), 6.0), sharex=True)
    wavenumbers = np.arange(summaries[spec.baseline]["psd_obs"].shape[-1])
    # Шаг 1 и последний: у 12-шаговых носителей это (0, 11), у 6-шагового IAM4VP (0, 5).
    last_step = summaries[spec.baseline]["psd_obs"].shape[0] - 1
    for row, step in enumerate((0, last_step)):
        for col, var in enumerate(UPPER_VARS):
            channel = channels.index(f"{var}500")
            ax = axes[row, col]
            ax.loglog(
                wavenumbers[1:],
                summaries[spec.baseline]["psd_obs"][step, channel, 1:],
                color=INK,
                lw=2.0,
                label="ERA5 (истина)",
            )
            for arm in spec.psd_arms:
                ax.loglog(
                    wavenumbers[1:],
                    summaries[arm]["psd_pred"][step, channel, 1:],
                    color=spec.psd_colors[arm],
                    lw=1.3,
                    ls="--",
                    label=spec.labels[arm],
                )
            if row == 0:
                ax.set_title(f"{var}500", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"мощность, шаг {step + 1}", fontsize=9)
            if row == 1:
                ax.set_xlabel("зональное волновое число m", fontsize=9)
            ax.grid(True, which="both", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        "Спектр мощности по зональному волновому числу m (сферические гармоники "
        "на региональном кропе не определены)\nпадение прогноза ниже истины на больших m "
        "= модель сглаживает мелкий масштаб",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def write_table(deltas: np.ndarray, channels: list[str], spec: LadderSpec, dst: Path) -> None:
    """Markdown-таблица: дельта каждого арма к контролю по всем метрикам с CI."""
    upper = upper_channels(channels)
    base = spec.baseline_label
    lines = [
        f"{spec.table_title}\n",
        f"Δ% к {base}, {len(upper)} upper-каналов, среднее по 12 шагам. "
        "CI — парный бутстрап (1000, 95 %).",
        "Значимо ⇔ интервал не накрывает 0. RMSE/W1/bias: <0 лучше. ACC/CSI/FSS: >0 лучше.\n",
        "| арм | " + " | ".join(label for label, _ in METRICS.values()) + " |",
        "|---" * (1 + len(METRICS)) + "|",
    ]
    for arm in spec.arm_order:
        cells = []
        for metric in METRICS:
            mean = delta_grid(deltas, arm, metric, "delta")[:, upper].mean()
            low = delta_grid(deltas, arm, metric, "ci_low")[:, upper].mean()
            high = delta_grid(deltas, arm, metric, "ci_high")[:, upper].mean()
            star = "**" if (low > 0 or high < 0) else ""
            cells.append(f"{star}{mean:+.2f}{star} [{low:+.2f}, {high:+.2f}]")
        lines.append(f"| {spec.labels[arm]} | " + " | ".join(cells) + " |")
    dst.write_text("\n".join(lines) + "\n")


def write_metric_outputs(
    spec: LadderSpec, results_dir: Path, figures_dir: Path, heatmaps_dir: Path
) -> None:
    """Собрать все фигуры и таблицу лестницы.

    Args:
        spec: специфика эксперимента (армы, подписи, эпоха).
        results_dir: вход — ``metrics_<arm>.npz`` + ``paired_deltas.npz``; туда же
            пишется ``metrics_table.md``.
        figures_dir: куда класть ``fig_ci_forest.png``, ``fig_levels_*.png``, ``fig_psd.png``.
        heatmaps_dir: корень пер-армовых хитмапов ``<метрика>/<арм>.png``.

    Side effects: пишет PNG и markdown на диск, печатает сводку.
    """
    summaries, deltas, channels = load(results_dir, spec)
    render_forest(deltas, channels, spec, figures_dir / "fig_ci_forest.png")
    for metric in METRICS:
        render_levels(deltas, metric, spec, figures_dir / f"fig_levels_{metric}.png")
    render_psd(summaries, channels, spec, figures_dir / "fig_psd.png")
    write_table(deltas, channels, spec, results_dir / "metrics_table.md")

    written = 0
    for metric in METRICS:
        metric_dir = heatmaps_dir / metric
        metric_dir.mkdir(parents=True, exist_ok=True)
        for arm in spec.arm_order:
            slug = arm.removeprefix(spec.arm_filename_prefix)
            render_arm_metric_heatmap(deltas, arm, metric, spec, metric_dir / f"{slug}.png")
            written += 1
    print(  # noqa: T201
        f"[metrics-figures] {len(summaries)} армов → {2 + len(METRICS)} сводных фигур, "
        f"таблица, {written} пер-армовых хитмапов в {heatmaps_dir}/<метрика>/<арм>.png"
    )
