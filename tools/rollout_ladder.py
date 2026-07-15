"""Фигуры многошагового rollout физической «лестницы» (общее ядро для экспериментов).

Раньше эта логика жила целиком внутри
``docs/experiments/16_model_ablation_ladder/rollout_figures.py``. Эксперимент 20
(PI-PredRNNv2) строит те же фигуры из тех же npz (схема — см.
:mod:`tools.rollout_common`) — отличаются только имена армов, подписи, цвета и
правило канонизации имени рана, — поэтому ядро вынесено сюда, а скрипты
экспериментов стали тонкими обёртками (тот же приём, что у
:mod:`tools.ladder_figures` и :mod:`tools.metrics_ladder`).

Вход: ``<rollout_dir>/rollout_*.npz`` (``rmse_free``/``rmse_forced`` формы
``(steps, C)``, ``channels``, ``n_samples``, ``checkpoint_epoch``,
``native_horizon``, ``arm``). Выход (:func:`write_rollout_outputs`):

  * ``fig_rollout_ratio_<baseline>.png`` — RMSE арма относительно контроля
    (÷контроль, среднее по 13 уровням) по шагам, панель на переменную;
  * ``fig_rollout_delta_steps.png`` — средняя по всем 69 каналам Δ% к контролю
    по шагам, free-running vs teacher-forced;
  * ``fig_rollout_heatmap.png`` — Δ% к контролю (free-running) по
    [уровень давления × шаг × арм], панель на переменную;
  * ``fig_rollout_heatmap_<arm>.png`` / ``fig_rollout_levels_<arm>.png`` — то же
    на один арм (второй — с числами в ячейках);
  * ``<rollout_dir>/level_step_table.md`` и ``rollout_index.json``.

Оконная механика параметризуется не флагом, а данными: ``native_horizon`` из npz
меньше числа шагов (exp16 t=6: 12 шагов = два окна по 6) → рисуется граница окон;
равен ему (exp20 t=12: один нативный форвард) → :func:`window_boundary` даёт
``None``, и ни линии-шва, ни «окна 2» в подписях не появляется.

Оформление зеркалит остальные фигуры лестниц: Okabe-Ito, врезка-пояснение.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UPPER_VARS = ("z", "t", "r", "u", "v")
VAR_UNITS = {"z": "м²/с²", "t": "K", "r": "%", "u": "м/с", "v": "м/с"}
TABLE_STEPS = (1, 6, 12)  # шаги прогноза (1-индексация) в markdown-таблице по уровням
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


@dataclass(frozen=True)
class RolloutSpec:
    """Специфика лестницы: армы, подписи, цвета, правило канонизации имён ранов.

    Attributes:
        baseline: канонический ключ контрольного арма (без физики); обязан
            совпадать с ``arm_order[0]`` — от него считаются все дельты.
        baseline_label: короткая подпись контроля в осях, заголовках и имени
            ratio-фигуры («R0», «P0»).
        arm_order: армы в порядке лестницы, контроль первым (порядок линий,
            блоков хитмапа и легенды).
        labels: подпись на арм; должна содержать и ключ ``baseline``.
        colors: hex-цвет на арм (Okabe-Ito).
        heatmap_arms: армы пер-армовых хитмапов ``fig_rollout_heatmap_<arm>.png``.
        level_arms: армы фигур с числами ``fig_rollout_levels_<arm>.png`` и строк
            ``level_step_table.md``.
        run_prefixes: префиксы имени рана, срезаемые при канонизации (первый
            подошедший; ``abl16L-`` перед ``abl16-``).
        run_suffixes: суффиксы имени рана, срезаемые при канонизации (первый
            подошедший; ``-t12-s0`` перед ``-s0``).
        single_mode: у MIMO-носителя (SimVPv2) весь горизонт выдаётся одним
            форвардом, teacher-forcing не определён (``rmse_forced == rmse_free``).
            При ``True`` :func:`render_delta_steps` рисует одну панель (free),
            а не вырожденную пару free/TF. Авторегрессивные лестницы (exp16/exp20)
            оставляют ``False`` — обе панели содержательны.
        arm_filename_prefix: префикс, срезаемый из ИМЕНИ пер-армовых PNG
            (``fig_rollout_{heatmap,levels}_<арм>.png``) — чтобы имена были чистыми
            как у exp16 (``..._r3-a2-exp13.png``). Ключ арма не меняется. Пусто →
            имя = ключ (exp16/exp20).
    """

    baseline: str
    baseline_label: str
    arm_order: tuple[str, ...]
    labels: dict[str, str]
    colors: dict[str, str]
    heatmap_arms: tuple[str, ...]
    level_arms: tuple[str, ...]
    run_prefixes: tuple[str, ...]
    run_suffixes: tuple[str, ...]
    single_mode: bool = False
    arm_filename_prefix: str = ""

    def __post_init__(self) -> None:
        assert self.arm_order[0] == self.baseline, (
            f"контроль {self.baseline!r} обязан быть первым в arm_order — "
            "иначе дельты и легенда разойдутся"
        )

    def canonical(self, name: str) -> str:
        """Свести имя рана к ключу арма (``abl16L-r0-no-physics-t12-s0`` → ``r0-no-physics``).

        Ключ волно-независим: у exp16 t=6 (``abl16-<arm>-s0``) и t=12
        (``abl16L-<arm>-t12-s0``) сводятся к одному стему.

        Args:
            name: имя эксперимента из npz.

        Returns:
            Канонический ключ арма (совпадает с ключами ``arm_order``).
        """
        stem = name
        for prefix in self.run_prefixes:
            if stem.startswith(prefix):
                stem = stem[len(prefix) :]
                break
        for suffix in self.run_suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        return stem


def delta_percent(arm_rmse: np.ndarray, base_rmse: np.ndarray) -> np.ndarray:
    """Поэлементная Δ% RMSE арма к baseline (− = арм лучше).

    Args:
        arm_rmse: RMSE арма, форма ``(steps, C)``.
        base_rmse: RMSE baseline той же формы.

    Returns:
        ``(steps, C)`` в процентах.
    """
    return 100.0 * (arm_rmse - base_rmse) / base_rmse


def level_matrix(
    rmse: np.ndarray, channels: list[str], base_var: str
) -> tuple[np.ndarray, list[int]]:
    """Срез по-уровневых каналов переменной → матрица [уровень × шаг].

    Args:
        rmse: ``(steps, C)`` — RMSE или Δ%.
        channels: имена C каналов (``z500``, ``t2``, ...).
        base_var: базовая переменная (``z``/``t``/``r``/``u``/``v``);
            поверхностные каналы (``t2``, ``u10``...) не включаются.

    Returns:
        ``(матрица (levels, steps), уровни по возрастанию гПа)``.
    """
    pairs = []
    for idx, name in enumerate(channels):
        prefix = name.rstrip("0123456789")
        if prefix != base_var or name == base_var:
            continue
        level = int(name[len(prefix) :])
        # поверхностные t2/u10/v10 маскируются под уровневые: их «уровень»
        # не входит в сетку давления 50..1000 из 13 значений ≥ 50
        if level < 50:
            continue
        pairs.append((level, idx))
    pairs.sort()
    levels = [level for level, _ in pairs]
    matrix = np.stack([rmse[:, idx] for _, idx in pairs], axis=0)
    return matrix, levels


def mean_delta_over_channels(delta: np.ndarray) -> np.ndarray:
    """Средняя по каналам Δ% на каждом шаге: ``(steps, C)`` → ``(steps,)``."""
    return delta.mean(axis=1)


def load_runs(rollout_dir: Path, spec: RolloutSpec) -> dict[str, dict]:
    """Прочитать все ``rollout_*.npz`` в ``{canonical_arm: {rmse_free, ...}}``.

    Ключ — канонический (см. :meth:`RolloutSpec.canonical`). ``native_horizon``
    (граница окон) и ``checkpoint_epoch`` берутся из npz; для старых файлов без
    ``native_horizon`` он выводится как половина числа шагов (двухоконный режим).

    Args:
        rollout_dir: каталог с ``rollout_*.npz``.
        spec: спецификация лестницы (нужна для канонизации имён).

    Returns:
        ``{арм: {rmse_free, rmse_forced, channels, n_samples, native_horizon,
        checkpoint_epoch}}``.
    """
    runs: dict[str, dict] = {}
    for path in sorted(rollout_dir.glob("rollout_*.npz")):
        data = np.load(path, allow_pickle=False)
        n_steps = int(data["rmse_free"].shape[0])
        runs[spec.canonical(str(data["arm"]))] = {
            "rmse_free": data["rmse_free"],
            "rmse_forced": data["rmse_forced"],
            "channels": [str(name) for name in data["channels"]],
            "n_samples": int(data["n_samples"]),
            "native_horizon": int(data["native_horizon"])
            if "native_horizon" in data
            else n_steps // 2,
            "checkpoint_epoch": int(data["checkpoint_epoch"]) if "checkpoint_epoch" in data else -1,
        }
    return runs


def window_boundary(runs: dict[str, dict], baseline: str) -> int | None:
    """Шаг границы окон (для пунктира) или ``None``, если прогноз одним окном.

    Граница = нативный горизонт, если он меньше полного числа шагов (t=6 → 6);
    для нативного одноконного прогноза (t=12 → 12 шагов из 12) границы нет.

    Args:
        runs: словарь прогонов из :func:`load_runs`.
        baseline: канонический ключ контрольного арма.

    Returns:
        Шаг границы окон или ``None``.
    """
    base = runs[baseline]
    n_steps = base["rmse_free"].shape[0]
    horizon = base["native_horizon"]
    return horizon if 0 < horizon < n_steps else None


def epoch_label(runs: dict[str, dict]) -> str:
    """Подпись чекпоинтов для заголовков.

    Волна с общего тега эпохи → ``эпоха N``. При отборе по валидации (``best.pt``)
    эпохи армов разные, и подпись обязана показать диапазон: эпоха одного baseline
    соврала бы про остальные армы.
    """
    epochs = sorted(
        {run["checkpoint_epoch"] + 1 for run in runs.values() if run["checkpoint_epoch"] >= 0}
    )
    if not epochs:
        return "эпоха ?"
    if len(epochs) == 1:
        return f"эпоха {epochs[0]}"
    return f"лучший val, эпохи {epochs[0]}–{epochs[-1]}"


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


def render_ratio_baseline(runs: dict[str, dict], spec: RolloutSpec, dst: Path) -> None:
    """Панель на переменную: RMSE арма относительно контроля (÷контроль) по шагам rollout.

    Абсолютный RMSE в t/r/u/v почти не различает армы (спред 1–2 % тонет в росте
    ошибки с лидом), поэтому нормируем на арм без физики: ``RMSE_arm / RMSE_base``.
    Контроль — линия 1.0; автомасштаб отношения магнифицирует вклад каждого арма.

    Args:
        runs: словарь прогонов из :func:`load_runs`.
        spec: спецификация лестницы.
        dst: путь PNG.

    Side effects:
        Пишет PNG в ``dst``.
    """
    baseline, name = spec.baseline, spec.baseline_label
    arms = [arm for arm in spec.arm_order if arm in runs]
    n_steps = runs[baseline]["rmse_free"].shape[0]
    boundary = window_boundary(runs, baseline)
    epoch = epoch_label(runs)
    fig, axes = plt.subplots(1, len(UPPER_VARS), figsize=(3.6 * len(UPPER_VARS), 3.2))
    steps = np.arange(1, n_steps + 1)
    base = {
        var: level_matrix(runs[baseline]["rmse_free"], runs[baseline]["channels"], var)[0].mean(
            axis=0
        )
        for var in UPPER_VARS
    }
    for ax, var in zip(axes, UPPER_VARS, strict=True):
        for arm in arms:
            matrix, _ = level_matrix(runs[arm]["rmse_free"], runs[arm]["channels"], var)
            ax.plot(
                steps,
                matrix.mean(axis=0) / base[var],
                color=spec.colors[arm],
                lw=1.7,
                marker="o",
                ms=3,
                label=spec.labels[arm],
            )
        ax.axhline(1.0, color=MUTED, lw=0.9, ls="--")
        if boundary is not None:
            ax.axvline(boundary + 0.5, color=MUTED, lw=0.9, ls=":")
        ax.set_title(f"{var} [{VAR_UNITS[var]}]", fontsize=10)
        ax.set_xlabel("шаг прогноза t+N")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel(f"RMSE / RMSE({name})   (÷{name}; <1 — лучше {name})")
    fig.suptitle(
        f"Вклад физики к {name} (÷{name}) по шагам {n_steps}-шагового free-running rollout "
        f"(полная вал-2004, сид 0, {epoch})",
        fontsize=12,
        y=1.12,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(arms),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    window_note = (
        f"шаги {boundary + 1}–{n_steps} — второе окно (вход = собственные прогнозы, пунктир —\n"
        f"граница окон {boundary}→{boundary})"
        if boundary is not None
        else f"весь прогноз — одно нативное окно из {n_steps} шагов"
    )
    add_caption(
        fig,
        f"Что показано: RMSE каждого арма, делённый на RMSE {name} "
        f"(÷{name}, среднее по 13 уровням),\n"
        f"по шагам free-running rollout; {window_note}. Пунктир 1.0 = {name}. "
        f"Ниже — лучше {name}.",
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def render_delta_steps(runs: dict[str, dict], spec: RolloutSpec, dst: Path) -> None:
    """Средняя по 69 каналам Δ% к контролю по шагам rollout.

    Две панели (free-running и teacher-forced) для авторегрессивных лестниц; при
    ``spec.single_mode`` — одна панель (free), т.к. у MIMO teacher-forcing не
    определён (``rmse_forced == rmse_free``), и вторая панель была бы копией первой.

    Args:
        runs: словарь прогонов из :func:`load_runs`.
        spec: спецификация лестницы.
        dst: путь PNG.

    Side effects:
        Пишет PNG в ``dst``.
    """
    baseline, name = spec.baseline, spec.baseline_label
    arms = [arm for arm in spec.arm_order if arm in runs and arm != baseline]
    base = runs[baseline]
    n_steps = base["rmse_free"].shape[0]
    boundary = window_boundary(runs, baseline)
    epoch = epoch_label(runs)
    win = "окно 2" if boundary is not None else "нативно"
    if spec.single_mode:
        modes = (("rmse_free", f"Нативный форвард MIMO ({win}: весь горизонт за один вызов)"),)
    else:
        modes = (
            ("rmse_free", f"Free-running ({win}: собственные прогнозы)"),
            ("rmse_forced", f"Teacher-forced ({win}: реальные кадры)"),
        )
    width = 11.2 if len(modes) == 2 else 6.2
    fig, axes = plt.subplots(1, len(modes), figsize=(width, 3.8), sharey=True, squeeze=False)
    axes = axes[0]
    steps = np.arange(1, n_steps + 1)
    for ax, (key, title) in zip(axes, modes, strict=True):
        for arm in arms:
            delta = delta_percent(runs[arm][key], base[key])
            ax.plot(
                steps,
                mean_delta_over_channels(delta),
                color=spec.colors[arm],
                lw=1.7,
                marker="o",
                ms=3,
                label=spec.labels[arm],
            )
        ax.axhline(0.0, color=INK, lw=0.9, ls="--")
        if boundary is not None:
            ax.axvline(boundary + 0.5, color=MUTED, lw=0.9, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("шаг прогноза t+N")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(f"средняя Δ% RMSE к {name} (69 каналов)")
    mode_word = "скользящим окном" if boundary is not None else "нативным окном"
    fig.suptitle(
        f"{n_steps}-шаговый rollout {mode_word}: Δ% к {name} «без физики» (вал-2004, {epoch})",
        fontsize=12,
        y=1.14,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=(len(arms) + 1) // 2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if spec.single_mode:
        tf_note = (
            "teacher-forcing не показан: MIMO выдаёт весь горизонт одним форвардом, "
            "без подачи промежуточных кадров, поэтому free ≡ TF."
        )
    elif boundary is not None:
        tf_note = (
            "teacher-forced изолирует по-оконную ошибку от накопления дрейфа "
            f"(шаги 1–{boundary} совпадают)."
        )
    else:
        tf_note = (
            "teacher-forced подаёт в историю реальные кадры → изолирует дрейф (шаг 1 совпадает)."
        )
    add_caption(
        fig,
        f"Что показано: Δ% lat-weighted RMSE к {name}, усреднённая по всем 69 каналам, "
        f"на каждом шаге\n"
        f"rollout. Ниже нуля — арм лучше {name}. Free-running — реальный "
        f"{n_steps}-шаговый прогноз;\n"
        f"{tf_note}",
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def render_heatmap(runs: dict[str, dict], spec: RolloutSpec, dst: Path, clip: float = 12.0) -> None:
    """Δ% к контролю (free-running): [уровень × (арм × шаг)] панель на переменную.

    Args:
        runs: словарь прогонов из :func:`load_runs`.
        spec: спецификация лестницы.
        dst: путь PNG.
        clip: предел шкалы цвета в процентах.

    Side effects:
        Пишет PNG в ``dst``.
    """
    baseline, name = spec.baseline, spec.baseline_label
    arms = [arm for arm in spec.arm_order if arm in runs and arm != baseline]
    base = runs[baseline]
    n_steps = base["rmse_free"].shape[0]
    boundary = window_boundary(runs, baseline)
    epoch = epoch_label(runs)
    mid = boundary - 1 if boundary is not None else n_steps // 2 - 1
    tick_offsets = sorted({0, mid, n_steps - 1})
    tick_labels = [str(off + 1) for off in tick_offsets]
    fig, axes = plt.subplots(1, len(UPPER_VARS), figsize=(3.4 * len(UPPER_VARS), 4.2), sharey=True)
    image = None
    for ax, var in zip(axes, UPPER_VARS, strict=True):
        base_matrix, levels = level_matrix(base["rmse_free"], base["channels"], var)
        blocks = []
        for arm in arms:
            arm_matrix, _ = level_matrix(runs[arm]["rmse_free"], runs[arm]["channels"], var)
            blocks.append(100.0 * (arm_matrix - base_matrix) / base_matrix)
        stacked = np.concatenate(blocks, axis=1)  # (levels, arms*steps)
        image = ax.imshow(
            stacked,
            cmap="RdBu_r",
            vmin=-clip,
            vmax=clip,
            aspect="auto",
            interpolation="nearest",
        )
        for arm_idx in range(1, len(arms)):
            ax.axvline(arm_idx * n_steps - 0.5, color=INK, lw=1.2)
        if boundary is not None:
            for arm_idx in range(len(arms)):
                ax.axvline(arm_idx * n_steps + boundary - 0.5, color=MUTED, lw=0.7, ls=":")
        ax.set_title(f"{var}", fontsize=10)
        ax.set_xticks(
            [arm_idx * n_steps + off for arm_idx in range(len(arms)) for off in tick_offsets],
            tick_labels * len(arms),
            fontsize=6,
        )
        ax.set_xlabel("шаг (блок = арм)")
        ax.grid(False)
    axes[0].set_yticks(range(len(levels)), [str(level) for level in levels], fontsize=7)
    axes[0].set_ylabel("уровень давления, гПа")
    fig.suptitle(
        f"Δ% RMSE к {name} по уровням и шагам free-running rollout (вал-2004, {epoch}): "
        f"синее = лучше {name}, красное = хуже",
        fontsize=12,
        y=1.06,
    )
    handles = [
        plt.Line2D([0], [0], color=spec.colors[arm], lw=4, label=spec.labels[arm]) for arm in arms
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=(len(arms) + 1) // 2,
    )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.012, pad=0.01)
    colorbar.set_label(f"Δ% RMSE к {name}")
    block_note = (
        f"в блоке {n_steps} шагов free-running (пунктир — граница окон {boundary}→{boundary})"
        if boundary is not None
        else f"в блоке {n_steps} шагов нативного free-running прогноза"
    )
    add_caption(
        fig,
        f"Что показано: Δ% lat-weighted RMSE к {name} на каждом "
        f"[уровень давления × шаг rollout];\n"
        f"внутри панели блоки слева направо — армы лестницы (порядок легенды), {block_note}.\n"
        f"Синее — арм лучше {name} (шкала ±{clip:.0f} %).",
        y=-0.06,
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def level_step_delta(
    runs: dict[str, dict], arm: str, var: str, baseline: str
) -> tuple[np.ndarray, list[int]]:
    """Δ% RMSE арма к контролю (free-running) на сетке [уровень давления × шаг].

    Args:
        runs: словарь прогонов из :func:`load_runs`.
        arm: канонический ключ арма.
        var: переменная из ``UPPER_VARS``.
        baseline: канонический ключ контрольного арма.

    Returns:
        ``(delta (13, steps), levels)`` — Δ% (<0 — арм лучше контроля) и уровни в
        гПа по возрастанию (порядок строк).
    """
    arm_matrix, levels = level_matrix(runs[arm]["rmse_free"], runs[arm]["channels"], var)
    base_matrix, _ = level_matrix(runs[baseline]["rmse_free"], runs[baseline]["channels"], var)
    return delta_percent(arm_matrix, base_matrix), levels


def render_arm_heatmap(
    runs: dict[str, dict], arm: str, spec: RolloutSpec, dst: Path, clip: float = 12.0
) -> None:
    """Один арм в стиле :func:`render_heatmap`: Δ% к контролю, [уровень × шаг], без чисел.

    Args:
        runs: словарь прогонов из :func:`load_runs`.
        arm: канонический ключ арма.
        spec: спецификация лестницы.
        dst: путь PNG.
        clip: предел шкалы цвета в процентах.

    Side effects:
        Пишет PNG в ``dst``.
    """
    baseline, name = spec.baseline, spec.baseline_label
    n_steps = runs[arm]["rmse_free"].shape[0]
    boundary = window_boundary(runs, baseline)
    epoch = epoch_label(runs)
    mid = boundary - 1 if boundary is not None else n_steps // 2 - 1
    tick_offsets = sorted({0, mid, n_steps - 1})
    tick_labels = [str(off + 1) for off in tick_offsets]
    fig, axes = plt.subplots(1, len(UPPER_VARS), figsize=(3.4 * len(UPPER_VARS), 4.2), sharey=True)
    image = None
    for ax, var in zip(axes, UPPER_VARS, strict=True):
        delta, levels = level_step_delta(runs, arm, var, baseline)
        image = ax.imshow(
            delta, cmap="RdBu_r", vmin=-clip, vmax=clip, aspect="auto", interpolation="nearest"
        )
        if boundary is not None:
            ax.axvline(boundary - 0.5, color=MUTED, lw=0.7, ls=":")
        ax.set_title(f"{var}", fontsize=10)
        ax.set_xticks(tick_offsets, tick_labels, fontsize=6)
        ax.set_xlabel("шаг прогноза")
        ax.grid(False)
    axes[0].set_yticks(range(len(levels)), [str(level) for level in levels], fontsize=7)
    axes[0].set_ylabel("уровень давления, гПа")
    fig.suptitle(
        f"{spec.labels[arm]}: Δ% RMSE к {name} по уровням и шагам free-running rollout "
        f"(вал-2004, {epoch}); синее = лучше {name}",
        fontsize=12,
        y=1.06,
    )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.012, pad=0.01)
    colorbar.set_label(f"Δ% RMSE к {name}")
    window_note = (
        f"пунктир — граница окон {boundary}→{boundary}"
        if boundary is not None
        else f"нативный {n_steps}-шаговый прогноз"
    )
    add_caption(
        fig,
        f"Что показано: Δ% lat-weighted RMSE к {name} на каждом "
        f"[уровень давления × шаг rollout];\n"
        f"{window_note}. Синее — арм лучше {name} (шкала ±{clip:.0f} %).",
        y=-0.06,
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def render_arm_level_heatmap(
    runs: dict[str, dict], arm: str, spec: RolloutSpec, dst: Path, clip: float = 12.0
) -> None:
    """Хитмап одного арма: Δ% к контролю по [уровень × шаг], панель на переменную.

    В отличие от :func:`render_heatmap` (все армы в одной панели блоками) — фигура
    на один арм, с числами в ячейках. Подпись несёт эпохи чекпоинтов: сравнение
    честно, только если эпоха арма совпадает с эпохой контроля.

    Args:
        runs: словарь прогонов из :func:`load_runs`.
        arm: канонический ключ арма.
        spec: спецификация лестницы.
        dst: путь PNG.
        clip: предел шкалы цвета в процентах.

    Side effects:
        Пишет PNG в ``dst``.
    """
    baseline, name = spec.baseline, spec.baseline_label
    n_steps = runs[arm]["rmse_free"].shape[0]
    arm_epoch, base_epoch = (
        runs[arm]["checkpoint_epoch"] + 1,
        runs[baseline]["checkpoint_epoch"] + 1,
    )
    fig, axes = plt.subplots(1, len(UPPER_VARS), figsize=(4.0 * len(UPPER_VARS), 4.4), sharey=True)
    image = None
    for ax, var in zip(axes, UPPER_VARS, strict=True):
        delta, levels = level_step_delta(runs, arm, var, baseline)
        image = ax.imshow(
            delta, cmap="RdBu_r", vmin=-clip, vmax=clip, aspect="auto", interpolation="nearest"
        )
        for row in range(delta.shape[0]):
            for col in range(delta.shape[1]):
                value = delta[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:+.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if abs(value) > 0.62 * clip else INK,
                )
        ax.set_title(var, fontsize=11)
        ax.set_xticks(range(n_steps), [str(step + 1) for step in range(n_steps)], fontsize=6)
        ax.set_xlabel("шаг прогноза")
        ax.grid(False)
    axes[0].set_yticks(range(len(levels)), [str(level) for level in levels], fontsize=7)
    axes[0].set_ylabel("уровень давления, гПа")
    matched = f"эпоха совпадает с {name} — сравнение без конфаунда"
    mismatched = f"эпоха НЕ совпадает с {name} ({base_epoch}) — сравнение конфаундировано глубиной"
    fig.suptitle(
        f"{spec.labels[arm]} против {name}: Δ% RMSE по уровням и шагам free-running "
        f"(вал-2004, эпоха {arm_epoch}); синее = физика лучше",
        fontsize=12,
        y=1.02,
    )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.012, pad=0.01)
    colorbar.set_label(f"Δ% RMSE к {name}")
    add_caption(
        fig,
        f"Что показано: Δ% lat-weighted RMSE арма к {name} на каждом [уровень × шаг]\n"
        f"{n_steps}-шагового free-running прогноза. Синее — физика лучше {name}, красное — хуже "
        f"(шкала ±{clip:.0f} %).\n"
        f"Чекпоинт арма — эпоха {arm_epoch}, {name} — эпоха {base_epoch}: "
        f"{matched if arm_epoch == base_epoch else mismatched}.",
        y=-0.08,
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def write_level_step_table(
    runs: dict[str, dict], arms: list[str], spec: RolloutSpec, dst: Path
) -> None:
    """Markdown-таблица Δ% к контролю по [уровень × шаг ``TABLE_STEPS``] на каждый арм.

    Args:
        runs: словарь прогонов из :func:`load_runs`.
        arms: армы-разделы таблицы.
        spec: спецификация лестницы.
        dst: путь .md.

    Side effects:
        Пишет markdown в ``dst``.
    """
    baseline, name = spec.baseline, spec.baseline_label
    lines: list[str] = [
        f"# Δ% RMSE к {name} по уровням давления (free-running, {epoch_label(runs)})\n",
        f"Отрицательное — физика лучше {name}. Шаги прогноза: "
        + ", ".join(str(step) for step in TABLE_STEPS)
        + ".\n",
        "**Конфаунд:** сравнивать с "
        f"{name} без поправки можно только арм с ТОЙ ЖЕ эпохой "
        f"чекпоинта ({name} — эпоха {runs[baseline]['checkpoint_epoch'] + 1}).\n",
    ]
    for arm in arms:
        lines.append(f"\n## {spec.labels[arm]} — эпоха {runs[arm]['checkpoint_epoch'] + 1}\n")
        header = "| гПа | " + " | ".join(
            f"{var} ш{step}" for var in UPPER_VARS for step in TABLE_STEPS
        )
        lines.append(header + " |")
        lines.append("|---" * (1 + len(UPPER_VARS) * len(TABLE_STEPS)) + "|")
        deltas = {var: level_step_delta(runs, arm, var, baseline) for var in UPPER_VARS}
        levels = deltas[UPPER_VARS[0]][1]
        for row, level in enumerate(levels):
            cells = [
                f"{deltas[var][0][row, step - 1]:+.1f}"
                for var in UPPER_VARS
                for step in TABLE_STEPS
            ]
            lines.append(f"| {level} | " + " | ".join(cells) + " |")
    dst.write_text("\n".join(lines) + "\n")


def write_rollout_outputs(
    spec: RolloutSpec, rollout_dir: Path, figures_dir: Path, suffix: str = ""
) -> None:
    """Собрать rollout-фигуры, пер-армовые хитмапы, таблицу по уровням и JSON-индекс.

    Args:
        spec: спецификация лестницы (армы, подписи, цвета, канонизация имён).
        rollout_dir: каталог ``rollout_*.npz``; туда же пишутся
            ``level_step_table.md`` и ``rollout_index.json``.
        figures_dir: каталог PNG (корень эксперимента).
        suffix: суффикс имён PNG (напр. ``_t12``) — чтобы разные волны одной
            лестницы не затирали фигуры друг друга.

    Raises:
        SystemExit: если в ``rollout_dir`` нет контрольного арма — дельты не от чего
            считать.

    Side effects:
        Пишет PNG, ``level_step_table.md``, ``rollout_index.json``; печатает сводку.
    """
    runs = load_runs(rollout_dir, spec)
    if spec.baseline not in runs:
        raise SystemExit(f"нет baseline {spec.baseline} в {rollout_dir}")
    ratio_name = f"fig_rollout_ratio_{spec.baseline_label.lower()}{suffix}.png"
    render_ratio_baseline(runs, spec, figures_dir / ratio_name)
    render_delta_steps(runs, spec, figures_dir / f"fig_rollout_delta_steps{suffix}.png")
    render_heatmap(runs, spec, figures_dir / f"fig_rollout_heatmap{suffix}.png")
    heatmap_arms = [arm for arm in spec.heatmap_arms if arm in runs]
    level_arms = [arm for arm in spec.level_arms if arm in runs]
    for arm in heatmap_arms:
        slug = arm.removeprefix(spec.arm_filename_prefix)
        render_arm_heatmap(runs, arm, spec, figures_dir / f"fig_rollout_heatmap_{slug}{suffix}.png")
    for arm in level_arms:
        slug = arm.removeprefix(spec.arm_filename_prefix)
        render_arm_level_heatmap(
            runs, arm, spec, figures_dir / f"fig_rollout_levels_{slug}{suffix}.png"
        )
    write_level_step_table(runs, level_arms, spec, rollout_dir / "level_step_table.md")
    index = {arm: {"n_samples": data["n_samples"]} for arm, data in sorted(runs.items())}
    (rollout_dir / "rollout_index.json").write_text(json.dumps(index, indent=2))
    print(  # noqa: T201
        f"[rollout-figures] {len(runs)} runs -> 3 figures{suffix} "
        f"+ {len(heatmap_arms)} пер-армовых heatmap + {len(level_arms)} levels-хитмапов "
        f"+ level_step_table.md"
    )
