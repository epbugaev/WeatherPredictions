"""Фигуры 12-шагового rollout exp16 из npz-файлов ``rollout_eval.py``.

Вход: ``results/rollout/rollout_<arm>.npz`` (rmse_free/rmse_forced (12, 69),
channels, n_samples). Выход — три фигуры в стиле инференс-диагностики
предыдущего поколения абляции:

  * ``fig_rollout_abs_rmse.png`` — абсолютный lat-weighted RMSE (физединицы)
    по шагам 1..12, панель на переменную (среднее по 13 уровням), линия на арм;
  * ``fig_rollout_delta_steps.png`` — средняя по всем 69 каналам Δ% к R0 по
    шагам, free-running vs teacher-forced;
  * ``fig_rollout_heatmap.png`` — Δ% к R0 (free-running) по
    [уровень давления × шаг × арм], панель на переменную.

Оформление зеркалит make_figures.py: Okabe-Ito, врезка-пояснение.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROLLOUT_DIR = HERE / "results" / "rollout"

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
BASELINE = ARM_ORDER[0]
UPPER_VARS = ("z", "t", "r", "u", "v")
VAR_UNITS = {"z": "м²/с²", "t": "K", "r": "%", "u": "м/с", "v": "м/с"}
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


def load_runs(rollout_dir: Path) -> dict[str, dict]:
    """Прочитать все rollout_<arm>.npz в ``{arm: {rmse_free, rmse_forced, ...}}``."""
    runs: dict[str, dict] = {}
    for path in sorted(rollout_dir.glob("rollout_*.npz")):
        data = np.load(path, allow_pickle=False)
        runs[str(data["arm"])] = {
            "rmse_free": data["rmse_free"],
            "rmse_forced": data["rmse_forced"],
            "channels": [str(name) for name in data["channels"]],
            "n_samples": int(data["n_samples"]),
        }
    return runs


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


def render_abs_rmse(runs: dict[str, dict], dst: Path) -> None:
    """Панель на переменную: абсолютный RMSE (среднее по уровням) по шагам."""
    arms = [arm for arm in ARM_ORDER if arm in runs]
    fig, axes = plt.subplots(1, len(UPPER_VARS), figsize=(3.6 * len(UPPER_VARS), 3.2))
    steps = np.arange(1, 13)
    for ax, var in zip(axes, UPPER_VARS, strict=True):
        for arm in arms:
            matrix, _ = level_matrix(runs[arm]["rmse_free"], runs[arm]["channels"], var)
            ax.plot(
                steps,
                matrix.mean(axis=0),
                color=ARM_COLORS[arm],
                lw=1.7,
                marker="o",
                ms=3,
                label=ARM_LABELS[arm],
            )
        ax.axvline(6.5, color=MUTED, lw=0.9, ls=":")
        ax.set_title(f"{var} [{VAR_UNITS[var]}]", fontsize=10)
        ax.set_xlabel("шаг прогноза t+N")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("lat-weighted RMSE (физединицы)")
    fig.suptitle(
        "Рост ошибки на 12-шаговом free-running rollout (полная вал-2004, сид 0, эпоха 18)",
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
        ncol=7,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    add_caption(
        fig,
        "Что показано: абсолютный lat-weighted RMSE (среднее по 13 уровням давления) по шагам\n"
        "free-running rollout; шаги 7–12 — второе окно, вход = собственные прогнозы (пунктир —\n"
        "граница окон 6→6). Все армы — чекпоинты одной эпохи (18). Ниже — лучше.",
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def render_delta_steps(runs: dict[str, dict], dst: Path) -> None:
    """Два режима rollout: средняя по 69 каналам Δ% к R0 по шагам."""
    arms = [arm for arm in ARM_ORDER if arm in runs and arm != BASELINE]
    base = runs[BASELINE]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), sharey=True)
    steps = np.arange(1, 13)
    modes = (
        ("rmse_free", "Free-running (окно 2 из собственных прогнозов)"),
        ("rmse_forced", "Teacher-forced (окно 2 из реальных кадров)"),
    )
    for ax, (key, title) in zip(axes, modes, strict=True):
        for arm in arms:
            delta = delta_percent(runs[arm][key], base[key])
            ax.plot(
                steps,
                mean_delta_over_channels(delta),
                color=ARM_COLORS[arm],
                lw=1.7,
                marker="o",
                ms=3,
                label=ARM_LABELS[arm],
            )
        ax.axhline(0.0, color=INK, lw=0.9, ls="--")
        ax.axvline(6.5, color=MUTED, lw=0.9, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("шаг прогноза t+N")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("средняя Δ% RMSE к R0 (69 каналов)")
    fig.suptitle(
        "12-шаговый rollout скользящим окном: Δ% к R0 «без физики» (вал-2004, эпоха 18)",
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
        ncol=6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    add_caption(
        fig,
        "Что показано: Δ% lat-weighted RMSE к R0, усреднённая по всем 69 каналам, на каждом шаге\n"
        "rollout. Ниже нуля — арм лучше R0. Free-running — реальный 12-шаговый прогноз;\n"
        "teacher-forced изолирует по-оконную ошибку от накопления дрейфа (шаги 1–6 совпадают).",
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def render_heatmap(runs: dict[str, dict], dst: Path, clip: float = 12.0) -> None:
    """Δ% к R0 (free-running): [уровень × (арм × шаг)] панель на переменную."""
    arms = [arm for arm in ARM_ORDER if arm in runs and arm != BASELINE]
    base = runs[BASELINE]
    n_steps = base["rmse_free"].shape[0]
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
        for arm_idx in range(len(arms)):
            ax.axvline(arm_idx * n_steps + 5.5, color=MUTED, lw=0.7, ls=":")
        ax.set_title(f"{var}", fontsize=10)
        ax.set_xticks(
            [arm_idx * n_steps + offset for arm_idx in range(len(arms)) for offset in (0, 5, 11)],
            ["1", "6", "12"] * len(arms),
            fontsize=6,
        )
        ax.set_xlabel("шаг (блок = арм)")
        ax.grid(False)
    axes[0].set_yticks(range(len(levels)), [str(level) for level in levels], fontsize=7)
    axes[0].set_ylabel("уровень давления, гПа")
    fig.suptitle(
        "Δ% RMSE к R0 по уровням и шагам free-running rollout (вал-2004, эпоха 18): "
        "синее = лучше R0, красное = хуже",
        fontsize=12,
        y=1.06,
    )
    handles = [
        plt.Line2D([0], [0], color=ARM_COLORS[arm], lw=4, label=ARM_LABELS[arm]) for arm in arms
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=6,
    )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.012, pad=0.01)
    colorbar.set_label("Δ% RMSE к R0")
    add_caption(
        fig,
        "Что показано: Δ% lat-weighted RMSE к R0 на каждом [уровень давления × шаг rollout];\n"
        "внутри панели блоки слева направо — армы лестницы (порядок легенды), в блоке 12 шагов\n"
        "free-running (пунктир — граница окон 6→6). Синее — арм лучше R0 (шкала ±"
        f"{clip:.0f} %).",
        y=-0.06,
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Собрать три rollout-фигуры и краткий JSON-индекс из results/rollout/."""
    runs = load_runs(ROLLOUT_DIR)
    if BASELINE not in runs:
        raise SystemExit(f"нет baseline {BASELINE} в {ROLLOUT_DIR}")
    render_abs_rmse(runs, HERE / "fig_rollout_abs_rmse.png")
    render_delta_steps(runs, HERE / "fig_rollout_delta_steps.png")
    render_heatmap(runs, HERE / "fig_rollout_heatmap.png")
    index = {arm: {"n_samples": data["n_samples"]} for arm, data in sorted(runs.items())}
    (ROLLOUT_DIR / "rollout_index.json").write_text(json.dumps(index, indent=2))
    print(f"[rollout-figures] {len(runs)} runs -> 3 figures")  # noqa: T201


if __name__ == "__main__":
    main()
