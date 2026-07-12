"""Фигуры 12-шагового rollout exp16 из npz-файлов ``rollout_eval.py``.

Вход: ``results/rollout/rollout_<arm>.npz`` (rmse_free/rmse_forced (12, 69),
channels, n_samples). Выход — три фигуры в стиле инференс-диагностики
предыдущего поколения абляции:

  * ``fig_rollout_ratio_r0.png`` — RMSE арма относительно R0 (÷R0, среднее по 13
    уровням) по шагам 1..12, панель на переменную, линия на арм (R0 = 1.0);
  * ``fig_rollout_delta_steps.png`` — средняя по всем 69 каналам Δ% к R0 по
    шагам, free-running vs teacher-forced;
  * ``fig_rollout_heatmap.png`` — Δ% к R0 (free-running) по
    [уровень давления × шаг × арм], панель на переменную.

Оформление зеркалит make_figures.py: Okabe-Ito, врезка-пояснение.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROLLOUT_DIR = HERE / "results" / "rollout"

# Реестр канонический: ключ независим от волны (t=6 ``abl16-<arm>-s0`` и
# t=12 ``abl16L-<arm>-t12-s0`` сводятся к одному стему ``canonical_arm``).
ARM_ORDER = (
    "r0-no-physics",
    "r1-legacy-hybrid",
    "r2a-a1-pre13",
    "r2-a2-pre13",
    "r3-a2-exp13",
    "r4-exp14",
    "r5-exp15",
    "r3a-no-diabatic",
    "r3q-diabatic-t-only",
)
ARM_LABELS = {
    "r0-no-physics": "R0 · без физики",
    "r1-legacy-hybrid": "R1 · легаси-hybrid",
    "r2a-a1-pre13": "R2a · A1 (до exp13)",
    "r2-a2-pre13": "R2 · A2 (до exp13)",
    "r3-a2-exp13": "R3 · A2 (exp13)",
    "r4-exp14": "R4 · +exp14",
    "r5-exp15": "R5 · +exp15",
    "r3a-no-diabatic": "R3a · R3 без Q_θ",
    "r3q-diabatic-t-only": "R3q · Q_θ только t",
}
ARM_COLORS = {
    "r0-no-physics": "#8a8a8a",
    "r1-legacy-hybrid": "#E69F00",
    "r2a-a1-pre13": "#56B4E9",
    "r2-a2-pre13": "#009E73",
    "r3-a2-exp13": "#0072B2",
    "r4-exp14": "#D55E00",
    "r5-exp15": "#CC79A7",
    "r3a-no-diabatic": "#000000",
    "r3q-diabatic-t-only": "#AA4499",
}
BASELINE = ARM_ORDER[0]


def canonical_arm(name: str) -> str:
    """Свести имя рана к волно-независимому стему (``abl16L-r0-…-t12-s0`` → ``r0-no-physics``).

    Args:
        name: имя эксперимента из npz (t=6 ``abl16-<arm>-s0`` или t=12
            ``abl16L-<arm>-t12-s0``).

    Returns:
        Канонический ключ арма (совпадает с ключами ``ARM_ORDER``).
    """
    stem = name
    for prefix in ("abl16L-", "abl16-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    for suffix in ("-t12-s0", "-s0"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


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
    """Прочитать все rollout_<arm>.npz в ``{canonical_arm: {rmse_free, ...}}``.

    Ключ — канонический (волно-независимый). ``native_horizon`` (граница окон)
    и ``checkpoint_epoch`` берутся из npz; для старых t=6-файлов без
    ``native_horizon`` он выводится как половина числа шагов (двухоконный режим).
    """
    runs: dict[str, dict] = {}
    for path in sorted(rollout_dir.glob("rollout_*.npz")):
        data = np.load(path, allow_pickle=False)
        n_steps = int(data["rmse_free"].shape[0])
        runs[canonical_arm(str(data["arm"]))] = {
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


def window_boundary(runs: dict[str, dict]) -> int | None:
    """Шаг границы окон (для пунктира) или ``None``, если прогноз одним окном.

    Граница = нативный горизонт, если он меньше полного числа шагов (t=6 → 6);
    для нативного одноконного прогноза (t=12 → 12 шагов из 12) границы нет.
    """
    base = runs[BASELINE]
    n_steps = base["rmse_free"].shape[0]
    horizon = base["native_horizon"]
    return horizon if 0 < horizon < n_steps else None


def epoch_label(runs: dict[str, dict]) -> str:
    """Подпись эпохи чекпоинта для заголовков (``эпоха N`` или ``эпоха ?``)."""
    epoch = runs[BASELINE]["checkpoint_epoch"]
    return f"эпоха {epoch + 1}" if epoch >= 0 else "эпоха ?"


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


def render_ratio_r0(runs: dict[str, dict], dst: Path) -> None:
    """Панель на переменную: RMSE арма относительно R0 (÷R0) по шагам rollout.

    Абсолютный RMSE в t/r/u/v почти не различает армы (спред 1–2 % тонет в росте
    ошибки с лидом), поэтому нормируем на арм без физики: ``RMSE_arm / RMSE_R0``.
    R0 — линия 1.0; автомасштаб отношения магнифицирует вклад каждого арма.
    """
    if BASELINE not in runs:
        return
    arms = [arm for arm in ARM_ORDER if arm in runs]
    n_steps = runs[BASELINE]["rmse_free"].shape[0]
    boundary = window_boundary(runs)
    epoch = epoch_label(runs)
    fig, axes = plt.subplots(1, len(UPPER_VARS), figsize=(3.6 * len(UPPER_VARS), 3.2))
    steps = np.arange(1, n_steps + 1)
    base = {
        var: level_matrix(runs[BASELINE]["rmse_free"], runs[BASELINE]["channels"], var)[0].mean(
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
                color=ARM_COLORS[arm],
                lw=1.7,
                marker="o",
                ms=3,
                label=ARM_LABELS[arm],
            )
        ax.axhline(1.0, color=MUTED, lw=0.9, ls="--")
        if boundary is not None:
            ax.axvline(boundary + 0.5, color=MUTED, lw=0.9, ls=":")
        ax.set_title(f"{var} [{VAR_UNITS[var]}]", fontsize=10)
        ax.set_xlabel("шаг прогноза t+N")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("RMSE / RMSE(R0)   (÷R0; <1 — лучше R0)")
    fig.suptitle(
        f"Вклад физики к R0 (÷R0) по шагам {n_steps}-шагового free-running rollout "
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
        "Что показано: RMSE каждого арма, делённый на RMSE R0 (÷R0, среднее по 13 уровням),\n"
        f"по шагам free-running rollout; {window_note}. Пунктир 1.0 = R0. Ниже — лучше R0.",
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def render_delta_steps(runs: dict[str, dict], dst: Path) -> None:
    """Два режима rollout: средняя по 69 каналам Δ% к R0 по шагам."""
    arms = [arm for arm in ARM_ORDER if arm in runs and arm != BASELINE]
    base = runs[BASELINE]
    n_steps = base["rmse_free"].shape[0]
    boundary = window_boundary(runs)
    epoch = epoch_label(runs)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), sharey=True)
    steps = np.arange(1, n_steps + 1)
    win = "окно 2" if boundary is not None else "нативно"
    modes = (
        ("rmse_free", f"Free-running ({win}: собственные прогнозы)"),
        ("rmse_forced", f"Teacher-forced ({win}: реальные кадры)"),
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
        if boundary is not None:
            ax.axvline(boundary + 0.5, color=MUTED, lw=0.9, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("шаг прогноза t+N")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("средняя Δ% RMSE к R0 (69 каналов)")
    mode_word = "скользящим окном" if boundary is not None else "нативным окном"
    fig.suptitle(
        f"{n_steps}-шаговый rollout {mode_word}: Δ% к R0 «без физики» (вал-2004, {epoch})",
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
    tf_note = (
        "teacher-forced изолирует по-оконную ошибку от накопления дрейфа (шаги 1–"
        f"{boundary} совпадают)."
        if boundary is not None
        else "teacher-forced подаёт в историю реальные кадры → изолирует дрейф (шаг 1 совпадает)."
    )
    add_caption(
        fig,
        "Что показано: Δ% lat-weighted RMSE к R0, усреднённая по всем 69 каналам, на каждом шаге\n"
        f"rollout. Ниже нуля — арм лучше R0. Free-running — реальный {n_steps}-шаговый прогноз;\n"
        f"{tf_note}",
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def render_heatmap(runs: dict[str, dict], dst: Path, clip: float = 12.0) -> None:
    """Δ% к R0 (free-running): [уровень × (арм × шаг)] панель на переменную."""
    arms = [arm for arm in ARM_ORDER if arm in runs and arm != BASELINE]
    base = runs[BASELINE]
    n_steps = base["rmse_free"].shape[0]
    boundary = window_boundary(runs)
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
        f"Δ% RMSE к R0 по уровням и шагам free-running rollout (вал-2004, {epoch}): "
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
        ncol=(len(arms) + 1) // 2,
    )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.012, pad=0.01)
    colorbar.set_label("Δ% RMSE к R0")
    block_note = (
        f"в блоке {n_steps} шагов free-running (пунктир — граница окон {boundary}→{boundary})"
        if boundary is not None
        else f"в блоке {n_steps} шагов нативного free-running прогноза"
    )
    add_caption(
        fig,
        "Что показано: Δ% lat-weighted RMSE к R0 на каждом [уровень давления × шаг rollout];\n"
        f"внутри панели блоки слева направо — армы лестницы (порядок легенды), {block_note}.\n"
        f"Синее — арм лучше R0 (шкала ±{clip:.0f} %).",
        y=-0.06,
    )
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Собрать три rollout-фигуры и JSON-индекс из указанного каталога npz.

    CLI: ``--rollout-dir`` (по умолч. results/rollout — волна t=6 USA) и
    ``--suffix`` (добавляется к именам PNG, напр. ``_t12``), чтобы разные
    волны не затирали фигуры друг друга.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", default=str(ROLLOUT_DIR), help="каталог rollout_*.npz")
    parser.add_argument("--suffix", default="", help="суффикс имён PNG (напр. _t12)")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    runs = load_runs(rollout_dir)
    if BASELINE not in runs:
        raise SystemExit(f"нет baseline {BASELINE} в {rollout_dir}")
    render_ratio_r0(runs, HERE / f"fig_rollout_ratio_r0{args.suffix}.png")
    render_delta_steps(runs, HERE / f"fig_rollout_delta_steps{args.suffix}.png")
    render_heatmap(runs, HERE / f"fig_rollout_heatmap{args.suffix}.png")
    index = {arm: {"n_samples": data["n_samples"]} for arm, data in sorted(runs.items())}
    (rollout_dir / "rollout_index.json").write_text(json.dumps(index, indent=2))
    print(f"[rollout-figures] {len(runs)} runs -> 3 figures{args.suffix}")  # noqa: T201


if __name__ == "__main__":
    main()
