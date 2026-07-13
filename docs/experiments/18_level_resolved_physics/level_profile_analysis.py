"""Эксперимент 18: связать per-level скилл обученных армов с физикой уравнений.

Наблюдаемое: rollout экспа 16 (``rollout_eval.py``) пишет lat-weighted RMSE
КАЖДОГО из 69 каналов = 13 уровней давления × {z,t,r,u,v} + 4 приземных, по
шагам прогноза. Отсюда — вертикальный профиль скилла каждого арма.

Объясняющее (на реальном ERA5): per-level относительная невязка уравнения
``residual_rel[level]`` из ``level_diagnostics.py`` (USA 2004) либо из готового
exp15 (USA 2000). Тезис: физ-арм улучшает уровень там, где невязка ядра <1
(тенденция ловит сигнал), и вредит/нейтрален там, где невязка ≫1 (тенденция
неверна) или ≈1 (физика инертна).

Скрипт:
  1. грузит 9 npz армов, строит per-level RMSE (среднее по шагам и шаг 1);
  2. дельта к R0 (%) по уровням: ``(RMSE_arm − RMSE_R0)/RMSE_R0·100``;
  3. корреляцию дельты с невязкой по 13 уровням (якоря R1↔base13, R4↔C_best,
     R5↔C15_now);
  4. фигуры: вертикальные профили дельты по переменным; scatter дельта×невязка;
     профили доминирующих членов;
  5. markdown-таблицы и JSON-сводку (вход для научного разбора).

Запуск:
    python level_profile_analysis.py \
        --rollout-dir results/rollout_t12 \
        --diag-json results/eq18_usa_2004.json \
        --out-dir figures
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PRESSURE_HPA: list[int] = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
UPPER_VARS: tuple[str, ...] = ("z", "t", "r", "u", "v")
VAR_BASE: dict[str, int] = {"z": 4, "t": 17, "r": 30, "u": 43, "v": 56}
VAR_LABEL: dict[str, str] = {
    "z": "z (геопотенциал)",
    "t": "T (температура)",
    "r": "r (отн. влажность)",
    "u": "u (зон. ветер)",
    "v": "v (мерид. ветер)",
}
# канонический арм → якорь-конфигурация ядра (для сопоставления с невязкой).
ARM_TO_ANCHOR: dict[str, str] = {
    "r1-legacy-hybrid": "base13",
    "r4-exp14": "C_best",
    "r5-exp15": "C15_now",
}
ARM_ORDER: tuple[str, ...] = (
    "r0-no-physics",
    "r1-legacy-hybrid",
    "r2a-a1-pre13",
    "r2-a2-pre13",
    "r3-a2-exp13",
    "r3a-no-diabatic",
    "r3q-diabatic-t-only",
    "r4-exp14",
    "r5-exp15",
)
ARM_LABEL: dict[str, str] = {
    "r0-no-physics": "R0 без физики",
    "r1-legacy-hybrid": "R1 легаси",
    "r2a-a1-pre13": "R2a A1 pre13",
    "r2-a2-pre13": "R2 A2 pre13",
    "r3-a2-exp13": "R3 +exp13",
    "r3a-no-diabatic": "R3a без диаб.",
    "r3q-diabatic-t-only": "R3q диаб. t",
    "r4-exp14": "R4 +exp14",
    "r5-exp15": "R5 +exp15",
}
ARM_COLOR: dict[str, str] = {
    "r0-no-physics": "#888888",
    "r1-legacy-hybrid": "#CC3311",
    "r2a-a1-pre13": "#EE7733",
    "r2-a2-pre13": "#EECC66",
    "r3-a2-exp13": "#009988",
    "r3a-no-diabatic": "#44BB99",
    "r3q-diabatic-t-only": "#AA4499",
    "r4-exp14": "#0077BB",
    "r5-exp15": "#332288",
}


def canonical_arm(name: str) -> str:
    """``abl16L-r5-exp15-t12-s0`` / ``abl16-r5-exp15-s0`` → ``r5-exp15``."""
    stem = name
    for prefix in ("abl16L-", "abl16-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    for suffix in ("-t12-s0", "-s0"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def load_runs(rollout_dir: Path) -> dict[str, dict]:
    """Считать все npz армов, ключ — канонический арм.

    Args:
        rollout_dir: папка с ``rollout_<arm>.npz``.

    Returns:
        ``{арм: {rmse_free (S,69), rmse_forced (S,69), channels, epoch}}``.
    """
    runs: dict[str, dict] = {}
    for npz_path in sorted(rollout_dir.glob("rollout_*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        arm = canonical_arm(str(data["arm"]))
        runs[arm] = {
            "rmse_free": data["rmse_free"],
            "rmse_forced": data["rmse_forced"],
            "channels": [str(c) for c in data["channels"]],
            "epoch": int(data["checkpoint_epoch"]),
        }
    return runs


def epoch_label(runs: dict[str, dict]) -> str:
    """Подпись чекпоинтов для заголовков и сводки.

    Волна с общего тега эпохи → ``эпоха N``. При отборе по валидации (``best.pt``)
    эпохи армов разные, и подпись обязана показать диапазон: эпоха произвольного
    арма (порядок словаря!) соврала бы про остальные.
    """
    epochs = sorted({run["epoch"] + 1 for run in runs.values() if run["epoch"] >= 0})
    if not epochs:
        return "эпоха ?"
    if len(epochs) == 1:
        return f"эпоха {epochs[0]}"
    return f"лучший val, эпохи {epochs[0]}–{epochs[-1]}"


def level_profile(rmse: np.ndarray, var: str, reduce_steps: str) -> np.ndarray:
    """13-уровневый профиль RMSE переменной ``var``.

    Args:
        rmse: ``(steps, 69)`` физ-единицы.
        var: одна из ``z,t,r,u,v``.
        reduce_steps: ``"mean"`` (среднее по шагам) или ``"first"`` (шаг 1).

    Returns:
        ``(13,)`` RMSE по уровням давления (порядок ``PRESSURE_HPA``).
    """
    base = VAR_BASE[var]
    block = rmse[:, base : base + 13]
    return block.mean(axis=0) if reduce_steps == "mean" else block[0]


def delta_percent(arm_prof: np.ndarray, base_prof: np.ndarray) -> np.ndarray:
    """Относительная дельта скилла к R0 в %: ``(arm − base)/base·100`` (<0 — лучше)."""
    return (arm_prof - base_prof) / (base_prof + 1e-30) * 100.0


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Корреляция Пирсона двух профилей (nan-безопасно)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def load_diagnostics(diag_json: Path) -> dict:
    """Считать per-level невязку из exp18 (``residual_rel_by_level``) либо exp15 (``by_level``)."""
    raw = json.loads(diag_json.read_text())
    if "residual_rel_by_level" in raw:
        return {
            "residual": raw["residual_rel_by_level"],
            "terms": raw.get("term_abs_over_obs_by_level", {}),
            "year": raw["meta"]["year"],
        }
    by_level = raw["by_level"]
    residual = {name: vals for name, vals in by_level.items() if name != "pressure_hpa"}
    return {"residual": residual, "terms": {}, "year": raw["meta"]["year"]}


def render_skill_profiles(runs: dict, out_path: Path, reduce_steps: str) -> None:
    """Вертикальные профили дельты скилла к R0 по 5 переменным (панель на переменную)."""
    if "r0-no-physics" not in runs:
        return
    y = np.arange(len(PRESSURE_HPA))
    fig, axes = plt.subplots(1, len(UPPER_VARS), figsize=(19, 6), sharey=True)
    base = {
        v: level_profile(runs["r0-no-physics"]["rmse_free"], v, reduce_steps) for v in UPPER_VARS
    }
    for ax, var in zip(axes, UPPER_VARS, strict=True):
        for arm in ARM_ORDER:
            if arm == "r0-no-physics" or arm not in runs:
                continue
            prof = level_profile(runs[arm]["rmse_free"], var, reduce_steps)
            ax.plot(
                delta_percent(prof, base[var]),
                y,
                marker="o",
                ms=3,
                lw=1.4,
                color=ARM_COLOR[arm],
                label=ARM_LABEL[arm],
            )
        ax.axvline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(VAR_LABEL[var])
        ax.set_xlabel("дельта RMSE к R0, %")
        ax.grid(alpha=0.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(PRESSURE_HPA)
    axes[0].set_ylabel("уровень давления, гПа")
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=9, frameon=False)
    reduce_ru = "среднее по 12 шагам" if reduce_steps == "mean" else "шаг 1 (короткий лид)"
    fig.suptitle(
        f"Вертикальный профиль вклада физики ({epoch_label(runs)}, {reduce_ru}); "
        f"<0 — физика улучшает",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_residual_scatter(runs: dict, diag: dict, out_path: Path) -> None:
    """Scatter «дельта скилла × невязка ядра» по 13 уровням для 3 якорей."""
    if "r0-no-physics" not in runs:
        return
    residual = diag["residual"]
    anchors = [
        (a, ARM_TO_ANCHOR[a]) for a in ARM_TO_ANCHOR if a in runs and ARM_TO_ANCHOR[a] in residual
    ]
    fig, axes = plt.subplots(1, len(UPPER_VARS), figsize=(19, 4.6))
    base = {v: level_profile(runs["r0-no-physics"]["rmse_free"], v, "mean") for v in UPPER_VARS}
    for ax, var in zip(axes, UPPER_VARS, strict=True):
        res_key = "q" if var == "r" else var  # невязка влажности считается по q
        for arm, anchor in anchors:
            if res_key not in residual[anchor]:
                continue
            prof = level_profile(runs[arm]["rmse_free"], var, "mean")
            d = delta_percent(prof, base[var])
            res = np.array(residual[anchor][res_key])
            ax.scatter(res, d, s=26, color=ARM_COLOR[arm], label=ARM_LABEL[arm], alpha=0.8)
        ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(1.0, color="gray", lw=0.8, ls=":", alpha=0.7)
        ax.set_xscale("log")
        ax.set_title(VAR_LABEL[var])
        ax.set_xlabel("невязка ядра residual_rel")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("дельта RMSE к R0, %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(
        f"Скилл физики против невязки уравнения на данных (ERA5 USA {diag['year']}); "
        f"невязка<1 (слева от пунктира) → физика ловит сигнал",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def build_summary(runs: dict, diag: dict) -> dict:
    """Числовая сводка (per-level дельты, невязки, корреляции) — вход разбора."""
    if "r0-no-physics" not in runs:
        return {}
    residual = diag["residual"]
    base_mean = {
        v: level_profile(runs["r0-no-physics"]["rmse_free"], v, "mean") for v in UPPER_VARS
    }
    base_first = {
        v: level_profile(runs["r0-no-physics"]["rmse_free"], v, "first") for v in UPPER_VARS
    }
    summary: dict = {
        "epochs": sorted({run["epoch"] + 1 for run in runs.values() if run["epoch"] >= 0}),
        "epoch_label": epoch_label(runs),
        "pressure_hpa": PRESSURE_HPA,
        "R0_rmse_mean": {v: base_mean[v].tolist() for v in UPPER_VARS},
        "R0_rmse_first": {v: base_first[v].tolist() for v in UPPER_VARS},
        "delta_pct_mean": {},
        "delta_pct_first": {},
        "residual_rel": residual,
        "residual_year": diag["year"],
        "corr_delta_vs_residual": {},
    }
    for arm in ARM_ORDER:
        if arm == "r0-no-physics" or arm not in runs:
            continue
        summary["delta_pct_mean"][arm] = {
            v: delta_percent(
                level_profile(runs[arm]["rmse_free"], v, "mean"), base_mean[v]
            ).tolist()
            for v in UPPER_VARS
        }
        summary["delta_pct_first"][arm] = {
            v: delta_percent(
                level_profile(runs[arm]["rmse_free"], v, "first"), base_first[v]
            ).tolist()
            for v in UPPER_VARS
        }
    for arm, anchor in ARM_TO_ANCHOR.items():
        if arm not in runs or anchor not in residual:
            continue
        summary["corr_delta_vs_residual"][arm] = {}
        for var in UPPER_VARS:
            res_key = "q" if var == "r" else var
            if res_key not in residual[anchor]:
                continue
            d = np.array(summary["delta_pct_mean"][arm][var])
            res = np.array(residual[anchor][res_key])
            summary["corr_delta_vs_residual"][arm][var] = pearson(res, d)
    return summary


def write_markdown_table(summary: dict, out_path: Path) -> None:
    """Компактная markdown-таблица дельт по уровням (step-mean) для README."""
    lines: list[str] = []
    lines.append(
        f"### Дельта RMSE к R0 по уровням (%, среднее по 12 шагам, {summary['epoch_label']})\n"
    )
    lines.append("Отрицательное = физика улучшает. Уровни в гПа.\n")
    for var in UPPER_VARS:
        lines.append(f"\n**{VAR_LABEL[var]}**\n")
        header = "| арм | " + " | ".join(str(p) for p in PRESSURE_HPA) + " |"
        sep = "|" + "---|" * (len(PRESSURE_HPA) + 1)
        lines.append(header)
        lines.append(sep)
        for arm in ARM_ORDER:
            if arm == "r0-no-physics" or arm not in summary["delta_pct_mean"]:
                continue
            vals = summary["delta_pct_mean"][arm][var]
            row = f"| {ARM_LABEL[arm]} | " + " | ".join(f"{x:+.1f}" for x in vals) + " |"
            lines.append(row)
    out_path.write_text("\n".join(lines))


def dominant_term(terms: dict, anchor: str, res_key: str, level_idx: int) -> str:
    """Член с наибольшей магнитудой на уровне ``level_idx`` для якоря/переменной."""
    if anchor not in terms or res_key not in terms[anchor]:
        return "—"
    per_term = {tn: vals[level_idx] for tn, vals in terms[anchor][res_key].items()}
    if not per_term:
        return "—"
    name = max(per_term, key=lambda k: abs(per_term[k]))
    return f"{name} {per_term[name]:.1f}"


def write_mechanism_digest(summary: dict, diag: dict, out_path: Path) -> None:
    """Совмещённая таблица: Δскилл армов × невязка ядра × доминирующий член.

    По одной таблице на переменную; строки — 13 уровней. Главный числовой
    артефакт эксперимента и вход для механистического разбора.
    """
    terms = diag.get("terms", {})
    residual = summary["residual_rel"]
    show_arms = ["r1-legacy-hybrid", "r3-a2-exp13", "r4-exp14", "r5-exp15"]
    lines: list[str] = [
        f"# Механистическая сводка exp18 ({summary['epoch_label']}, "
        f"невязка ERA5 USA {summary['residual_year']})\n",
        "Δ — дельта RMSE к R0 (%, среднее по 12 шагам; <0 — физика лучше). "
        "resid — невязка уравнения ядра residual_rel (<1 — ловит сигнал, ≫1 — шум). "
        "top-член — доминирующий член ПЧ у C15_now (|член|/|obs_t|).\n",
    ]
    for var in UPPER_VARS:
        res_key = "q" if var == "r" else var
        r0 = summary["R0_rmse_mean"][var]
        lines.append(f"\n## {VAR_LABEL[var]}\n")
        head = (
            "| гПа | R0 RMSE | "
            + " | ".join(f"Δ{ARM_LABEL[a].split()[0]}" for a in show_arms)
            + " | resid R1 | resid R5 | top-член R5 |"
        )
        lines.append(head)
        lines.append("|" + "---|" * (len(show_arms) + 4))
        for i, hpa in enumerate(PRESSURE_HPA):
            deltas = []
            for a in show_arms:
                if a in summary["delta_pct_mean"]:
                    deltas.append(f"{summary['delta_pct_mean'][a][var][i]:+.1f}")
                else:
                    deltas.append("—")
            res_r1 = residual.get("base13", {}).get(res_key, [float("nan")] * 13)[i]
            res_r5 = residual.get("C15_now", {}).get(res_key, [float("nan")] * 13)[i]
            top = dominant_term(terms, "C15_now", res_key, i)
            row = (
                f"| {hpa} | {r0[i]:.2f} | "
                + " | ".join(deltas)
                + f" | {res_r1:.1f} | {res_r5:.2f} | {top} |"
            )
            lines.append(row)
    out_path.write_text("\n".join(lines))


def main() -> None:
    """CLI: грузит rollout+диагностику, пишет фигуры, JSON-сводку, md-таблицу."""
    args = parse_args()
    rollout_dir = Path(args.rollout_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(rollout_dir)
    print(f"[exp18] загружено армов: {sorted(runs)}")
    diag = load_diagnostics(Path(args.diag_json))

    render_skill_profiles(runs, out_dir / "fig_level_skill_mean.png", "mean")
    render_skill_profiles(runs, out_dir / "fig_level_skill_first.png", "first")
    render_residual_scatter(runs, diag, out_dir / "fig_residual_vs_skill.png")

    summary = build_summary(runs, diag)
    (results_dir / "level_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    write_markdown_table(summary, results_dir / "level_delta_table.md")
    write_mechanism_digest(summary, diag, results_dir / "mechanism_digest.md")
    print("[exp18] корреляция дельта×невязка (step-mean):")
    for arm, corrs in summary.get("corr_delta_vs_residual", {}).items():
        print(f"  {arm}: " + "  ".join(f"{v}={c:+.2f}" for v, c in corrs.items()))
    print(f"[exp18] фигуры → {out_dir}; сводка → {results_dir}")


def parse_args() -> Namespace:
    """CLI-аргументы: папка rollout, JSON диагностики, папка фигур."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True, help="папка с rollout_<arm>.npz")
    parser.add_argument(
        "--diag-json", required=True, help="exp18 eq18_*.json либо exp15 eq15_*.json"
    )
    parser.add_argument("--out-dir", default="figures", help="папка для фигур")
    return parser.parse_args()


if __name__ == "__main__":
    main()
