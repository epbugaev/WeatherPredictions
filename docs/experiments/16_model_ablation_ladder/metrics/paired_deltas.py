"""Парные бутстрап-дельты каждого арма к R0 по всем метрикам → компактный npz.

Все армы прогоняются по **одним и тем же** 727 валидационным сэмплам, поэтому
неопределённость их разницы считается парным ресэмплом: один набор индексов на оба
арма. Общая изменчивость («трудный день» тяжёл для всех) при этом сокращается, и
интервал разницы получается в разы уже независимых CI. По независимым CI судить,
отличаются ли армы между собой, **нельзя** — они переоценивают неопределённость.

Читает по-сэмпльные npz (`metrics_eval.py --out-per-sample`), пишет одну сводку с
дельтами и их CI. По-сэмпльные файлы весят десятки МБ на арм и остаются на кластере.

Запуск (кластер, CPU):
    python paired_deltas.py --per-sample-dir ~/abl16_metrics_e500_raw \
        --out ~/abl16_metrics_e500/paired_deltas.npz
"""

from __future__ import annotations

import importlib.util
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp16_metrics_lib", HERE / "metrics_lib.py")
ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ml)

BASELINE = "r0-no-physics"
# Метрики, где меньше = лучше, считаем как есть; ACC/CSI/FSS — «больше = лучше»,
# знак дельты у них читается наоборот (в фигурах это учитывается явно).
MEAN_METRICS = ("rmse", "acc", "bias", "w1", "std_pred")


def canonical_arm(name: str) -> str:
    """``abl16L-r3-a2-exp13-t12-s0`` → ``r3-a2-exp13``."""
    stem = name.removeprefix("abl16L-").removeprefix("abl16-")
    return stem.removesuffix("-t12-s0").removesuffix("-s0")


def load_per_sample(directory: Path) -> dict[str, dict[str, np.ndarray]]:
    """Прочитать по-сэмпльные npz всех армов в ``{арм: {метрика: (S, ...)}}``."""
    runs: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(directory.glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        arm = canonical_arm(path.stem.replace("metrics_", "").replace("_per_sample", ""))
        runs[arm] = {key: data[key] for key in data.files}
    return runs


def compute_deltas(
    runs: dict[str, dict[str, np.ndarray]], args: Namespace
) -> dict[str, np.ndarray]:
    """Парные дельты к R0 (в %) и их бутстрап-CI для каждой метрики и арма."""
    base = runs[BASELINE]
    out: dict[str, np.ndarray] = {}
    for arm, values in sorted(runs.items()):
        if arm == BASELINE:
            continue
        for metric in MEAN_METRICS:
            mean, std, low, high = ml.bootstrap_paired_delta_ci(
                values[metric], base[metric], args.bootstrap, seed=args.seed
            )
            out |= {
                f"{arm}__{metric}__delta": mean,
                f"{arm}__{metric}__std": std,
                f"{arm}__{metric}__ci_low": low,
                f"{arm}__{metric}__ci_high": high,
            }
        # CSI и FSS — отношения сумм: парную дельту берём от восстановленных
        # по-сэмпльных значений скилла, чтобы ресэмпл шёл по тем же индексам.
        csi_arm = values["csi_tp"] / (values["csi_tp"] + values["csi_fp"] + values["csi_fn"])
        csi_base = base["csi_tp"] / (base["csi_tp"] + base["csi_fp"] + base["csi_fn"])
        fss_arm = 1.0 - values["fss_num"] / np.where(
            values["fss_den"] > 0, values["fss_den"], np.nan
        )
        fss_base = 1.0 - base["fss_num"] / np.where(base["fss_den"] > 0, base["fss_den"], np.nan)
        for metric, (arm_values, base_values) in (
            ("csi", (csi_arm, csi_base)),
            ("fss", (fss_arm, fss_base)),
        ):
            mean, std, low, high = ml.bootstrap_paired_delta_ci(
                arm_values, base_values, args.bootstrap, seed=args.seed
            )
            out |= {
                f"{arm}__{metric}__delta": mean,
                f"{arm}__{metric}__std": std,
                f"{arm}__{metric}__ci_low": low,
                f"{arm}__{metric}__ci_high": high,
            }
        print(f"[paired] {arm}: готово")  # noqa: T201
    return out


def main() -> None:
    """CLI: каталог по-сэмпльных npz → npz с парными дельтами и CI."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--per-sample-dir", required=True, help="каталог *_per_sample.npz")
    parser.add_argument("--out", required=True, help="выходной npz с дельтами")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs = load_per_sample(Path(args.per_sample_dir))
    assert BASELINE in runs, f"нет baseline {BASELINE} в {args.per_sample_dir}"
    print(f"[paired] армов: {len(runs)}, сэмплов: {runs[BASELINE]['rmse'].shape[0]}")  # noqa: T201

    deltas = compute_deltas(runs, args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **deltas, n_bootstrap=args.bootstrap, baseline=BASELINE)
    print(f"[paired] записано {out_path}")  # noqa: T201


if __name__ == "__main__":
    main()
