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
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp16_metrics_lib", HERE / "metrics_lib.py")
ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ml)

BASELINE = "r0-no-physics"
# Режим сравнения у каждой метрики свой — иначе получаются бессмысленные числа:
#   * RELATIVE (%)  — положительные величины без нуля: RMSE, W1, std поля;
#   * ABSOLUTE (пункты ×100) — ОГРАНИЧЕННЫЕ оценки: ACC, CSI, FSS. Относительная
#     дельта у них взрывается (у R0 CSI бывает ровно 0 на p99 → деление на ноль);
#   * BIAS — знакопеременная величина около нуля: относительное изменение не
#     интерпретируется. Сравниваем |bias|, нормированный на σ наблюдённого поля,
#     и берём АБСОЛЮТНУЮ разницу (в % от σ).
RELATIVE_METRICS = ("rmse", "w1", "std_pred")
ABSOLUTE_METRICS = ("acc",)


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


def _skill_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Отношение с NaN там, где знаменателя нет (сэмпл без событий), а не с нулём."""
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def compute_deltas(
    runs: dict[str, dict[str, np.ndarray]], args: Namespace
) -> dict[str, np.ndarray]:
    """Парные дельты к R0 и их бутстрап-CI. Режим сравнения — свой у каждой метрики."""
    base = runs[BASELINE]
    out: dict[str, np.ndarray] = {}
    for arm, values in sorted(runs.items()):
        if arm == BASELINE:
            continue

        # Ограниченные оценки восстанавливаем по-сэмпльно, чтобы ресэмпл шёл
        # по одним и тем же индексам, что и у остальных метрик.
        csi = {
            "arm": _skill_ratio(
                values["csi_tp"], values["csi_tp"] + values["csi_fp"] + values["csi_fn"]
            ),
            "base": _skill_ratio(base["csi_tp"], base["csi_tp"] + base["csi_fp"] + base["csi_fn"]),
        }
        fss = {
            "arm": 1.0 - _skill_ratio(values["fss_num"], values["fss_den"]),
            "base": 1.0 - _skill_ratio(base["fss_num"], base["fss_den"]),
        }

        statistics: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        for metric in RELATIVE_METRICS:
            statistics[metric] = _relative(values[metric], base[metric])
        for metric in ABSOLUTE_METRICS:
            statistics[metric] = _absolute(values[metric], base[metric])
        statistics["csi"] = _absolute(csi["arm"], csi["base"])
        statistics["fss"] = _absolute(fss["arm"], fss["base"])
        statistics["bias"] = _bias_over_sigma(values["bias"], base["bias"], base["std_obs"])
        # Агрегаты (mCSI = среднее по порогам; FSS с окном 3) — самостоятельные метрики:
        # усреднять НАДО до бутстрапа. Иначе CI агрегата пришлось бы «усреднять» из
        # интервалов слагаемых, а это не CI. FSS_NEIGHBOURHOOD_INDEX=1 → окно 3 ячейки.
        statistics["mcsi"] = _absolute(
            np.nanmean(csi["arm"], axis=-1), np.nanmean(csi["base"], axis=-1)
        )
        statistics["fss3"] = _absolute(
            np.nanmean(fss["arm"][..., 1], axis=-1), np.nanmean(fss["base"][..., 1], axis=-1)
        )

        n_samples = base["rmse"].shape[0]
        for metric, statistic in statistics.items():
            mean, std, low, high = ml.bootstrap_paired_statistic_ci(
                statistic, n_samples, args.bootstrap, seed=args.seed
            )
            out |= {
                f"{arm}__{metric}__delta": mean,
                f"{arm}__{metric}__std": std,
                f"{arm}__{metric}__ci_low": low,
                f"{arm}__{metric}__ci_high": high,
            }
        print(f"[paired] {arm}: готово")  # noqa: T201
    return out


def _relative(arm: np.ndarray, base: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Относительная дельта в % (для положительных величин: RMSE, W1, std)."""

    def statistic(idx: np.ndarray) -> np.ndarray:
        return 100.0 * (np.nanmean(arm[idx], axis=0) / np.nanmean(base[idx], axis=0) - 1.0)

    return statistic


def _absolute(arm: np.ndarray, base: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Абсолютная разница ×100 «в пунктах» (для ограниченных оценок: ACC, CSI, FSS)."""

    def statistic(idx: np.ndarray) -> np.ndarray:
        return 100.0 * (np.nanmean(arm[idx], axis=0) - np.nanmean(base[idx], axis=0))

    return statistic


def _bias_over_sigma(
    arm: np.ndarray, base: np.ndarray, sigma: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Изменение |bias| в % от σ наблюдённого поля (знакопеременную величину так и надо)."""

    def statistic(idx: np.ndarray) -> np.ndarray:
        scale = np.nanmean(sigma[idx], axis=0)
        arm_bias = np.abs(np.nanmean(arm[idx], axis=0))
        base_bias = np.abs(np.nanmean(base[idx], axis=0))
        return 100.0 * (arm_bias - base_bias) / scale

    return statistic


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
