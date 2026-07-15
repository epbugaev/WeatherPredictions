"""Склейка парных дельт под ДВА контроля «без физики» → один npz для фигур.

Зачем это только в exp21. У IAM4VP (exp16) и PredRNNv2 (exp20) авторегрессивная
подача встроена — контроль «без физики» ровно один (R0/P0), и вся лестница честно
меряется к нему. У SimVPv2 подача — свободный рычаг (`physics_coupling`), и «без
физики» существует в ДВУХ вариантах:

* ``S0``  — batched без физики (нативная MIMO-подача: residual-корректору идёт
  собственный сырой прогноз кадра t−1);
* ``S0c`` — chained без физики (корректору идёт уже скорректированный кадр t−1).

Одна лишь связка `chained` двигает скилл на порядок сильнее любой физики (§8.5,
−14.7 % на плато). Поэтому мерить физику к общему S0 нечестно: у chained-арма в
дельту к S0 подмешан гигантский вклад связки, и лестница физики тонет. Чтобы форест
показывал ТОЛЬКО вклад физики (как у exp16), каждый арм сравнивается с контролем
СВОЕЙ связки: batched-армы — с S0, S3c — с S0c.

Механически: ``paired_deltas.py`` прогоняется дважды (``--baseline`` = S0 и = S0c),
оба раза с ``--seed 0`` и на одних 727 сэмплах, поэтому наборы бутстрап-индексов
идентичны и парная структура сохраняется через склейку. Здесь мы берём поля
batched-армов из прогона-к-S0, поля S3c — из прогона-к-S0c, а сам S0c из фореста
выпадает (он теперь контроль, а не арм — как R0 у exp16).

Запуск (локально, после двух прогонов `paired_deltas.py`):
    python merge_physics_baselines.py \
        --vs-s0  results/paired_deltas_vs_s0.npz \
        --vs-s0c results/paired_deltas_vs_s0c.npz \
        --out    results/paired_deltas.npz
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

# Batched-армы физической лестницы — их дельта берётся из прогона-к-S0.
BATCHED_ARMS = (
    "exp21L-s1-legacy-hybrid",
    "exp21L-s3a-no-diabatic",
    "exp21L-s3-a2-exp13",
    "exp21L-s4-exp14",
    "exp21L-s5-exp15",
)
# Chained-арм с физикой — его дельта берётся из прогона-к-S0c (изолирует физику).
CHAINED_PHYSICS_ARM = "exp21L-s3c-a2-exp13-chained"
# Первичный контроль: его summary несёт каналы и PSD-истину для фигур (load()).
PRIMARY_BASELINE = "exp21L-s0-no-physics"


def _arm_of(field: str) -> str:
    """Ключ арма из имени поля ``<арм>__<метрика>__<key>`` (скаляры → пустая строка)."""
    return field.split("__", 1)[0] if "__" in field else ""


def merge(vs_s0: Path, vs_s0c: Path, out: Path) -> None:
    """Собрать физически-изолированный npz дельт из двух прогонов с разными контролями.

    Args:
        vs_s0: парные дельты к S0 (``paired_deltas.py --baseline exp21L-s0-no-physics``).
        vs_s0c: парные дельты к S0c (``--baseline exp21L-s0c-no-physics-chained``).
        out: куда писать склейку — её читает ``metrics_figures.py``.

    Side effects: пишет ``out`` (npz).
    """
    a = np.load(vs_s0, allow_pickle=False)
    b = np.load(vs_s0c, allow_pickle=False)
    merged = {f: a[f] for f in a.files if _arm_of(f) in BATCHED_ARMS}
    merged |= {f: b[f] for f in b.files if _arm_of(f) == CHAINED_PHYSICS_ARM}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **merged, n_bootstrap=a["n_bootstrap"], baseline=PRIMARY_BASELINE)
    arms = sorted({_arm_of(f) for f in merged})
    print(f"[merge] {len(arms)} армов → {out}: {', '.join(arms)}")  # noqa: T201


def main() -> None:
    """CLI: два npz дельт (к S0 и к S0c) → один склеенный npz для фигур."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--vs-s0", type=Path, required=True)
    parser.add_argument("--vs-s0c", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    merge(args.vs_s0, args.vs_s0c, args.out)


if __name__ == "__main__":
    main()
