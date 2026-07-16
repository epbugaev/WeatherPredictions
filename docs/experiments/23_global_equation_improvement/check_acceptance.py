"""Проверка критерия принятия вариантов эксперимента 15 по results-JSON.

Критерий: вариант принят, если целевая переменная улучшается относительно
базы (C_best) на ОБОИХ доменах во ВСЕ годы, а остальные переменные не
деградируют более чем на 1 %. Для вариантов с полями из данных
(S1/S2/S12/C15_full) критерий применяется к 2001–2002 (2000 — in-sample).

Запуск: python check_acceptance.py [results_dir]
"""

import json
import sys
from pathlib import Path

DOMS = ("usa", "globe")
YEARS = (2000, 2001, 2002)
ALL_VARS = ("u", "v", "t", "q", "z")
BASE = "C_best"
DATA_DRIVEN = {"S1_zm", "S2_map", "S12", "C15_full"}
TARGETS = {
    "W_plain": ("t",),
    "W_obrien": ("t", "z"),
    "T_hs": ("t",),
    "T_lh": ("t",),
    "Z_ps": ("z",),
    "NoW_t": ("t",),
    "NoW_tq": ("t", "q"),
    "NoW_all": ("t", "q", "u", "v"),
    "C15_now": ("t", "q"),
    # Клим-варианты: цель — переменные с содержательным клим-сигналом
    # (u-климатология почти нулевая — u оценивается допуском «остальных» 1 %).
    "S1_zm": ("v",),
    "S2_map": ("t", "q", "z"),
    "S12": ("v", "t", "q", "z"),
    "C15_full": ("v", "t", "q", "z"),
}


def main() -> None:
    """Печатает по каждому варианту: улучшение целевых, деградации, вердикт."""
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
    res = {}
    for dom in DOMS:
        for year in YEARS:
            path = results_dir / f"eq15_{dom}_{year}.json"
            if path.exists():
                res[(dom, year)] = json.loads(path.read_text())["residual_rel"]
    print(f"загружено {len(res)} JSON из {results_dir}")  # noqa: T201
    variants = set()
    for r in res.values():
        variants.update(r.keys())
    for name in sorted(variants - {BASE, "base13"}):
        years = tuple(y for y in YEARS if y != 2000) if name in DATA_DRIVEN else YEARS
        cells = [(d, y) for d in DOMS for y in years if (d, y) in res and name in res[(d, y)]]
        if not cells:
            continue
        targets = TARGETS.get(name, ALL_VARS)
        rows = []
        ok = True
        for var in ALL_VARS:
            deltas = [
                (res[c][name][var] - res[c][BASE][var]) / res[c][BASE][var] * 100 for c in cells
            ]
            worst = max(deltas)
            best = min(deltas)
            is_target = var in targets
            if is_target and worst >= 0.0:
                ok = False
            if not is_target and worst > 1.0:
                ok = False
            rows.append(f"    {var}: Δ {best:+.1f}%..{worst:+.1f}% {'target' if is_target else ''}")
        verdict = "ПРИНЯТ" if ok else "ОТКЛОНЁН"
        print(f"{name} [{'+'.join(map(str, years))}] → {verdict}")  # noqa: T201
        for row in rows:
            print(row)  # noqa: T201


if __name__ == "__main__":
    main()
