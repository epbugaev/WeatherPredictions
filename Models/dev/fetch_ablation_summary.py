"""Сводка кластерного ablation-sweep из Comet → готовые строки для
experiments/E*.md и README.

Тянет последние эксперименты, отбирает ablation-методы (`*_E0..E5*`) и якорь
fixC, печатает по каждому метрики ровно в той форме, что в Ek.md-таблицах:
weighted_rmse/{z500,t850,u500,v500,q700} + acc/z500 + frac_ic_blown_up
на h∈{6,24,48}, плюс Comet URL.

Использование (локально, читает .env):
    .venv/bin/python Models/dev/fetch_ablation_summary.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = REPO_ROOT / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from comet_ml.api import API  # noqa: E402

API_KEY = os.environ["COMET_API_KEY"]
WORKSPACE = os.environ.get("COMET_WORKSPACE", "buzaev-fedor")
PROJECT = os.environ.get("COMET_PROJECT_NAME", "weatherpredictions")

HOURS = (6, 24, 48)
METRICS = (
    "weighted_rmse/z/500hPa",
    "weighted_rmse/t/850hPa",
    "weighted_rmse/u/500hPa",
    "weighted_rmse/v/500hPa",
    "weighted_rmse/q/700hPa",
    "acc/z/500hPa",
    "frac_ic_blown_up",
)
# Метод абляции: ..._E0 / ..._E1+... ; либо якорь fixC.
ABL_RE = re.compile(r"_E[0-5](\+|_|$)|fix.?C|fixmode.?C|mode.?C", re.IGNORECASE)


def _at(ms: list, h: int) -> float:
    for m in sorted(ms, key=lambda x: int(x["step"] or 0)):
        if int(m["step"] or 0) == h:
            v = m["metricValue"]
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def main() -> None:
    api = API(api_key=API_KEY)
    exps = api.get_experiments(workspace=WORKSPACE, project_name=PROJECT)
    exps = sorted(exps, key=lambda e: e.start_server_timestamp or 0, reverse=True)[:30]

    seen: set[str] = set()
    picked = []
    for e in exps:
        others = e.get_others_summary("method")
        method = others[0] if others else (e.get_name() or e.id)
        if not ABL_RE.search(method) or method in seen:
            continue
        seen.add(method)
        picked.append((method, e))

    if not picked:
        print("[!] ablation-методы не найдены в последних 30 экспериментах")
        return

    print(f"=== Ablation summary ({WORKSPACE}/{PROJECT}) ===\n")
    for method, e in sorted(picked, key=lambda x: x[0]):
        cache = {mn: (e.get_metrics(mn) or []) for mn in METRICS}
        print(f"## {method}")
        print(f"  url: https://www.comet.com/{WORKSPACE}/{PROJECT}/{e.id}")
        hdr = "  | метрика | " + " | ".join(f"h={h}" for h in HOURS) + " |"
        print(hdr)
        for mn in METRICS:
            vals = " | ".join(f"{_at(cache[mn], h):.3e}" for h in HOURS)
            print(f"  | {mn} | {vals} |")
        print()


if __name__ == "__main__":
    main()
