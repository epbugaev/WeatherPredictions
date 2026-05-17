"""Скачать метрики последних N экспериментов в Comet и проверить
что разные методы дают разные значения (не идентичные NaN-baseline).

Использование:
    .venv/bin/python Models/dev/fetch_comet_metrics.py
"""

from __future__ import annotations

import collections
import os
import sys
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

KEY_METRICS = (
    "weighted_rmse/u/500hPa",
    "weighted_rmse/z/500hPa",
    "weighted_rmse/t/500hPa",
    "weighted_rmse/q/700hPa",
    "acc/u/500hPa",
    "acc/z/500hPa",
    "frac_ic_blown_up",
)


def main() -> None:
    api = API(api_key=API_KEY)
    exps = api.get_experiments(workspace=WORKSPACE, project_name=PROJECT)
    print(f"[init] total experiments in {WORKSPACE}/{PROJECT}: {len(exps)}")

    # filter recent (latest 20) with non-trivial duration
    exps_sorted = sorted(exps, key=lambda e: e.start_server_timestamp or 0, reverse=True)[:20]
    rows = []
    for e in exps_sorted:
        name = e.get_name() or e.id
        others = e.get_others_summary("method")  # list[str]
        method = others[0] if others else "?"
        # get final-step values for each KEY metric
        finals: dict[str, float] = {}
        for metric_name in KEY_METRICS:
            ms = e.get_metrics(metric_name) or []
            if not ms:
                continue
            ms_by_step = sorted(ms, key=lambda m: int(m["step"] or 0))
            for h in (0, 1, 6, 24, 48):
                for m in ms_by_step:
                    if int(m["step"] or 0) == h:
                        try:
                            finals[f"{metric_name}@h{h}"] = float(m["metricValue"])
                        except (TypeError, ValueError):
                            finals[f"{metric_name}@h{h}"] = float("nan")
                        break
        rows.append((method, name, e.id, finals))

    # Group by method, print h=1 and h=48 comparison
    print("\n=== Per-method snapshot (latest experiment per method_name) ===")
    by_method: dict[str, dict] = {}
    for method, name, eid, finals in rows:
        if method not in by_method:
            by_method[method] = {"name": name, "id": eid, "finals": finals}

    header = (
        f"{'method':<40s}"
        + "".join(f"{'wrmse_u500@h1':>14s}{'wrmse_z500@h1':>14s}{'acc_u500@h1':>13s}{'wrmse_u500@h48':>15s}{'acc_u500@h48':>13s}".split())
    )
    # Simpler print
    print(f"\n{'method':<35s}{'h=1 wrmse(u500)':>18s}{'h=1 wrmse(z500)':>18s}{'h=1 acc(u500)':>16s}{'h=48 wrmse(u500)':>20s}{'h=48 acc(u500)':>16s}")
    for method, info in by_method.items():
        f = info["finals"]
        wu1 = f.get("weighted_rmse/u/500hPa@h1", float("nan"))
        wz1 = f.get("weighted_rmse/z/500hPa@h1", float("nan"))
        au1 = f.get("acc/u/500hPa@h1", float("nan"))
        wu48 = f.get("weighted_rmse/u/500hPa@h48", float("nan"))
        au48 = f.get("acc/u/500hPa@h48", float("nan"))
        print(f"{method:<35s}{wu1:>18.3e}{wz1:>18.3e}{au1:>16.3e}{wu48:>20.3e}{au48:>16.3e}")

    # Cross-check: at h=1, do at least 2 finite values differ?
    print("\n=== Identical-pair check at h=1 ===")
    h1_vals = {m: info["finals"].get("weighted_rmse/u/500hPa@h1") for m, info in by_method.items()}
    finite = {m: v for m, v in h1_vals.items() if v is not None and v == v and not (v != v)}
    print(f"  methods with finite wrmse(u@500)@h1: {len(finite)}/{len(by_method)}")
    vals = list(finite.values())
    if len(set(round(v, 4) for v in vals)) == 1 and len(vals) > 1:
        print(f"  !!! BUG: all {len(vals)} methods have identical wrmse(u@500)@h1 = {vals[0]}")
    else:
        print(f"  OK: methods are distinguishable ({len(set(round(v, 4) for v in vals))} unique rounded values)")

    print("\n=== Experiment URLs ===")
    for method, info in by_method.items():
        print(f"  {method:<35s}  https://www.comet.com/{WORKSPACE}/{PROJECT}/{info['id']}")


if __name__ == "__main__":
    main()
