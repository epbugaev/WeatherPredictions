"""Локальный smoke-gate абляции: прогон check_physics_new_kernel на синтетике.

Гейт «Шаг 3» из плана абляции (`~/.claude/plans/radiant-booping-charm.md`):
после прохождения CPU-sanity и до кластерного 48h-прогона каждый Ek быстро
проверяется на синтетическом ERA5 memmap с короткими IC, чтобы поймать
интеграционные регрессии без обращения к кластеру/реальному memmap.

Синтетический memmap (`Models/dev/make_synthetic_era5.py`) хранит 240 ч, в него
влезают только январские IC, поэтому `default_initial_conditions`
monkey-патчится на 2 даты (как в `Models/dev/run_checkers_offline.py`).

Все аргументы после имени скрипта пробрасываются в
`tools/check_physics_new_kernel.py`. Недостающие
`--memmap-path/--memmap-meta-path/--mean-std-path/--offline/--horizon-hours`
подставляются дефолтами (synthetic, offline, 6 ч).

Запуск:
    .venv/bin/python Models/dev/smoke_ablation.py \
        --stencil fd4 --time-scheme euler --coriolis spherical --abl-label E0
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.check_physics_common as cpc  # noqa: E402

SYN_DAT = Path("/tmp/syn_era5.dat")
SYN_META = Path("/tmp/syn_era5.meta.json")
CHECKER = "tools/check_physics_new_kernel.py"
SYN_BUILDER = "Models/dev/make_synthetic_era5.py"


def _short_ic(year: int = 2005, hour: int = 0) -> list[pd.Timestamp]:
    """2 IC в пределах 240-часового синтетического memmap (h=0 и h=72)."""
    return [
        pd.Timestamp(year=2005, month=1, day=1, hour=0),
        pd.Timestamp(year=2005, month=1, day=4, hour=0),
    ]


def main() -> None:
    """Собрать argv, при необходимости построить синтетику, запустить чекер."""
    if not SYN_DAT.exists() or not SYN_META.exists():
        # runpy (а не import) — чтобы не тянуть Models/__init__.py (timm).
        print(f"[smoke] synthetic memmap not found → building {SYN_DAT}")
        saved = sys.argv
        sys.argv = [SYN_BUILDER, str(SYN_DAT)]
        runpy.run_path(str(REPO_ROOT / SYN_BUILDER), run_name="__main__")
        sys.argv = saved

    cpc.default_initial_conditions = _short_ic

    passthrough = sys.argv[1:]
    argv = [CHECKER, *passthrough]
    defaults = {
        "--memmap-path": str(SYN_DAT),
        "--memmap-meta-path": str(SYN_META),
        "--mean-std-path": "",
        "--horizon-hours": "6",
    }
    for flag, value in defaults.items():
        if flag not in argv:
            argv += [flag, value]
    if "--offline" not in argv:
        argv.append("--offline")

    print(f"[smoke] argv={argv[1:]}")
    sys.argv = argv
    runpy.run_path(str(REPO_ROOT / CHECKER), run_name="__main__")
    print("[smoke] OK — no exception, see metrics above")


if __name__ == "__main__":
    main()
