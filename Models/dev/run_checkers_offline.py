"""Прогнать все 4 чекера на синтетическом memmap, оффлайн в Comet.

Используется для самопроверки pipeline. На синтетике вместо 12 IC использует 2
(2005-01-01, 2005-01-04), чтобы влезть в `/tmp/synthetic_era5_2005.dat` (240 ч).

Запуск:
    .venv/bin/python Models/dev/run_checkers_offline.py
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

MEMMAP = "/tmp/synthetic_era5_2005.dat"


# Override IC: 2 даты, влезают в 240-часовой memmap.
def _short_ic(year: int = 2005, hour: int = 0) -> list[pd.Timestamp]:
    return [
        pd.Timestamp(year=2005, month=1, day=1, hour=0),
        pd.Timestamp(year=2005, month=1, day=4, hour=0),
    ]


cpc.default_initial_conditions = _short_ic

COMMON = ["--memmap-path", MEMMAP, "--offline", "--mean-std-path", "", "--horizon-hours", "48"]

CHECKERS = [
    ("check_physics_weathergft", "tools/check_physics_weathergft.py", []),
    ("check_physics_predformergft", "tools/check_physics_predformergft.py", []),
    (
        "check_physics_new_kernel_fd4",
        "tools/check_physics_new_kernel.py",
        ["--stencil", "fd4", "--time-scheme", "euler", "--coriolis", "spherical"],
    ),
    (
        "check_physics_new_kernel_weno5",
        "tools/check_physics_new_kernel.py",
        ["--stencil", "weno5", "--time-scheme", "euler", "--coriolis", "beta_plane"],
    ),
]


def main() -> None:
    saved_argv = sys.argv
    for name, script, extra in CHECKERS:
        print(f"\n=== {name} ===")
        sys.argv = [script] + extra + COMMON
        runpy.run_path(str(REPO_ROOT / script), run_name="__main__")
        print(f"[{name}] OK")
    sys.argv = saved_argv
    print("\n=== All checkers passed ===")


if __name__ == "__main__":
    main()
