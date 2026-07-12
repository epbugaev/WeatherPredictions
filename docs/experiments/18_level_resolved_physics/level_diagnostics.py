"""Эксперимент 18: per-level физическая диагностика ядра уравнений на данных.

Мотивация: rollout экспа 16 (обученные армы t=12) даёт скилл (RMSE) КАЖДОГО из
13 уровней давления по отдельности. Чтобы объяснить *почему* физ-ядро на одних
высотах помогает, а на других вредит, нужен объясняющий слой на реальном ERA5:

  * **residual_rel[level]** — относительная невязка уравнения на уровне
    ``Σ|obs_t − rhs_t| / Σ|obs_t|``. <1 → тенденция ядра ловит сигнал (физика
    права); ≈1 → тенденция ≈0 или ошибка = сигнал (физика инертна/бесполезна);
    ≫1 → тенденция ядра больше самого сигнала и неверна (физика впрыскивает шум).
  * **term_abs_over_obs[level]** — магнитуда каждого члена правой части на уровне,
    делённая на |obs_t|. Показывает, КАКОЙ член доминирует невязку на данной
    высоте (гидростатика/адиабата/адвекция/трение/Кориолис).

Три конфигурации ядра = якоря лестницы экспа 16 (те же уравнения, что и в
обученных армах, но здесь считаются напрямую диагностически):

  * ``base13``   — легаси-уравнения (до знаковых фиксов) ≈ арм **R1** (legacy_hybrid);
  * ``C_best``   — геометрия сферы exp14 (metric/spherical/rayleigh, exact dx,
                   mass-consistent ω) ≈ арм **R4** (exp14);
  * ``C15_now``  — + скрытое тепло + omega_free (exp15) ≈ арм **R5** (exp15).

R3/R3a/R3q (только знаковые фиксы exp13, без геометрии exp14) — промежуточная
ступень: их невязки лежат между base13 и C_best. Отдельным якорем не считаются,
чтобы не выдумывать неточное отображение конфигурации.

Считается на USA-кропе (32×64), год из env (по умолчанию 2004 — val-год rollout),
часовые тройки снимков (fwd-разность как «наблюдённая» тенденция).

Запуск (кластер):  OUT=results/eq18_usa_2004.json YEAR=2004 STRIDE=1 \
                   REPO_ROOT=~/wt_fix_v2 python level_diagnostics.py
Локальный smoke:    OUT=/tmp/x.json SYNTHETIC=1 MAX_TRIPLES=4 python level_diagnostics.py

Лоджика членов (eq_terms) — самодостаточная копия из
``15_deep_equation_improvement/physics_stats_eq.py``: каждая подпапка экспов
самодостаточна (см. docs/experiments/README.md), а агрегация здесь другая
(per-level магнитуда членов, а не скалярная).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_env_root = os.environ.get("REPO_ROOT")
REPO_ROOT = Path(_env_root) if _env_root else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import h5netcdf  # noqa: E402

from utils.physics import Grid, GridConfig, PurePDEKernel  # noqa: E402
from utils.physics_hybrid import relative_to_specific_humidity  # noqa: E402

ERA5_ROOT = os.environ.get("ERA5_ROOT", "/home/fratnikov/weather_bench/1.40625deg/")
YEAR = int(os.environ.get("YEAR", "2004"))
STRIDE = int(os.environ.get("STRIDE", "1"))
MAX_TRIPLES = int(os.environ.get("MAX_TRIPLES", "0"))
SYNTHETIC = os.environ.get("SYNTHETIC", "0") == "1"
OUT = os.environ["OUT"]
DT_OBS = 3600.0
N_LEVELS = 13
PRESSURE_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

USA = {"H": 32, "W": 64, "rows": slice(75, 107), "cols": slice(164, 228), "boundary_x": "replicate"}
VARS = (
    ("geopotential", "z"),
    ("temperature", "t"),
    ("relative_humidity", "r"),
    ("u_component_of_wind", "u"),
    ("v_component_of_wind", "v"),
)
ALL_KEYS = ("u", "v", "t", "q", "z")

S2N = {"rows_south_to_north": True}
CBEST_KW = dict(
    S2N,
    metric_terms=True,
    spherical_divergence=True,
    rayleigh_friction=True,
    w_diagnostic="mass_consistent",
)
C15_KW = dict(CBEST_KW, latent_heating_coupling=True, omega_free=("t", "q"))
# имя → (тип сетки, kwargs ядра); отображение на армы R1 / R4 / R5.
VARIANTS: dict[str, tuple[str, dict]] = {
    "base13": ("legacy", {}),
    "C_best": ("exact", CBEST_KW),
    "C15_now": ("exact", C15_KW),
}


class LevelAccum:
    """Стриминговое отношение Σ|num|/Σ|den| по каждому уровню давления."""

    def __init__(self, n_levels: int) -> None:
        self.num = torch.zeros(n_levels, dtype=torch.float64)
        self.den = torch.zeros(n_levels, dtype=torch.float64)

    def add(self, num: torch.Tensor, den: torch.Tensor) -> None:
        self.num += num.double().abs().sum(dim=(0, 2, 3))
        self.den += den.double().abs().sum(dim=(0, 2, 3))

    def ratios(self) -> list[float]:
        return (self.num / (self.den + 1e-30)).tolist()


class Era5Reader:
    """Чтение состояний ERA5 USA-кропа с кэшем часовых снимков."""

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.files = {
            folder: h5netcdf.File(f"{ERA5_ROOT}{folder}/{folder}_{YEAR}_1.40625deg.nc", "r")
            for folder, _ in VARS
        }
        first = next(iter(self.files.values()))
        self.n_hours = first.variables[VARS[0][1]].shape[0]
        self.cache: dict[int, dict[str, torch.Tensor]] = {}

    def close(self) -> None:
        for f in self.files.values():
            f.close()

    def state(self, hour: int) -> dict[str, torch.Tensor]:
        """Состояние {u,v,t,q,z} формы (1, 13, 32, 64) в физических единицах."""
        if hour in self.cache:
            return self.cache[hour]
        fields = {}
        for folder, short in VARS:
            arr = np.asarray(
                self.files[folder].variables[short][hour, 0:13, USA["rows"], USA["cols"]],
                dtype=np.float32,
            )
            fields[short] = torch.from_numpy(arr)[None]
        q = relative_to_specific_humidity(fields["r"], fields["t"], self.grid.pressure)
        state = {
            "u": fields["u"],
            "v": fields["v"],
            "t": fields["t"],
            "q": q.clamp_min(1e-8),
            "z": fields["z"],
        }
        self.cache[hour] = state
        for old in [h for h in self.cache if h < hour - 2]:
            del self.cache[old]
        return state


def synthetic_state(seed: int) -> dict[str, torch.Tensor]:
    """Квази-реалистичное состояние для локального smoke (без ERA5)."""
    gen = torch.Generator().manual_seed(seed)
    h, w = USA["H"], USA["W"]
    phi = torch.tensor(
        [
            199300.0,
            157400,
            134000,
            116600,
            102300,
            90000,
            69700,
            54100,
            40700,
            28600,
            13500,
            7300,
            800,
        ]
    ).reshape(1, 13, 1, 1)
    t = torch.tensor([217.0, 208, 213, 218, 223, 229, 242, 253, 262, 270, 279, 283, 287]).reshape(
        1, 13, 1, 1
    )
    return {
        "u": 10 + 5 * torch.randn(1, 13, h, w, generator=gen),
        "v": 5 * torch.randn(1, 13, h, w, generator=gen),
        "t": t + 2 * torch.randn(1, 13, h, w, generator=gen),
        "q": (3e-3 * (1 + 0.3 * torch.randn(1, 13, h, w, generator=gen))).clamp_min(1e-8),
        "z": phi + 300 * torch.randn(1, 13, h, w, generator=gen),
    }


def data_latitudes_deg() -> tuple[float, ...]:
    """Реальные широты 32 строк USA-кропа: из nc либо аналитически (smoke)."""
    if SYNTHETIC:
        return tuple(16.171875 + 1.40625 * i for i in range(USA["H"]))
    path = f"{ERA5_ROOT}geopotential/geopotential_{YEAR}_1.40625deg.nc"
    with h5netcdf.File(path, "r") as f:
        lat = np.asarray(f.variables["lat"][USA["rows"]], dtype=np.float64)
    return tuple(float(x) for x in lat)


def build_kernel(grid_kind: str, kernel_kwargs: dict) -> PurePDEKernel:
    """Ядро exp13-фиксов на legacy- либо exact-сетке (exact = геометрия exp14)."""
    if grid_kind == "legacy":
        grid = Grid(GridConfig(H=USA["H"], W=USA["W"], lat_range_deg=(16.875, 63.28125)))
    else:
        grid = Grid(
            GridConfig(
                H=USA["H"],
                W=USA["W"],
                latitudes_deg=data_latitudes_deg(),
                lon_step_deg=1.40625,
            )
        )
    return PurePDEKernel(
        grid,
        stencil="fd4",
        coriolis="spherical",
        block_dt=300.0,
        boundary_x=USA["boundary_x"],
        t_t_formulation="adiabatic_omega",
        **kernel_kwargs,
    ).eval()


def eq_terms(k: PurePDEKernel, s: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Слагаемые правой части всех пяти уравнений (для per-level магнитуд).

    Копия логики ``physics_stats_eq.eq_terms`` (exp15), удержана самодостаточной.

    Args:
        k: ядро (его конфигурация задаёт состав членов).
        s: состояние {u,v,t,q,z}, каждое ``(B, 13, H, W)``.

    Returns:
        ``{переменная: {имя члена: тензор (B, 13, H, W)}}``.
    """
    u, v, t, q, z = s["u"], s["v"], s["t"], s["q"], s["z"]
    w = k.get_w(u, v)
    pressure = k.grid.pressure
    omega_pa = -100.0 * w
    terms_u = {
        "pgf": -k.diff.d_x(z),
        "coriolis": k.f_field * v,
        "adv_h": k._horiz_adv(u, u, v),
        "adv_v": -w * k._d_z(u),
    }
    terms_v = {
        "pgf": -k.diff.d_y(z),
        "coriolis": -k.f_field * u,
        "adv_h": k._horiz_adv(v, u, v),
        "adv_v": -w * k._d_z(v),
    }
    if k.metric_terms:
        terms_u["metric"] = k.tan_phi_over_a * u * v
        terms_v["metric"] = -k.tan_phi_over_a * u * u
    if k.rayleigh_friction:
        terms_u["friction"] = -k.rayleigh_k * u
        terms_v["friction"] = -k.rayleigh_k * v
    terms_t = {
        "adv_h": k._horiz_adv(t, u, v),
        "adv_v": -w * k._d_z(t),
        "adiabatic": k.consts.R_d * t * omega_pa / (k.consts.c_p * pressure),
    }
    cond = k._condensation_source(t, q, w)
    if k.latent_heating_coupling:
        terms_t["latent"] = -(k.consts.L / k.consts.c_p) * cond
    terms_q = {
        "adv_h": k._horiz_adv(q, u, v),
        "adv_v": -w * k._d_z(q),
        "cond": cond,
    }
    t_t_full = sum(terms_t.values())
    z_from_adv = k.get_z_t(terms_t["adv_h"] + terms_t["adv_v"])
    terms_z = {
        "from_adv": z_from_adv,
        "from_nonadv": k.get_z_t(t_t_full) - z_from_adv,
    }
    if "t" in k.omega_free:
        del terms_t["adv_v"], terms_t["adiabatic"]
    if "q" in k.omega_free:
        del terms_q["adv_v"]
    if "u" in k.omega_free:
        del terms_u["adv_v"]
    if "v" in k.omega_free:
        del terms_v["adv_v"]
    return {"u": terms_u, "v": terms_v, "t": terms_t, "q": terms_q, "z": terms_z}


def run() -> dict:
    """Прогон троек снимков; per-level невязка и магнитуды членов по 3 якорям."""
    kernels = {name: build_kernel(gk, kw) for name, (gk, kw) in VARIANTS.items()}
    exact_grid = kernels["C_best"].grid

    reader = None
    if SYNTHETIC:
        n_hours = 72
    else:
        reader = Era5Reader(exact_grid)
        n_hours = reader.n_hours
    triple_starts = list(range(0, n_hours - 2, STRIDE))
    if MAX_TRIPLES > 0:
        triple_starts = triple_starts[:MAX_TRIPLES]

    residual = {name: {v: LevelAccum(N_LEVELS) for v in ALL_KEYS} for name in VARIANTS}
    term_mag = {name: {v: {} for v in ALL_KEYS} for name in VARIANTS}

    wall_start = time.time()
    for i, h in enumerate(triple_starts):
        if SYNTHETIC:
            s0, s1 = synthetic_state(2 * i), synthetic_state(2 * i + 1)
        else:
            s0, s1 = reader.state(h), reader.state(h + 1)
        obs_fwd = {key: (s1[key] - s0[key]) / DT_OBS for key in ALL_KEYS}
        with torch.no_grad():
            for name in VARIANTS:
                rhs = kernels[name].rhs(**s0)
                for var in ALL_KEYS:
                    residual[name][var].add(obs_fwd[var] - rhs[f"{var}_t"], obs_fwd[var])
                terms = eq_terms(kernels[name], s0)
                for var in ALL_KEYS:
                    for t_name, t_val in terms[var].items():
                        acc = term_mag[name][var].setdefault(t_name, LevelAccum(N_LEVELS))
                        acc.add(t_val, obs_fwd[var])
        if (i + 1) % 500 == 0 or i == 0:
            rate = (i + 1) / (time.time() - wall_start)
            print(f"[usa/{YEAR}] triple {i + 1}/{len(triple_starts)} rate={rate:.2f}/s", flush=True)

    if reader is not None:
        reader.close()

    return {
        "meta": {
            "year": YEAR,
            "domain": "usa",
            "stride_hours": STRIDE,
            "n_triples": len(triple_starts),
            "dt_obs_s": DT_OBS,
            "synthetic": SYNTHETIC,
            "pressure_hpa": PRESSURE_HPA,
            "variant_to_arm": {"base13": "R1", "C_best": "R4", "C15_now": "R5"},
            "wall_seconds": time.time() - wall_start,
        },
        "residual_rel_by_level": {
            name: {v: acc.ratios() for v, acc in accs.items()} for name, accs in residual.items()
        },
        "term_abs_over_obs_by_level": {
            name: {
                v: {t_name: acc.ratios() for t_name, acc in terms.items()}
                for v, terms in vars.items()
            }
            for name, vars in term_mag.items()
        },
    }


def main() -> None:
    """Прогоняет USA/YEAR, пишет JSON в $OUT, печатает per-level невязку z и t."""
    results = run()
    Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("WROTE", OUT)
    for name in VARIANTS:
        print(f"--- {name}: residual_rel by level (z; t) ---")
        z = results["residual_rel_by_level"][name]["z"]
        t = results["residual_rel_by_level"][name]["t"]
        print("  z: " + " ".join(f"{x:.2f}" for x in z))
        print("  t: " + " ".join(f"{x:.2f}" for x in t))


if __name__ == "__main__":
    main()
