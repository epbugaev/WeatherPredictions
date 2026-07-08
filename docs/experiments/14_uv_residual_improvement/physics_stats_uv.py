"""Эксперимент 14: диагностика и снижение невязки уравнений движения (u, v).

Меряет относительную невязку (residual_rel = Σ|obs − model| / Σ|obs|) и RMS
тенденций всех пяти переменных на парах часовых снимков ERA5 для матрицы
вариантов уравнений движения (см. VARIANTS). Базовый вариант ``base``
воспроизводит конфигурацию эксперимента 13 бит-в-бит. Дополнительно для
фокус-вариантов (base, V12_geom, C_full) снимает:

  1. Разложение невязки u/v по членам уравнения: кумулятивная сборка
     (PGF+Кориолис → +гор. адвекция → +верт. адвекция → +кривизна → +трение)
     и «выкинуть один член» (drop-one).
  2. Разбивку невязки по уровням давления (13) и широтным поясам.
  3. Широтные профили невязки u/v и interior-невязку (без 2 краевых
     строк/столбцов).
  4. Контроль временной схемы: forward-разность за 1 ч против центрированной
     за 2 ч и трапецеидального RHS (½·(RHS(t)+RHS(t+1))).
  5. Проверку дискретной бездивергентности: RMS членов ∂u/∂x, ∂v/∂y,
     метрического −v·tanφ/a и их суммы.
  6. Карты накопленной относительной невязки (Σ_t,P |err| / Σ_t,P |obs| в
     каждой ячейке) для base и C_full по всем 5 переменным — NPZ в $MAPS_OUT
     (плюс lat/lon/lsm для отрисовки).

Запуск (кластер):  OUT=... YEAR=2000 DOMAIN=globe STRIDE=4 python physics_stats_uv.py
Локальный smoke:   OUT=/tmp/x.json SYNTHETIC=1 MAX_TRIPLES=2 python physics_stats_uv.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO_ROOT))

import h5netcdf  # noqa: E402

from utils.physics import Grid, GridConfig, PurePDEKernel  # noqa: E402
from utils.physics_hybrid import relative_to_specific_humidity  # noqa: E402

ERA5_ROOT = os.environ.get("ERA5_ROOT", "/home/fratnikov/weather_bench/1.40625deg/")
YEAR = int(os.environ.get("YEAR", "2000"))
DOMAIN = os.environ.get("DOMAIN", "globe")
STRIDE = int(os.environ.get("STRIDE", "4"))
MAX_TRIPLES = int(os.environ.get("MAX_TRIPLES", "0"))  # 0 = все доступные
SYNTHETIC = os.environ.get("SYNTHETIC", "0") == "1"
OUT = os.environ["OUT"]
MAPS_OUT = os.environ.get("MAPS_OUT", "")
DT_OBS = 3600.0  # шаг между снимками, с
MAP_VARIANTS = ("base", "C_best")  # «до» и «после» для карт невязки

# Домены exp 13: (H, W, legacy lat_range для Grid, срезы в глобальном поле,
# boundary_x). Реальные широты строк читаются из nc (exact-сетка).
DOMAINS = {
    "usa": {
        "H": 32,
        "W": 64,
        "lat_range": (16.875, 63.28125),
        "rows": slice(75, 107),
        "cols": slice(164, 228),
        "boundary_x": "replicate",
    },
    "globe": {
        "H": 128,
        "W": 256,
        "lat_range": (-90.0, 90.0),
        "rows": slice(None),
        "cols": slice(None),
        "boundary_x": "periodic",
    },
}

VARS = (
    ("geopotential", "z"),
    ("temperature", "t"),
    ("relative_humidity", "r"),
    ("u_component_of_wind", "u"),
    ("v_component_of_wind", "v"),
)

# Матрица вариантов: имя → (сетка, kwargs ядра). Сетка: 'legacy' — как в
# exp 13 (linspace lat_range), 'exact' — реальные широты строк данных.
# Изолированные улучшения V3–V8 меряются ПОВЕРХ V12_geom (корректная
# ориентация d_y + точные широты): без правильного знака ∂/∂y их эффект
# тонет в знаковой ошибке PGF.
S2N = {"rows_south_to_north": True}
VARIANTS: dict[str, tuple[str, dict]] = {
    "base": ("legacy", {}),
    "base_mc": ("legacy", {"w_diagnostic": "mass_consistent"}),
    "V1_dy": ("legacy", dict(S2N)),
    "V2_lat": ("exact", {}),
    "V12_geom": ("exact", dict(S2N)),
    "V3_metric": ("exact", dict(S2N, metric_terms=True)),
    "V4_dz": ("exact", dict(S2N, vertical_scheme="lagrange3")),
    "V5_frict": ("exact", dict(S2N, rayleigh_friction=True)),
    "V6_flux": ("exact", dict(S2N, advection_form="flux")),
    "V7_mc": ("exact", dict(S2N, w_diagnostic="mass_consistent")),
    "V8_sphdiv": ("exact", dict(S2N, spherical_divergence=True)),
    "C_geo": ("exact", dict(S2N, metric_terms=True, spherical_divergence=True)),
    "C_geo_dz": (
        "exact",
        dict(S2N, metric_terms=True, spherical_divergence=True, vertical_scheme="lagrange3"),
    ),
    "C_full": (
        "exact",
        dict(
            S2N,
            metric_terms=True,
            spherical_divergence=True,
            vertical_scheme="lagrange3",
            rayleigh_friction=True,
        ),
    ),
    "C_full_mc": (
        "exact",
        dict(
            S2N,
            metric_terms=True,
            spherical_divergence=True,
            vertical_scheme="lagrange3",
            rayleigh_friction=True,
            w_diagnostic="mass_consistent",
        ),
    ),
    "C_full_flux": (
        "exact",
        dict(
            S2N,
            metric_terms=True,
            spherical_divergence=True,
            vertical_scheme="lagrange3",
            rayleigh_friction=True,
            advection_form="flux",
        ),
    ),
    # Кандидат финальной комбинации: без lagrange3 и flux (дымовой прогон
    # показал, что они ухудшают), с массо-согласованной ω.
    "C_best": (
        "exact",
        dict(
            S2N,
            metric_terms=True,
            spherical_divergence=True,
            rayleigh_friction=True,
            w_diagnostic="mass_consistent",
        ),
    ),
}
FOCUS = ("base", "V12_geom", "C_best")  # варианты с полной диагностикой
ALL_KEYS = ("u", "v", "t", "q", "z")


class RatioAccum:
    """Стриминговое отношение Σ|num|/Σ|den| и RMS числителя (float64)."""

    def __init__(self) -> None:
        self.num = self.den = self.sq = 0.0
        self.count = 0.0

    def add(self, num: torch.Tensor, den: torch.Tensor) -> None:
        self.num += float(num.double().abs().sum())
        self.den += float(den.double().abs().sum())
        self.sq += float((num.double() ** 2).sum())
        self.count += num.numel()

    def ratio(self) -> float:
        return self.num / (self.den + 1e-30)

    def rms(self) -> float:
        return float(np.sqrt(self.sq / max(self.count, 1.0)))


class LevelAccum:
    """Невязка по уровням давления: Σ|·| по (B, H, W) на каждый уровень."""

    def __init__(self, n_levels: int) -> None:
        self.num = torch.zeros(n_levels, dtype=torch.float64)
        self.den = torch.zeros(n_levels, dtype=torch.float64)
        self.count = 0.0

    def add(self, num: torch.Tensor, den: torch.Tensor) -> None:
        self.num += num.double().abs().sum(dim=(0, 2, 3))
        self.den += den.double().abs().sum(dim=(0, 2, 3))
        self.count += num.numel() / num.shape[1]

    def ratios(self) -> list[float]:
        return (self.num / (self.den + 1e-30)).tolist()

    def absmeans(self) -> list[float]:
        return (self.num / max(self.count, 1.0)).tolist()


class RowAccum:
    """Невязка по строкам широты: Σ|·| по (B, P, W) на каждую строку."""

    def __init__(self, n_rows: int) -> None:
        self.num = torch.zeros(n_rows, dtype=torch.float64)
        self.den = torch.zeros(n_rows, dtype=torch.float64)

    def add(self, num: torch.Tensor, den: torch.Tensor) -> None:
        self.num += num.double().abs().sum(dim=(0, 1, 3))
        self.den += den.double().abs().sum(dim=(0, 1, 3))

    def ratios(self) -> list[float]:
        return (self.num / (self.den + 1e-30)).tolist()


class Era5Reader:
    """Чтение состояний ERA5 с кэшем открытых файлов и часовых состояний."""

    def __init__(self, dom: dict, grid: Grid) -> None:
        self.dom = dom
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
        """Состояние {u,v,t,q,z} формы (1, 13, H, W) в физических единицах."""
        if hour in self.cache:
            return self.cache[hour]
        fields = {}
        for folder, short in VARS:
            arr = np.asarray(
                self.files[folder].variables[short][
                    hour, 0:13, self.dom["rows"], self.dom["cols"]
                ],
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


def synthetic_state(dom: dict, seed: int) -> dict[str, torch.Tensor]:
    """Синтетическое квази-реалистичное состояние для локального smoke."""
    gen = torch.Generator().manual_seed(seed)
    h, w = dom["H"], dom["W"]
    phi = torch.tensor(
        [199300.0, 157400, 134000, 116600, 102300, 90000, 69700, 54100, 40700, 28600, 13500, 7300, 800]
    ).reshape(1, 13, 1, 1)
    t = torch.tensor(
        [217.0, 208, 213, 218, 223, 229, 242, 253, 262, 270, 279, 283, 287]
    ).reshape(1, 13, 1, 1)
    return {
        "u": 10 + 5 * torch.randn(1, 13, h, w, generator=gen),
        "v": 5 * torch.randn(1, 13, h, w, generator=gen),
        "t": t + 2 * torch.randn(1, 13, h, w, generator=gen),
        "q": (3e-3 * (1 + 0.3 * torch.randn(1, 13, h, w, generator=gen))).clamp_min(1e-8),
        "z": phi + 300 * torch.randn(1, 13, h, w, generator=gen),
    }


def data_latitudes_deg(dom: dict) -> tuple[float, ...]:
    """Реальные широты строк домена: из nc-файла либо аналитически (smoke)."""
    if SYNTHETIC:
        start = -89.296875 if dom["H"] == 128 else 16.171875
        return tuple(start + 1.40625 * i for i in range(dom["H"]))
    path = f"{ERA5_ROOT}geopotential/geopotential_{YEAR}_1.40625deg.nc"
    with h5netcdf.File(path, "r") as f:
        lat = np.asarray(f.variables["lat"][dom["rows"]], dtype=np.float64)
    return tuple(float(x) for x in lat)


def build_kernel(dom: dict, grid_kind: str, kernel_kwargs: dict) -> PurePDEKernel:
    """Ядро на legacy- либо exact-сетке; остальное — конфигурация exp 13."""
    if grid_kind == "legacy":
        grid = Grid(GridConfig(H=dom["H"], W=dom["W"], lat_range_deg=dom["lat_range"]))
    else:
        grid = Grid(GridConfig(H=dom["H"], W=dom["W"], latitudes_deg=data_latitudes_deg(dom)))
    return PurePDEKernel(
        grid,
        stencil="fd4",
        coriolis="spherical",
        block_dt=300.0,
        boundary_x=dom["boundary_x"],
        t_t_formulation="adiabatic_omega",
        **kernel_kwargs,
    ).eval()


def uv_terms(k: PurePDEKernel, s: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Члены уравнений движения по отдельности для разложения невязки.

    Args:
        k: ядро (конфигурация задаёт формы членов).
        s: состояние {u,v,t,q,z}, каждое (B, P, H, W).

    Returns:
        {"u": {имя члена: тензор}, "v": {...}} — слагаемые правой части.
    """
    u, v, z = s["u"], s["v"], s["z"]
    w = k.get_w(u, v)
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
    return {"u": terms_u, "v": terms_v}


CUMULATIVE_STAGES = (
    ("pgf_cor", ("pgf", "coriolis")),
    ("+adv_h", ("pgf", "coriolis", "adv_h")),
    ("+adv_v", ("pgf", "coriolis", "adv_h", "adv_v")),
    ("+metric", ("pgf", "coriolis", "adv_h", "adv_v", "metric")),
    ("+friction", ("pgf", "coriolis", "adv_h", "adv_v", "metric", "friction")),
)


def latitude_bands(lat_deg: torch.Tensor) -> dict[str, torch.Tensor]:
    """Индексы строк широтных поясов: tropics |φ|<23.5, midlat, polar |φ|≥66.5."""
    bands = {
        "tropics": (lat_deg.abs() < 23.5).nonzero().flatten(),
        "midlat": ((lat_deg.abs() >= 23.5) & (lat_deg.abs() < 66.5)).nonzero().flatten(),
        "polar": (lat_deg.abs() >= 66.5).nonzero().flatten(),
    }
    return {name: rows for name, rows in bands.items() if rows.numel() > 0}


def run() -> dict:
    """Полный набор статистик по одному домену и году."""
    dom = DOMAINS[DOMAIN]
    n_levels = 13
    kernels = {name: build_kernel(dom, gk, kw) for name, (gk, kw) in VARIANTS.items()}
    exact_grid = kernels["V12_geom"].grid
    lat_deg = exact_grid.latitudes * 180.0 / torch.pi

    reader = None
    if SYNTHETIC:
        n_hours = 48
    else:
        reader = Era5Reader(dom, exact_grid)
        n_hours = reader.n_hours
    triple_starts = list(range(0, n_hours - 2, STRIDE))
    if MAX_TRIPLES > 0:
        triple_starts = triple_starts[:MAX_TRIPLES]

    residual = {name: {v: RatioAccum() for v in ALL_KEYS} for name in VARIANTS}
    interior = {name: {v: RatioAccum() for v in ALL_KEYS} for name in VARIANTS}
    by_level = {name: {v: LevelAccum(n_levels) for v in ("u", "v")} for name in FOCUS}
    obs_level = {v: LevelAccum(n_levels) for v in ("u", "v")}
    bands = latitude_bands(lat_deg)
    by_band = {
        name: {band: {v: RatioAccum() for v in ALL_KEYS} for band in bands} for name in FOCUS
    }
    lat_profile = {name: {v: RowAccum(dom["H"]) for v in ("u", "v")} for name in FOCUS}
    terms_cum = {
        name: {v: {stage: RatioAccum() for stage, _ in CUMULATIVE_STAGES} for v in ("u", "v")}
        for name in FOCUS
    }
    terms_drop = {
        name: {
            v: {
                t: RatioAccum()
                for t in ("pgf", "coriolis", "adv_h", "adv_v", "metric", "friction")
            }
            for v in ("u", "v")
        }
        for name in FOCUS
    }
    term_rms = {
        name: {
            v: {
                t: RatioAccum()
                for t in ("pgf", "coriolis", "adv_h", "adv_v", "metric", "friction")
            }
            for v in ("u", "v")
        }
        for name in FOCUS
    }
    temporal = {
        name: {v: {m: RatioAccum() for m in ("fwd", "centered", "trapezoid")} for v in ("u", "v")}
        for name in FOCUS
    }
    div_check = {t: RatioAccum() for t in ("u_x", "v_y", "metric", "sum_plain", "sum_spherical")}
    map_num = {
        name: {v: torch.zeros(dom["H"], dom["W"], dtype=torch.float64) for v in ALL_KEYS}
        for name in MAP_VARIANTS
    }
    map_den = {v: torch.zeros(dom["H"], dom["W"], dtype=torch.float64) for v in ALL_KEYS}

    wall_start = time.time()
    for i, h in enumerate(triple_starts):
        if SYNTHETIC:
            s0, s1, s2 = (synthetic_state(dom, seed=3 * i + j) for j in range(3))
        else:
            s0, s1, s2 = reader.state(h), reader.state(h + 1), reader.state(h + 2)
        obs_fwd = {key: (s1[key] - s0[key]) / DT_OBS for key in ALL_KEYS}
        obs_cen = {key: (s2[key] - s0[key]) / (2 * DT_OBS) for key in ("u", "v")}

        with torch.no_grad():
            rhs0 = {name: kernels[name].rhs(**s0) for name in VARIANTS}
            for name in VARIANTS:
                for var in ALL_KEYS:
                    err = obs_fwd[var] - rhs0[name][f"{var}_t"]
                    residual[name][var].add(err, obs_fwd[var])
                    interior[name][var].add(
                        err[..., 2:-2, 2:-2], obs_fwd[var][..., 2:-2, 2:-2]
                    )
                    if name in MAP_VARIANTS:
                        map_num[name][var] += err.double().abs().sum(dim=(0, 1))
            for var in ALL_KEYS:
                map_den[var] += obs_fwd[var].double().abs().sum(dim=(0, 1))

            for name in FOCUS:
                k = kernels[name]
                rhs_f = rhs0[name]
                for var in ("u", "v"):
                    err = obs_fwd[var] - rhs_f[f"{var}_t"]
                    by_level[name][var].add(err, obs_fwd[var])
                    lat_profile[name][var].add(err, obs_fwd[var])
                for band, rows in bands.items():
                    for var in ALL_KEYS:
                        err = obs_fwd[var] - rhs_f[f"{var}_t"]
                        by_band[name][band][var].add(
                            err.index_select(2, rows), obs_fwd[var].index_select(2, rows)
                        )
                # Разложение по членам
                terms = uv_terms(k, s0)
                for var in ("u", "v"):
                    full = sum(terms[var].values())
                    for stage, keys in CUMULATIVE_STAGES:
                        active = [terms[var][t] for t in keys if t in terms[var]]
                        model = sum(active)
                        terms_cum[name][var][stage].add(obs_fwd[var] - model, obs_fwd[var])
                    for t_name, t_val in terms[var].items():
                        terms_drop[name][var][t_name].add(
                            obs_fwd[var] - (full - t_val), obs_fwd[var]
                        )
                        term_rms[name][var][t_name].add(t_val, obs_fwd[var])
                # Временная схема
                rhs1 = k.rhs(**s1)
                for var in ("u", "v"):
                    temporal[name][var]["fwd"].add(
                        obs_fwd[var] - rhs_f[f"{var}_t"], obs_fwd[var]
                    )
                    temporal[name][var]["centered"].add(
                        obs_cen[var] - rhs1[f"{var}_t"], obs_cen[var]
                    )
                    trap = 0.5 * (rhs_f[f"{var}_t"] + rhs1[f"{var}_t"])
                    temporal[name][var]["trapezoid"].add(obs_fwd[var] - trap, obs_fwd[var])

            for var in ("u", "v"):
                obs_level[var].add(obs_fwd[var], obs_fwd[var])

            # Бездивергентность (на exact-сетке с корректным d_y)
            kg = kernels["V12_geom"]
            u_x = kg.diff.d_x(s0["u"])
            v_y = kg.diff.d_y(s0["v"])
            metric = -s0["v"] * kernels["C_geo"].tan_phi_over_a
            div_check["u_x"].add(u_x, u_x)
            div_check["v_y"].add(v_y, v_y)
            div_check["metric"].add(metric, metric)
            div_check["sum_plain"].add(u_x + v_y, u_x)
            div_check["sum_spherical"].add(u_x + v_y + metric, u_x)

        if (i + 1) % 50 == 0 or i == 0:
            rate = (i + 1) / (time.time() - wall_start)
            print(
                f"[{DOMAIN}/{YEAR}] triple {i + 1}/{len(triple_starts)} "
                f"hour={h} rate={rate:.2f}/s",
                flush=True,
            )

    if reader is not None:
        reader.close()

    if MAPS_OUT:
        arrays: dict[str, np.ndarray] = {
            "lat": lat_deg.numpy().astype(np.float32),
            "lon": (np.arange(256, dtype=np.float32) * 1.40625)[dom["cols"]],
        }
        if SYNTHETIC:
            arrays["lsm"] = np.zeros((dom["H"], dom["W"]), dtype=np.float32)
        else:
            with h5netcdf.File(f"{ERA5_ROOT}constants/constants_1.40625deg.nc", "r") as f:
                lsm_global = np.asarray(f.variables["lsm"], dtype=np.float32)
            arrays["lsm"] = lsm_global[dom["rows"], dom["cols"]]
        for name in MAP_VARIANTS:
            for var in ALL_KEYS:
                ratio = map_num[name][var] / (map_den[var] + 1e-30)
                arrays[f"resmap_{name}_{var}"] = ratio.numpy().astype(np.float32)
        np.savez_compressed(MAPS_OUT, **arrays)
        print("WROTE", MAPS_OUT)

    return {
        "meta": {
            "year": YEAR,
            "domain": DOMAIN,
            "stride_hours": STRIDE,
            "n_triples": len(triple_starts),
            "n_hours_in_year": n_hours,
            "dt_obs_s": DT_OBS,
            "synthetic": SYNTHETIC,
            "stencil": "fd4",
            "variants": {name: [gk, kw] for name, (gk, kw) in VARIANTS.items()},
            "wall_seconds": time.time() - wall_start,
        },
        "residual_rel": {
            name: {v: acc.ratio() for v, acc in accs.items()} for name, accs in residual.items()
        },
        "residual_rms": {
            name: {v: acc.rms() for v, acc in accs.items()} for name, accs in residual.items()
        },
        "residual_rel_interior": {
            name: {v: acc.ratio() for v, acc in accs.items()} for name, accs in interior.items()
        },
        "by_level": {
            "pressure_hpa": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
            **{
                name: {v: acc.ratios() for v, acc in accs.items()}
                for name, accs in by_level.items()
            },
            "obs_absmean": {v: obs_level[v].absmeans() for v in ("u", "v")},
        },
        "by_band": {
            name: {
                band: {v: acc.ratio() for v, acc in accs.items()}
                for band, accs in band_accs.items()
            }
            for name, band_accs in by_band.items()
        },
        "lat_profile": {
            "lat_deg": lat_deg.tolist(),
            **{
                name: {v: acc.ratios() for v, acc in accs.items()}
                for name, accs in lat_profile.items()
            },
        },
        "term_decomposition": {
            name: {
                v: {
                    "cumulative": {s: acc.ratio() for s, acc in terms_cum[name][v].items()},
                    "drop_one": {
                        t: acc.ratio() for t, acc in terms_drop[name][v].items() if acc.count > 0
                    },
                    "term_abs_over_obs": {
                        t: acc.ratio() for t, acc in term_rms[name][v].items() if acc.count > 0
                    },
                }
                for v in ("u", "v")
            }
            for name in FOCUS
        },
        "temporal_scheme": {
            name: {
                v: {m: acc.ratio() for m, acc in accs.items()}
                for v, accs in temporal[name].items()
            }
            for name in FOCUS
        },
        "divergence_check_rms": {t: acc.rms() for t, acc in div_check.items()},
    }


def main() -> None:
    """Гоняет домен/год, пишет JSON в $OUT, печатает сводку."""
    results = run()
    Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("WROTE", OUT)
    print(f"--- {DOMAIN}/{YEAR}: residual_rel (u, v, t, q, z) ---")
    for name in VARIANTS:
        r = results["residual_rel"][name]
        print(
            f"  {name:12s}: "
            + "  ".join(f"{var}={r[var]:7.3f}" for var in ALL_KEYS)
        )


if __name__ == "__main__":
    main()
