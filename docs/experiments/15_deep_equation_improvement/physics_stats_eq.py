"""Эксперимент 15: почленный аудит и улучшение всех пяти уравнений ядра.

Меряет относительную невязку (residual_rel = Σ|obs − model| / Σ|obs|) и RMS
тенденций u, v, T, q, z на тройках часовых снимков ERA5 для матрицы вариантов
уравнений (VARIANTS). База ``C_best`` — принятая геометрия эксперимента 14
(точные широты, d_y юг→север, кривизна, сферическая континуити, трение
Хелда–Суареса, массо-согласованная ω); ``base13`` воспроизводит эксперимент 13.

Новые члены эксперимента 15 (opt-in параметры PurePDEKernel):

  * ``T_hs`` — ньютоновская релаксация Хелда–Суареса (диабатика);
  * ``T_lh`` — скрытое тепло конденсации в T (энергетическая связка с q);
  * ``W_obrien`` — поправка О'Брайена к кинематической ω;
  * ``Z_ps`` — баротропный якорь гидростатики из кинематической ∂p_s/∂t;
  * ``S1_zm``/``S2_map``/``S12`` — климатологические поправки-источники
    (Q₁/Q₂-климатология), построенные ТОЛЬКО на 2000 годе (CLIM_OUT), затем
    оцениваемые вне выборки на 2001–2002 (CLIM_IN).

Диагностика (focus-варианты): разложение невязки по членам каждого уравнения
(кумулятивно и drop-one), по уровням давления, широтным поясам и строкам
широты; контроль временной схемы (fwd 1 ч / centered 2 ч / trapezoid);
ω-тест против residual-implied ω; гидростатический контроль якоря z;
карты накопленной невязки (NPZ).

Запуск (кластер):  OUT=... YEAR=2000 DOMAIN=globe STRIDE=1 CLIM_OUT=... python physics_stats_eq.py
Оценка вне выборки: OUT=... YEAR=2001 DOMAIN=globe CLIM_IN=<clim-2000.npz> python physics_stats_eq.py
Локальный smoke:    OUT=/tmp/x.json SYNTHETIC=1 MAX_TRIPLES=4 python physics_stats_eq.py
"""

import calendar
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
STRIDE = int(os.environ.get("STRIDE", "1"))
MAX_TRIPLES = int(os.environ.get("MAX_TRIPLES", "0"))  # 0 = все доступные
SYNTHETIC = os.environ.get("SYNTHETIC", "0") == "1"
OUT = os.environ["OUT"]
MAPS_OUT = os.environ.get("MAPS_OUT", "")
CLIM_OUT = os.environ.get("CLIM_OUT", "")  # построить климатологию (только 2000)
CLIM_IN = os.environ.get("CLIM_IN", "")  # применить готовую климатологию
DT_OBS = 3600.0  # шаг между снимками, с

DOMAINS = {
    "usa": {
        "H": 32,
        "W": 64,
        "rows": slice(75, 107),
        "cols": slice(164, 228),
        "boundary_x": "replicate",
    },
    "globe": {
        "H": 128,
        "W": 256,
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
ALL_KEYS = ("u", "v", "t", "q", "z")
CLIM_VARS = ("u", "v", "t", "q", "z")  # зонально-месячная климатология
MAP_CLIM_VARS = ("t", "q", "z")  # суточная карта-климатология

# Конфигурация базы = C_best эксперимента 14 (принятые геометрия и члены).
S2N = {"rows_south_to_north": True}
CBEST_KW = dict(
    S2N,
    metric_terms=True,
    spherical_divergence=True,
    rayleigh_friction=True,
    w_diagnostic="mass_consistent",
)
# Комбинация эксперимента 15 по итогам USA-2000 (job 4169269): скрытое тепло
# конденсации + исключение ω-членов из выходных тенденций T и q (кинематическая
# ω поточечно бесполезна: r < 0.1 c implied-ω; z интегрирует ПОЛНЫЙ t_t).
# Ньютоновская релаксация (T_hs, эффект +0.1%) и кинематический якорь
# (Z_ps, z 3.05 → 40.7) в комбинацию не вошли — оставлены в матрице как
# документированные отрицательные результаты.
C15_KW = dict(CBEST_KW, latent_heating_coupling=True, omega_free=("t", "q"))
# имя → (сетка, kwargs ядра). 'legacy' — linspace-широты exp 13; 'exact' —
# реальные широты строк данных (принято в exp 14).
VARIANTS: dict[str, tuple[str, dict]] = {
    "base13": ("legacy", {}),
    "C_best": ("exact", CBEST_KW),
    "W_plain": ("exact", dict(CBEST_KW, w_diagnostic="plain")),
    "W_obrien": ("exact", dict(CBEST_KW, w_diagnostic="obrien")),
    "T_hs": ("exact", dict(CBEST_KW, newtonian_relaxation=True)),
    "T_lh": ("exact", dict(CBEST_KW, latent_heating_coupling=True)),
    "Z_ps": ("exact", dict(CBEST_KW, z_anchor="kinematic_ps")),
    "NoW_t": ("exact", dict(CBEST_KW, omega_free=("t",))),
    "NoW_tq": ("exact", dict(CBEST_KW, omega_free=("t", "q"))),
    "NoW_all": ("exact", dict(CBEST_KW, omega_free=("t", "q", "u", "v"))),
    "C15_now": ("exact", C15_KW),
}
# Клим-варианты (только при CLIM_IN): базовое ядро + внешние источники.
SOURCE_VARIANTS = {
    "S1_zm": ("C_best", "zm"),
    "S2_map": ("C_best", "map"),
    "S12": ("C_best", "both"),
    "C15_full": ("C15_now", "both"),
}
FOCUS = ("base13", "C_best", "C15_now")
MAP_SET = ("base13", "C_best", "C15_now", "C15_full", "S12")
CLIM_REFS = ("C_best", "C15_now")  # против каких ядер копится климатология


class RatioAccum:
    """Стриминговое отношение Σ|num|/Σ|den| и RMS числителя (float64)."""

    def __init__(self) -> None:
        self.num = self.den = self.sq = self.den_sq = 0.0
        self.count = 0.0

    def add(self, num: torch.Tensor, den: torch.Tensor) -> None:
        self.num += float(num.double().abs().sum())
        self.den += float(den.double().abs().sum())
        self.sq += float((num.double() ** 2).sum())
        self.den_sq += float((den.double() ** 2).sum())
        self.count += num.numel()

    def ratio(self) -> float:
        return self.num / (self.den + 1e-30)

    def rms(self) -> float:
        return float(np.sqrt(self.sq / max(self.count, 1.0)))

    def rms_ratio(self) -> float:
        return float(np.sqrt(self.sq / (self.den_sq + 1e-30)))


class LevelAccum:
    """Невязка по уровням давления: Σ|·| по (B, H, W) на каждый уровень."""

    def __init__(self, n_levels: int) -> None:
        self.num = torch.zeros(n_levels, dtype=torch.float64)
        self.den = torch.zeros(n_levels, dtype=torch.float64)

    def add(self, num: torch.Tensor, den: torch.Tensor) -> None:
        self.num += num.double().abs().sum(dim=(0, 2, 3))
        self.den += den.double().abs().sum(dim=(0, 2, 3))

    def ratios(self) -> list[float]:
        return (self.num / (self.den + 1e-30)).tolist()


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


class CorrAccum:
    """Стриминговая корреляция Пирсона (float64)."""

    def __init__(self) -> None:
        self.n = 0.0
        self.sa = self.sb = self.saa = self.sbb = self.sab = 0.0

    def add(self, a: torch.Tensor, b: torch.Tensor) -> None:
        a64 = a.double().flatten()
        b64 = b.double().flatten()
        self.n += a64.numel()
        self.sa += float(a64.sum())
        self.sb += float(b64.sum())
        self.saa += float((a64 * a64).sum())
        self.sbb += float((b64 * b64).sum())
        self.sab += float((a64 * b64).sum())

    def corr(self) -> float:
        if self.n == 0:
            return float("nan")
        cov = self.sab - self.sa * self.sb / self.n
        va = self.saa - self.sa * self.sa / self.n
        vb = self.sbb - self.sb * self.sb / self.n
        return cov / (np.sqrt(max(va, 0.0) * max(vb, 0.0)) + 1e-30)


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
        lat_range = (16.875, 63.28125) if dom["H"] == 32 else (-90.0, 90.0)
        grid = Grid(GridConfig(H=dom["H"], W=dom["W"], lat_range_deg=lat_range))
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


def month_of_hour(hour: int, year: int) -> int:
    """Месяц (0..11) по часу года (UTC), с учётом високосности."""
    days = np.cumsum([0] + [calendar.monthrange(year, m)[1] for m in range(1, 13)])
    day = hour // 24
    return int(np.searchsorted(days, day, side="right") - 1)


def eq_terms(
    k: PurePDEKernel, s: dict[str, torch.Tensor]
) -> dict[str, dict[str, torch.Tensor]]:
    """Члены всех пяти уравнений по отдельности для разложения невязки.

    Args:
        k: ядро (его конфигурация задаёт состав членов).
        s: состояние {u,v,t,q,z}, каждое (B, P, H, W).

    Returns:
        {переменная: {имя члена: тензор}} — слагаемые правой части.
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
    if k.newtonian_relaxation:
        terms_t["hs_relax"] = -k.hs_k_t * (t - k.hs_t_eq)
    cond = k._condensation_source(t, q, w)
    if k.latent_heating_coupling:
        terms_t["latent"] = -(k.consts.L / k.consts.c_p) * cond
    terms_q = {
        "adv_h": k._horiz_adv(q, u, v),
        "adv_v": -w * k._d_z(q),
        "cond": cond,
    }
    # z: вклад членов T-уравнения через гидростатический интеграл + якорь.
    # ВСЕГДА от полного t_t (omega_free не трогает интегранд z).
    t_t_full = sum(terms_t.values())
    z_from_adv = k.get_z_t(terms_t["adv_h"] + terms_t["adv_v"])
    terms_z = {
        "from_adv": z_from_adv,
        "from_nonadv": k.get_z_t(t_t_full) - z_from_adv,
    }
    # omega_free: ω-члены исключаются из ВЫХОДНЫХ тенденций соответствующих
    # уравнений — декомпозиция обязана собирать те же члены, что и rhs.
    if "t" in k.omega_free:
        del terms_t["adv_v"], terms_t["adiabatic"]
    if "q" in k.omega_free:
        del terms_q["adv_v"]
    if "u" in k.omega_free:
        del terms_u["adv_v"]
    if "v" in k.omega_free:
        del terms_v["adv_v"]
    if k.z_anchor == "kinematic_ps":
        dps_dt = 100.0 * k._raw_column_w_top(u, v)
        terms_z["baro_anchor"] = (
            k.consts.R_d * t[:, -1:] / pressure[:, -1:] * dps_dt
        ).expand_as(t)
    return {"u": terms_u, "v": terms_v, "t": terms_t, "q": terms_q, "z": terms_z}


def latitude_bands(lat_deg: torch.Tensor) -> dict[str, torch.Tensor]:
    """Индексы строк широтных поясов: tropics |φ|<23.5, midlat, polar |φ|≥66.5."""
    bands = {
        "tropics": (lat_deg.abs() < 23.5).nonzero().flatten(),
        "midlat": ((lat_deg.abs() >= 23.5) & (lat_deg.abs() < 66.5)).nonzero().flatten(),
        "polar": (lat_deg.abs() >= 66.5).nonzero().flatten(),
    }
    return {name: rows for name, rows in bands.items() if rows.numel() > 0}


class ClimBuilder:
    """Накопление климатологии невязок против reference-ядра (только 2000).

    Два поля на каждую переменную:
      * ``zm``: (12 месяцев, P, H) — зонально-месячное среднее невязки;
      * ``map``: (24 часа UTC, P, H, W) — суточно-годовое среднее (t, q, z).
    Комбинация без двойного счёта: S12 = zm[m] + map[hh] − annual (annual —
    среднее zm по месяцам с весами выборки).
    """

    def __init__(self, n_levels: int, height: int, width: int) -> None:
        self.zm_sum = {
            v: torch.zeros(12, n_levels, height, dtype=torch.float64) for v in CLIM_VARS
        }
        self.zm_cnt = torch.zeros(12, dtype=torch.float64)
        self.map_sum = {
            v: torch.zeros(24, n_levels, height, width, dtype=torch.float32)
            for v in MAP_CLIM_VARS
        }
        self.map_cnt = torch.zeros(24, dtype=torch.float64)
        self.width = width

    def add(self, month: int, hour_of_day: int, err: dict[str, torch.Tensor]) -> None:
        for v in CLIM_VARS:
            self.zm_sum[v][month] += err[v].double().sum(dim=(0, 3)) / self.width
        self.zm_cnt[month] += 1
        for v in MAP_CLIM_VARS:
            self.map_sum[v][hour_of_day] += err[v][0].float()
        self.map_cnt[hour_of_day] += 1

    def finalize(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        zm_cnt = self.zm_cnt.clamp_min(1.0)
        map_cnt = self.map_cnt.clamp_min(1.0)
        total = float(self.zm_cnt.sum().clamp_min(1.0))
        for v in CLIM_VARS:
            zm = self.zm_sum[v] / zm_cnt.reshape(12, 1, 1)
            out[f"zm_{v}"] = zm.float().numpy()
            annual = (self.zm_sum[v].sum(dim=0) / total).float()
            out[f"annual_{v}"] = annual.numpy()
        for v in MAP_CLIM_VARS:
            out[f"map_{v}"] = (
                self.map_sum[v] / map_cnt.reshape(24, 1, 1, 1).float()
            ).numpy()
        out["zm_count"] = self.zm_cnt.numpy()
        out["map_count"] = self.map_cnt.numpy()
        return out


class ClimSource:
    """Выдаёт внешние источники rhs по (месяц, час UTC) из готовой климатологии."""

    def __init__(self, arrays: dict[str, np.ndarray], prefix: str) -> None:
        self.zm = {
            v: torch.from_numpy(arrays[f"{prefix}zm_{v}"]) for v in CLIM_VARS
        }  # (12,P,H)
        self.annual = {v: torch.from_numpy(arrays[f"{prefix}annual_{v}"]) for v in CLIM_VARS}
        self.map = {
            v: torch.from_numpy(arrays[f"{prefix}map_{v}"]) for v in MAP_CLIM_VARS
        }  # (24,P,H,W)

    def sources(self, mode: str, month: int, hour_of_day: int) -> dict[str, torch.Tensor]:
        """Источники {var: (1,P,H,1|W)}: 'zm' | 'map' | 'both' (без двойного счёта)."""
        if mode == "zm":
            return {v: self.zm[v][month].unsqueeze(0).unsqueeze(-1) for v in CLIM_VARS}
        if mode == "map":
            return {v: self.map[v][hour_of_day].unsqueeze(0) for v in MAP_CLIM_VARS}
        out: dict[str, torch.Tensor] = {}
        for v in CLIM_VARS:
            zm_part = self.zm[v][month].unsqueeze(0).unsqueeze(-1)
            if v in MAP_CLIM_VARS:
                out[v] = (
                    zm_part
                    + self.map[v][hour_of_day].unsqueeze(0)
                    - self.annual[v].unsqueeze(0).unsqueeze(-1)
                )
            else:
                out[v] = zm_part
        return out


def run() -> dict:
    """Полный набор статистик по одному домену и году."""
    dom = DOMAINS[DOMAIN]
    n_levels = 13
    kernels = {name: build_kernel(dom, gk, kw) for name, (gk, kw) in VARIANTS.items()}
    exact_grid = kernels["C_best"].grid
    lat_deg = exact_grid.latitudes * 180.0 / torch.pi

    clim_src: dict[str, ClimSource] | None = None
    active_source_variants: dict[str, tuple[str, str]] = {}
    if CLIM_IN:
        arrays = dict(np.load(CLIM_IN))
        clim_src = {ref: ClimSource(arrays, f"{ref}__") for ref in CLIM_REFS}
        active_source_variants = dict(SOURCE_VARIANTS)

    builders: dict[str, ClimBuilder] | None = None
    if CLIM_OUT:
        builders = {
            ref: ClimBuilder(n_levels, dom["H"], dom["W"]) for ref in CLIM_REFS
        }

    reader = None
    if SYNTHETIC:
        n_hours = 72
    else:
        reader = Era5Reader(dom, exact_grid)
        n_hours = reader.n_hours
    triple_starts = list(range(0, n_hours - 2, STRIDE))
    if MAX_TRIPLES > 0:
        triple_starts = triple_starts[:MAX_TRIPLES]

    all_names = list(VARIANTS) + list(active_source_variants)
    residual = {name: {v: RatioAccum() for v in ALL_KEYS} for name in all_names}
    interior = {name: {v: RatioAccum() for v in ALL_KEYS} for name in all_names}
    by_level = {name: {v: LevelAccum(n_levels) for v in ALL_KEYS} for name in FOCUS}
    bands = latitude_bands(lat_deg)
    by_band = {
        name: {band: {v: RatioAccum() for v in ALL_KEYS} for band in bands} for name in FOCUS
    }
    lat_profile = {name: {v: RowAccum(dom["H"]) for v in ALL_KEYS} for name in FOCUS}
    terms_cum = {name: {v: {} for v in ALL_KEYS} for name in FOCUS}
    terms_drop = {name: {v: {} for v in ALL_KEYS} for name in FOCUS}
    term_mag = {name: {v: {} for v in ALL_KEYS} for name in FOCUS}
    temporal = {
        name: {v: {m: RatioAccum() for m in ("fwd", "centered", "trapezoid")} for v in ALL_KEYS}
        for name in FOCUS
    }
    omega_corr = {name: CorrAccum() for name in ("W_plain", "C_best", "W_obrien")}
    omega_mag = {name: RatioAccum() for name in ("W_plain", "C_best", "W_obrien", "implied")}
    hydro = {
        "fixed": CorrAccum(),
        "kinematic_ps": CorrAccum(),
        "perfect_baro": CorrAccum(),
        "mag_fixed": RatioAccum(),
        "mag_perfect": RatioAccum(),
    }
    map_names = [n for n in MAP_SET if n in all_names]
    map_num = {
        name: {v: torch.zeros(dom["H"], dom["W"], dtype=torch.float64) for v in ALL_KEYS}
        for name in map_names
    }
    map_den = {v: torch.zeros(dom["H"], dom["W"], dtype=torch.float64) for v in ALL_KEYS}

    consts = kernels["C_best"].consts
    pressure = exact_grid.pressure

    wall_start = time.time()
    for i, h in enumerate(triple_starts):
        if SYNTHETIC:
            s0, s1, s2 = (synthetic_state(dom, seed=3 * i + j) for j in range(3))
        else:
            s0, s1, s2 = reader.state(h), reader.state(h + 1), reader.state(h + 2)
        month = month_of_hour(h, YEAR)
        hour_of_day = h % 24
        obs_fwd = {key: (s1[key] - s0[key]) / DT_OBS for key in ALL_KEYS}
        obs_cen = {key: (s2[key] - s0[key]) / (2 * DT_OBS) for key in ALL_KEYS}

        with torch.no_grad():
            rhs0 = {name: kernels[name].rhs(**s0) for name in VARIANTS}
            model_t = {name: {v: rhs0[name][f"{v}_t"] for v in ALL_KEYS} for name in VARIANTS}
            if clim_src is not None:
                for sname, (ref, mode) in active_source_variants.items():
                    src = clim_src[ref].sources(mode, month, hour_of_day)
                    model_t[sname] = {
                        v: model_t[ref][v] + src[v] if v in src else model_t[ref][v]
                        for v in ALL_KEYS
                    }
            for name in all_names:
                for var in ALL_KEYS:
                    err = obs_fwd[var] - model_t[name][var]
                    residual[name][var].add(err, obs_fwd[var])
                    interior[name][var].add(
                        err[..., 2:-2, 2:-2], obs_fwd[var][..., 2:-2, 2:-2]
                    )
                    if name in map_num:
                        map_num[name][var] += err.double().abs().sum(dim=(0, 1))
            for var in ALL_KEYS:
                map_den[var] += obs_fwd[var].double().abs().sum(dim=(0, 1))

            if builders is not None:
                for ref in CLIM_REFS:
                    builders[ref].add(
                        month,
                        hour_of_day,
                        {v: obs_fwd[v] - model_t[ref][v] for v in CLIM_VARS},
                    )

            for name in FOCUS:
                k = kernels[name]
                rhs_f = rhs0[name]
                for var in ALL_KEYS:
                    err = obs_fwd[var] - rhs_f[f"{var}_t"]
                    by_level[name][var].add(err, obs_fwd[var])
                    lat_profile[name][var].add(err, obs_fwd[var])
                    for band, rows in bands.items():
                        by_band[name][band][var].add(
                            err.index_select(2, rows), obs_fwd[var].index_select(2, rows)
                        )
                terms = eq_terms(k, s0)
                for var in ALL_KEYS:
                    tnames = list(terms[var])
                    full = sum(terms[var].values())
                    for j in range(1, len(tnames) + 1):
                        stage = "+".join(tnames[:j])
                        acc = terms_cum[name][var].setdefault(stage, RatioAccum())
                        acc.add(obs_fwd[var] - sum(terms[var][t] for t in tnames[:j]), obs_fwd[var])
                    for t_name, t_val in terms[var].items():
                        acc = terms_drop[name][var].setdefault(t_name, RatioAccum())
                        acc.add(obs_fwd[var] - (full - t_val), obs_fwd[var])
                        mag = term_mag[name][var].setdefault(t_name, RatioAccum())
                        mag.add(t_val, obs_fwd[var])
                rhs1 = kernels[name].rhs(**s1)
                for var in ALL_KEYS:
                    temporal[name][var]["fwd"].add(
                        obs_fwd[var] - rhs_f[f"{var}_t"], obs_fwd[var]
                    )
                    temporal[name][var]["centered"].add(
                        obs_cen[var] - rhs1[f"{var}_t"], obs_cen[var]
                    )
                    trap = 0.5 * (rhs_f[f"{var}_t"] + rhs1[f"{var}_t"])
                    temporal[name][var]["trapezoid"].add(obs_fwd[var] - trap, obs_fwd[var])

            # ω-тест: kinematic (plain / mc / obrien) против residual-implied.
            k = kernels["C_best"]
            t0 = s0["t"]
            t_z = k._d_z(t0)
            coef = consts.R_d * t0 / (consts.c_p * pressure) - t_z * (-0.01)
            rhs_obs = obs_fwd["t"] - k._horiz_adv(t0, s0["u"], s0["v"])
            omega_implied = (
                rhs_obs
                / torch.where(coef.abs() < 1e-9, torch.full_like(coef, 1e-9), coef)
            ).clamp(-10.0, 10.0)
            omega_mag["implied"].add(omega_implied, omega_implied)
            for name in ("W_plain", "C_best", "W_obrien"):
                om = -100.0 * rhs0[name]["w"]
                omega_corr[name].add(om, omega_implied)
                omega_mag[name].add(om, om)

            # Гидростатический контроль якоря: z_t из НАБЛЮДЁННОЙ T_t.
            z_t_hyd = k.get_z_t(obs_fwd["t"])
            hydro["fixed"].add(z_t_hyd, obs_fwd["z"])
            hydro["mag_fixed"].add(z_t_hyd - obs_fwd["z"], obs_fwd["z"])
            dps_dt = 100.0 * k._raw_column_w_top(s0["u"], s0["v"])
            baro_kin = consts.R_d * t0[:, -1:] / pressure[:, -1:] * dps_dt
            hydro["kinematic_ps"].add(z_t_hyd + baro_kin, obs_fwd["z"])
            pz = exact_grid.pixel_z
            baro_perfect = ((obs_fwd["z"] - z_t_hyd) * pz).sum(dim=1, keepdim=True) / pz.sum()
            hydro["perfect_baro"].add(z_t_hyd + baro_perfect, obs_fwd["z"])
            hydro["mag_perfect"].add(z_t_hyd + baro_perfect - obs_fwd["z"], obs_fwd["z"])

        if (i + 1) % 200 == 0 or i == 0:
            rate = (i + 1) / (time.time() - wall_start)
            print(
                f"[{DOMAIN}/{YEAR}] triple {i + 1}/{len(triple_starts)} "
                f"hour={h} rate={rate:.2f}/s",
                flush=True,
            )

    if reader is not None:
        reader.close()

    if builders is not None:
        arrays: dict[str, np.ndarray] = {}
        for ref in CLIM_REFS:
            for key, arr in builders[ref].finalize().items():
                arrays[f"{ref}__{key}"] = arr
        arrays["lat_deg"] = lat_deg.numpy().astype(np.float32)
        np.savez_compressed(CLIM_OUT, **arrays)
        print("WROTE", CLIM_OUT)

    if MAPS_OUT:
        arrays = {
            "lat": lat_deg.numpy().astype(np.float32),
            "lon": (np.arange(256, dtype=np.float32) * 1.40625)[dom["cols"]],
        }
        if SYNTHETIC:
            arrays["lsm"] = np.zeros((dom["H"], dom["W"]), dtype=np.float32)
        else:
            with h5netcdf.File(f"{ERA5_ROOT}constants/constants_1.40625deg.nc", "r") as f:
                lsm_global = np.asarray(f.variables["lsm"], dtype=np.float32)
            arrays["lsm"] = lsm_global[dom["rows"], dom["cols"]]
        for name in map_num:
            for var in ALL_KEYS:
                ratio = map_num[name][var] / (map_den[var] + 1e-30)
                arrays[f"resmap_{name}_{var}"] = ratio.numpy().astype(np.float32)
        np.savez_compressed(MAPS_OUT, **arrays)
        print("WROTE", MAPS_OUT)

    def ratios_of(accs: dict) -> dict:
        return {key: acc.ratio() for key, acc in accs.items()}

    return {
        "meta": {
            "year": YEAR,
            "domain": DOMAIN,
            "stride_hours": STRIDE,
            "n_triples": len(triple_starts),
            "n_hours_in_year": n_hours,
            "dt_obs_s": DT_OBS,
            "synthetic": SYNTHETIC,
            "clim_in": CLIM_IN,
            "clim_out": CLIM_OUT,
            "variants": {name: [gk, kw] for name, (gk, kw) in VARIANTS.items()},
            "source_variants": active_source_variants,
            "wall_seconds": time.time() - wall_start,
        },
        "residual_rel": {
            name: {v: acc.ratio() for v, acc in accs.items()} for name, accs in residual.items()
        },
        "residual_rms": {
            name: {v: acc.rms() for v, acc in accs.items()} for name, accs in residual.items()
        },
        "residual_rms_rel": {
            name: {v: acc.rms_ratio() for v, acc in accs.items()}
            for name, accs in residual.items()
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
                var: {
                    "cumulative": ratios_of(terms_cum[name][var]),
                    "drop_one": ratios_of(terms_drop[name][var]),
                    "term_abs_over_obs": ratios_of(term_mag[name][var]),
                }
                for var in ALL_KEYS
            }
            for name in FOCUS
        },
        "temporal_scheme": {
            name: {
                var: {m: acc.ratio() for m, acc in accs.items()}
                for var, accs in temporal[name].items()
            }
            for name in FOCUS
        },
        "omega_test": {
            "corr_vs_implied": {name: acc.corr() for name, acc in omega_corr.items()},
            "absmean_pa_s": {
                name: acc.num / max(acc.count, 1.0) for name, acc in omega_mag.items()
            },
        },
        "hydrostatic_anchor_test": {
            "corr_fixed": hydro["fixed"].corr(),
            "corr_kinematic_ps": hydro["kinematic_ps"].corr(),
            "corr_perfect_baro": hydro["perfect_baro"].corr(),
            "rel_err_fixed": hydro["mag_fixed"].ratio(),
            "rel_err_perfect_baro": hydro["mag_perfect"].ratio(),
        },
    }


def main() -> None:
    """Гоняет домен/год, пишет JSON в $OUT, печатает сводку."""
    results = run()
    Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("WROTE", OUT)
    print(f"--- {DOMAIN}/{YEAR}: residual_rel (u, v, t, q, z) ---")
    for name, r in results["residual_rel"].items():
        print(f"  {name:14s}: " + "  ".join(f"{var}={r[var]:7.3f}" for var in ALL_KEYS))


if __name__ == "__main__":
    main()
