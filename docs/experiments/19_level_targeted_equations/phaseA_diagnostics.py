"""Эксперимент 19, фаза A: Экмановское трение — снижает ли оно per-level невязку импульса.

Мотивация: exp18 (`18_level_resolved_physics/level_diagnostics.py`) показал, что
физика PurePDEKernel (якорь ``C15_now`` ≈ арм R5) помогает в свободной
тропосфере (``residual_rel ≲ 1``), но вредит в погранслое ≲850 гПа
(``residual_rel`` → 3.6 для импульса). Гипотеза фазы A: недостающий член —
турбулентное перемешивание импульса (Экмановское трение), которое в реальной
атмосфере доминирует именно в погранслое и отсутствует в текущем уравнении
импульса (там есть только Рэлеевское демпфирование, не диффузия по ``p``).

Формулировка: ∂/∂p(K(p) ∂u/∂p), ∂/∂p(K(p) ∂v/∂p) — вертикальная диффузия
импульса с коэффициентом K, пикующим у поверхности и нулевым в свободной
тропосфере (``ekman_profile``). Считается через тот же вертикальный оператор
``k._d_z`` (= −∂/∂p, идентичен ``eq_terms`` exp18), чтобы калибровка K была
согласована с обучаемым physics-ядром.

Это ОФФЛАЙН-ЗАМЕР на реальном ERA5, не обучение (стиль exp18): для якоря
``C15_now`` (R5) прогоняется sweep по K0 ∈ [0, 5e5, 1e6, 2e6, 5e6, 1e7] и
накапливается per-level невязка ``residual_rel`` u,v с добавкой Экман-члена
к правой части. Критерий приёмки: падение средней невязки на 925/1000 гПа
(индексы 11,12) ≥20% против K0=0 при росте невязки на ≤700 гПа (индексы 0..9)
не более 5%.

Дополнительно (r-часть): по тому же якорю C15_now сравниваются две схемы
восстановления тенденции относительной влажности r=q/q_s из модельных
``q_t``, ``t_t`` — ``as_is`` (полная T-зависимость q_s через ∂q_s/∂T·t_t) и
``cc_bridge`` (T-зависимость считается уже снятой мостом Клаузиуса-Клапейрона
в ``q_evolution``, см. ``utils/physics_hybrid.py``). Обе сравниваются с
наблюдаемой ERA5-тенденцией r. Критерий приёмки (печать): невязка cc_bridge
ниже as_is на уровнях 600–1000 гПа (индексы 8..12).

Каркас (REPO_ROOT-ленивый импорт, ``Era5Reader``, ``synthetic_state``,
``LevelAccum``, ``build_kernel``, ``data_latitudes_deg``) — самодостаточная
копия из ``18_level_resolved_physics/level_diagnostics.py`` (каждая подпапка
экспов самодостаточна, см. docs/experiments/README.md).

Запуск (кластер):  OUT=results/eq19A_usa_2004.json YEAR=2004 STRIDE=1 \
                   REPO_ROOT=~/wt_fix_v2 python phaseA_diagnostics.py
Локальный smoke:    OUT=/tmp/x.json SYNTHETIC=1 MAX_TRIPLES=4 python phaseA_diagnostics.py
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
from utils.physics_hybrid import (  # noqa: E402
    relative_to_specific_humidity,
    saturation_specific_humidity,
    specific_to_relative_humidity,
)

ERA5_ROOT = os.environ.get("ERA5_ROOT", "/home/fratnikov/weather_bench/1.40625deg/")
YEAR = int(os.environ.get("YEAR", "2004"))
STRIDE = int(os.environ.get("STRIDE", "1"))
MAX_TRIPLES = int(os.environ.get("MAX_TRIPLES", "0"))
SYNTHETIC = os.environ.get("SYNTHETIC", "0") == "1"
OUT = os.environ["OUT"]
DT_OBS = 3600.0
N_LEVELS = 13
PRESSURE_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
K0_VALUES = [0.0, 5e5, 1e6, 2e6, 5e6, 1e7]
# Все *_IDX ниже — полу-открытые срезы (lo, hi) в смысле Python [lo:hi).
BOUNDARY_LAYER_IDX = (11, 13)  # 925, 1000 hPa (индексы 11..12 включительно)
FREE_TROP_IDX = (0, 10)  # 50..700 hPa (индексы 0..9 включительно)
R_RESIDUAL_IDX = (8, 13)  # 600..1000 hPa (индексы 8..12 включительно)

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


def level_range_mean(*value_lists: list[float], idx_range: tuple[int, int]) -> float:
    """Среднее по полу-открытому срезу ``[lo:hi)`` уровней для одного/нескольких списков.

    Общий helper для критериев приёмки (Экман-sweep u,v; r-невязка as_is/cc_bridge)
    и для соответствующих print-блоков в ``main`` — избегает дублирования логики
    среза/усреднения (CLAUDE.md §1 DRY).

    Args:
        *value_lists: один или несколько списков per-level значений ``residual_rel``
            одинаковой длины (например ``entry["u"]``, ``entry["v"]``).
        idx_range: ``(lo, hi)`` — полу-открытый срез индексов уровней давления.

    Returns:
        float — среднее значение по объединённому срезу всех переданных списков.
    """
    lo, hi = idx_range
    combined: list[float] = []
    for values in value_lists:
        combined.extend(values[lo:hi])
    return float(np.mean(combined))


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
        """Состояние {u,v,t,q,z,r} формы (1, 13, 32, 64) в физических единицах.

        ``r`` — сырая относительная влажность ERA5 в процентах (для r-space
        диагностики фазы A); ``q`` — конвертированная удельная влажность,
        используемая ядром физики.
        """
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
            "r": fields["r"],
        }
        self.cache[hour] = state
        for old in [h for h in self.cache if h < hour - 2]:
            del self.cache[old]
        return state


def synthetic_state(seed: int) -> dict[str, torch.Tensor]:
    """Квази-реалистичное состояние для локального smoke (без ERA5).

    ``r`` строится из синтетических ``q``, ``t`` через
    ``specific_to_relative_humidity`` (тот же Магнус, что и q_s в
    ``r_tendency_from_q``), чтобы r-часть диагностики имела физически
    согласованный вход даже без реальных данных.
    """
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
    t_clim = torch.tensor(
        [217.0, 208, 213, 218, 223, 229, 242, 253, 262, 270, 279, 283, 287]
    ).reshape(1, 13, 1, 1)
    t = t_clim + 2 * torch.randn(1, 13, h, w, generator=gen)
    q = (3e-3 * (1 + 0.3 * torch.randn(1, 13, h, w, generator=gen))).clamp_min(1e-8)
    pressure_pa = torch.tensor(PRESSURE_HPA, dtype=torch.float32).reshape(1, 13, 1, 1) * 100.0
    return {
        "u": 10 + 5 * torch.randn(1, 13, h, w, generator=gen),
        "v": 5 * torch.randn(1, 13, h, w, generator=gen),
        "t": t,
        "q": q,
        "z": phi + 300 * torch.randn(1, 13, h, w, generator=gen),
        "r": specific_to_relative_humidity(q, t, pressure_pa),
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


# K(σ)-профиль: пиковый в погранслое, ноль в свободной тропосфере.
# σ = p/1000; profile[i] = K0 · max(0, (σ_i − σ_b)/(1 − σ_b)).
def ekman_profile(K0: float, sigma_b: float = 0.7) -> torch.Tensor:
    """Экмановский коэффициент вертикальной диффузии импульса K(σ).

    Args:
        K0: масштаб коэффициента диффузии (м²/с в единицах давления-стенсила).
        sigma_b: σ верхней границы погранслоя (по умолчанию 0.7 → ~700 гПа).

    Returns:
        torch.Tensor формы (1, 13, 1, 1) — профиль K по уровням давления.
    """
    sigma = torch.tensor(PRESSURE_HPA, dtype=torch.float32).reshape(1, -1, 1, 1) / 1000.0
    return K0 * torch.clamp((sigma - sigma_b) / (1.0 - sigma_b), min=0.0)


def ekman_term(k: PurePDEKernel, s: dict[str, torch.Tensor], K: torch.Tensor) -> dict[str, torch.Tensor]:
    """Экмановское трение ∂/∂p(K ∂/∂p u,v) через k._d_z (= −∂/∂p).

    Args:
        k: ядро PurePDEKernel (задаёт вертикальный оператор ``_d_z``).
        s: состояние {u,v,t,q,z}, каждое ``(B, 13, H, W)``.
        K: профиль коэффициента диффузии, ``(1, 13, 1, 1)`` (см. ``ekman_profile``).

    Returns:
        Добавки к тенденциям импульса: ``{"u_t": Tensor, "v_t": Tensor}``,
        каждая ``(B, 13, H, W)``.
    """
    return {
        "u_t": k._d_z(K * k._d_z(s["u"])),
        "v_t": k._d_z(K * k._d_z(s["v"])),
    }


def run_ekman_sweep(k: PurePDEKernel, triple_states: list[tuple[dict, dict]]) -> dict[str, dict[str, list[float]]]:
    """Per-level residual_rel u,v для якоря C15_now при добавке Экман-члена, sweep по K0.

    Args:
        k: ядро якоря C15_now (арм R5).
        triple_states: список пар (s0, s1) состояний, разделённых DT_OBS.

    Returns:
        ``{str(K0): {"u": [13 значений residual_rel], "v": [...]}}``.
    """
    k_profiles = {k0: ekman_profile(k0) for k0 in K0_VALUES}
    accum = {k0: {"u": LevelAccum(N_LEVELS), "v": LevelAccum(N_LEVELS)} for k0 in K0_VALUES}
    for s0, s1 in triple_states:
        obs_u = (s1["u"] - s0["u"]) / DT_OBS
        obs_v = (s1["v"] - s0["v"]) / DT_OBS
        with torch.no_grad():
            rhs = k.rhs(**{key: s0[key] for key in ALL_KEYS})
            for k0 in K0_VALUES:
                ek = ekman_term(k, s0, k_profiles[k0])
                u_t = rhs["u_t"] + ek["u_t"]
                v_t = rhs["v_t"] + ek["v_t"]
                accum[k0]["u"].add(obs_u - u_t, obs_u)
                accum[k0]["v"].add(obs_v - v_t, obs_v)
    return {str(k0): {"u": accum[k0]["u"].ratios(), "v": accum[k0]["v"].ratios()} for k0 in K0_VALUES}


def r_tendency_from_q(
    q_t: torch.Tensor,
    t_t: torch.Tensor,
    q: torch.Tensor,
    t: torch.Tensor,
    q_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Тенденция относительной влажности r=q/q_s из модельных q_t,t_t, две схемы.

    as_is: r_t = d(q/q_s)/dt ≈ q_t/q_s − q·q_s_t/q_s² (полная T-зависимость q_s
    через частную производную по времени). cc_bridge: r_t = q_t/q_s
    (T-зависимость снята мостом Клаузиуса-Клапейрона — см. ``q_evolution`` в
    ``utils/physics_hybrid.py``). q_s_t = q_s·(L/(R_v·T²))·t_t.

    Args:
        q_t: тенденция удельной влажности, ``(B, 13, H, W)``, кг/(кг·с).
        t_t: тенденция температуры, ``(B, 13, H, W)``, К/с.
        q: удельная влажность, ``(B, 13, H, W)``, кг/кг.
        t: температура, ``(B, 13, H, W)``, К.
        q_s: насыщенная удельная влажность, ``(B, 13, H, W)``, кг/кг.

    Returns:
        Кортеж ``(r_as_is, r_cc)`` тенденций r (доля/с), каждый формы ``q_t``.
    """
    L, R_v = 2.5e6, 461.5
    q_s_t = q_s * (L / (R_v * t * t)) * t_t
    r_as_is = q_t / q_s - q * q_s_t / (q_s * q_s)
    r_cc = q_t / q_s
    return r_as_is, r_cc


def r_space_residual(
    k: PurePDEKernel, triple_states: list[tuple[dict, dict]]
) -> dict[str, list[float]]:
    """Per-level residual_rel влажности r для схем as_is и cc_bridge, якорь C15_now.

    Сравнивает две схемы восстановления тенденции r=q/q_s из модельных q_t,t_t
    (``r_tendency_from_q``) с наблюдаемой ERA5-тенденцией r (``Era5Reader.state``
    сохраняет сырое поле ``"r"``, %).

    Args:
        k: ядро якоря C15_now (арм R5).
        triple_states: список пар (s0, s1) состояний, разделённых DT_OBS; каждое
            состояние содержит ключи u,v,t,q,z,r (``(B, 13, H, W)``, r в %).

    Returns:
        ``{"as_is": [13 значений residual_rel], "cc_bridge": [...]}``.
    """
    accum = {"as_is": LevelAccum(N_LEVELS), "cc_bridge": LevelAccum(N_LEVELS)}
    for s0, s1 in triple_states:
        obs_r = (s1["r"] - s0["r"]) / DT_OBS / 100.0
        with torch.no_grad():
            rhs = k.rhs(**{key: s0[key] for key in ALL_KEYS})
            q_s = saturation_specific_humidity(s0["t"], k.grid.pressure)
            r_as_is, r_cc = r_tendency_from_q(rhs["q_t"], rhs["t_t"], s0["q"], s0["t"], q_s)
        accum["as_is"].add(obs_r - r_as_is, obs_r)
        accum["cc_bridge"].add(obs_r - r_cc, obs_r)
    return {name: accumulator.ratios() for name, accumulator in accum.items()}


def evaluate_acceptance(ekman_sweep: dict[str, dict[str, list[float]]]) -> dict:
    """Критерий приёмки: лучший K0 по невязке в погранслое без порчи свободной тропосферы.

    Ищет K0, минимизирующий среднюю невязку u,v на 925/1000 гПа (индексы 11,12);
    принят, если падение ≥20% против K0=0 И невязка на ≤700 гПа (индексы 0..9)
    не выросла более чем на 5%.

    Args:
        ekman_sweep: результат ``run_ekman_sweep`` (ключи — str(K0)).

    Returns:
        ``{"best_k0": float, "boundary_drop_pct": float, "free_trop_increase_pct": float,
        "accepted": bool}``.
    """

    def boundary_score(entry: dict[str, list[float]]) -> float:
        return level_range_mean(entry["u"], entry["v"], idx_range=BOUNDARY_LAYER_IDX)

    def free_trop_score(entry: dict[str, list[float]]) -> float:
        return level_range_mean(entry["u"], entry["v"], idx_range=FREE_TROP_IDX)

    baseline_key = str(0.0)
    baseline_boundary = boundary_score(ekman_sweep[baseline_key])
    baseline_free_trop = free_trop_score(ekman_sweep[baseline_key])

    best_k0 = 0.0
    best_boundary = baseline_boundary
    for k0 in K0_VALUES:
        boundary = boundary_score(ekman_sweep[str(k0)])
        if boundary < best_boundary:
            best_boundary = boundary
            best_k0 = k0

    best_free_trop = free_trop_score(ekman_sweep[str(best_k0)])
    boundary_drop_pct = 100.0 * (baseline_boundary - best_boundary) / (baseline_boundary + 1e-30)
    free_trop_increase_pct = 100.0 * (best_free_trop - baseline_free_trop) / (baseline_free_trop + 1e-30)
    accepted = boundary_drop_pct >= 20.0 and free_trop_increase_pct <= 5.0
    return {
        "best_k0": best_k0,
        "boundary_drop_pct": boundary_drop_pct,
        "free_trop_increase_pct": free_trop_increase_pct,
        "accepted": accepted,
    }


def run() -> dict:
    """Прогон троек снимков для якоря C15_now (R5); Экман-sweep по K0."""
    kernel = build_kernel("exact", C15_KW)
    grid = kernel.grid

    reader = None
    if SYNTHETIC:
        n_hours = 72
    else:
        reader = Era5Reader(grid)
        n_hours = reader.n_hours
    triple_starts = list(range(0, n_hours - 2, STRIDE))
    if MAX_TRIPLES > 0:
        triple_starts = triple_starts[:MAX_TRIPLES]

    wall_start = time.time()
    triple_states: list[tuple[dict, dict]] = []
    for i, h in enumerate(triple_starts):
        if SYNTHETIC:
            s0, s1 = synthetic_state(2 * i), synthetic_state(2 * i + 1)
        else:
            s0, s1 = reader.state(h), reader.state(h + 1)
        triple_states.append((s0, s1))
        if (i + 1) % 500 == 0 or i == 0:
            rate = (i + 1) / (time.time() - wall_start)
            print(f"[usa/{YEAR}] triple {i + 1}/{len(triple_starts)} rate={rate:.2f}/s", flush=True)

    if reader is not None:
        reader.close()

    ekman_sweep = run_ekman_sweep(kernel, triple_states)
    acceptance = evaluate_acceptance(ekman_sweep)
    r_residual = r_space_residual(kernel, triple_states)

    return {
        "meta": {
            "year": YEAR,
            "domain": "usa",
            "stride_hours": STRIDE,
            "n_triples": len(triple_starts),
            "dt_obs_s": DT_OBS,
            "synthetic": SYNTHETIC,
            "pressure_hpa": PRESSURE_HPA,
            "anchor": "C15_now",
            "anchor_arm": "R5",
            "k0_values": K0_VALUES,
            "wall_seconds": time.time() - wall_start,
        },
        "ekman_sweep": ekman_sweep,
        "acceptance": acceptance,
        "r_residual": r_residual,
    }


def main() -> None:
    """Прогоняет USA/YEAR, пишет JSON в $OUT, печатает критерии приёмки."""
    results = run()
    Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("WROTE", OUT)
    print("--- ekman_sweep: residual_rel boundary-layer (925/1000 hPa) mean by K0 ---")
    for k0 in K0_VALUES:
        entry = results["ekman_sweep"][str(k0)]
        mean_bl = level_range_mean(entry["u"], entry["v"], idx_range=BOUNDARY_LAYER_IDX)
        print(f"  K0={k0:.1e}  boundary_mean={mean_bl:.4f}")
    acc = results["acceptance"]
    print(
        f"--- acceptance: best_k0={acc['best_k0']:.1e} "
        f"boundary_drop={acc['boundary_drop_pct']:.1f}% "
        f"free_trop_increase={acc['free_trop_increase_pct']:.1f}% "
        f"accepted={acc['accepted']} ---"
    )
    r_res = results["r_residual"]
    as_is_mean = level_range_mean(r_res["as_is"], idx_range=R_RESIDUAL_IDX)
    cc_mean = level_range_mean(r_res["cc_bridge"], idx_range=R_RESIDUAL_IDX)
    print(
        f"--- r_residual (600-1000 hPa): as_is={as_is_mean:.4f} cc_bridge={cc_mean:.4f} "
        f"cc_better={cc_mean < as_is_mean} ---"
    )


if __name__ == "__main__":
    main()
