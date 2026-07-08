"""Пост-фикс статистика физики на реальном ERA5 за 2000 год: USA-кроп vs глобус.

Прогоняет исправленное ядро (аудит 13: B1 знак адиабаты, B2 знак гидростатики,
B3 конденсация с реальным давлением) на парах часовых снимков ERA5 и снимает:

  1. PDE-невязки по переменным (u, v, t, q, z): Σ|obs−model|/Σ|obs| и RMS —
     для plain и mass_consistent ω; плюс эмуляция ДО-фиксовых знаков
     (t_t_pre = adv − adia; z_t_pre = −get_z_t(t_t_pre); q без конденсации),
     чтобы измерить вклад фикса на реальных данных.
  2. ω-тест (exp 10 test A c ПРАВИЛЬНОЙ конвенцией ω = −100·w): корреляция и
     величины кинематической ω (plain / mass_consistent) против
     residual-implied ω из адиабатического T-бюджета.
  3. Гидростатический тест (exp 10 test C): z_t_hyd = get_z_t(T_t_obs) против
     наблюдаемой z_t — корреляция (до фикса была −0.35) и |ratio|.
  4. Статистика конденсации: доля насыщенных точек (q ≥ q_s при реальном p),
     доля активного триггера δ = (ω<0 ∧ q≥q_s), величина источника и его
     отношение к адвекции q.
  5. Переоценка диабатического остатка Q (exp 10 test B, был ×48 при
     перевёрнутой адиабате): Σ|Q_est| / Σ|adia|.
  6. CFL по реальным ветрам (dt = 300 c, референс pure-physics).

Запуск (кластер): OUT=... N_PAIRS=96 python physics_stats_2000.py
Локальный smoke:   OUT=/tmp/x.json SYNTHETIC=1 N_PAIRS=2 python physics_stats_2000.py
"""

import json
import os
import sys
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
N_PAIRS = int(os.environ.get("N_PAIRS", "96"))
SYNTHETIC = os.environ.get("SYNTHETIC", "0") == "1"
OUT = os.environ["OUT"]
DT_OBS = 3600.0  # шаг между снимками, с
DT_CFL = 300.0  # референсный шаг pure-physics для CFL

# Домены: (имя, H, W, lat_range для Grid, срезы в глобальном поле, boundary_x).
# USA-кроп: строки 75:107 → широты 18.28125..61.875 c шагом 1.40625 ю→с;
# lat_range подобран так, что linspace(low, high, H+2)[1:-1] воспроизводит их
# точно: low = 18.28125 − 1.40625, high = 18.28125 + 32·1.40625.
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


def read_var(folder: str, name: str, hour: int, rows: slice, cols: slice) -> torch.Tensor:
    """Читает 13 уровней переменной ERA5 на час ``hour``, форма (1, 13, H, W)."""
    path = f"{ERA5_ROOT}{folder}/{folder}_{YEAR}_1.40625deg.nc"
    with h5netcdf.File(path, "r") as f:
        arr = np.asarray(f.variables[name][hour, 0:13, rows, cols], dtype=np.float32)
    return torch.from_numpy(arr)[None]


def load_state(hour: int, dom: dict, grid: Grid) -> dict[str, torch.Tensor]:
    """Состояние {u,v,t,q,z} в физических единицах (q — кг/кг из r%)."""
    fields = {
        short: read_var(folder, short, hour, dom["rows"], dom["cols"])
        for folder, short in VARS
    }
    q = relative_to_specific_humidity(fields["r"], fields["t"], grid.pressure).clamp_min(1e-8)
    return {"u": fields["u"], "v": fields["v"], "t": fields["t"], "q": q, "z": fields["z"]}


def synthetic_state(dom: dict, grid: Grid, seed: int) -> dict[str, torch.Tensor]:
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


class CorrAccum:
    """Стриминговая корреляция Пирсона по конкатенации всех пар (float64)."""

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


class RatioAccum:
    """Стриминговое отношение Σ|num|/Σ|den| и RMS числителя."""

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


def run_domain(name: str, dom: dict, hours: list[int]) -> dict:
    """Полный набор статистик по одному домену."""
    grid = Grid(GridConfig(H=dom["H"], W=dom["W"], lat_range_deg=dom["lat_range"]))
    kernels = {
        diag: PurePDEKernel(
            grid,
            stencil="fd4",
            coriolis="spherical",
            block_dt=DT_CFL,
            boundary_x=dom["boundary_x"],
            t_t_formulation="adiabatic_omega",
            w_diagnostic=diag,
        ).eval()
        for diag in ("plain", "mass_consistent")
    }
    k = kernels["plain"]
    pressure = grid.pressure  # (1, P, 1, 1), Па
    consts = k.consts

    residual = {
        mode: {v: RatioAccum() for v in ("u", "v", "t", "q", "z")}
        for mode in ("fixed_plain", "fixed_mc", "prefix_emulated")
    }
    omega_corr = {"plain": CorrAccum(), "mc": CorrAccum()}
    omega_mag = {"kin_plain": RatioAccum(), "kin_mc": RatioAccum(), "implied": RatioAccum()}
    hydro_corr = CorrAccum()
    hydro_mag = RatioAccum()
    cond = {
        "sat_frac_sum": 0.0,
        "active_frac_sum": 0.0,
        "n_pairs": 0,
        "cond_vs_qadv": RatioAccum(),
        "q_source_abs_sum": 0.0,
        "q_source_count": 0.0,
    }
    diabatic = {"plain": RatioAccum(), "mc": RatioAccum()}
    cfl = {"adv_max": 0.0, "adv_mean_sum": 0.0}

    for i, hour in enumerate(hours):
        if SYNTHETIC:
            s0 = synthetic_state(dom, grid, seed=2 * i)
            s1 = synthetic_state(dom, grid, seed=2 * i + 1)
        else:
            s0 = load_state(hour, dom, grid)
            s1 = load_state(hour + 1, dom, grid)
        u, v, t, q, z = s0["u"], s0["v"], s0["t"], s0["q"], s0["z"]
        obs = {key: (s1[key] - s0[key]) / DT_OBS for key in ("u", "v", "t", "q", "z")}

        with torch.no_grad():
            rhs_plain = kernels["plain"].rhs(u, v, t, q, z)
            rhs_mc = kernels["mass_consistent"].rhs(u, v, t, q, z)
            for var in ("u", "v", "t", "q", "z"):
                residual["fixed_plain"][var].add(obs[var] - rhs_plain[f"{var}_t"], obs[var])
                residual["fixed_mc"][var].add(obs[var] - rhs_mc[f"{var}_t"], obs[var])

            # --- Разложение t_t и эмуляция до-фиксовых знаков (B1/B2/B3) ---
            w = rhs_plain["w"]
            omega = -100.0 * w  # Па/с, правильная конвенция
            t_z = k.diff.d_z(t)
            adv_t = k._horiz_adv(t, u, v) - w * t_z
            adia = consts.R_d * t * omega / (consts.c_p * pressure)
            t_t_pre = adv_t - adia  # B1: перевёрнутая адиабата
            z_t_pre = -k.get_z_t(t_t_pre)  # B2: перевёрнутый интеграл
            q_z = k.diff.d_z(q)
            q_adv = k._horiz_adv(q, u, v) - w * q_z
            q_source = rhs_plain["q_t"] - q_adv  # конденсация (только фикс-ядро)
            residual["prefix_emulated"]["t"].add(obs["t"] - t_t_pre, obs["t"])
            residual["prefix_emulated"]["z"].add(obs["z"] - z_t_pre, obs["z"])
            residual["prefix_emulated"]["q"].add(obs["q"] - q_adv, obs["q"])
            residual["prefix_emulated"]["u"].add(obs["u"] - rhs_plain["u_t"], obs["u"])
            residual["prefix_emulated"]["v"].add(obs["v"] - rhs_plain["v_t"], obs["v"])

            # --- ω-тест (правильная конвенция) ---
            coef = consts.R_d * t / (consts.c_p * pressure) - t_z * (-0.01)
            # t_z = d_z(t) = −∂T/∂p_гПа → ∂T/∂p_Па = −t_z/100; coef = R_d·T/(c_p·p) − ∂T/∂p_Па
            rhs_obs = obs["t"] - k._horiz_adv(t, u, v)
            omega_implied = rhs_obs / torch.where(
                coef.abs() < 1e-9, torch.full_like(coef, 1e-9), coef
            )
            omega_implied = omega_implied.clamp(-10.0, 10.0)
            omega_mc = -100.0 * rhs_mc["w"]
            omega_corr["plain"].add(omega, omega_implied)
            omega_corr["mc"].add(omega_mc, omega_implied)
            omega_mag["kin_plain"].add(omega, omega)
            omega_mag["kin_mc"].add(omega_mc, omega_mc)
            omega_mag["implied"].add(omega_implied, omega_implied)

            # --- Гидростатический тест: z_t из наблюдаемой T_t ---
            z_t_hyd = k.get_z_t(obs["t"])
            hydro_corr.add(z_t_hyd, obs["z"])
            hydro_mag.add(z_t_hyd, obs["z"])

            # --- Конденсация ---
            q_s = torch.maximum(k._get_qs(pressure, t), torch.full_like(q, 1e-6))
            sat = (q >= q_s).double()
            active = ((omega < 0) & (q >= q_s)).double()
            cond["sat_frac_sum"] += float(sat.mean())
            cond["active_frac_sum"] += float(active.mean())
            cond["n_pairs"] += 1
            cond["cond_vs_qadv"].add(q_source, q_adv)
            cond["q_source_abs_sum"] += float(q_source.double().abs().sum())
            cond["q_source_count"] += q_source.numel()

            # --- Диабатический остаток Q/c_p против адиабаты ---
            diabatic["plain"].add(obs["t"] - rhs_plain["t_t"], adia)
            adia_mc = consts.R_d * t * omega_mc / (consts.c_p * pressure)
            diabatic["mc"].add(obs["t"] - rhs_mc["t_t"], adia_mc)

            # --- CFL ---
            cfl_x = (u.abs() * DT_CFL / grid.pixel_x).max()
            cfl_y = (v.abs() * DT_CFL / grid.pixel_y).max()
            cfl["adv_max"] = max(cfl["adv_max"], float(cfl_x), float(cfl_y))
            cfl["adv_mean_sum"] += float((u.abs() * DT_CFL / grid.pixel_x).mean())
        print(f"[{name}] pair {i + 1}/{len(hours)} hour={hour} done", flush=True)

    n = max(cond["n_pairs"], 1)
    return {
        "residual_rel": {
            mode: {var: acc.ratio() for var, acc in accs.items()}
            for mode, accs in residual.items()
        },
        "residual_rms": {
            mode: {var: acc.rms() for var, acc in accs.items()}
            for mode, accs in residual.items()
        },
        "omega_test": {
            "corr_kin_plain_vs_implied": omega_corr["plain"].corr(),
            "corr_kin_mc_vs_implied": omega_corr["mc"].corr(),
            "absmean_kin_plain_pa_s": omega_mag["kin_plain"].num
            / max(omega_mag["kin_plain"].count, 1.0),
            "absmean_kin_mc_pa_s": omega_mag["kin_mc"].num / max(omega_mag["kin_mc"].count, 1.0),
            "absmean_implied_pa_s": omega_mag["implied"].num
            / max(omega_mag["implied"].count, 1.0),
        },
        "hydrostatic_test": {
            "corr_z_t_hyd_vs_obs": hydro_corr.corr(),
            "mag_ratio_hyd_over_obs": hydro_mag.ratio(),
        },
        "condensation": {
            "saturated_fraction": cond["sat_frac_sum"] / n,
            "active_trigger_fraction": cond["active_frac_sum"] / n,
            "source_absmean_kg_kg_s": cond["q_source_abs_sum"] / max(cond["q_source_count"], 1.0),
            "source_over_advection": cond["cond_vs_qadv"].ratio(),
        },
        "diabatic_residual_over_adiabatic": {
            "plain": diabatic["plain"].ratio(),
            "mass_consistent": diabatic["mc"].ratio(),
        },
        "cfl_dt300": {"adv_max": cfl["adv_max"], "adv_mean": cfl["adv_mean_sum"] / n},
        "n_pairs": cond["n_pairs"],
    }


def main() -> None:
    """Считает статистику по обоим доменам и пишет JSON в $OUT."""
    hours = [int(h) for h in np.linspace(0, 8760 - 2, N_PAIRS)]
    results = {
        "meta": {
            "year": YEAR,
            "n_pairs": N_PAIRS,
            "dt_obs_s": DT_OBS,
            "dt_cfl_s": DT_CFL,
            "synthetic": SYNTHETIC,
            "stencil": "fd4",
            "note": (
                "fixed = ядро после аудита 13; prefix_emulated = знаки B1/B2 "
                "перевёрнуты и конденсация отключена (алгебраическая эмуляция "
                "до-фиксового кода); omega = -100*w (правильная конвенция)"
            ),
        }
    }
    for name, dom in DOMAINS.items():
        print(f"=== domain {name} ===", flush=True)
        results[name] = run_domain(name, dom, hours)
    Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("WROTE", OUT)
    for name in DOMAINS:
        r = results[name]
        print(f"--- {name}: rel residual (fixed_plain / prefix_emulated) ---")
        for var in ("u", "v", "t", "q", "z"):
            fixed = r["residual_rel"]["fixed_plain"][var]
            pre = r["residual_rel"]["prefix_emulated"][var]
            print(f"  {var}: {fixed:8.3f}  /  {pre:8.3f}")
        print(
            f"  omega corr plain={r['omega_test']['corr_kin_plain_vs_implied']:+.3f} "
            f"mc={r['omega_test']['corr_kin_mc_vs_implied']:+.3f}; "
            f"hydro corr={r['hydrostatic_test']['corr_z_t_hyd_vs_obs']:+.3f}"
        )


if __name__ == "__main__":
    main()
