"""Знаковые и единичные конвенции физических ядер (аудит 2026-07-08).

Внутренняя вертикальная конвенция обоих ядер — ζ = −p: ``d_z`` возвращает
−∂/∂p, а ``get_w`` — вверх-положительную скорость w = −ω (гПа/с). Адвективные
члены в этой конвенции самосогласованы (двойные инверсии знака гасятся) и
запинены здесь золотыми значениями. Тесты ниже фиксируют места, где код обязан
конвертировать w → ω = −100·w (адиабатический член, конденсация) и знак
гидростатического интеграла z_t (см. docs/experiments/13_sign_convention_fix).

Запуск: ``.venv/bin/python -m unittest tests/test_physics_sign_conventions.py``.
"""

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.physics import Grid, GridConfig, PurePDEKernel  # noqa: E402
from utils.physics_hybrid import PDE_kernel, pressure  # noqa: E402

H, W, P = 8, 16, 13
PHI_STD = torch.tensor(
    [199300.0, 157400, 134000, 116600, 102300, 90000, 69700, 54100, 40700, 28600, 13500, 7300, 800]
).reshape(1, P, 1, 1)
T_STD = torch.tensor([217.0, 208, 213, 218, 223, 229, 242, 253, 262, 270, 279, 283, 287]).reshape(
    1, P, 1, 1
)


def make_hybrid_kernel(t_t_formulation: str = "adiabatic_omega") -> PDE_kernel:
    """Ядро physics_hybrid на геометрии латента USA-v4 (8x16, 5.625 град)."""
    return PDE_kernel(
        in_dim=65,
        physics_part_coef=0.5,
        block_dt=400.0,
        lat_start_deg=18.28125,
        dlat_deg=5.625,
        dlon_deg=5.625,
        grid_h=H,
        t_t_formulation=t_t_formulation,
    )


def make_pure_kernel() -> PurePDEKernel:
    """PurePDEKernel из utils.physics на сопоставимом региональном кропе."""
    grid = Grid(GridConfig(H=H, W=W, lat_range_deg=(16.17, 59.77)))
    return PurePDEKernel(grid, boundary_x="replicate")


def divergent_wind(kernel: PDE_kernel, div: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Поле (u, v) с однородной горизонтальной дивергенцией ``div`` [1/с]."""
    x_idx = torch.arange(W, dtype=torch.float32).reshape(1, 1, 1, W)
    u = (div * kernel.pixel_x * x_idx).expand(1, P, H, W).clone()
    return u, torch.zeros_like(u)


def interior(field: torch.Tensor) -> torch.Tensor:
    """Внутренность поля вне зоны влияния WENO-паддинга (2 ячейки с краёв)."""
    return field[..., 2:-2, 4:-4]


class HybridConventionPins(unittest.TestCase):
    """Пины существующих КОРРЕКТНЫХ конвенций physics_hybrid (должны жить вечно)."""

    def test_get_w_is_upward_positive_minus_omega(self) -> None:
        """Однородная дивергенция (физически ω>0, субсиденция) даёт w < 0."""
        kernel = make_hybrid_kernel()
        u, v = divergent_wind(kernel, 1e-5)
        w = kernel.get_w(u, v)
        self.assertLess(float(interior(w[:, 6]).mean()), -1e-3)

    def test_momentum_vertical_advection_sign(self) -> None:
        """Субсиденция при du/dp<0 разгоняет ветер: du/dt = -omega*du/dp > 0."""
        kernel = make_hybrid_kernel()
        p_hpa = pressure.float()
        u = (60.0 - 0.04 * p_hpa).expand(1, P, H, W).clone()
        v = torch.zeros_like(u)
        w_code = torch.full_like(u, -1e-3)  # omega = +0.1 Па/с (субсиденция)
        kernel.share_z_dxyz(torch.zeros_like(u))
        u_t, _ = kernel.get_uv_dt(u, v, w_code)
        got = float(interior(u_t[:, 6]).mean())
        self.assertAlmostEqual(got, 4.0e-5, delta=1.0e-5)

    def test_temperature_vertical_advection_sign(self) -> None:
        """Субсиденция при dT/dp>0 даёт -omega*dT/dp = -9e-5 K/с (адвекция)."""
        kernel = make_hybrid_kernel(t_t_formulation="legacy_paper")
        p_hpa = pressure.float()
        t = (200.0 + 0.09 * p_hpa).expand(1, P, H, W).clone()
        zeros = torch.zeros_like(t)
        w_code = torch.full_like(t, -1e-3)
        kernel.share_z_dxyz(zeros)  # z=0 => legacy Q-член зануляется, остаётся адвекция
        t_t = kernel.get_t_t(zeros, zeros, w_code, t)
        got = float(interior(t_t[:, 6]).mean())
        self.assertAlmostEqual(got, -9.0e-5, delta=2.0e-5)

    def test_legacy_paper_t_t_is_byte_stable(self) -> None:
        """Легаси-ветка t_t (регрессионный флаг) не должна меняться фиксами."""
        torch.manual_seed(7)
        gen = torch.Generator().manual_seed(7)
        p_hpa = pressure.float()
        phi = PHI_STD.expand(1, P, H, W).clone()
        t0 = (200.0 + 0.09 * p_hpa).expand(1, P, H, W) + 2 * torch.randn(1, P, H, W, generator=gen)
        u0 = 10 + 5 * torch.randn(1, P, H, W, generator=gen)
        v0 = 5 * torch.randn(1, P, H, W, generator=gen)
        kernel = make_hybrid_kernel(t_t_formulation="legacy_paper")
        w0 = kernel.get_w(u0, v0)
        kernel.share_z_dxyz(phi)
        t_t = kernel.get_t_t(u0, v0, w0, t0)
        self.assertAlmostEqual(float(t_t.double().sum()), -87058.82842332125, delta=1e-3)
        self.assertAlmostEqual(float(t_t[0, 6, 3, 7]), -1238.197265625, delta=1e-3)

    def test_momentum_tendency_is_byte_stable(self) -> None:
        """get_uv_dt (нетронутый фиксами путь) бит-стабилен на пиновом состоянии."""
        torch.manual_seed(7)
        gen = torch.Generator().manual_seed(7)
        p_hpa = pressure.float()
        phi = PHI_STD.expand(1, P, H, W).clone()
        t0 = (200.0 + 0.09 * p_hpa).expand(1, P, H, W) + 2 * torch.randn(1, P, H, W, generator=gen)
        del t0  # состояние генератора должно совпасть со скриптом захвата голденов
        u0 = 10 + 5 * torch.randn(1, P, H, W, generator=gen)
        v0 = 5 * torch.randn(1, P, H, W, generator=gen)
        kernel = make_hybrid_kernel()
        w0 = kernel.get_w(u0, v0)
        kernel.share_z_dxyz(phi)
        u_t, v_t = kernel.get_uv_dt(u0, v0, w0)
        self.assertAlmostEqual(float(w0[0, 6, 3, 7]), 0.0026476336643099785, delta=1e-9)
        self.assertAlmostEqual(float(u_t[0, 6, 3, 7]), 0.00021699792705476284, delta=1e-9)
        self.assertAlmostEqual(float(v_t[0, 6, 3, 7]), -0.0013110688887536526, delta=1e-9)


class HybridSignFixes(unittest.TestCase):
    """Знаковые фиксы physics_hybrid: адиабата, гидростатика, конденсация."""

    def test_adiabatic_subsidence_warms(self) -> None:
        """Однородная дивергенция => субсиденция => адиабатический НАГРЕВ (t_t>0)."""
        kernel = make_hybrid_kernel()
        u, v = divergent_wind(kernel, 1e-5)
        w = kernel.get_w(u, v)
        t = torch.full((1, P, H, W), 250.0)
        kernel.share_z_dxyz(torch.zeros_like(t))
        t_t = kernel.get_t_t(u, v, w, t)
        self.assertGreater(float(interior(t_t[:, 6]).mean()), 1e-4)

    def test_hydrostatic_warming_raises_geopotential(self) -> None:
        """Однородный прогрев столба увеличивает толщину: z_t > 0 выше поверхности."""
        kernel = make_hybrid_kernel()
        kernel.t_t = torch.full((1, P, H, W), 1e-4)
        z_t = kernel.get_z_t()
        self.assertGreater(float(z_t[0, 0, 0, 0]), 0.0)  # 50 гПа
        self.assertGreater(float(z_t[0, 7, 0, 0]), 0.0)  # 500 гПа
        self.assertGreater(float(z_t[0, 0, 0, 0]), float(z_t[0, 7, 0, 0]))

    def _q_dt_for_prescribed_w(self, w_value: float, q_value: float) -> torch.Tensor:
        """q_t при нулевых ветрах, T=280K, однородном q и заданной w [гПа/с].

        Однородное по уровням q зануляет всю адвекцию (q_x=q_y=q_z=0), поэтому
        ненулевой q_t может дать только конденсационный источник. При T=280K
        q_s(1000 гПа) ~= 6.2e-3: q=6.5e-3 насыщает только придонный уровень.
        """
        kernel = make_hybrid_kernel()
        t = torch.full((1, P, H, W), 280.0)
        zeros = torch.zeros_like(t)
        q = torch.full_like(t, q_value)
        w = torch.full_like(t, w_value)
        kernel.share_z_dxyz(PHI_STD.expand(1, P, H, W).clone())
        kernel.t_t = torch.zeros_like(t)
        kernel.get_z_t()
        return kernel.get_q_dt(zeros, zeros, t, w, q)

    def test_saturated_ascent_condenses(self) -> None:
        """Насыщенный подъём (w>0, т.е. ω<0) сушит придонный уровень физично."""
        q_t = self._q_dt_for_prescribed_w(w_value=1e-3, q_value=6.5e-3)
        bottom = float(interior(q_t[:, 12]).mean())
        self.assertLess(bottom, 0.0)
        self.assertGreater(abs(bottom), 1e-9)  # F*|omega|/p ~ 1e-8 кг/кг/с
        self.assertLess(abs(bottom), 1e-6)
        self.assertLess(float(q_t[:, 7].abs().max()), 1e-12)  # 500 гПа не насыщен

    def test_subsaturated_ascent_has_no_condensation(self) -> None:
        """Без насыщения источник конденсации равен нулю (q_t = чистая адвекция = 0)."""
        q_t = self._q_dt_for_prescribed_w(w_value=1e-3, q_value=3e-3)
        self.assertLess(float(q_t.abs().max()), 1e-12)

    def test_saturated_descent_does_not_condense(self) -> None:
        """Опускание (w<0, ω>0) не должно вызывать конденсацию даже при q>=q_s."""
        q_t = self._q_dt_for_prescribed_w(w_value=-1e-3, q_value=6.5e-3)
        self.assertLess(float(q_t.abs().max()), 1e-12)


class PureKernelSignFixes(unittest.TestCase):
    """Те же конвенции в диагностической библиотеке utils.physics."""

    def test_adiabatic_subsidence_warms(self) -> None:
        """rhs: однородная дивергенция даёт адиабатический нагрев (t_t > 0)."""
        kernel = make_pure_kernel()
        x_idx = torch.arange(W, dtype=torch.float32).reshape(1, 1, 1, W)
        u = (1e-5 * kernel.grid.pixel_x * x_idx).expand(1, P, H, W).clone()
        state = {
            "u": u,
            "v": torch.zeros_like(u),
            "t": torch.full_like(u, 250.0),
            "q": torch.full_like(u, 1e-4),
            "z": PHI_STD.expand(1, P, H, W).clone(),
        }
        rhs = kernel.rhs(**state)
        self.assertGreater(float(interior(rhs["t_t"][:, 6]).mean()), 1e-4)

    def test_hydrostatic_warming_raises_geopotential(self) -> None:
        """get_z_t: прогрев столба даёт z_t > 0 с ростом к верхним уровням."""
        kernel = make_pure_kernel()
        z_t = kernel.get_z_t(torch.full((1, P, H, W), 1e-4))
        self.assertGreater(float(z_t[0, 0, 0, 0]), 0.0)
        self.assertGreater(float(z_t[0, 0, 0, 0]), float(z_t[0, 7, 0, 0]))

    def test_qs_is_realistic_at_500hpa(self) -> None:
        """_get_qs(50000 Па, 253 K) в физичном диапазоне ~1.5e-3 кг/кг."""
        kernel = make_pure_kernel()
        q_s = kernel._get_qs(torch.full((1, 1, 1, 1), 5e4), torch.full((1, 1, 1, 1), 253.0))
        self.assertGreater(float(q_s), 1e-3)
        self.assertLess(float(q_s), 2.5e-3)

    def test_qs_is_batch_independent(self) -> None:
        """q_s сэмпла 0 не зависит от температуры сэмпла 1 (нет batch-remap)."""
        kernel = make_pure_kernel()
        p = torch.full((2, 1, 1, 1), 5e4)
        t_a = torch.tensor([280.0, 250.0]).reshape(2, 1, 1, 1)
        t_b = torch.tensor([280.0, 320.0]).reshape(2, 1, 1, 1)
        qs_a = kernel._get_qs(p, t_a)[0]
        qs_b = kernel._get_qs(p, t_b)[0]
        self.assertAlmostEqual(float(qs_a), float(qs_b), delta=1e-12)

    def test_saturated_ascent_condenses_via_rhs(self) -> None:
        """Конвергенция (подъём) + насыщение у поверхности => q_t < 0 внизу."""
        kernel = make_pure_kernel()
        x_idx = torch.arange(W, dtype=torch.float32).reshape(1, 1, 1, W)
        u = (-1e-5 * kernel.grid.pixel_x * x_idx).expand(1, P, H, W).clone()
        state = {
            "u": u,
            "v": torch.zeros_like(u),
            "t": torch.full_like(u, 280.0),
            "q": torch.full_like(u, 6.5e-3),  # насыщает только нижние уровни
            "z": PHI_STD.expand(1, P, H, W).clone(),
        }
        rhs = kernel.rhs(**state)
        bottom = float(interior(rhs["q_t"][:, 12]).mean())
        self.assertLess(bottom, 0.0)
        self.assertGreater(abs(bottom), 1e-9)  # F*|omega|/p ~ 1e-8 кг/кг/с
        self.assertLess(abs(bottom), 1e-6)
        self.assertLess(float(rhs["q_t"][:, 7].abs().max()), 1e-12)  # 500 гПа не насыщен


class Exp14MomentumVariants(unittest.TestCase):
    """Знаки, размерности и бит-стабильность членов эксперимента 14.

    Все новые члены уравнений движения — opt-in параметры
    :class:`PurePDEKernel`; дефолтный путь запинен голdenами
    (``test_default_rhs_is_byte_stable``).
    """

    @staticmethod
    def _kernel(**kwargs) -> PurePDEKernel:
        grid = Grid(GridConfig(H=H, W=W, lat_range_deg=(16.17, 59.77)))
        return PurePDEKernel(grid, boundary_x="replicate", coriolis="spherical", **kwargs)

    @staticmethod
    def _pinned_state() -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(14)
        phi = PHI_STD.expand(1, P, H, W).clone()
        return {
            "u": 10 + 5 * torch.randn(1, P, H, W, generator=gen),
            "v": 5 * torch.randn(1, P, H, W, generator=gen),
            "t": 250 + 10 * torch.randn(1, P, H, W, generator=gen),
            "q": (3e-3 * (1 + 0.3 * torch.randn(1, P, H, W, generator=gen))).clamp_min(1e-8),
            "z": phi + 300 * torch.randn(1, P, H, W, generator=gen),
        }

    def test_default_rhs_is_byte_stable(self) -> None:
        """Дефолтный rhs PurePDEKernel не меняется от opt-in параметров exp 14."""
        kernel = self._kernel()
        rhs = kernel.rhs(**self._pinned_state())
        golden = {
            "u_t": -0.009438502850116492,
            "v_t": -1.4592431275868876,
            "t_t": -0.014798275890939294,
            "q_t": -1.227921801313231e-05,
            "z_t": -5.975435962147742,
            "w": 0.06491834346371661,
        }
        for key, value in golden.items():
            self.assertAlmostEqual(float(rhs[key].double().sum()), value, delta=1e-9)
        self.assertAlmostEqual(float(rhs["u_t"][0, 6, 3, 7]), 0.0004705985775217414, delta=1e-12)
        self.assertAlmostEqual(float(rhs["v_t"][0, 6, 3, 7]), -0.0010004773503169417, delta=1e-12)

    def test_d_y_orientation_on_ascending_rows(self) -> None:
        """rows_south_to_north=True: d_y(f=j) = +1/Δy; легаси даёт −1/Δy."""
        legacy = self._kernel()
        s2n = self._kernel(rows_south_to_north=True)
        field = torch.arange(H, dtype=torch.float32).reshape(1, 1, H, 1).expand(1, 1, H, W).clone()
        dy = float(legacy.grid.pixel_y)
        self.assertAlmostEqual(float(legacy.diff.d_y(field)[0, 0, 4, 8]) * dy, -1.0, delta=1e-4)
        self.assertAlmostEqual(float(s2n.diff.d_y(field)[0, 0, 4, 8]) * dy, 1.0, delta=1e-4)

    def test_metric_terms_sign_and_magnitude(self) -> None:
        """Кривизна: u_t += u·v·tanφ/a, v_t −= u²·tanφ/a (Holton 2.19–2.20)."""
        base = self._kernel()
        metric = self._kernel(metric_terms=True)
        u = torch.full((1, P, H, W), 20.0)
        v = torch.full((1, P, H, W), 5.0)
        zeros = torch.zeros_like(u)
        u_t0, v_t0 = base.get_uv_dt(u, v, zeros, zeros, zeros)
        u_t1, v_t1 = metric.get_uv_dt(u, v, zeros, zeros, zeros)
        tan_over_a = float(torch.tan(base.grid.latitudes[4]) / base.grid.config.radius)
        self.assertAlmostEqual(float((u_t1 - u_t0)[0, 6, 4, 8]), 100.0 * tan_over_a, delta=1e-9)
        self.assertAlmostEqual(float((v_t1 - v_t0)[0, 6, 4, 8]), -400.0 * tan_over_a, delta=1e-9)

    def test_rayleigh_friction_only_in_boundary_layer(self) -> None:
        """Трение Held–Suarez: нуль выше σ_b=0.7, −k_f·u на 1000 гПа."""
        base = self._kernel()
        frict = self._kernel(rayleigh_friction=True)
        u = torch.full((1, P, H, W), 20.0)
        zeros = torch.zeros_like(u)
        u_t0, _ = base.get_uv_dt(u, zeros, zeros, zeros, zeros)
        u_t1, _ = frict.get_uv_dt(u, zeros, zeros, zeros, zeros)
        diff = (u_t1 - u_t0)[0, :, 4, 8]
        self.assertEqual(float(diff[:10].abs().max()), 0.0)  # до 700 гПа включительно
        self.assertAlmostEqual(float(diff[12]), -20.0 / 86400.0, delta=1e-10)  # 1000 гПа

    def test_lagrange3_vertical_derivative_is_exact_on_linear_profile(self) -> None:
        """lagrange3: d_z(2p) = −2 на всех уровнях нерегулярной сетки."""
        kernel = self._kernel(vertical_scheme="lagrange3")
        p_hpa = kernel.grid.pressure.reshape(-1) / 100.0
        profile = (2.0 * p_hpa).reshape(1, P, 1, 1).expand(1, P, H, W).clone()
        d_z = kernel._d_z(profile)
        self.assertTrue(torch.allclose(d_z, torch.full_like(d_z, -2.0), atol=1e-4))

    def test_spherical_divergence_reduces_divergence_for_northward_wind(self) -> None:
        """Континуити на сфере: однородный v>0 в СП даёт div = −v·tanφ/a < 0."""
        base = self._kernel()
        sph = self._kernel(spherical_divergence=True)
        v = torch.full((1, P, H, W), 10.0)
        zeros = torch.zeros_like(v)
        w_diff = sph.get_w(zeros, v) - base.get_w(zeros, v)
        # div уменьшился → w = ∫(−div) больше у поверхности (СП: tanφ > 0).
        self.assertGreater(float(w_diff[0, 12, 4, 8]), 0.0)

    def test_grid_explicit_latitudes(self) -> None:
        """GridConfig.latitudes_deg: точные широты строк и pixel_y из шага."""
        lats = tuple(16.171875 + 1.40625 * i for i in range(H))
        grid = Grid(GridConfig(H=H, W=W, latitudes_deg=lats))
        got = grid.latitudes * 180.0 / torch.pi
        self.assertAlmostEqual(float(got[0]), 16.171875, delta=1e-4)
        self.assertAlmostEqual(float(got[-1]), lats[-1], delta=1e-4)
        expected_dy = 1.40625 / 180.0 * torch.pi * grid.config.radius
        self.assertAlmostEqual(float(grid.pixel_y), expected_dy, delta=0.5)


class Exp15EquationVariants(unittest.TestCase):
    """Знаки и размерности членов эксперимента 15 (все — opt-in параметры).

    Дефолтный путь по-прежнему запинен голденами
    ``Exp14MomentumVariants.test_default_rhs_is_byte_stable``.
    """

    @staticmethod
    def _kernel(**kwargs) -> PurePDEKernel:
        grid = Grid(GridConfig(H=H, W=W, lat_range_deg=(16.17, 59.77)))
        return PurePDEKernel(grid, boundary_x="replicate", coriolis="spherical", **kwargs)

    @staticmethod
    def _state(**overrides) -> dict[str, torch.Tensor]:
        base = {
            "u": torch.zeros(1, P, H, W),
            "v": torch.zeros(1, P, H, W),
            "t": T_STD.expand(1, P, H, W).clone(),
            "q": torch.full((1, P, H, W), 1e-4),
            "z": PHI_STD.expand(1, P, H, W).clone(),
        }
        base.update(overrides)
        return base

    def test_newtonian_relaxation_cools_warm_anomaly(self) -> None:
        """T > T_eq у поверхности → релаксация даёт отрицательную добавку t_t."""
        base = self._kernel()
        relax = self._kernel(newtonian_relaxation=True)
        hot = self._state(t=torch.full((1, P, H, W), 330.0))
        diff = relax.rhs(**hot)["t_t"] - base.rhs(**hot)["t_t"]
        self.assertLess(float(diff[0, 12, 4, 8]), 0.0)  # 1000 гПа: k_s-зона
        # Величина ~k_s·ΔT: ΔT ≈ 330−T_eq(≈300) ≈ 30 К, k_T ≤ 1/4 сут.
        self.assertLess(abs(float(diff[0, 12, 4, 8])), 40.0 / (4 * 86400.0) * 2)

    def test_newtonian_relaxation_teq_range_is_physical(self) -> None:
        """T_eq ограничена снизу 200 К и не превышает 315 К."""
        relax = self._kernel(newtonian_relaxation=True)
        self.assertGreaterEqual(float(relax.hs_t_eq.min()), 200.0 - 1e-4)
        self.assertLessEqual(float(relax.hs_t_eq.max()), 315.0 + 1e-4)

    def test_latent_heating_warms_where_condensation_dries(self) -> None:
        """Конденсация (q_t|cond < 0) обязана греть: Δt_t = −(L/c_p)·cond > 0."""
        base = self._kernel()
        coupled = self._kernel(latent_heating_coupling=True)
        x_idx = torch.arange(W, dtype=torch.float32).reshape(1, 1, 1, W)
        u_conv = (-1e-5 * base.grid.pixel_x * x_idx).expand(1, P, H, W).clone()
        state = self._state(
            u=u_conv, t=torch.full((1, P, H, W), 280.0), q=torch.full((1, P, H, W), 6.5e-3)
        )
        rhs_b = base.rhs(**state)
        rhs_c = coupled.rhs(**state)
        cond_bottom = interior((rhs_b["q_t"])[:, 12])  # адвекция q тут нулевая
        dt_bottom = interior((rhs_c["t_t"] - rhs_b["t_t"])[:, 12])
        self.assertLess(float(cond_bottom.mean()), 0.0)
        self.assertGreater(float(dt_bottom.mean()), 0.0)
        ratio = float(dt_bottom.mean()) / float(cond_bottom.mean())
        self.assertAlmostEqual(ratio, -2.5e6 / 1005.0, delta=1.0)

    def test_obrien_zeroes_top_w_and_keeps_surface(self) -> None:
        """О'Брайен: w(50 гПа) = 0 точно, w(1000 гПа) не изменяется."""
        plain = self._kernel()
        obrien = self._kernel(w_diagnostic="obrien")
        gen = torch.Generator().manual_seed(15)
        u = 10 + 5 * torch.randn(1, P, H, W, generator=gen)
        v = 5 * torch.randn(1, P, H, W, generator=gen)
        w_plain = plain.get_w(u, v)
        w_ob = obrien.get_w(u, v)
        self.assertLess(float(w_ob[:, 0].abs().max()), 1e-12)
        self.assertGreater(float(w_plain[:, 0].abs().max()), 0.0)
        self.assertTrue(torch.allclose(w_ob[:, 12], w_plain[:, 12], atol=1e-12))

    def test_kinematic_ps_anchor_raises_z_on_convergence(self) -> None:
        """Конвергенция в столбе → ∂p_s/∂t > 0 → z_t выше fixed-якоря на всех уровнях."""
        fixed = self._kernel()
        anchored = self._kernel(z_anchor="kinematic_ps")
        x_idx = torch.arange(W, dtype=torch.float32).reshape(1, 1, 1, W)
        u_conv = (-1e-5 * fixed.grid.pixel_x * x_idx).expand(1, P, H, W).clone()
        state = self._state(u=u_conv)
        diff = anchored.rhs(**state)["z_t"] - fixed.rhs(**state)["z_t"]
        self.assertGreater(float(interior(diff[:, 0]).min()), 0.0)
        # Баротропный член одинаков на всех уровнях.
        self.assertTrue(torch.allclose(diff[:, 0], diff[:, 12], atol=1e-10))

    def test_omega_free_t_strips_omega_terms_but_keeps_z(self) -> None:
        """omega_free=('t',): t_t — чистая гор. адвекция; z_t не меняется."""
        base = self._kernel()
        nf = self._kernel(omega_free=("t",))
        gen = torch.Generator().manual_seed(151)
        state = self._state(
            u=10 + 5 * torch.randn(1, P, H, W, generator=gen),
            v=5 * torch.randn(1, P, H, W, generator=gen),
            t=T_STD.expand(1, P, H, W) + 2 * torch.randn(1, P, H, W, generator=gen),
        )
        rhs_b = base.rhs(**state)
        rhs_n = nf.rhs(**state)
        adv_h = base._horiz_adv(state["t"], state["u"], state["v"])
        self.assertTrue(torch.allclose(rhs_n["t_t"], adv_h, atol=1e-5))
        self.assertTrue(torch.allclose(rhs_n["z_t"], rhs_b["z_t"], atol=0.0))
        self.assertFalse(torch.allclose(rhs_n["t_t"], rhs_b["t_t"], atol=1e-8))

    def test_omega_free_q_uv_strip_vertical_advection(self) -> None:
        """omega_free=('q','u','v'): верт. адвекция исключена, остальное на месте."""
        base = self._kernel()
        nf = self._kernel(omega_free=("q", "u", "v"))
        gen = torch.Generator().manual_seed(152)
        state = self._state(
            u=10 + 5 * torch.randn(1, P, H, W, generator=gen),
            v=5 * torch.randn(1, P, H, W, generator=gen),
            q=(3e-3 * (1 + 0.3 * torch.randn(1, P, H, W, generator=gen))).clamp_min(1e-8),
        )
        rhs_b = base.rhs(**state)
        rhs_n = nf.rhs(**state)
        w = base.get_w(state["u"], state["v"])
        for var, field in (("u", state["u"]), ("v", state["v"]), ("q", state["q"])):
            expected = rhs_b[f"{var}_t"] + w * base._d_z(field)
            self.assertTrue(torch.allclose(rhs_n[f"{var}_t"], expected, atol=1e-6))
        self.assertTrue(torch.equal(rhs_n["t_t"], rhs_b["t_t"]))

    def test_omega_free_rejects_unknown_keys(self) -> None:
        """omega_free с 'z' или опечаткой — явная ошибка конфигурации."""
        with self.assertRaises(ValueError):
            self._kernel(omega_free=("z",))

    def test_rhs_external_sources_are_added_verbatim(self) -> None:
        """sources={'t': S} даёт t_t + S точно; остальные тенденции не тронуты."""
        kernel = self._kernel()
        state = self._state()
        src = torch.full((1, P, H, W), 3.14e-5)
        rhs0 = kernel.rhs(**state)
        rhs1 = kernel.rhs(**state, sources={"t": src})
        self.assertTrue(torch.allclose(rhs1["t_t"], rhs0["t_t"] + src, atol=0.0))
        self.assertTrue(torch.equal(rhs1["q_t"], rhs0["q_t"]))
        self.assertTrue(torch.equal(rhs1["z_t"], rhs0["z_t"]))


class Exp23EquationVariants(unittest.TestCase):
    """Знаки, размерности, бит-стабильность членов эксперимента 23.

    Новые уравнения — наблюдаемый форсинг: виртуальная температура в
    гидростатике (``virtual_temperature``), квадратичное bulk-трение
    пограничного слоя (``bulk_drag``) и латентное тепло осадков
    (:func:`latent_heating_profile`, вход через ``sources``). Дефолтный путь
    по-прежнему запинен ``Exp14MomentumVariants.test_default_rhs_is_byte_stable``.
    """

    @staticmethod
    def _kernel(**kwargs) -> PurePDEKernel:
        grid = Grid(GridConfig(H=H, W=W, lat_range_deg=(16.17, 59.77)))
        return PurePDEKernel(grid, boundary_x="replicate", coriolis="spherical", **kwargs)

    @staticmethod
    def _pinned_state() -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(23)
        phi = PHI_STD.expand(1, P, H, W).clone()
        return {
            "u": 10 + 5 * torch.randn(1, P, H, W, generator=gen),
            "v": 5 * torch.randn(1, P, H, W, generator=gen),
            "t": 250 + 10 * torch.randn(1, P, H, W, generator=gen),
            "q": (3e-3 * (1 + 0.3 * torch.randn(1, P, H, W, generator=gen))).clamp_min(1e-8),
            "z": phi + 300 * torch.randn(1, P, H, W, generator=gen),
        }

    def test_exp23_flags_off_are_byte_stable(self) -> None:
        """Явные дефолты новых флагов exp23 не меняют rhs бит-в-бит."""
        base = self._kernel()
        off = self._kernel(virtual_temperature=False, bulk_drag=False)
        state = self._pinned_state()
        rhs_b = base.rhs(**state)
        rhs_o = off.rhs(**state)
        for key in ("u_t", "v_t", "t_t", "q_t", "z_t", "w"):
            self.assertTrue(torch.equal(rhs_b[key], rhs_o[key]), key)

    # ----- E23-B: виртуальная температура в гидростатике -----

    def test_virtual_temperature_is_dry_zt_when_q_zero(self) -> None:
        """Golden: q≡0 ⇒ z_t виртуально-температурного ядра = сухому бит-в-бит."""
        base = self._kernel()
        vt = self._kernel(virtual_temperature=True)
        state = self._pinned_state()
        state["q"] = torch.zeros(1, P, H, W)
        self.assertTrue(torch.equal(vt.rhs(**state)["z_t"], base.rhs(**state)["z_t"]))

    def test_virtual_temperature_raises_z_tendency_in_moist_warming(self) -> None:
        """T_v = T(1+0.608q) > T при q>0 ⇒ |интегранд| и z_t растут при прогреве.

        Чистый прогрев (t_t>0) без адвекции: столб с влажностью даёт
        z_t выше сухого на положительную величину 0.608·(вклад q·t_t + T·q_t).
        """
        base = self._kernel()
        vt = self._kernel(virtual_temperature=True)
        # источник тепла у поверхности через sources ⇒ t_t>0, q фикс. большой.
        state = self._pinned_state()
        state["q"] = torch.full((1, P, H, W), 8e-3)
        src = {"t": torch.full((1, P, H, W), 1e-3)}  # равномерный прогрев К/с
        z_dry = base.rhs(**state, sources=src)["z_t"]
        z_vt = vt.rhs(**state, sources=src)["z_t"]
        # z_t собирается от уровня к поверхности; прогрев столба поднимает
        # геопотенциал сильнее во влажном случае на верхних уровнях.
        self.assertGreater(float(interior((z_vt - z_dry)[:, 0]).mean()), 0.0)

    # ----- E23-D: bulk-трение пограничного слоя -----

    def test_bulk_drag_is_antiparallel_to_wind(self) -> None:
        """Трение тормозит ветер: (u_t_drag)·u ≤ 0 и (v_t)·v ≤ 0 поточечно."""
        base = self._kernel()
        drag = self._kernel(bulk_drag=True)
        state = self._pinned_state()
        du = drag.rhs(**state)["u_t"] - base.rhs(**state)["u_t"]
        dv = drag.rhs(**state)["v_t"] - base.rhs(**state)["v_t"]
        mask = (drag.bulk_drag_mask > 0).expand(1, P, H, W)
        self.assertLessEqual(float((du * state["u"])[mask].max()), 1e-12)
        self.assertLessEqual(float((dv * state["v"])[mask].max()), 1e-12)

    def test_bulk_drag_confined_to_boundary_layer(self) -> None:
        """Выше σ_b=0.7 (уровни ≤700 гПа) добавка трения строго нулевая."""
        base = self._kernel()
        drag = self._kernel(bulk_drag=True)
        state = self._pinned_state()
        du = drag.rhs(**state)["u_t"] - base.rhs(**state)["u_t"]
        # уровни 0..9 = 50..700 гПа: σ ≤ 0.7 ⇒ маска 0.
        self.assertEqual(float(du[:, :10].abs().max()), 0.0)
        self.assertGreater(float(du[:, 12].abs().max()), 0.0)  # 1000 гПа

    def test_bulk_drag_zero_for_zero_wind(self) -> None:
        """Нулевой ветер ⇒ нулевое трение (квадратично по скорости)."""
        drag = self._kernel(bulk_drag=True)
        state = self._pinned_state()
        state["u"] = torch.zeros(1, P, H, W)
        state["v"] = torch.zeros(1, P, H, W)
        du, dv = drag._bulk_drag(state["u"], state["v"], state["t"])
        self.assertEqual(float(du.abs().max()), 0.0)
        self.assertEqual(float(dv.abs().max()), 0.0)

    def test_bulk_drag_mutually_exclusive_with_rayleigh(self) -> None:
        """bulk_drag и rayleigh_friction вместе — явная ошибка конфигурации."""
        with self.assertRaises(ValueError):
            self._kernel(bulk_drag=True, rayleigh_friction=True)

    # ----- E23-C: латентное тепло осадков -----

    def test_latent_heating_conserves_column_energy(self) -> None:
        """∫(Q/c_p)·(dp/g) по столбу = L·ρ_w·P/c_p (сохранение энергии)."""
        from utils.physics import PhysicsConstants, latent_heating_profile

        kernel = self._kernel()
        consts = PhysicsConstants()
        precip = torch.zeros(1, 1, H, W)
        precip[0, 0, 3, 7] = 1e-6  # ~3.6 мм/ч
        pz = kernel.grid.pixel_z
        p_hpa = kernel.grid.pressure.reshape(-1) / 100.0
        q_t = latent_heating_profile(precip, pz, consts, pressure_hpa=p_hpa)
        dp_pa = pz.reshape(-1) * 100.0
        col = float((q_t[0, :, 3, 7] * consts.c_p * dp_pa / consts.g).sum())
        expect = consts.L * consts.rho_w * 1e-6
        self.assertAlmostEqual(col / expect, 1.0, delta=1e-4)

    def test_latent_heating_zero_without_precip_and_positive_with(self) -> None:
        """P=0 ⇒ нулевой источник; P>0 ⇒ строго положительный прогрев столба."""
        from utils.physics import PhysicsConstants, latent_heating_profile

        kernel = self._kernel()
        consts = PhysicsConstants()
        pz = kernel.grid.pixel_z
        p_hpa = kernel.grid.pressure.reshape(-1) / 100.0
        precip = torch.zeros(1, 1, H, W)
        precip[0, 0, 3, 7] = 2e-6
        q_t = latent_heating_profile(precip, pz, consts, pressure_hpa=p_hpa)
        self.assertEqual(float(q_t[0, :, 0, 0].abs().max()), 0.0)  # нет осадков
        self.assertGreater(float(q_t[0, :, 3, 7].sum()), 0.0)  # есть осадки

    def test_latent_heating_default_profile_peaks_mid_troposphere(self) -> None:
        """Дефолтный η(σ)=sin(πσ): максимум в средней тропосфере, нули по краям."""
        from utils.physics import PhysicsConstants, latent_heating_profile

        kernel = self._kernel()
        consts = PhysicsConstants()
        pz = kernel.grid.pixel_z
        p_hpa = kernel.grid.pressure.reshape(-1) / 100.0
        precip = torch.zeros(1, 1, H, W)
        precip[0, 0, 3, 7] = 1e-6
        q_t = latent_heating_profile(precip, pz, consts, pressure_hpa=p_hpa)
        col = q_t[0, :, 3, 7]
        # максимум не на верхнем (50 гПа) и не на приземном (1000 гПа) уровне.
        self.assertNotIn(int(col.argmax()), (0, P - 1))
        self.assertGreater(float(col[6]), float(col[0]))  # 400 гПа > 50 гПа
        self.assertGreater(float(col[6]), float(col[12]))  # 400 гПа > 1000 гПа


if __name__ == "__main__":
    unittest.main()
