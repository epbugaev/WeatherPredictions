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


if __name__ == "__main__":
    unittest.main()
