"""Пины порта exp14/15-термов в PDE_kernel (эксперимент 16).

Главный инвариант: все новые флаги ВЫКЛЮЧЕНЫ по умолчанию, и дефолтный
``physics_only_forward`` остаётся бит-в-бит с до-портовым голденом
(``tests/goldens/exp16_kernel_default_out.pt``). Знаковые/структурные пины
новых термов зеркалят эталонные реализации ``utils.physics.PurePDEKernel``.
"""

import torch

from utils.physics_hybrid import PDE_kernel


def _state(seed: int = 0) -> torch.Tensor:
    """Физически правдоподобное состояние (B, 5·P, H, W) = [z, t, q, u, v]."""
    g = torch.Generator().manual_seed(seed)
    B, P, H, W = 2, 13, 8, 16
    z = 5.0e4 + 1.0e3 * torch.randn(B, P, H, W, generator=g)
    t = 260.0 + 15.0 * torch.randn(B, P, H, W, generator=g)
    q = (5.0e-3 + 2.0e-3 * torch.randn(B, P, H, W, generator=g)).clamp_min(1e-6)
    u = 10.0 * torch.randn(B, P, H, W, generator=g)
    v = 10.0 * torch.randn(B, P, H, W, generator=g)
    return torch.cat([z, t, q, u, v], dim=1)


def _kernel(**kw) -> PDE_kernel:
    """USA-v4 латент-геометрия (8×16), mass-consistent ω, passthrough-режим."""
    torch.manual_seed(0)
    kernel = PDE_kernel(
        in_dim=65,
        variable_dim=13,
        physics_part_coef=0.5,
        w_diagnostic="mass_consistent",
        lat_start_deg=18.28125,
        dlat_deg=5.625,
        dlon_deg=5.625,
        grid_h=8,
        physical_passthrough=True,
        **kw,
    )
    return kernel.eval()


def test_defaults_bitexact_golden() -> None:
    """Все новые флаги выключены → physics_only_forward бит-в-бит с до-портовым."""
    kernel = _kernel()
    with torch.no_grad():
        out = kernel.physics_only_forward(_state())
    golden = torch.load("tests/goldens/exp16_kernel_default_out.pt", weights_only=True)
    assert torch.equal(out, golden)


def test_new_kwargs_accepted() -> None:
    """Все exp14-флаги включаются одновременно и дают конечный выход."""
    kernel = _kernel(
        advection_form="advective",
        metric_terms=True,
        spherical_divergence=True,
        rayleigh_friction=True,
        vertical_scheme="lagrange3",
    )
    with torch.no_grad():
        out = kernel.physics_only_forward(_state())
    assert torch.isfinite(out).all()


def test_metric_terms_sign() -> None:
    """+u·v·tanφ/a в u_t: при u,v>0 в северном полушарии u_t растёт."""
    base, metric = _kernel(), _kernel(metric_terms=True)
    state = _state()
    z, t, q, u, v = state.chunk(5, dim=1)
    u, v = u.abs() + 1.0, v.abs() + 1.0
    w = base.get_w(u, v)
    base.share_z_dxyz(z)
    metric.share_z_dxyz(z)
    ut_base, _ = base.get_uv_dt(u, v, w)
    ut_metric, _ = metric.get_uv_dt(u, v, w)
    assert (ut_metric - ut_base).min() > 0


def test_rayleigh_only_boundary_layer() -> None:
    """k_v = 0 при σ<0.7: верхние уровни не трогаются трением."""
    fric = _kernel(rayleigh_friction=True)
    assert fric.rayleigh_k[0, :5].abs().max() == 0
    assert fric.rayleigh_k[0, -1].item() > 0


def test_advective_form_changes_output() -> None:
    """Адвективная форма — другой оператор, выход отличается от flux-дефолта."""
    base, adv = _kernel(), _kernel(advection_form="advective")
    state = _state()
    with torch.no_grad():
        out_base = base.physics_only_forward(state)
        out_adv = adv.physics_only_forward(state)
    assert not torch.equal(out_base, out_adv)
