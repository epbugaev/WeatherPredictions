"""Пины порта exp14/15-термов в PDE_kernel (эксперимент 16).

Главный инвариант: все новые флаги ВЫКЛЮЧЕНЫ по умолчанию, и дефолтный
``physics_only_forward`` остаётся бит-в-бит с до-портовым голденом
(``tests/goldens/exp16_kernel_default_out.pt``). Знаковые/структурные пины
новых термов зеркалят эталонные реализации ``utils.physics.PurePDEKernel``.
"""

import torch

from Models.IAM4VP import IAM4VP
from utils.physics_hybrid import HybridBlock, PDE_kernel


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


def test_omega_free_t_keeps_full_tt_for_z() -> None:
    """omega_free('t'): тенденция t усечена, но z интегрирует ПОЛНЫЙ t_t."""
    base, ofree = _kernel(), _kernel(omega_free=("t",))
    state = _state()
    with torch.no_grad():
        out_base = base.physics_only_forward(state)
        out_free = ofree.physics_only_forward(state)
    z_base, t_base = out_base.chunk(5, dim=1)[:2]
    z_free, t_free = out_free.chunk(5, dim=1)[:2]
    assert torch.equal(z_base, z_free)  # z не изменился (полный t_t в интеграле)
    assert not torch.equal(t_base, t_free)  # выходная t-тенденция усечена


def test_omega_free_q_keeps_condensation() -> None:
    """omega_free('q') убирает w·q_z, но конденсация остаётся (зеркало physics.py)."""
    base, ofree = _kernel(), _kernel(omega_free=("q",))
    state = _state()
    z, t, q, u, v = state.chunk(5, dim=1)
    w = base.get_w(u, v)
    q_t_base = base.get_q_dt(u, v, t, w, q)
    q_t_free = ofree.get_q_dt(u, v, t, w, q)
    assert torch.isfinite(q_t_free).all()
    assert not torch.equal(q_t_base, q_t_free)


def test_latent_heating_warms_saturated_column() -> None:
    """Скрытое тепло: t_t(latent) − t_t(base) = −(L/c_p)·cond ≥ 0 (cond ≤ 0)."""
    base, lat = _kernel(), _kernel(latent_heating_coupling=True)
    state = _state()
    z, t, q, u, v = state.chunk(5, dim=1)
    q = torch.full_like(q, 0.02)  # форсируем насыщение
    w = base.get_w(u, v)
    base.share_z_dxyz(z)
    lat.share_z_dxyz(z)
    base._q_for_latent = q
    lat._q_for_latent = q
    tt_base = base.get_t_t(u, v, w, t)
    tt_lat = lat.get_t_t(u, v, w, t)
    assert (tt_lat - tt_base).min() >= 0
    assert (tt_lat - tt_base).max() > 0


def test_kwargs_reach_kernel_through_hybrid_block() -> None:
    """Сквозной проброс exp14/15-kwargs: HybridBlock → PDE_block → PDE_kernel.

    Регрессия на потерю kwargs в промежуточной обёртке (юнит-тесты ядра
    такой разрыв не ловят).
    """
    torch.manual_seed(0)
    block = HybridBlock(
        dim=65,
        zquvtw_channel=13,
        depth=2,
        block_dt=300.0,
        inverse_time=False,
        physics_part_coef=0.5,
        w_diagnostic="mass_consistent",
        lat_start_deg=18.28125,
        dlat_deg=5.625,
        dlon_deg=5.625,
        grid_h=8,
        physical_passthrough=True,
        advection_form="advective",
        metric_terms=True,
        rayleigh_friction=True,
        vertical_scheme="lagrange3",
        omega_free=("t", "q"),
        latent_heating_coupling=True,
    )
    for kernel in block.pde_block.PDE_kernels:
        assert kernel.advection_form == "advective"
        assert kernel.metric_terms and kernel.rayleigh_friction
        assert kernel.vertical_scheme == "lagrange3"
        assert kernel.omega_free == ("t", "q")
        assert kernel.latent_heating_coupling


def test_clim_sources_pooled_buffer() -> None:
    """Клим-источники: annual (13, 32) → усреднение широты до (1, 13, 8, 1)."""
    import numpy as np

    rng = np.random.default_rng(0)
    arrays = {
        f"C15_now__annual_{k}": rng.normal(0, 1e-5, size=(13, 32)).astype("float32")
        for k in ("t", "q", "z")
    }
    np.savez("tests/goldens/exp16_clim_stub.npz", **arrays)
    kernel = _kernel(clim_sources_path="tests/goldens/exp16_clim_stub.npz")
    assert kernel.clim_src_t.shape == (1, 13, 8, 1)
    expected_row0 = arrays["C15_now__annual_t"][:, 0:4].mean(axis=1)
    assert torch.allclose(kernel.clim_src_t[0, :, 0, 0], torch.from_numpy(expected_row0))
    with torch.no_grad():
        out = kernel.physics_only_forward(_state())
    assert torch.isfinite(out).all()


def test_diabatic_mask_t_only_covers_only_t_block() -> None:
    """exp16-long R3q: маска ``t_only`` — единицы ровно на T-блоке (13 каналов)."""
    mask = IAM4VP._build_diabatic_mask("t_only", 65, 0)
    assert mask.shape == (1, 65, 1, 1)
    assert mask[0, 13:26].sum().item() == 13.0
    assert mask.sum().item() == 13.0
    # регрессия существующих режимов
    assert IAM4VP._build_diabatic_mask("t_and_q", 65, 0).sum().item() == 26.0
    assert IAM4VP._build_diabatic_mask("all_upper_air", 65, 0).sum().item() == 65.0
    # смещение surface-каналов сдвигает блок
    shifted = IAM4VP._build_diabatic_mask("t_only", 69, 4)
    assert shifted[0, 17:30].sum().item() == 13.0
    assert shifted.sum().item() == 13.0
