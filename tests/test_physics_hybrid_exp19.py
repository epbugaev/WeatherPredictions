import torch

from utils.physics_hybrid import PDE_kernel


def _state(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    B, P, H, W = 2, 13, 8, 16
    z = 5.0e4 + 1.0e3 * torch.randn(B, P, H, W, generator=g)
    t = 260.0 + 15.0 * torch.randn(B, P, H, W, generator=g)
    q = (5.0e-3 + 2.0e-3 * torch.randn(B, P, H, W, generator=g)).clamp_min(1e-6)
    u = 10.0 * torch.randn(B, P, H, W, generator=g)
    v = 10.0 * torch.randn(B, P, H, W, generator=g)
    return torch.cat([z, t, q, u, v], dim=1)


def _kernel(**kw) -> PDE_kernel:
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


def test_mask_hard_zeros_boundary_layer_u_only() -> None:
    """physics_level_mask {'u':700}: u не меняется на 850/925/1000; v,t,q — как без маски."""
    base = _kernel()
    masked = _kernel(physics_level_mask={"u": 700})
    state = _state()
    with torch.no_grad():
        out_base = base.physics_only_forward(state)
        out_mask = masked.physics_only_forward(state)
    z_b, t_b, q_b, u_b, v_b = out_base.chunk(5, dim=1)
    z_m, t_m, q_m, u_m, v_m = out_mask.chunk(5, dim=1)
    z_in, t_in, q_in, u_in, v_in = state.chunk(5, dim=1)
    # u заморожен на индексах 10,11,12 (850/925/1000):
    assert torch.equal(u_m[:, 10:], u_in[:, 10:])
    # u выше отсечки эволюционирует (как в base):
    assert torch.equal(u_m[:, :10], u_b[:, :10])
    # v,t,q не затронуты маской:
    assert torch.equal(v_m, v_b) and torch.equal(t_m, t_b) and torch.equal(q_m, q_b)


def test_mask_learnable_gate_grad_and_init() -> None:
    """Обучаемый гейт: параметр (4,13) с градиентом; α≈жёсткой маске при init ±4."""
    masked = _kernel(
        physics_level_mask={"u": 700, "v": 700, "t": 850, "q": 850},
        physics_level_mask_learnable=True,
    )
    assert masked.level_gate_logit.shape == (4, 13)
    assert masked.level_gate_logit.requires_grad
    alpha = torch.sigmoid(masked.level_gate_logit)
    # u (строка 0): выше 700 (индексы 0..9) ≈1, ниже ≈0
    assert alpha[0, :10].min() > 0.95 and alpha[0, 10:].max() < 0.05
    # t (строка 2): отсечка 850 → индексы 0..10 ≈1, 11,12 ≈0
    assert alpha[2, :11].min() > 0.95 and alpha[2, 11:].max() < 0.05


def test_mask_learnable_default_off_bitexact() -> None:
    """Без learnable-флага гейт не создаётся; выход бит-в-бит с немаскированным."""
    base = _kernel()
    assert not hasattr(base, "level_gate_logit")
    with torch.no_grad():
        out = base.physics_only_forward(_state())
    golden = torch.load("tests/goldens/exp16_kernel_default_out.pt", weights_only=True)
    assert torch.equal(out, golden)
