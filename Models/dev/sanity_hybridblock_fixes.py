"""Single-card sanity checks for the PI-IAM4VP integration fixes (F1-F5, F8, F3b).

Verifies, on CPU in well under two minutes, the working-tree fixes to
``Models/PredFormerGFT_HybridBlock.py`` and ``Models/PI_IAM4VP.py`` audited in
``docs/PI_IAM4VP_integration_audit_ru.md``:

    1. check_equations_fixed  (F1/F8/T-1) — adiabatic_omega thermodynamics is
       bounded while legacy_paper blows up by orders of magnitude; universal-R
       vs dry-R hydrostatic scaling.
    2. check_geometry         (F2/P-1)   — crop-aware Coriolis/pixel metric on
       the USA window vs the preserved global default geometry.
    3. check_dt_semantics     (F3/B-1)   — physical_clip makes the increment
       scale with block_dt; scale_diff is dt-invariant (legacy artefact).
    4. check_batch_independence (F3)     — physical_clip does not couple samples
       across a batch (tendency-limiter fix); scale_diff leaks across the batch.
    5. check_latent_tendency  (F4)       — tendency-on-latent carries physics
       only, not the coarse<->fine resampling residual.
    6. check_prior_detach     (F5)       — physics_prior_detach cuts task-loss
       gradient into the HybridBlock feature generator.
    7. check_zero_init_identity          — zero-init corrector emits an exact
       identity prediction at step 0 with zeroed physics diagnostics.
    8. check_block_dt_accounting (F3b)   — block_dt splits the physics horizon
       across depth*hybrid_steps kernel calls.
    9. check_mass_consistent_invariant   — column-integrated corrected
       divergence vanishes and get_w matches integral_z(-div_corr) (E-omega).
    10. check_diabatic_mask              — diabatic_apply_to='t_and_q' masks
        exactly the T and humidity blocks of the Q_theta head output.

The script prints ``[PASS]``/``[FAIL] <name>: <detail>`` per assertion, a
``N passed, M failed`` summary, and exits with code 1 if any assertion failed.
It generates all data with ``torch.randn`` (no ERA5) and follows LBYL control
flow (no ``try``/``except``).

Run:
    python Models/dev/sanity_hybridblock_fixes.py --device cpu
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.PI_IAM4VP import IAM4VP
from Models.PredFormerGFT_HybridBlock import HybridBlock, PDE_kernel, integral_z, pixel_z

EARTH_RADIUS_M = 6371.0 * 1000.0
DEG2RAD = math.pi / 180.0
R_DRY = 287.0
R_UNIVERSAL = 8.314
ATOL = 1e-5

# USA-crop latent geometry (32x64 crop -> 8x16 latent, ~18.3N..57.7N).
USA_LAT_START_DEG = 18.28125
USA_DLAT_DEG = 5.625
USA_DLON_DEG = 5.625
USA_GRID_H = 8


def report(results: list[tuple[str, bool]], name: str, passed: bool, detail: str) -> None:
    """Print one ``[PASS]``/``[FAIL]`` line and record the outcome.

    Args:
        results: accumulator of ``(name, passed)`` mutated in place.
        name: dotted check name shown in the line.
        passed: whether the assertion held.
        detail: short human-readable evidence string.

    Side effects:
        Appends ``(name, passed)`` to ``results`` and writes one line to stdout.
    """
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")
    results.append((name, passed))


def build_physical_fields(
    batch_size: int,
    grid_h: int,
    grid_w: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build plausible physical (z, t, q, u, v) fields for kernel-level checks.

    Args:
        batch_size: number of samples B.
        grid_h: latent height H.
        grid_w: latent width W.
        device: device to place the tensors on.

    Returns:
        Tuple ``(z, t, q, u, v)``, each ``torch.Tensor`` of shape
        ``(B, 13, H, W)``. ``t`` is a temperature (~250 K), ``u``/``v`` are winds
        (~10 m/s), ``q`` a positive humidity (~1e-3), and ``z`` a monotone
        hydrostatic-like geopotential profile from 2e4 down to 5e2.
    """
    levels = 13
    shape = (batch_size, levels, grid_h, grid_w)
    t = 250.0 + 10.0 * torch.randn(shape, device=device)
    u = 10.0 * torch.randn(shape, device=device)
    v = 10.0 * torch.randn(shape, device=device)
    q = (1e-3 * torch.rand(shape, device=device)).clamp(min=1e-6)
    profile = torch.linspace(2e4, 5e2, levels, device=device).view(1, levels, 1, 1)
    z = profile * (1.0 + 0.01 * torch.randn(shape, device=device))
    return z, t, q, u, v


def pack_channels_last(
    z: torch.Tensor,
    t: torch.Tensor,
    q: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Concatenate the 5 blocks into the HybridBlock ``[B, H, W, 65]`` layout.

    Args:
        z, t, q, u, v: each ``torch.Tensor`` of shape ``(B, 13, H, W)``.

    Returns:
        ``torch.Tensor`` of shape ``(B, H, W, 65)`` with channel order z,t,q,u,v.
    """
    channels_first = torch.cat([z, t, q, u, v], dim=1)
    return channels_first.permute(0, 2, 3, 1).contiguous()


def plausible_physics_normalization() -> tuple[torch.Tensor, torch.Tensor]:
    """Return plausible per-channel (mean, std) for the 69-channel state.

    Returns:
        Tuple ``(mean, std)``, each ``torch.Tensor`` of shape ``(69,)``. Channel
        layout: surface 0:4, z 4:17, t 17:30, r 30:43, u 43:56, v 56:69.
    """
    mean = torch.zeros(69)
    mean[4:17] = 50000.0
    mean[17:30] = 280.0
    mean[30:43] = 50.0
    std = torch.ones(69)
    std[4:17] = 3000.0
    std[17:30] = 15.0
    std[30:43] = 25.0
    std[43:56] = 10.0
    std[56:69] = 10.0
    return mean, std


def build_b1_like_model(device: torch.device, **overrides: object) -> IAM4VP:
    """Build a B1-like residual-corrector IAM4VP (config v4 + USA geometry).

    Mirrors ``configs/pi_iam4vp_residual_usa_v4.yaml`` (stable_physical residual
    corrector, physical input space, relative->specific humidity, tendency clip
    8.0) with mass-consistent omega and the USA-crop physics geometry, and calls
    ``set_physics_normalization`` with plausible stats so the physical branch is
    ready.

    Args:
        device: device for the model and its normalization buffers.
        **overrides: keyword overrides forwarded to ``IAM4VP`` (e.g.
            ``physics_tendency_on_latent`` or ``physics_prior_detach``).

    Returns:
        A constructed ``IAM4VP`` on ``device`` with physics normalization set.
    """
    params: dict[str, object] = {
        "use_physics": False,
        "use_physics_residual_corrector": True,
        "physics_residual_hidden_channels": 128,
        "physics_residual_apply_to": "upper_air_only",
        "physics_residual_zero_init": True,
        "physics_residual_lambda_l1": 0.0001,
        "physics_feature_mode": "tendency",
        "physics_residual_shuffle": "none",
        "physics_residual_hybrid_steps": 3,
        "physics_residual_hybrid_mode": "stable_physical",
        "physics_residual_input_space": "physical",
        "physics_residual_humidity_mode": "relative_to_specific",
        "physics_residual_tendency_clip": 8.0,
        "physics_w_diagnostic": "mass_consistent",
        "physics_lat_start_deg": USA_LAT_START_DEG,
        "physics_dlat_deg": USA_DLAT_DEG,
        "physics_dlon_deg": USA_DLON_DEG,
    }
    params.update(overrides)
    model = IAM4VP(**params).to(device)
    mean, std = plausible_physics_normalization()
    model.set_physics_normalization(mean, std)
    return model


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two tensors flattened to a single vector.

    Args:
        a: first ``torch.Tensor`` (any shape).
        b: second ``torch.Tensor`` broadcastable/flattenable to ``a``'s size.

    Returns:
        Scalar cosine similarity as a Python float.
    """
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=1).item()


def check_equations_fixed(device: torch.device, results: list[tuple[str, bool]]) -> None:
    """F1/F8/T-1: adiabatic thermodynamics bounded; legacy blows up; R scaling.

    Args:
        device: compute device.
        results: outcome accumulator (mutated).

    Side effects:
        Appends three ``report`` outcomes.
    """
    torch.manual_seed(0)
    z, t, q, u, v = build_physical_fields(batch_size=2, grid_h=8, grid_w=16, device=device)

    def run_kernel(t_t_formulation: str, use_universal_R: bool) -> PDE_kernel:
        kernel = PDE_kernel(
            in_dim=65,
            physics_part_coef=0.5,
            block_dt=400,
            grid_h=8,
            t_t_formulation=t_t_formulation,
            use_universal_R=use_universal_R,
        ).to(device)
        kernel.eval()
        kernel.share_z_dxyz(z)
        w = kernel.get_w(u, v)
        kernel.get_t_t(u, v, w, t)
        return kernel

    kernel_adiabatic = run_kernel("adiabatic_omega", use_universal_R=False)
    kernel_legacy = run_kernel("legacy_paper", use_universal_R=False)
    max_adiabatic = kernel_adiabatic.t_t.abs().max().item()
    max_legacy = kernel_legacy.t_t.abs().max().item()
    blowup_ratio = max_legacy / max_adiabatic

    report(
        results,
        "check_equations_fixed.adiabatic_bounded",
        max_adiabatic < 1e2,
        f"max|t_t_adiabatic|={max_adiabatic:.3e} K/s (<1e2)",
    )
    report(
        results,
        "check_equations_fixed.legacy_blows_up",
        blowup_ratio > 1e3,
        f"max|t_t_legacy|/max|t_t_adiabatic|={blowup_ratio:.3e} (>1e3)",
    )

    kernel_universal = run_kernel("adiabatic_omega", use_universal_R=True)
    z_zt_dry = kernel_adiabatic.get_z_zt()
    z_zt_universal = kernel_universal.get_z_zt()
    r_ratio = z_zt_dry.abs().max().item() / z_zt_universal.abs().max().item()
    expected_ratio = R_DRY / R_UNIVERSAL
    report(
        results,
        "check_equations_fixed.hydrostatic_R_ratio",
        abs(r_ratio - expected_ratio) / expected_ratio < 0.05,
        f"z_zt(R_d)/z_zt(R)={r_ratio:.3f} vs 287/8.314={expected_ratio:.3f}",
    )


def check_geometry(device: torch.device, results: list[tuple[str, bool]]) -> None:
    """F2/P-1: crop-aware USA geometry vs preserved global default geometry.

    Args:
        device: compute device.
        results: outcome accumulator (mutated).

    Side effects:
        Appends four ``report`` outcomes.
    """
    usa = PDE_kernel(
        in_dim=65,
        physics_part_coef=0.5,
        grid_h=USA_GRID_H,
        lat_start_deg=USA_LAT_START_DEG,
        dlat_deg=USA_DLAT_DEG,
        dlon_deg=USA_DLON_DEG,
    ).to(device)
    f_field = usa.f_field.flatten()
    coriolis_ok = bool(
        (f_field > 0).all()
        and (f_field >= 4e-5).all()
        and (f_field <= 1.3e-4).all()
        and (f_field[1:] > f_field[:-1]).all()
    )
    report(
        results,
        "check_geometry.usa_coriolis",
        coriolis_ok,
        f"f in [{f_field.min():.2e},{f_field.max():.2e}] positive & monotone S->N",
    )

    pixel_y = float(usa.pixel_y)
    expected_pixel_y = EARTH_RADIUS_M * DEG2RAD * USA_DLAT_DEG
    report(
        results,
        "check_geometry.usa_pixel_y",
        abs(pixel_y - expected_pixel_y) / expected_pixel_y < 0.01,
        f"pixel_y={pixel_y:.1f} m vs R*d2r(5.625)={expected_pixel_y:.1f} m",
    )

    pixel_x_row0 = float(usa.pixel_x.flatten()[0])
    expected_pixel_x = (
        EARTH_RADIUS_M * math.cos(USA_LAT_START_DEG * DEG2RAD) * DEG2RAD * USA_DLON_DEG
    )
    report(
        results,
        "check_geometry.usa_pixel_x_row0",
        abs(pixel_x_row0 - expected_pixel_x) / expected_pixel_x < 0.01,
        f"pixel_x[0]={pixel_x_row0:.1f} m vs R*cos(lat0)*d2r(5.625)={expected_pixel_x:.1f} m",
    )

    default = PDE_kernel(in_dim=65, physics_part_coef=0.5, grid_h=8).to(device)
    default_f = default.f_field.flatten()
    both_signs = bool((default_f > 0).any() and (default_f < 0).any())
    default_pixel_y = float(default.pixel_y)
    expected_default_pixel_y = math.pi * EARTH_RADIUS_M / 9.0
    default_rel = abs(default_pixel_y - expected_default_pixel_y) / expected_default_pixel_y
    default_pixel_y_ok = default_rel < 0.01
    report(
        results,
        "check_geometry.default_global_grid",
        both_signs and default_pixel_y_ok,
        f"f has both signs={both_signs}; pixel_y={default_pixel_y:.1f} vs pi*R/9="
        f"{expected_default_pixel_y:.1f}",
    )


def check_dt_semantics(device: torch.device, results: list[tuple[str, bool]]) -> None:
    """F3/B-1: physical_clip scales with block_dt; scale_diff is dt-invariant.

    Args:
        device: compute device.
        results: outcome accumulator (mutated).

    Side effects:
        Appends three ``report`` outcomes.
    """
    torch.manual_seed(0)
    z, t, q, u, v = build_physical_fields(batch_size=2, grid_h=8, grid_w=16, device=device)
    # Scale winds to ~1 m/s so physical_clip tendencies stay below the caps.
    u = u * 0.1
    v = v * 0.1

    def u_increment(
        block_dt: float, limiter: str, reference: PDE_kernel | None
    ) -> tuple[torch.Tensor, PDE_kernel]:
        kernel = PDE_kernel(
            in_dim=65,
            physics_part_coef=0.5,
            block_dt=block_dt,
            grid_h=8,
            tendency_limiter=limiter,
        ).to(device)
        if reference is not None:
            kernel.load_state_dict(reference.state_dict())
        kernel.eval()
        kernel.share_z_dxyz(z)
        w = kernel.get_w(u, v)
        u_new, _ = kernel.uv_evolution(u, v, w)
        return u_new - u, kernel

    inc_clip_400, ref_clip = u_increment(400.0, "physical_clip", None)
    inc_clip_40, _ = u_increment(40.0, "physical_clip", ref_clip)
    clip_ratio = (inc_clip_400.norm() / inc_clip_40.norm()).item()

    report(
        results,
        "check_dt_semantics.physical_clip_scales_with_dt",
        not torch.allclose(inc_clip_400, inc_clip_40, atol=ATOL),
        f"increments differ across block_dt 400 vs 40 (norm ratio={clip_ratio:.3f})",
    )
    report(
        results,
        "check_dt_semantics.physical_clip_ratio_10x",
        5.0 <= clip_ratio <= 15.0,
        f"|Δu(400)|/|Δu(40)|={clip_ratio:.3f} in [5, 15]",
    )

    inc_scale_400, ref_scale = u_increment(400.0, "scale_diff", None)
    inc_scale_40, _ = u_increment(40.0, "scale_diff", ref_scale)
    report(
        results,
        "check_dt_semantics.scale_diff_dt_invariant",
        torch.allclose(inc_scale_400, inc_scale_40, atol=ATOL),
        "scale_diff increments identical across block_dt 400 vs 40 (legacy artefact)",
    )


def check_batch_independence(device: torch.device, results: list[tuple[str, bool]]) -> None:
    """F3: physical_clip decouples samples; scale_diff leaks across the batch.

    The physical_clip tendency limiter is element-wise and ``get_qs`` clamps the
    Magnus exponent elementwise, so in ``eval()`` mode every sample's HybridBlock
    output (both the conv x-path and the evolved ``zquvtw``, all 65 channels) is
    identical whether the sample runs in a batch of 2 or solo. The legacy
    ``scale_diff`` limiter normalizes increments by batch-global min/max and thus
    couples samples: scaling sample 1 changes sample 0's output.

    Args:
        device: compute device.
        results: outcome accumulator (mutated).

    Side effects:
        Appends two ``report`` outcomes.
    """
    torch.manual_seed(0)
    z, t, q, u, v = build_physical_fields(batch_size=2, grid_h=8, grid_w=16, device=device)
    x = pack_channels_last(z, t, q, u, v)

    block_clip = HybridBlock(
        dim=65,
        zquvtw_channel=13,
        depth=1,
        block_dt=400,
        inverse_time=False,
        physics_part_coef=0.5,
        grid_h=8,
        tendency_limiter="physical_clip",
    ).to(device)
    block_clip.eval()
    with torch.no_grad():
        x_batch, zquvtw_batch = block_clip(x, x)
        batch_independent = True
        max_diff = 0.0
        for sample in range(x.shape[0]):
            single = x[sample : sample + 1]
            x_single, zquvtw_single = block_clip(single, single)
            x_ok = torch.allclose(x_batch[sample : sample + 1], x_single, atol=ATOL)
            zquvtw_ok = torch.allclose(zquvtw_batch[sample : sample + 1], zquvtw_single, atol=ATOL)
            batch_independent = batch_independent and x_ok and zquvtw_ok
            max_diff = max(
                max_diff,
                (x_batch[sample : sample + 1] - x_single).abs().max().item(),
                (zquvtw_batch[sample : sample + 1] - zquvtw_single).abs().max().item(),
            )

    report(
        results,
        "check_batch_independence.physical_clip_full",
        batch_independent,
        f"x-out & zquvtw (all 65 ch) batch-vs-solo max diff={max_diff:.3e} (atol={ATOL:g})",
    )

    block_scale = HybridBlock(
        dim=65,
        zquvtw_channel=13,
        depth=1,
        block_dt=400,
        inverse_time=False,
        physics_part_coef=0.5,
        grid_h=8,
        tendency_limiter="scale_diff",
    ).to(device)
    block_scale.eval()
    x_scaled = x.clone()
    x_scaled[1] = x_scaled[1] * 100.0
    with torch.no_grad():
        _, zquvtw_batch_scaled = block_scale(x_scaled, x_scaled)
        solo = x_scaled[0:1]
        _, zquvtw_solo = block_scale(solo, solo)
    # z block stays finite at 100x (u/v overflow); it cleanly shows the leakage.
    z_block_batch = zquvtw_batch_scaled[0:1].chunk(5, dim=-1)[0]
    z_block_solo = zquvtw_solo.chunk(5, dim=-1)[0]
    leak = (z_block_batch - z_block_solo).abs().max().item()
    report(
        results,
        "check_batch_independence.scale_diff_leaks",
        math.isfinite(leak) and leak > ATOL,
        f"sample-0 z-block shifts by {leak:.3e} when sample-1 scaled 100x (>{ATOL:g})",
    )


def check_latent_tendency(device: torch.device, results: list[tuple[str, bool]]) -> None:
    """F4: tendency-on-latent carries physics only, not the resampling residual.

    Uses a difference-of-modes formulation. Before the final sanitizer, the
    legacy prior is ``up(evolved_latent)`` and the on-latent prior is
    ``input + up(evolved_latent - input_latent)``, so with IDENTICAL weights the
    two priors differ by exactly the coarse<->fine resampling highpass of the
    input. A direct cosine of the legacy delta against the highpass (the earlier
    formulation) is no longer meaningful: since the elementwise-clamp ``get_qs``
    fix, the humidity path is active on synthetic data and the (coarse-grid,
    low-frequency) physics deltas exceed the highpass by ~10x, diluting the
    cosine without touching the resampling mechanism under test.

    Measured on the z block only and with the check's own models built with
    ``physics_residual_tendency_clip=0.0``: z is the one variable group whose
    sanitize bounds this synthetic data never hits and which skips the nonlinear
    r->q->r humidity round-trip, and the ±clip saturation would otherwise erase
    the highpass signature (t/u/v are attenuated to ~0.75 by output clamps).

    Args:
        device: compute device.
        results: outcome accumulator (mutated).

    Side effects:
        Appends three ``report`` outcomes.
    """
    # Identical weights across modes: reseed before each build.
    torch.manual_seed(0)
    model_on_latent = build_b1_like_model(
        device, physics_tendency_on_latent=True, physics_residual_tendency_clip=0.0
    )
    torch.manual_seed(0)
    model_legacy = build_b1_like_model(
        device, physics_tendency_on_latent=False, physics_residual_tendency_clip=0.0
    )
    model_on_latent.eval()
    model_legacy.eval()
    weights_equal = all(
        torch.equal(param_a, param_b)
        for (_, param_a), (_, param_b) in zip(
            model_on_latent.state_dict().items(),
            model_legacy.state_dict().items(),
            strict=True,
        )
    )

    torch.manual_seed(1)
    grid_h, grid_w = 32, 64
    prev_state = 0.5 * torch.randn(2, 69, grid_h, grid_w, device=device)
    row = torch.arange(grid_h, device=device).view(grid_h, 1)
    col = torch.arange(grid_w, device=device).view(1, grid_w)
    checker = 1.0 - 2.0 * ((row + col) % 2).to(torch.float32)  # +1/-1 checkerboard
    prev_state[:, 4:] = prev_state[:, 4:] + 0.5 * checker

    prev_upper_air = prev_state[:, 4:]
    downsampled = F.interpolate(prev_upper_air, size=(8, 16), mode="bilinear")
    reconstructed = F.interpolate(downsampled, size=(grid_h, grid_w), mode="bilinear")
    highpass = prev_upper_air - reconstructed

    with torch.no_grad():
        delta_on_latent = (model_on_latent._physics_prior_from_state(prev_state) - prev_state)[
            :, 4:
        ]
        delta_legacy = (model_legacy._physics_prior_from_state(prev_state) - prev_state)[:, 4:]

    cos_on_latent = _cosine(delta_on_latent, -highpass)
    report(
        results,
        "check_latent_tendency.on_latent_no_resampling_residual",
        cos_on_latent < 0.3,
        f"cos(delta_on_latent, -highpass)={cos_on_latent:.3f} (<0.3)",
    )

    z_slice = slice(0, 13)  # z block within the upper-air layout
    mode_diff_z = (delta_legacy - delta_on_latent)[:, z_slice]
    highpass_z = highpass[:, z_slice]
    cos_diff = _cosine(mode_diff_z, -highpass_z)
    report(
        results,
        "check_latent_tendency.legacy_minus_on_latent_is_highpass",
        weights_equal and cos_diff > 0.9,
        f"cos(delta_legacy - delta_on_latent, -highpass) on z={cos_diff:.3f} (>0.9), "
        f"identical_weights={weights_equal}",
    )

    magnitude_ratio = (mode_diff_z.norm() / highpass_z.norm()).item()
    report(
        results,
        "check_latent_tendency.mode_difference_magnitude",
        0.8 <= magnitude_ratio <= 1.2,
        f"|delta_legacy - delta_on_latent|/|highpass| on z={magnitude_ratio:.3f} in [0.8, 1.2]",
    )


def check_prior_detach(device: torch.device, results: list[tuple[str, bool]]) -> None:
    """F5: physics_prior_detach cuts task-loss gradient into the HybridBlock.

    Both runs nudge the zero-init corrector's final conv weight to a small
    constant so the feature path carries gradient and the comparison is
    meaningful.

    Args:
        device: compute device.
        results: outcome accumulator (mutated).

    Side effects:
        Appends two ``report`` outcomes; runs backward on two models.
    """
    torch.manual_seed(0)
    x_raw = torch.randn(2, 6, 69, 32, 64, device=device)
    t = torch.full((2,), 100.0, device=device)

    def hybrid_block_grad_norm(detach: bool) -> float:
        model = build_b1_like_model(device, physics_prior_detach=detach)
        with torch.no_grad():
            model.physics_residual_corrector.net[-1].weight.fill_(1e-3)
        prediction = model(x_raw, [], t)
        loss = prediction.square().mean()
        loss.backward()
        total = 0.0
        for name, param in model.named_parameters():
            if name.startswith("hybrid_block") and param.grad is not None:
                total += param.grad.abs().sum().item()
        return total

    grad_detached = hybrid_block_grad_norm(detach=True)
    grad_connected = hybrid_block_grad_norm(detach=False)
    report(
        results,
        "check_prior_detach.detach_true_no_gradient",
        grad_detached == 0.0,
        f"sum|grad| into hybrid_block with detach=True is {grad_detached:.3e} (==0)",
    )
    report(
        results,
        "check_prior_detach.detach_false_has_gradient",
        grad_connected > 0.0,
        f"sum|grad| into hybrid_block with detach=False is {grad_connected:.3e} (>0)",
    )


def check_zero_init_identity(device: torch.device, results: list[tuple[str, bool]]) -> None:
    """Zero-init corrector is an exact identity with zeroed physics diagnostics.

    Args:
        device: compute device.
        results: outcome accumulator (mutated).

    Side effects:
        Appends one ``report`` outcome; runs one forward pass.
    """
    torch.manual_seed(0)
    model = build_b1_like_model(device)
    model.eval()
    x_raw = torch.randn(2, 6, 69, 32, 64, device=device)
    t = torch.full((2,), 100.0, device=device)
    with torch.no_grad():
        model(x_raw, [], t)
    diagnostics = model.physics_residual_diagnostics()
    correction_rms = diagnostics["physics_residual_correction_rms"].item()
    has_router = "physics_router_weight_abs" in diagnostics
    has_gamma = "physics_hybrid_bn_gamma_drift" in diagnostics
    router_zero = has_router and diagnostics["physics_router_weight_abs"].item() == 0.0
    gamma_zero = has_gamma and diagnostics["physics_hybrid_bn_gamma_drift"].item() == 0.0
    edge_key = "physics_residual_delta_edge_interior_ratio"
    edge_finite = edge_key in diagnostics and bool(torch.isfinite(diagnostics[edge_key]).item())
    edge_value = diagnostics[edge_key].item() if edge_key in diagnostics else float("nan")
    report(
        results,
        "check_zero_init_identity.zero_correction_and_diagnostics",
        correction_rms == 0.0 and router_zero and gamma_zero and edge_finite,
        f"correction_rms={correction_rms:g}, router_weight_abs=0, bn_gamma_drift=0, "
        f"edge/interior ratio finite ({edge_value:.3f}) at init",
    )


def check_mass_consistent_invariant(device: torch.device, results: list[tuple[str, bool]]) -> None:
    """mass_consistent w: column-integrated corrected divergence vanishes.

    Verifies the algebraic invariant behind exp 12 / E-omega on the USA-crop
    grid: after subtracting the pixel_z-weighted column mean, the
    pixel_z-weighted column sum of the divergence is ~0 for every column, and
    ``get_w`` equals ``integral_z(-div_corrected)``.

    Args:
        device: compute device.
        results: outcome accumulator (mutated).

    Side effects:
        Appends two ``report`` outcomes.
    """
    torch.manual_seed(0)
    _, _, _, u, v = build_physical_fields(batch_size=2, grid_h=8, grid_w=16, device=device)
    kernel = PDE_kernel(
        in_dim=65,
        physics_part_coef=0.5,
        block_dt=400,
        grid_h=8,
        lat_start_deg=18.28125,
        dlat_deg=5.625,
        dlon_deg=5.625,
        w_diagnostic="mass_consistent",
    ).to(device)
    kernel.eval()
    div = kernel._d_x(u) + kernel._d_y(v)
    pz = pixel_z.reshape(1, -1, 1, 1).to(dtype=div.dtype, device=div.device)
    div_corrected = div - (div * pz).sum(dim=1, keepdim=True) / pz.sum()
    column_residual = (div_corrected * pz).sum(dim=1).abs().max().item()
    column_scale = (div.abs() * pz).sum(dim=1).max().item()
    report(
        results,
        "check_mass_consistent_invariant.column_divergence_zero",
        column_residual < 1e-6 * max(column_scale, 1e-30),
        f"max |sum(div_corr*pz)|={column_residual:.3e} vs scale {column_scale:.3e}",
    )
    w_kernel = kernel.get_w(u, v)
    w_manual = integral_z(-div_corrected)
    w_match = torch.allclose(w_kernel, w_manual, atol=1e-6, rtol=1e-5)
    report(
        results,
        "check_mass_consistent_invariant.get_w_matches_integral",
        w_match,
        f"get_w == integral_z(-div_corr): {w_match}",
    )


def check_diabatic_mask(device: torch.device, results: list[tuple[str, bool]]) -> None:
    """diabatic_apply_to: t_and_q masks exactly the T and humidity blocks.

    Args:
        device: compute device (mask logic is device-free; kept for symmetry).
        results: outcome accumulator (mutated).

    Side effects:
        Appends two ``report`` outcomes.
    """
    mask_all = IAM4VP._build_diabatic_mask("all_upper_air", 65, 0)
    mask_tq = IAM4VP._build_diabatic_mask("t_and_q", 65, 0)
    mask_tq_surface = IAM4VP._build_diabatic_mask("t_and_q", 69, 4)
    all_ones = bool((mask_all == 1.0).all().item())
    flat_tq = mask_tq.flatten()
    tq_correct = bool(
        (flat_tq[13:39] == 1.0).all().item()
        and (flat_tq[:13] == 0.0).all().item()
        and (flat_tq[39:] == 0.0).all().item()
    )
    flat_tq_s = mask_tq_surface.flatten()
    tq_surface_correct = bool(
        (flat_tq_s[17:43] == 1.0).all().item()
        and (flat_tq_s[:17] == 0.0).all().item()
        and (flat_tq_s[43:] == 0.0).all().item()
    )
    report(
        results,
        "check_diabatic_mask.blocks",
        all_ones and tq_correct and tq_surface_correct,
        "all_upper_air=ones; t_and_q selects [13:39) (upper-air) / [17:43) (with surface)",
    )
    model = IAM4VP(
        use_physics=False,
        use_physics_residual_corrector=True,
        physics_feature_mode="no_physics",
        diabatic_apply_to="t_and_q",
    )
    report(
        results,
        "check_diabatic_mask.flag_stored",
        model.diabatic_apply_to == "t_and_q",
        f"diabatic_apply_to stored: {model.diabatic_apply_to}",
    )


def check_block_dt_accounting(device: torch.device, results: list[tuple[str, bool]]) -> None:
    """F3b: block_dt splits the horizon across depth*hybrid_steps kernel calls.

    Args:
        device: compute device.
        results: outcome accumulator (mutated).

    Side effects:
        Appends one ``report`` outcome.
    """
    model = build_b1_like_model(device)
    kernel_block_dt = model.hybrid_block.pde_block.PDE_kernels[0].block_dt
    expected = 3600.0 / (3 * 3)
    report(
        results,
        "check_block_dt_accounting.horizon_split",
        kernel_block_dt == expected,
        f"PDE_kernel.block_dt={kernel_block_dt} == 3600/(3*3)={expected}",
    )


def main() -> None:
    """Parse ``--device``, run every check, print a summary, and exit.

    Side effects:
        Writes results to stdout and calls ``sys.exit(1)`` if any check failed.
    """
    parser = argparse.ArgumentParser(description="PI-IAM4VP HybridBlock fix sanity checks")
    parser.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
    args = parser.parse_args()
    device = torch.device(args.device)

    warnings.filterwarnings("ignore", category=UserWarning)
    torch.manual_seed(0)
    print(f"Device: {device}")

    results: list[tuple[str, bool]] = []
    check_equations_fixed(device, results)
    check_geometry(device, results)
    check_dt_semantics(device, results)
    check_batch_independence(device, results)
    check_latent_tendency(device, results)
    check_prior_detach(device, results)
    check_zero_init_identity(device, results)
    check_mass_consistent_invariant(device, results)
    check_diabatic_mask(device, results)
    check_block_dt_accounting(device, results)

    failures = [name for name, passed in results if not passed]
    passed_count = len(results) - len(failures)
    print(f"\n{passed_count} passed, {len(failures)} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
