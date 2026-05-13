"""Single-card sanity check for PredFormer with neural-operator integrations.

Covers five configurations on the USA-crop input shape (B=1, T=12, C=69, 32x64):

  1. baseline:               pos=sinusoidal, space schedule = all attn
  2. fourier_2d_pe:          pos=fourier_2d, space schedule = all attn
  3. hybrid_spectral_inner:  pos=sinusoidal, space schedule = [attn, spectral..., attn]
  4. fourier_pe + hybrid:    pos=fourier_2d, space schedule = hybrid
  5. dual_bridge:            pos=sinusoidal, space schedule = [attn, dual..., attn];
                             alpha is swept across {1.0, 0.5, 0.0} per run

For each config:
  - instantiate model on a single CUDA device (falls back to CPU if unavailable)
  - run one forward pass on a random input
  - check output shape, finiteness (no NaN/Inf), peak GPU memory
  - run one backward pass on a simple L2 loss and check that grads are finite

This is NOT a training script. It is intended to validate that the new modules
plug into the existing PredFormer cleanly before kicking off any cluster run.

Run on the cluster (where the constants netCDF is available):

    python -m Models.dev.sanity_predformer_no \
        --path-to-constants \
        /home/fratnikov/weather_bench/1.40625deg/constants/constants_1.40625deg.nc

Optional: ``--full-grid`` runs the same configs on the global 128x256 grid
with no cut, to verify large-N behavior of SpectralMixer2D.
"""

import argparse
import copy
import time

import torch

from Models.PredFormer import PredFormer_Model


def build_base_config(path_to_constants: str, full_grid: bool) -> dict:
    """Return a baseline PredFormer model_config matching the USA-crop run."""
    if full_grid:
        cfg = {
            'height': 128,
            'width': 256,
            'cut': None,
            'Ndepth': 4,
        }
    else:
        cfg = {
            'height': 32,
            'width': 64,
            'cut': [[75, 107], [164, 228]],
            'Ndepth': 4,
        }
    cfg.update({
        'num_channels': 69,
        'pre_seq': 12,
        'after_seq': 12,
        'patch_size': 8,
        'dim': 256,
        'heads': 8,
        'dim_head': 32,
        'dropout': 0.0,
        'attn_dropout': 0.0,
        'drop_path': 0.0,
        'scale_dim': 4,
        'depth': 1,
        'path_to_constants': path_to_constants,
    })
    return cfg


def _hybrid_schedule_with(op: str, ndepth: int, num_edge_attn: int = 1) -> list[str]:
    """First/last ``num_edge_attn`` layers stay attn, middle becomes ``op``."""
    num_edge_attn = min(num_edge_attn, ndepth // 2)
    num_inner = ndepth - 2 * num_edge_attn
    schedule = ['attn'] * num_edge_attn + [op] * num_inner + ['attn'] * num_edge_attn
    assert len(schedule) == ndepth
    return schedule


def _configs(base_cfg: dict) -> dict[str, dict]:
    ndepth = base_cfg['Ndepth']
    hybrid_spectral = _hybrid_schedule_with('spectral', ndepth, num_edge_attn=1)
    hybrid_bridge = _hybrid_schedule_with('dual_bridge', ndepth, num_edge_attn=1)
    spectral_kwargs = {
        'channel_blocks': 8,
        'soft_thresh_lambda': 0.01,
        'use_local_branch': True,
    }
    pe_kwargs = {'pos_encoding_type': 'fourier_2d', 'pe_K_lat': 16, 'pe_K_lon': 16, 'pe_K_t': 8}

    cfg_baseline = copy.deepcopy(base_cfg)

    cfg_fourier_pe = copy.deepcopy(base_cfg)
    cfg_fourier_pe.update(pe_kwargs)

    cfg_hybrid_spec = copy.deepcopy(base_cfg)
    cfg_hybrid_spec['space_op_schedule'] = hybrid_spectral
    cfg_hybrid_spec['spectral_kwargs'] = dict(spectral_kwargs)

    cfg_both = copy.deepcopy(cfg_hybrid_spec)
    cfg_both.update(pe_kwargs)

    cfg_dual_bridge = copy.deepcopy(base_cfg)
    cfg_dual_bridge['space_op_schedule'] = hybrid_bridge
    cfg_dual_bridge['spectral_kwargs'] = dict(spectral_kwargs)

    return {
        'baseline': cfg_baseline,
        'fourier_2d_pe': cfg_fourier_pe,
        'hybrid_spectral_inner': cfg_hybrid_spec,
        'fourier_pe + hybrid': cfg_both,
        'dual_bridge': cfg_dual_bridge,
    }


def _count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _run_one(name: str, cfg: dict, device: torch.device, alpha: float | None = None) -> None:
    print(f"\n=== {name} ===")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    model = PredFormer_Model(cfg).to(device)
    if alpha is not None:
        n_bridges = model.set_dual_bridge_alpha(alpha)
        print(f"  set alpha={alpha} on {n_bridges} dual_bridge(s)")
    total, trainable = _count_params(model)
    print(f"  params: total={total:,}  trainable={trainable:,}  build_time={time.time() - t0:.2f}s")

    B, T, C, H, W = 1, cfg['pre_seq'], cfg['num_channels'], cfg['height'], cfg['width']
    x = torch.randn(B, T, C, H, W, device=device)
    target = torch.randn_like(x)

    model.train()
    t0 = time.time()
    out = model(x)
    fwd_t = time.time() - t0

    if out.shape != (B, T, C, H, W):
        raise AssertionError(f"output shape {tuple(out.shape)} != expected {(B, T, C, H, W)}")
    if not torch.isfinite(out).all():
        nan_count = (~torch.isfinite(out)).sum().item()
        raise AssertionError(f"forward produced {nan_count} non-finite values")

    loss = (out - target).pow(2).mean()
    t0 = time.time()
    loss.backward()
    bwd_t = time.time() - t0

    bad_grad = []
    for p_name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad_grad.append(p_name)
    if bad_grad:
        raise AssertionError(
            f"non-finite gradients in {len(bad_grad)} params, first: {bad_grad[:3]}"
        )

    peak_mem = None
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    mem_str = f"  peak_mem={peak_mem:.1f}MB" if peak_mem is not None else ""
    print(f"  forward: {fwd_t * 1000:.1f}ms  backward: {bwd_t * 1000:.1f}ms"
          f"  loss={loss.item():.4f}{mem_str}")
    print("  OK: shapes, finiteness, gradients")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--path-to-constants', required=True,
        help='Path to constants_*.nc file used by PredFormer for static masks and lat/lon.',
    )
    parser.add_argument(
        '--full-grid', action='store_true',
        help='Run on the full 128x256 grid with no cut instead of the USA crop.',
    )
    parser.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA is available.')
    args = parser.parse_args()

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device('cuda', 0) if use_cuda else torch.device('cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    base_cfg = build_base_config(args.path_to_constants, args.full_grid)
    print(f"Input shape: B=1 T={base_cfg['pre_seq']} C={base_cfg['num_channels']} "
          f"H={base_cfg['height']} W={base_cfg['width']}  Ndepth={base_cfg['Ndepth']}")

    for name, cfg in _configs(base_cfg).items():
        if name == 'dual_bridge':
            for alpha in (1.0, 0.5, 0.0):
                _run_one(f"{name} (alpha={alpha})", cfg, device, alpha=alpha)
        else:
            _run_one(name, cfg, device)

    print("\nAll configurations passed sanity checks.")


if __name__ == '__main__':
    main()
