# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `sh_files/bench_dataloader.sh` — SLURM submit script for diagnostic
  dataloader benchmarks. Exports `BENCH_MAX_STEPS` and `PROFILE_TRACE_DIR`
  so `trainer.py` runs a short profiled epoch and exits before validation
  / checkpoint writes. Writes `trace.json` (torch.profiler), `dmon.csv`
  (`nvidia-smi dmon`), `bench.log`, `config_snapshot.yaml` and `git_sha.txt`
  to `bench_logs/<TS>_<host>_<jobid>_<tag>/`.
- `configs/bench_predformer_usa.yaml` — short bench config: `max_epoch=1`,
  logging silenced (`log_every_n_steps=1000`), early-stopping disabled,
  trimmed val date range. Paired with `BENCH_MAX_STEPS` for a hard
  step-count cap.
- `bench_logs/` — gitignored output directory for bench artefacts.
- `tools/repack_era5.py` — one-shot packer that turns the per-variable
  per-year netCDF tree under `/home/fratnikov/weather_bench/1.40625deg/`
  into a single contiguous float32 `np.memmap`. The output already has the
  channel filter (`variables_list`, 110→69) and the spatial cut applied,
  so `WeatherBench128Memmap.__getitem__` is a couple of row slices plus
  normalisation — no per-sample `h5netcdf` parse, no spatial cut, no
  channel filter.
- `Data.weatherbench_128_v3.WeatherBench128Memmap` — subclass that reads
  from the memmap built by `tools/repack_era5.py`. Registered as the
  `v3_memmap` dataset version via `Data/__init__.py`.
- `configs/bench_memmap.yaml` — bench config that points at the §2.5
  memmap (`/home/fa.buzaev/era5_memmap/predformer_usa_2000_2004.dat`).
- `sh_files/repack_era5.sh` — SLURM submit script that runs the packer
  with env-driven year/cut/output overrides.
- `train.build_dataset`: also forwards `data.memmap_path` and
  `data.memmap_meta_path` when set (alongside the previously-added
  `data_folder` / `input_folder` overrides), so the same path through
  `build_dataset` works for v3 and v3_memmap without per-version code.
- `trainer.py` with a pure-PyTorch `Trainer` (DDP via `DistributedDataParallel`,
  AMP-ready via `torch.amp`, step-cadenced LR scheduler, EarlyStopping, top-1
  and `last.pt` checkpointing).
- `training_strategies/` package with five `StepStrategy` subclasses
  (`SimpleStep`, `TimestepSelectStep`, `AutoregressiveStep`,
  `IterativeManualStep`, `PredRNNStep`) replacing the nine LitModels.
- `utils/registry.py` — module registries for models / datasets / strategies,
  eliminating the local-import + if-chain in `train.py`.
- `utils/distributed.py`, `utils/experiment.py`, `utils/early_stopping.py`,
  `utils/checkpointing.py` — building blocks for the new trainer.
- `utils/checkpointing.convert_lightning_checkpoint` plus CLI:
  `python -m utils.checkpointing convert <src.ckpt> <dst.pt>`.
- `train/_common.run_legacy_training` helper used by every script under
  `train/` and `train/dev/` to share the Trainer boilerplate.
- `utils/normalize.WeatherNormalize` — `nn.Module` that owns per-channel
  `mean`/`std` as `register_buffer`. Single source of truth for ERA5
  z-score normalisation: round-trips through `state_dict`, follows
  `.to(device)` and DDP replication, and stays fp32 under autocast.
- `Data.weatherbench_128_v4.WeatherBench128V4` — raw-memmap dataset that
  returns un-normalised float32 tensors. Subclass of `WeatherBench128Memmap`
  overriding only `__getitem__`; reuses parent's time list, valid_idx,
  mean/std loading and memmap mmap. Registered as `v4` via
  `Data/__init__.py`. Numerical equivalence with `v3_memmap` confirmed
  bitwise via `tools/verify_v4_normalize.py`.
- `tools/verify_v4_normalize.py` — equivalence test: reads one sample from
  `v3_memmap` (normalise-in-dataset) and `v4` + `WeatherNormalize`, asserts
  bit-exact match.
- `configs/predformer_usa_v4.yaml` — production PredFormer-USA config that
  uses the v4 dataset and trainer-side normalisation.
- `sh_files/train_PredFormer_USA_2gpu_v4.sh` — 2-GPU DDP launcher with
  STAGE_MEMMAP staging that targets the v4 config.

### Changed

- Pinned `xarray==2024.7.0`, `h5netcdf==1.6.1` and `h5py==3.15.1` (the
  previous `xarray==2025.6.1` / `h5netcdf==1.8.1` pins required `numpy>=2`,
  which broke the rest of the stack). `numpy==1.26.3` is unchanged.
- Migrated the training stack from PyTorch Lightning to pure PyTorch.
  Existing YAML configs continue to work without edits; Lightning-specific
  keys (`trainer.precision`, `trainer.static_graph`,
  `trainer.log_every_n_steps`, `trainer.num_sanity_val_steps`,
  `trainer.float32_matmul_precision`) are reinterpreted by the new trainer.
- `sh_files/launch_train.sh` now invokes `torchrun` instead of `python` so
  `RANK`/`LOCAL_RANK`/`WORLD_SIZE` are set in the environment for
  `utils.distributed.setup_distributed`.
- Moved OpenSTL-derived layer blocks (HorNet, MogaNet, PoolFormer, UniFormer, VAN)
  from `Models/layers_openstl/` to a new package `Models/blocks/`. Public class
  names preserved; attribution to upstream repositories kept in the file headers.
- Rephrased `README.md` to credit upstream sources directly without framing the
  project as a fork of OpenSTL.
- `train.py`: `DataLoader` now reads `pin_memory`, `persistent_workers`,
  `prefetch_factor` and per-split `drop_last` from the YAML `data` block.
  Previously these keys were silently ignored, forcing the defaults
  (`pin_memory=False`, `persistent_workers=False`) regardless of what the
  config specified.
- `Data/weatherbench_128_v3.WeatherBench128` accepts `data_folder` and
  `input_folder` as `__init__` kwargs (previously hardcoded to
  `/home/fratnikov/weather_bench/...`); `train.build_dataset` forwards them
  from `data.data_folder` / `data.input_folder` when set, leaving the
  defaults intact for existing configs.
- `trainer.py` and `Data/weatherbench_128_v3.py`: instrumented the
  training-loop hot path and `__getitem__` with
  `torch.profiler.record_function` tags (`data_wait`, `to_device`,
  `forward`, `backward`, `optimizer_step`, `custom_np_load_x/_y`,
  `normalization_x/_y`, `from_numpy_x/_y`, `stack_x/_y`) so per-phase
  cost is visible in profiler traces. Tags are no-ops when the profiler
  is disabled.
- `trainer.py`: when `PROFILE_TRACE_DIR` is set, the train loop is wrapped
  in `torch.profiler.profile` (Chrome trace dumped after
  warmup=10 + active=40 steps). When `BENCH_MAX_STEPS` is set, the loop
  breaks after that many steps and `fit()` skips validation and
  checkpoint writes. Both env vars default unset, so production behaviour
  is unchanged.
- `trainer.Trainer.fit`: new optional `normalize: nn.Module | None` kwarg.
  When provided (v4 datasets), it is moved to the trainer's device once
  and applied to every batch right after `_to_device`, so strategies and
  models keep seeing normalised tensors. `_normalize_batch` walks
  tuples/lists/dicts to preserve batch structure.
- `train.py`: optional `training.seed` field — when set to a non-negative
  int, fixes model init / sampler / worker RNGs via `torch.manual_seed`
  and `torch.cuda.manual_seed_all`. Default unset = non-deterministic init.
  Used for A/B-style numerical comparisons (v3_memmap vs v4).
- `train.py`: AdamW now uses `fused=True` on CUDA (single fused kernel for
  grad update + momentum + weight decay; ~3-5% on `optimizer.step`).
- `trainer.py`: `GradScaler` is now gated on `amp_dtype == float16`.
  bf16 has fp32's dynamic range and needs no loss scaling — keeping
  scaler enabled for bf16 was adding per-step no-op overhead.
- `configs/predformer_usa_v4.yaml`: `trainer.precision` switched from
  `32` to `bf16` for A100's native bf16 tensor cores
  (~2-3x speedup on forward+backward, frees ~40 GB of activation memory).
- `configs/predformer_usa_v4.yaml`: `data.train.batch_size` and
  `data.val.batch_size` bumped 90 -> 144 (+60%) using the activation
  headroom freed by bf16. Expected memory ~73 GB / 85 GB.
- `configs/predformer_usa_v4.yaml`: `num_workers` 4 -> 8 and
  `prefetch_factor` 2 -> 4 to feed the bigger batch. With 4 workers,
  batch=144 was throughput-capped at 30 samples/sec (vs 41.8 at
  batch=90) — collate dominated CPU. Pairs with `--cpus-per-task=32`
  in `sh_files/train_PredFormer_USA_2gpu_v4.sh`.
- `sh_files/train_PredFormer_USA_2gpu_v4.sh`: `--cpus-per-task` 16 -> 32
  (16 CPU/rank, type_e nodes have 128 CPU available).
- `configs/predformer_globe_v4.yaml` — full-globe production config on
  the 5.625deg (32x64) grid, 2000-2017 train / 2018 val (19 years
  total). Same model/optimiser recipe as `predformer_usa_v4.yaml`; only
  the grid (no cut) and `memmap_path` differ.
- `sh_files/train_PredFormer_Globe_2gpu_v4.sh` — 2-GPU DDP launcher
  with STAGE_MEMMAP staging targeting the globe config.
- `tools/repack_era5.py` — new `--res-suffix` flag (default `1.40625deg`,
  pass `5.625deg` for the coarser grid) so the same packer handles both
  WeatherBench resolutions.
- `sh_files/repack_era5.sh` — accepts new `REPACK_RES_SUFFIX` env
  (forwarded to `--res-suffix`); `--time` raised 1h -> 2h to fit the
  19-year globe pack.
- `train.build_optimizer_and_scheduler`: optional linear LR warmup via
  `training.warmup_ratio` (fraction of total steps) and
  `training.warmup_start_factor` (default `1e-3`). When `warmup_ratio > 0`
  the schedule becomes `SequentialLR(LinearLR -> CosineAnnealingLR)`;
  otherwise the legacy pure-cosine schedule is preserved. Used at
  larger batch sizes where a cold `lr` causes the loss to "jump off"
  the basin found in epoch 1 (job 3991270 anomaly: val_loss 0.5707 -> 0.624).
- `configs/predformer_usa_v4.yaml`: `training.warmup_ratio: 0.05` and
  `training.warmup_start_factor: 0.001` to stabilise batch=144 + bf16.
- `configs/predformer_usa_v4.yaml`: `training.max_epoch` 20 -> 2000 and
  `training.early_stopping_patience` 5 -> 100 for long production
  runs. Job 3991527 showed val_loss still falling at epoch 20
  (0.1968 -> 0.1951) — room to converge further.
- `configs/predformer_usa_v4.yaml`: `model.params.drop_path` 0.0 -> 0.15
  to delay overfit. Job 3991633 (141 epochs) showed val_loss bottom
  at epoch 41 (0.1867) then climbed to 0.2103 while train_loss kept
  falling 0.187 -> 0.111 — classic overfit with zero regularization.
  Stochastic depth at 0.15 across the 24 attention blocks should
  push the val_loss minimum further out.
- `sh_files/train_PredFormer_USA_2gpu_v4.sh`: `#SBATCH --time` 2-18:00:00
  -> 6-00:00:00 to fit the 2000-epoch run (~2.1 min/epoch × 2000 ≈ 70 h).
- `trainer.TrainerConfig`: new `grad_clip_norm: float | None` and
  `skip_non_finite_loss: bool` (default True). When `grad_clip_norm`
  is set, the training loop calls `clip_grad_norm_` after unscaling
  gradients and before `optimizer.step()`. When `skip_non_finite_loss`
  is True, batches whose forward produces NaN/Inf loss are skipped
  (no backward, no step) — defends bf16 runs against a single bad
  sample poisoning the optimiser state for the rest of training.
- `configs/predformer_usa_v4.yaml`: `trainer.grad_clip_norm: 1.0` and
  `trainer.skip_non_finite_loss: true` to harden bf16 training.

### Fixed

- `Data/weatherbench_128_v3.py`: moved the `import json` from inside
  `WeatherBench128.get_mean_std` to the module top (CLAUDE.md §2 forbids
  local imports).
- `Data/weatherbench_128_v4.py`: silenced the `"NumPy array is not
  writable"` `UserWarning` raised by `torch.from_numpy` on read-only
  memmap views. The view is never written to (a fresh contiguous buffer
  is allocated by `torch.stack`), so the warning was noise that flooded
  stderr on every worker startup.
- `configs/predformer_usa_v4.yaml`: set `persistent_workers: false`
  while keeping `pin_memory: true`. PyTorch's pin_memory thread leaked
  sockets when workers were persistent, crashing at epoch ~5 under bf16
  (job 3990855: `RuntimeError: Pin memory thread exited unexpectedly`).
  Disabling pin_memory entirely (job 3990956) dropped GPU util from
  78% to 58%, so we keep pin_memory and instead recreate workers each
  epoch — the ~30-60 sec startup is amortised over a ~3-min epoch.
- `configs/*.yaml`: removed the committed `logging.comet_api_key` field
  (the literal API key was leaked in repo history). The Comet API key is
  now read from `$COMET_API_KEY`, loaded by
  `sh_files/_shell_contract.sh` from a gitignored `.env`. The leaked key
  must be rotated in Comet — git history still contains it.
- `configs/*.yaml`: retargeted `logging.checkpoint_base` from the previous
  user-specific paths (`/home/ebugaev/checkpoints/`,
  `/home/epbugaev/checkpoints/`) to `/home/fa.buzaev/WeatherPredictions/checkpoints/`,
  which is gitignored via the existing top-level `checkpoints/` rule. The
  old paths caused `PermissionError` on `os.makedirs` at the first
  checkpoint save under the current user.
- `utils/experiment.build_experiment` docstring updated to document the
  env-based API-key flow.

### Removed

- `LitModels/` package (nine `LightningModule` wrappers replaced by
  `training_strategies/`).
- `lightning==2.2.0`, `lightning-utilities==0.10.1` and
  `pytorch-lightning==2.6.1` from `requirements.txt`.
- `Models/layers_openstl/` package (replaced by `Models/blocks/`).
- `Models/openstl_utils.py` (was empty, no imports).
- `train/dev/train_PredFormer_v0.py` (broken: depended on a non-existent
  `PredFormer/openstl/` path).

### Migration notes

- Existing Lightning checkpoints can be reused once converted:
  `python -m utils.checkpointing convert <old.ckpt> <new.pt>`.
  The converter strips the `model.` prefix from state-dict keys; the
  inference notebook (`Inference_and_plots.ipynb`) keeps working because it
  already loads via `torch.load` and slices the prefix manually.
