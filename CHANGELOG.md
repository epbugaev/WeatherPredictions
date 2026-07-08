# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **PI-IAM4VP inline-physics equation-variant and geometry flags** on
  `PDE_kernel` / `HybridBlock` / `IAM4VP`: `t_t_formulation`,
  `use_universal_R`, `coriolis_formulation`, `lat_start_deg` / `dlat_deg` /
  `dlon_deg`, `tendency_limiter` / `tendency_caps`,
  `physics_tendency_on_latent`, `physics_prior_detach` and
  `physics_horizon_seconds`. Defaults select the fixed physics; the legacy
  behaviour stays reachable by flag (see the `legacy_hybrid` arm).
- **Hybrid drift diagnostics** logged by the residual corrector:
  `physics_router_weight_abs` and `physics_hybrid_bn_gamma_drift`.
- **`diabatic_apply_to` flag** on `IAM4VP` (`all_upper_air` default /
  `t_and_q`): restricts the Q_theta head output to the T and humidity blocks
  per E9'/exp 10 (the physics deficit is diabatic heating/moistening); the A2
  config uses `t_and_q`. Per-block diagnostics `physics_diabatic_rms_{z,t,r,u,v}`.
- **Edge-contamination diagnostic** `physics_residual_delta_edge_interior_ratio`
  (RMS of `delta_phys` on the one-latent-cell border ring vs interior) —
  observability for boundary artefacts of the 8x16 WENO stencil on the crop.
- **Legacy-grid warning**: constructing the residual corrector with the
  default fake-global latent geometry now emits a `UserWarning` (expected only
  for the `legacy_hybrid` regression arm).
- Sanity checks for the mass-consistent column-divergence invariant and the
  diabatic channel mask in `Models/dev/sanity_hybridblock_fixes.py`.
- **fixedeq sbatch guard** now checks `Models/PredFormerGFT_HybridBlock.py`
  (the file PI-IAM4VP actually uses) instead of `Models/WeatherGFT.py`.
- **DDP guard for residual warmup** (`IterativeManualStep._set_residual_warmup`):
  raises `ValueError` when `freeze_iam4vp_for_residual_warmup` is requested with a
  positive warmup under `DistributedDataParallel`, because toggling `requires_grad`
  after DDP wrapping breaks gradient bucketing (buckets are built once at
  construction time) — audit F11.
- `docs/ideas/01`–`05` and the PI-IAM4VP integration audit report
  (`docs/PI_IAM4VP_integration_audit_ru.md`).
- **Per-variable val RMSE для всех 69 каналов.** `BASEMODEL_INDEX_MAP` /
  `MULTIOUT_INDEX_MAP` (`training_strategies/_index_maps.py`) расширены с
  13-канального суррогата до полного 69-канального покрытия (surface +
  z/t/r/u/v × 13 уровней), построенного алгоритмически из канонического
  v4-layout. Цель Парето-улучшения по всем каналам теперь наблюдаема в
  Comet (namespace без префикса, наложится на baseline там, где индекс не
  менялся, напр. `z500`). Затрагивает все `SimpleStep`-/multiout-стратегии
  (значения существующих корректных ключей не изменились).
- **`Models/dev/sanity_train_probe.py`** — single-card training-пробник:
  прогоняет реальный per-batch путь (PredFormer + `WeatherNormalize` +
  `SimpleStep` + AdamW + `Metrics.WRMSE`) N optimiser-шагов на
  синтетическом батче, печатает траекторию loss, train/val gap и per-69
  RMSE; падает на NaN/Inf или не убывающем loss. Дешёвый go/no-go перед
  6-суточным DDP-раном (loss-scale при MAE→MSE, регрессы pipeline).
- **`training.weight_decay`** — поле конфига, пробрасывается в
  `AdamW(weight_decay=...)` (`train.py`). Default `0.0` сохраняет
  numerical parity; включается как L2-регуляризатор против overfit
  (job 3991633).

- **Pure-physics stabilization ablation (E1–E5).** Forward-ablation of
  `utils.physics.PurePDEKernel` to make pure-physics rollout numerically
  stable (post-БАГ-1 it blows up at h=1 on real ERA5). Each idea is an
  opt-in flag with a backward-compatible default (kernel behaviour
  unchanged unless enabled), a CPU sanity gate in `Models/dev/`, and a
  dedicated Comet experiment:
  - `time_scheme="ssp_rk3"` — Shu–Osher 3-stage SSP Runge–Kutta
    (E1; resolves docs/physics.md open question #5 «WENO-5 + Forward
    Euler → RK3-TVD»). Default stays `euler`.
  - `hyperdiffusion`, `hyperdiff_tau_hours` — scale-selective ∇⁴
    hyperdiffusion via a 3-point second difference (NOT `d_x∘d_x`,
    which has a zero Nyquist response); K4 calibrated to e-fold the
    2Δx mode in `tau` (E2; principled replacement for `scale_diff`,
    docs/physics.md open question #4).
  - `polar_filter`, `polar_filter_lat_deg` — zonal Fourier filter
    truncating `m > (W/2)·cosφ/cosφ₀` poleward of φ₀ (E3; removes
    polar-convergence CFL of the regular lat-lon grid). Adds
    `import torch.fft`.
  - `advection_form="flux"` — conservative divergence form
    `−∇·(VX)` (E5; conserves ∑X to round-off on the periodic grid;
    `advective` default unchanged).
  - `w_diagnostic="mass_consistent"` — subtract the p-weighted column-
    mean divergence so ∫div·dp≈0 per column (E5).
  - `tools.check_physics_common.balance_initial_state` + `prepare_hook`
    parameter of `run_72h_rollout` — IC balancing (E4): `geostrophic`
    (wind from mass) and forward-only stabilized `dfi`
    (Lynch–Huang-style digital filter on `kernel.step`), `--balance-ic`
    default `none`. LBYL fallback to the raw IC on non-finite output.
  - `time_scheme="semi_implicit"` — Crank–Nicolson на линейной быстрой
    подсистеме (PGF↔div↔hydrostatic), сведённой к Гельмгольцу для z;
    решается спектрально с **масс-взвешенно-симметризованным** модальным
    разложением (cumsum-`integral_z`-вертикаль не самосопряжена → нужна
    Δp-симметризация для well-conditioned модального базиса) (E6). Default
    остаётся `euler`.
- `tools/check_physics_new_kernel.py`: CLI flags `--time-scheme ssp_rk3|
  semi_implicit`, `--hyperdiffusion[/-tau-hours]`, `--polar-filter[/-lat-deg]`,
  `--balance-ic`, `--advection-form`, `--w-diagnostic`, `--abl-label`;
  `method_name`/tags carry the active stabilizers so each ablation step
  is a distinct Comet experiment.
- `sh_files/check_physics_ablation.sh` — cumulative E0..E6 + fixC anchor
  runner (cpu-e-quick); `Models/dev/smoke_ablation.py` local synthetic
  smoke gate; `Models/dev/sanity_{ssp_rk3,hyperdiffusion,polar_filter,
  ic_balance,conservation,semi_implicit}.py` per-idea invariants;
  `Models/dev/fetch_ablation_summary.py` Comet→table; `experiments/`
  ablation log (`README.md` + `E0..E6_*.md`).
- **Ablation outcome (job 3998911, real ERA5, 48h, 12 IC):** ни одна из
  6 идей (E1–E6) ни их полный стек не предотвращают и не задерживают
  взрыв pure-physics — `frac_ic_blown_up` 0→1.0 на h=1 тождественно для
  E0..E5; E6-машинерия корректна (Helmholtz roundtrip 9e-7) но scoped SI
  не безусловно устойчив (λ-зависимая метрика + cumsum-вертикаль). Только
  `fixC` (`scale_diff`+`.detach()`) численно конечен, но физически
  расходится (acc/z 0.44→0.24). Подтверждает архитектуру WeatherGFT
  (scale_diff + NN-коррекция обязательны). Детали — `experiments/`.
- `ruff.toml`: expanded the lint selects to `F, E, W, B, I, UP, SIM,
  T100, T201` and pinned `target-version = "py310"`. New rule families
  enforce CLAUDE.md requirements: `UP` for PEP 604 (`|`, `list[...]`),
  `SIM` for ≤3-level nesting / control-flow simplifications, `T201`
  for the no-`print()` rule (legacy entry points keep their
  `per-file-ignores`). Models/dev/* and tools/* get tolerated rule
  baselines so the one-shot lint pass doesn't fight archival code.
- `.pre-commit-config.yaml`: two hooks pinned to ruff 0.15.9
  (`ruff --fix` + `ruff-format`). Install with
  `uv tool install pre-commit && pre-commit install`.
- `utils/regions.USA_CROP`: `Final[list[list[int]]]` constant for the
  USA index window (was duplicated across five train scripts).
- `training_strategies.base.StepStrategy._build_val_metrics`: protected
  helper that bundles `val_loss` + per-variable RMSE for the first /
  last predicted timestep. All five strategies now call it instead of
  rolling their own copy.
- `Models.WeatherGFT.PDE_kernel._build_grid_buffers` and the four
  finite-difference / integral methods (`integral_z`, `d_x`, `d_y`,
  `d_z`) replace the module-level globals and free functions of the
  same names. Same code in `Models/WeatherGFTSingle.py`.
- `IterativeManualStep._iterate_timesteps`: shared per-timestep
  prediction loop used by both `train_step` (with manual backward) and
  `val_step` (forward only).
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

- PI-IAM4VP `block_dt` is now derived from `physics_horizon_seconds`
  (`3600 s / (depth * steps)` = 400 s) instead of the hardcoded 1200 s
  (a nominal 3 h per 1 h autoregressive step) — audit F3b.
- `configs/pi_iam4vp_residual_diabatic_usa_v4.yaml`: `diabatic_lambda_l1`
  1e-5 → 1e-4 (equal to `physics_residual_lambda_l1`; removes the 10x
  incentive for Q_θ to absorb non-diabatic correction mass) — audit F7b.
- `configs/pi_iam4vp_residual_*.yaml`: residual experiment names suffixed
  `-fixv2` so post-fix Comet runs never mix with the pre-fix (poisoned) runs.
- PI-IAM4VP construction-time channel-layout prints converted to warnings.
- Унифицирован namespace per-variable RMSE-метрик: multiout-стратегии
  (`autoregressive`, `predrnn`, `iterative_manual`, `timestep_select`)
  больше не добавляют legacy-префикс `"f "` к ключам — теперь все модели
  (включая SimVP/PredFormer-семейство на `SimpleStep`) логируют
  `RMSE_<var>_first/last` под одним именем. Без этого на общих панелях
  Comet модели не накладывались (SimVP писал `RMSE_z500_last`, остальные
  — `f RMSE_z500_last`). Старые раны multiout-моделей сохраняют префикс
  `"f "` в истории; новые — единый ключ.
- `tools/check_physics_common.py`: surface-метрики переименованы
  `weighted_rmse/surface/{t2m,u10,v10,tp}` → `persistence/surface/{...}`.
  Физика их не прогнозирует (passthrough IC) → метрика тождественна у
  ВСЕХ методов = persistence-ошибка данных; новый префикс явно отделяет
  baseline-пол от метод-различающих `weighted_rmse/{prog}/{lvl}hPa`
  (иначе совпадающие кривые читались как «баг-дубликат»).
- `utils.physics.Grid` Coriolis variants теперь все используют единое
  `2Ω·sin(...)` приближение: `f_constant = 2Ω·sin(45°) ≈ 1.03e-4`,
  `f_beta_plane = 2Ω·sin(45°) + β·R·φ`, `f_spherical = 2Ω·sin(φ)`. Был
  «зоопарк» с f=Ω (без множителя 2 и без sin) в beta_plane и константном
  варианте. Старая физика байт-в-байт сохранена в `utils/old_physics.py`.
- `utils.physics.PurePDEKernel` гидростатика: дефолт сменился с
  универсальной `R=8.314 Дж/(моль·К)` на `R_d=287 Дж/(кг·К)` — это
  физически корректная константа для воздуха per-mass. Флаг
  `use_R_d_in_hydrostatic` переименован в `use_universal_R` (default
  `False`); установка `True` восстанавливает старое поведение для
  регрессии чекпоинтов. CLI-флаг `--use-R-d` → `--use-universal-R` в
  `tools/check_physics_new_kernel.py` и `tools/physics_baseline.py`.
- `tools/physics_baseline.py` теперь имеет явный CLI-флаг
  `--memmap-is-normalized` (default False, т.к. v4 raw — текущий стандарт).
  Раньше денормализация `x*std + mean` срабатывала только по факту передачи
  `--mean-std-path`, что было неявным и портило данные при ошибочной комбинации
  «v4 raw + переданный mean_std» (вело к ×std + mean поверх raw, физическая
  чушь). Теперь: без флага и без mean_std_path memmap читается как есть
  (v4 raw); `--memmap-is-normalized` без `--mean-std-path` → `ValueError` на
  старте.
- `utils.physics.GridConfig` параметризован полем `lat_range_deg: (low, high)`
  (дефолт `(-90, 90)`). Раньше широты строились хардкодом от -90 до 90 без
  возможности задать региональный кроп, что давало ошибочные `pixel_x` /
  `pixel_y` / `f_spherical` для не-глобальных данных. Аналогично в
  `tools.check_physics_common.GeometryCPU`. Все check-скрипты и
  `tools/physics_baseline.py` получили CLI-флаг `--lat-range-deg LOW HIGH`.
  Поле `GridConfig.lat_scheme` (с веткой `'arange'`) удалено — на равномерной
  сетке оно численно идентично `linspace`, ветка только вводила в заблуждение.
- `utils.physics.PurePDEKernel.get_t_t` теперь по умолчанию использует
  физически корректную адиабатику `dT/dt|_adia = R_d·T·ω/(c_p·p)` (где
  `ω = 100·w`, w из `get_w` в hPa/s). Старая формула `(Q − z_z·w)/c_p` с
  `Q = −L·z_z·w` — это legacy_paper-вариант через параметр
  `t_t_formulation='legacy_paper'`. Старый дефолт давал 7-порядковый
  overflow за один substep на ERA5. Те же изменения в
  `tools/check_physics_weathergft.py`, `check_physics_predformergft.py`,
  `check_physics_weathergft_3.py` (используют новый helper
  `tools.check_physics_common.adiabatic_temperature_tendency`).
  `tools/check_physics_fix.py` сохранён как тест трёх вариантов (включая
  legacy broken Q как контроль).
- Magnus saturation теперь работает в SI везде: `tools.check_physics_common.magnus_qs`
  и `relhum_to_specific` принимают давление в Па (раньше в гПа), формула
  `e_s = 611.2·exp(...)`. Согласовано с `utils.physics.PurePDEKernel._get_qs`
  и `Grid.pressure`. Поле `GeometryCPU.pressure_hpa_t` удалено (использовалось
  только во внутреннем расчёте); все вызовы используют `pressure_pa_t`.
- `utils.physics.FiniteDifference` и `WENO5` теперь принимают три значения
  boundary: `'periodic' | 'reflect' | 'replicate'`. Старое 'periodic' для оси
  H (lat) и оси P (pressure) делало cat-pad первых/последних строк — это
  было *replicate*, а не периодика; имя теперь соответствует поведению.
  По умолчанию `boundary_x='periodic'` (lon циклична), `boundary_y='replicate'`,
  `boundary_z='replicate'` (`'periodic'` по оси давления запрещён в __init__).
  `PurePDEKernel` параметр `boundary_horiz` распался на отдельные `boundary_x`
  и `boundary_y`; CLI флаги `--boundary-h` → `--boundary-x`/`--boundary-y` в
  `tools/check_physics_new_kernel.py` и `tools/physics_baseline.py`.
- `tools.check_physics_common.coriolis_constant` default value сменился с
  `7.29e-5` (Ω) на `2Ω·sin(45°) ≈ 1.03e-4`; `coriolis_beta_plane` дефолт
  `f0` так же. `coriolis_spherical` теперь использует `omega=7.2921e-5`
  (явный SI-литерал, не округлённый 7.29e-5). Скрипты
  `tools/check_physics_weathergft.py` и `tools/check_physics_predformergft.py`
  пробрасывают `7.29e-5` явно через `--coriolis-value`/`--f0` — они
  гоняют регрессию старой физики через `utils.old_physics`.
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
- `trainer._to_device`: reverted to the legacy ``.to(device,
  non_blocking=True)`` after the in-thread ``pin_memory()`` experiment
  (commit 16fcb6c) regressed GPU util 73% → 52% on job 3993815. The
  synchronous ``tensor.pin_memory()`` allocated a fresh ~50 MB pinned
  buffer per tensor per batch and blocked the trainer for ~30 ms/step
  (≈30% of the bf16 step time). DataLoader's own pin_memory thread is
  cheaper despite the cross-epoch socket leak.
- `configs/predformer_usa_v4.yaml`: keep `pin_memory: true` and
  `persistent_workers: false`. The ~30-60 s/epoch worker-restart cost
  is the lesser evil compared to either the +30 ms/step in-thread pin
  regression or the pin_memory thread leak.
- `trainer.TrainerConfig`: new `val_every_n_epochs: int = 1`. When set
  to N>1 the trainer skips validation + checkpoint + early-stopping
  on (N-1)/N epochs, freeing those gaps. Last epoch always validates.
- `configs/predformer_usa_v4.yaml`: `trainer.val_every_n_epochs: 3` and
  `training.early_stopping_patience: 100 -> 34` (34*3 ≈ same 100-epoch
  plateau budget).
- `trainer.Trainer`: rank-0 owns a 1-thread ThreadPoolExecutor for
  checkpoint writes. ``_write_checkpoints`` now snapshots state_dicts
  on CPU synchronously (~1-3 s) then submits the actual ``torch.save``
  plus atomic rename to the background pool, so the DDP barrier after
  this method no longer waits on disk I/O (was ~5-15 s/epoch). The
  next epoch's call awaits any in-flight write before clobbering paths.
  ``atexit`` drains the pool on shutdown.
- `trainer._log_train_metrics`: logs rank-0 local train_loss directly,
  no cross-rank ``reduce_dict``. Validation metric reduction (val_loss
  feeds early-stopping) is unchanged.
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
- `Models/WeatherGFT.py` and `Models/WeatherGFTSingle.py`: moved the
  module-level grid constants (`M_z`, `pixel_x`, `pixel_y`, `pixel_z`,
  `pressure`) into `PDE_kernel.__init__` via `register_buffer`. The four
  free FD/integral helpers (`integral_z`, `d_x`, `d_y`, `d_z`) became
  methods on `PDE_kernel` and read the buffers via `self.*`. Effect:
  one `.to(device)` per init instead of one per `forward`, and the
  module follows DDP replication and `nn.Module.to` semantics. The
  latent-grid shape (`(8, 16)` for the legacy model, `(32, 64)` for the
  Single variant) is now an `__init__` parameter `latents_size` rather
  than a module-level global.

### Fixed

- **PI-IAM4VP inline-physics / integration fixes (branch `fix_inline_v2`).**
  - Re-applied the 4 inline-equation fixes silently reverted by `5ae159c`
    (Coriolis `f0`, adiabatic `t_t`, `R_d` hydrostatics, `avoid_inf`) — audit F1.
  - Crop-aware `HybridBlock` geometry: the kernel used a fake global -70..70
    latent grid on the USA crop (negative Coriolis rows, dx/dy off 3-5x) — F2.
  - Tendency limiter `physical_clip` restores `block_dt` semantics and removes
    the cross-batch min/max leakage of `scale_diff` — F3.
  - Physics tendency is now built on the latent grid (no resampling high-pass
    contamination of `delta_phys`) — F4.
  - Validation now logs the real physics aux loss (was always 0 in val) — F10.
  - Router weight naming (`weight_physics` / `weight_ai`) matched to the actual
    physics / AI paths — F9.
- **Неверные legacy-индексы `BASEMODEL_INDEX_MAP`** для v4-layout
  (`training_strategies/_index_maps.py`, используется `SimpleStep` —
  PredFormer/SimVP). Прежние hand-written индексы ссылались на старый
  LitModels-layout: напр. `t500`→канал 20 (= t@200 hPa) вместо 24,
  `u500`→32 (= r@150 hPa) вместо 50, `t50`→23 вместо 17, `t1000`→19
  вместо 29; корректным был только `z500` (11). Per-variable RMSE в
  Comet для PredFormer-USA-v4 (вкл. job 3991633) под этими именами
  измерял **не те физические каналы**. Исправлено алгоритмическим
  построением из канонического 69-layout (см. Added).

- **Ablation stabilizer bugs B1–B4** (`utils/physics.py`,
  `tools/check_physics_*`). Аудит показал, что прежний результат «E0–E6
  тождественно взрываются@h1» (job 3998911) был во многом BUG-артефактом,
  а не чистой физикой:
  - **B1** — `hyperdiffusion` был ЯВНЫМ `−K4·∇⁴` с единым K4 на самой
    крупной ячейке; т.к. ∇⁴-собств.знач. ∝ 1/Δx⁴, на полюсе явный
    множитель ≈ −172 → гипердиффузия УСИЛИВАЛА 2Δx ×172/шаг (E2–E5
    активно хуже E0). Заменено безусловно-устойчивым НЕЯВНЫМ зональным
    спектральным фильтром `1/(1+(dt/τ)·((2−2cos kx)/4)²)` в `_finalize`;
    удалены `_biharmonic`/`_laplacian`/`hyperdiff_k4`.
  - **B2** — semi-implicit `lam.clamp_min(0)` зануляло БЫСТРЕЙШИЕ
    гравитационные моды (внешняя c≈309 м/с → λ<0 из-за знака
    cumsum-вертикали) → `_si_solve` ≈ identity (E6 был no-op).
    Заменено `lam.abs()` (все моды); implicit-strength стал O(1)=2.4.
  - **B4** — CN ∇²-несогласованность (спектральный const-Δx implicit vs
    физический `_laplacian` в RHS, 52%); + const-Δx «не видел» полюсные
    стиффовые моды. Заменено ЗОНАЛЬНО-неявной схемой с λ-зависимым Δx
    (rfft по долготе, символ ∂²ₓ зависит от широты), один и тот же ∂²ₓ
    (`_si_xlap`/`_si_solve`) по обе стороны CN.
  - **B3** — DFI span=6ч → 145 forward-шагов на нестабильном ядре →
    всегда NaN → silent fallback (E4≡E3). Дефолт span 6ч→1ч
    (`balance_initial_state`, `--balance-span-hours`, ABL[E4–E6]).
  Проверено КОРРЕКТНЫМ (не баг): БАГ-1 фикс, единицы Pa/hPa, проводка
  флагов, знаки flux/polar/geostrophic, GeometryCPU↔Grid. Все 6
  `Models/dev/sanity_*.py` переписаны/усилены (ловят B1: контракция ≤1
  при любых τ/dt; B2: implicit O(1) не identity) и PASS. Локальный
  re-run на гладком сбалансированном synthetic
  (`Models/dev/{make_synthetic_era5_smooth,rerun_ablation_local}.py`)
  показал: после фикса методы РАЗЛИЧАЮТСЯ (blow@substep 3..26, не
  идентично). Кластерный re-run исправленного кода — pending.
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
- `utils.metrics.weighted_latitude_weighting_factor_torch`: dead helper
  (no callers across the codebase).
- `utils.metrics.type_weighted_{bias,activity}_torch{,_channels}`: removed
  the `metric_type="all"` parameter (was unused, only kept "for
  compatibility"). Internal callers in `Metrics.Bias`/`Metrics.Activity`
  updated accordingly.

### Migration notes

- Existing Lightning checkpoints can be reused once converted:
  `python -m utils.checkpointing convert <old.ckpt> <new.pt>`.
  The converter strips the `model.` prefix from state-dict keys; the
  inference notebook (`Inference_and_plots.ipynb`) keeps working because it
  already loads via `torch.load` and slices the prefix manually.
