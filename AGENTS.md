# AGENTS.md

Orientation map for AI agents. Coding rules live in `CLAUDE.md` (source of
truth); this file describes what the repo *is*. Keep both in sync with reality.

## What is this project?

Graduate thesis (VKR, HSE LAMBDA lab) on weather prediction with
physics-informed neural networks on WeatherBench (1.40625deg ERA5). Pure
PyTorch + DDP (no Lightning). Exactly three model families are active; each
has (or will get) a physics-informed "PI-" variant powered by a shared
trainable physics core.

| Family | Registry keys | Files |
| --- | --- | --- |
| IAM4VP | `IAM4VP` / `PI-IAM4VP` (aliases of one class; physics via params) | `Models/IAM4VP.py`, `Models/IAM4VP_utils.py` |
| SimVP | `SimVP` | `Models/SimVP.py`, `Models/SimVP_utils.py`, `Models/SimVP_blocks/` |
| PredRNN | `PredRNN`, `PredRNNv2` | `Models/PredRNN.py`, `Models/PredRNN_utils.py` |

The trainable physics core (HybridBlock router, PDE kernels with WENO-5
derivatives, `PhysicsTendencyResidualCorrector`, humidity conversions) is
model-independent and lives in `utils/physics_hybrid.py`. The non-trainable
physics diagnostics library (FD/WENO analysis, PDE residuals) is
`utils/physics.py` — deliberately separate, currently without in-repo callers.

## Repository layout

```text
configs/               YAML experiment configs (model, data, training, trainer, logging)
train.py               Unified training entry point (torchrun + YAML)
trainer.py             Training loop: DDP, AMP, scheduler, checkpointing, early stop
Models/                The three model families + model registry (Models/__init__.py)
training_strategies/   Per-batch train/val StepStrategy classes (registry keys below)
Data/                  WeatherBench datasets: v1, v3, v3_memmap, v4
utils/                 metrics, normalize, registry, distributed, checkpointing,
                       paths, physics.py, physics_hybrid.py
train/                 Legacy standalone per-model launchers (hardcoded params)
tools/                 Data repack / verification CLI scripts
sh_files/              Slurm launchers, remote submit, .env contract
tests/                 test_model_registry.py (registry contract; unittest)
docs/                  Dated experiment notes and audits (historical record —
                       file paths in old notes may predate renames)
```

## How to run

```bash
torchrun --standalone --nproc_per_node=1 train.py --config configs/simvp_usa_v4.yaml
bash sh_files/launch_train.sh simvp_usa            # local helper
sbatch sh_files/train_usa_2gpu.sh pi_iam4vp_usa    # cluster (see README)
```

Copy `.env.example` to `.env` per machine (Comet credentials, cluster paths).
Components are looked up by string keys from `utils/registry.py`; importing
`Models` / `Data` / `training_strategies` populates the registries.

## Registry keys

- Models: `IAM4VP`, `PI-IAM4VP`, `SimVP`, `PredRNN`, `PredRNNv2`
- Datasets (`data.dataset_version`): `v1`, `v3`, `v3_memmap`, `v4`
- Strategies (`training.litmodel`): `mutiout`, `mutiout_f`, `multiout_double`,
  `mutiout_imvp`, `mutiout_imvp_small_world`, `mutiout_predrnn`
  (the `mutiout` spelling is intentional legacy — do not rename)

Typical pairing: SimVP→`mutiout_f`, IAM4VP/PI-IAM4VP→`mutiout_imvp_small_world`
(manual per-timestep backward), PredRNN/v2→`mutiout_predrnn`.

## Data facts

- 69 channels: 4 surface + 5 upper-air vars x 13 pressure levels; channel
  layout surface=0:4, z=4:17, t=17:30, r=30:43, u=43:56, v=56:69.
- Active region: USA crop `cut=[[75, 107], [164, 228]]` → 32x64 pixels.
- v1/v3/v3_memmap return normalized tensors; v4 returns raw memmap rows and
  `trainer.py` applies `WeatherNormalize` on the batch.
- Cluster data roots come from `.env` (`WEATHERBENCH_ROOT`,
  `WEATHERPRED_USA_MEMMAP`, ...); defaults point to the shared HSE cluster.

## Things to be aware of

- **DDP graph:** models with parameters outside the task loss (PI-IAM4VP
  residual mode, PredRNNv2) need `trainer.static_graph: false` +
  `find_unused_parameters: true` in the config; plain SimVP runs
  `static_graph: true`.
- **Checkpoint compatibility:** state_dict keys derive from module attribute
  names — never rename/reorder `self.x = nn.Module(...)` lines in `__init__`
  of live models (also changes RNG init order).
- **Physics geometry:** the hybrid core currently hardcodes the 8x16 latent
  grid (32x64 crop / patch 4); `IAM4VP` raises on other latent sizes. The
  fixedeq Slurm launcher greps `utils/physics_hybrid.py` for the
  `adiabatic_omega` default as a provenance guard.
- **Optimizer parity:** AdamW `betas=(0.9, 0.9)`, `weight_decay` from config
  (default 0.0), cosine schedule (optionally after linear warmup).
- **Tests:** `conda run -n <env> python -m unittest tests.test_model_registry`.
- **Lint:** `ruff check .` and `ruff format .` must stay clean (`ruff.toml`).
