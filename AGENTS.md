# AGENTS.md

## What is this project?

Graduate thesis (VKR, HSE LAMBDA lab) on weather prediction using physics-informed neural networks. It combines physics-based PDE modules with attention-based architectures, trained on the WeatherBench dataset for the South Atlantic Ocean region.

Two novel architectures are proposed:
- **PredFormerGFT** — WeatherGFT physics module + PredFormer (video prediction transformer)
- **PI-IAM4VP** — WeatherGFT physics module + IAM4VP (masked autoregressive vision model)

Baselines: WeatherGFT, PredFormer, SimVP, PredRNN.

## Repository layout

```
configs/                YAML experiment configs (model, data, training params)
train.py                Unified training entry point (reads YAML configs)
Data/                   Dataset classes for WeatherBench (v1, v2, v3 with temporal windows)
Models/                 Neural network architectures (WeatherGFT, PredFormerGFT, PI_IAM4VP, SimVP, PredRNN, etc.)
Models/layers_openstl/  Vision backbone layers from OpenSTL (HorNet, PoolFormer, VAN, etc.)
Models/dev/             Experimental model variants
LitModels/              PyTorch Lightning wrappers (training_step/validation_step for each model type)
train/                  Legacy per-model training scripts (hardcoded params)
train/dev/              Development training scripts
utils/                  Metrics (lat-weighted RMSE/ACC), losses, DDP dataloaders
Research/               Jupyter notebooks for research
example_data/           Sample data and normalization statistics (mean_std.json)
Inference_and_plots.ipynb  Main inference and visualization notebook
```

## Tech stack

- **Python** with **PyTorch** and **PyTorch Lightning**
- **einops**, **timm** for tensor ops and vision model components
- **fairscale** for gradient checkpointing
- **xarray**, **h5netcdf**, **numpy**, **pandas** for climate data I/O
- **Comet ML** for experiment tracking (primary), **wandb** (secondary)
- DDP (Distributed Data Parallel) for multi-GPU/multi-node training

There is no requirements.txt or setup.py. Dependencies must be installed manually.

## How to run training

Use the unified `train.py` with a YAML config:

```bash
# Single GPU
python train.py --config configs/simvp.yaml

# Multi-GPU (8 GPUs, 1 node)
python train.py --config configs/predformergft.yaml --gpus_per_node 8 --nodes 1
```

### Creating a new experiment

Copy an existing config and modify what you need:

```bash
cp configs/simvp.yaml configs/simvp_usa.yaml
# Edit the new file: change cut, experiment name, etc.
python train.py --config configs/simvp_usa.yaml
```

### Config file structure

```yaml
experiment:
  name: SimVP-USA                 # experiment name (used for checkpoints and Comet)

model:
  type: SimVP                     # SimVP | WeatherGFT | PredFormerGFT | PI-IAM4VP | PredFormer | PredRNN | WeatherGFTSingle
  params:                         # passed directly to the model constructor
    in_shape: [12, 69, 32, 64]

data:
  dataset_version: v3             # v1 (single-step) | v2 | v3 (temporal windows + cut support)
  cut: [[75, 107], [164, 228]]    # spatial crop [lat_start:lat_end, lon_start:lon_end]
  start_time_x: 0                 # v3 only: input temporal window
  end_time_x: 11
  start_time_y: 12                # v3 only: target temporal window
  end_time_y: 23
  train:                          # split-specific overrides
    start_time: "2000-01-01 00:00:00"
    end_time: "2003-12-30 00:00:00"
    batch_size: 32
  val:
    start_time: "2004-01-01 00:00:00"
    end_time: "2004-12-30 00:00:00"
    batch_size: 32

training:
  litmodel: mutiout_f             # which LitModel wrapper to use
  lr: 5e-4
  max_epoch: 20
  loss_type: MAE
  extra_kwargs:                   # additional kwargs passed to LitModel
    muti_out_nums: 6

logging:
  checkpoint_base: /home/ebugaev/checkpoints/
  comet_api_key: ""
```

### Available configs

| Config | Model | Dataset | Region |
|--------|-------|---------|--------|
| `simvp.yaml` | SimVP | v3 | Original |
| `simvp_usa.yaml` | SimVP | v3 | USA |
| `weathergft.yaml` | WeatherGFT | v1 | Original |
| `weathergft_single.yaml` | WeatherGFTSingle | v1 | Original |
| `predformergft.yaml` | PredFormerGFT | v3 | Original |
| `pi_iam4vp.yaml` | PI-IAM4VP | v3 | Original |
| `predformer.yaml` | PredFormer | v1 | Full globe |
| `predrnn.yaml` | PredRNN | v3 | Original |

### Model-to-litmodel mapping

Each model type has a specific LitModel wrapper it should use:

| Model type | litmodel | Notes |
|-----------|----------|-------|
| SimVP | `mutiout_f` | |
| WeatherGFT | `multiout_double` | |
| WeatherGFTSingle | `mutiout` | |
| PredFormerGFT | `mutiout_f` | |
| PredFormer | `mutiout_f` | |
| PI-IAM4VP | `mutiout_imvp_small_world` | Manual optimization, iterative prediction |
| PredRNN | `mutiout_predrnn` | Supports `subset_step` in train data config |

### Legacy training scripts

The original per-model scripts in `train/` are still available but have hardcoded parameters:

```bash
python train/train_WeatherGFT.py --gpus_per_node 1 --nodes 1
```

## Data

- **Source:** WeatherBench at 1.40625deg resolution (128x256 grid)
- **Format:** `.npy` files per timestep, `.nc` NetCDF for raw data
- **Channels:** 69 weather variables (4 surface + 5 vars x 13 pressure levels)
- **Spatial crop:** South Atlantic Ocean `cut=[[36,68],[125,189]]` (32x64 pixels)
- **Normalization:** Z-score with mean/std from training set
- **Train period:** 2000-01-01 to 2003-12-25
- **Val period:** 2004-01-01 to 2004-12-25

Data paths are hardcoded to the HSE HPC cluster:
- `/home/fratnikov/weather_bench/` — raw data and npy files
- `/home/epbugaev/weather_bench/` — mean_std normalization stats

## Key patterns and conventions

### Model architecture
All models take tensors of shape `[B, T, C, H, W]` or `[B, C, H, W]` and output predictions of the same spatial shape. The WeatherGFT physics module uses 5th-order WENO finite difference schemes for spatial derivatives and learnable PDE kernels with configurable time steps (`block_dt=300s`).

### Training wrappers (LitModels/)
Different models need different `LitModel` wrappers due to varying input/output formats:
- `basemodel.py` — standard single-step models
- `mutiout*.py` — multi-step output models (WeatherGFT outputs 6 steps per forward pass)
- `mutiout_imvp*.py` — IAM4VP iterative autoregressive training (manual optimization)
- `mutiout_predrnn.py` — PredRNN-specific handling

### Optimizer config (all models)
- AdamW with `weight_decay=0.0, betas=(0.9, 0.9)`
- CosineAnnealingLR scheduler (step-level)
- MAE loss (default), MSE available
- Early stopping: patience=5 on `val_loss`

### Metrics (`utils/metrics.py`)
All metrics are latitude-weighted to account for Earth's spherical geometry:
- **WRMSE** — weighted RMSE (primary evaluation metric)
- **WACC** — weighted anomaly correlation coefficient
- **Bias**, **Activity** — additional diagnostic metrics
- **RQE** — relative quantile error for extreme events

### Dataset versioning
- `weatherbench_128.py` (v1) — single-step input/output
- `weatherbench_128_v2.py` (v2) — adds multi-step targets
- `weatherbench_128_v3.py` (v3) — configurable temporal windows (`start_time_x/y`, `end_time_x/y`); used by PredFormerGFT and PI-IAM4VP
- Data loading uses a preload cache (`self.preload`) that reads chunks of hours from NetCDF and caches them in memory

### Imports
Training scripts use `sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` to add the project root. Some files import as `from WeatherPredictions.X import Y` (absolute) and others use relative imports — be consistent with whichever style the file already uses.

### Toggle physics
For IAM4VP, the `use_physics` constructor argument switches between plain IAM4VP and PI-IAM4VP.

## Things to be aware of when modifying

- **Hardcoded paths:** Data paths, checkpoint save paths, and `log_code()` file paths all point to HSE HPC locations. Update these when running elsewhere.
- **COMET_API_KEY** is set to empty string in training scripts — must be configured for experiment tracking.
- **No tests:** There is no test suite. Validate changes by running training and checking metrics.
- **Typo in naming:** Several files use `mutiout` (not `multiout`) — this is intentional legacy naming, don't rename without updating all references.
- **`train/dev/`** contains experimental work — treat as unstable.
- **Static DDP graph:** All training scripts use `DDPStrategy(static_graph=True)` — model forward pass graph must not change between iterations.
- **Precision:** All training runs at `precision=32` (FP32). Mixed precision (`16-mixed`) is commented out but available.
