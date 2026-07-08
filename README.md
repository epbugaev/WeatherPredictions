# WeatherPredictions

Code for HSE VKR weather-forecasting experiments on WeatherBench. The repository
is now trimmed to the active model families used for training:

- `PI-IAM4VP` / IAM4VP, implemented in `Models/PI_IAM4VP.py`.
- `SimVP`, implemented in `Models/SimVP.py`.
- `PredRNN` and `PredRNNv2`, implemented in `Models/PredRNN.py`.

`Models/PredFormerGFT_HybridBlock.py` is kept as an internal PI-IAM4VP physics
dependency. It is not registered as a standalone training architecture.

## Repository Layout

- `configs/` — YAML configs for active IAM4VP, SimVP, and PredRNN runs.
- `train.py` — unified pure-PyTorch training entry point.
- `trainer.py` — training loop, DDP, checkpointing, validation and Comet logs.
- `Data/` — WeatherBench datasets (`v1`, `v3`, `v3_memmap`, `v4`).
- `Models/` — active architectures and the model registry.
- `training_strategies/` — training-step strategies for the active models.
- `utils/` — metrics, normalization, registries, distributed helpers and
  checkpoint utilities.
- `train/` — small per-model wrappers for IAM4VP, SimVP and PredRNN.
- `sh_files/` — Slurm launchers and remote-submit helpers.
- `docs/` — retained PI-IAM4VP notes and current idea sketches.

## Running Training

Use a config stem from `configs/`:

```bash
bash sh_files/launch_train.sh simvp_usa
```

On the cluster, submit the generic USA launcher with an active config stem:

```bash
sbatch sh_files/train_usa_2gpu.sh simvp_usa
sbatch sh_files/train_usa_2gpu.sh pi_iam4vp_usa
sbatch sh_files/train_usa_2gpu.sh predrnn_usa
```

For v4 memmap runs, use the dedicated scripts:

```bash
sbatch sh_files/train_simvp_usa_v4_2gpu.sh
sbatch sh_files/train_pi_iam4vp_usa_v4_2gpu.sh
sbatch sh_files/train_predrnn_usa_v4_2gpu.sh
```

The remote helper pushes the current branch to cHARISMa and submits a job:

```bash
bash sh_files/remote_submit.sh sh_files/train_usa_2gpu.sh simvp_usa
```

## Config Basics

Each YAML config has these blocks:

```yaml
experiment:
  name: SimVP-USA-v4

model:
  type: SimVP
  params:
    in_shape: [12, 69, 32, 64]

data:
  dataset_version: v4
  cut: [[75, 107], [164, 228]]
  sample_stride: 6
  frame_interval: 1
  start_time_x: 0
  end_time_x: 11
  start_time_y: 12
  end_time_y: 23
  train:
    start_time: "2000-01-01 00:00:00"
    end_time: "2003-12-30 00:00:00"
    batch_size: 8
  val:
    start_time: "2004-01-01 00:00:00"
    end_time: "2004-12-30 00:00:00"
    batch_size: 8

training:
  litmodel: mutiout_f
  lr: 1e-4
  max_epoch: 2000

logging:
  checkpoint_base: "${WEATHERPRED_CHECKPOINT_BASE:-./checkpoints}"
  comet_project: "${COMET_PROJECT_NAME:-WeatherPredictions}"
```

Important details:

- Before running on a cluster account, copy `.env.example` to `.env` and set
  `COMET_API_KEY`, `COMET_WORKSPACE`, `REPO_ROOT`, `CONDA_ENV_BIN`,
  `WEATHERPRED_CHECKPOINT_BASE`, and memmap paths for v4 runs.
- USA configs use `data.cut: [[75, 107], [164, 228]]`; the original South
  Atlantic crop configs were removed with the old experiment archive.
- For `v3`, `v3_memmap`, and `v4`, temporal window indices are frame offsets.
  With `frame_interval: 1`, `start_time_x: 0`, `end_time_x: 11` means hourly
  frames `t..t+11h`, and `start_time_y: 12`, `end_time_y: 23` means hourly
  targets `t+12h..t+23h`.
- `v3_memmap` and `v4` require an explicit packed memmap path. Pass
  `MEMMAP_PATH_OVERRIDE`, or set `WEATHERPRED_USA_MEMMAP` /
  `WEATHERPRED_GLOBE_MEMMAP` in `.env`.
- `v1`, `v3` and `v3_memmap` datasets return normalized tensors. `v4` returns
  raw memmap tensors; `trainer.py` applies `WeatherNormalize` on the batch.

## Model and Strategy Keys

| Model key | Typical strategy (`training.litmodel`) |
| --- | --- |
| `SimVP` | `mutiout_f` |
| `PI-IAM4VP` | `mutiout_imvp_small_world` |
| `PredRNN` | `mutiout_predrnn` |
| `PredRNNv2` | `mutiout_predrnn` |

The `mutiout` spelling is legacy and intentional.

## PI-IAM4VP Residual Corrector

`PI-IAM4VP` supports three related modes through `model.params`:

- Plain IAM4VP: `use_physics: false`, `use_physics_residual_corrector: false`.
- Legacy PI-IAM4VP: `use_physics: true`, `use_physics_residual_corrector: false`.
- Physics-tendency residual corrector:
  `use_physics: false`, `use_physics_residual_corrector: true`.

The residual-corrector mode treats the inherited HybridBlock as a tendency
feature generator rather than a trusted forecast. The IAM4VP decoder first
predicts `y_nn`; then a small zero-initialized convolutional head receives
`[y_nn, prev_state, y_nn - prev_state, delta_phys]` and predicts an output-space
correction. By default only upper-air channels are corrected, while surface
channels remain equal to `y_nn`.

Useful retained configs:

- `configs/iam4vp_usa_v4.yaml`
- `configs/pi_iam4vp_usa_v4.yaml`
- `configs/pi_iam4vp_residual_usa_v4.yaml`
- `configs/pi_iam4vp_residual_no_physics_usa_v4.yaml`
- `configs/pi_iam4vp_residual_shuffled_usa_v4.yaml`
- `configs/pi_iam4vp_residual_legacy_hybrid_usa_v4.yaml`
- `configs/pi_iam4vp_residual_usa_v4_fixedeq.yaml`

## Data

WeatherBench data paths in committed configs default to the shared HSE cluster
root via `WEATHERBENCH_ROOT`. Override that in `.env` for another account or
storage layout. The v4 production configs expect packed memmaps, usually staged
to node-local `/tmp` by the matching Slurm scripts before training starts.

The memmap loader validates channel count, variable order, crop metadata and
time coverage on startup so a mismatched file fails early instead of training on
the wrong window.
