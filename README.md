# WeatherPredictions

Code for the HSE VKR weather-forecasting experiments on WeatherBench. The
project trains neural weather models on regional and global WeatherBench
windows, including physics-informed variants based on WeatherGFT.

Main model families:

- `PredFormerGFT` and `PredFormerGFT_HybridBlock`
- `PI-IAM4VP`
- baselines: `WeatherGFT`, `WeatherGFTSingle`, `PredFormer`, `SimVP`,
  `PredRNN`, `PredRNNv2`

## Repository Layout

- `configs/` — YAML experiment configs. This is the primary place to define a
  run.
- `train.py` — unified pure-PyTorch training entry point.
- `trainer.py` — training loop, DDP, checkpointing, validation and Comet logs.
- `Data/` — WeatherBench datasets (`v1`, `v3`, `v3_memmap`, `v4`).
- `Models/` — architectures and model registry.
- `training_strategies/` — replacements for the old Lightning wrappers.
- `utils/` — metrics, normalization, registries, distributed helpers,
  checkpoint utilities.
- `train/` — legacy per-model scripts kept for compatibility.
- `sh_files/` — Slurm launchers and remote-submit helpers.
- `docs/` — cluster workflow, smoke tests and migration notes.

## Running Training

Use a config stem from `configs/`:

```bash
bash sh_files/launch_train.sh simvp_usa
```

On the cluster, submit one of the Slurm scripts:

```bash
sbatch sh_files/train_usa_2gpu.sh simvp_usa
sbatch sh_files/train_usa_2gpu.sh predformergft_usa
```

The local-to-cluster workflow is documented in
[`docs/cluster_workflow.md`](docs/cluster_workflow.md). In short: commit the
branch locally, then run:

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
  dataset_version: v3
  cut: [[75, 107], [164, 228]]
  sample_stride: 6      # hours between training sample starts
  frame_interval: 1     # hours between neighboring frames inside one sample
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
- `data.cut` is the spatial crop. USA configs use `[[75, 107], [164, 228]]`;
  the original South Atlantic crop is `[[36, 68], [125, 189]]`.
- For `v3`, `v3_memmap`, and `v4`, the temporal window indices are frame
  offsets. With `frame_interval: 1`, `start_time_x: 0`, `end_time_x: 11`
  means hourly frames `t..t+11h`, and `start_time_y: 12`, `end_time_y: 23`
  means hourly targets `t+12h..t+23h` before any `lead_time` offset.
  `sample_stride` controls how far apart neighboring training sample starts
  are. The legacy key `interval` is still accepted as a sample-start stride.
- `PredFormerGFT` and `PredFormerGFT_HybridBlock` also need
  `model.params.cut` to crop their static constants to the same window as the
  data.
- `v1` and `v3` use legacy WeatherBench files. Set `WEATHERBENCH_ROOT` or the
  fine-grained `WEATHERBENCH_INPUT_ROOT` / `WEATHERBENCH_NPY_ROOT` overrides
  if your files are not under the shared HSE path.
- `v3_memmap` and `v4` require an explicit packed memmap path. The committed
  v4 configs use `memmap_path: null`; pass `MEMMAP_PATH_OVERRIDE`, or set
  `WEATHERPRED_USA_MEMMAP` / `WEATHERPRED_GLOBE_MEMMAP` in `.env`. The v4
  Slurm scripts still accept `ORIG_MEMMAP` for one-off overrides.
- `v1`, `v3` and `v3_memmap` datasets return normalized tensors. `v4` returns
  raw memmap tensors; `trainer.py` applies `WeatherNormalize` on the batch.
- Validation logs `RMSE_<channel>_first`, `RMSE_<channel>_last`, and
  `RMSE_<channel>_mean` to Comet. Channel names follow the exact 69-channel
  WeatherBench layout after filtering: `t2`, `u10`, `v10`, `tp`, then
  `{z,t,r,u,v}{50,100,150,200,250,300,400,500,600,700,850,925,1000}`.
  Intermediate pressure levels such as `t450` are not direct channels. To
  restrict mean-channel logging, set `training.extra_kwargs.validation_channels`
  to a list such as `[z500, t500, t850, u500, v500]`.
- Storage paths can be overridden without editing configs:
  `DATA_FOLDER_OVERRIDE`, `INPUT_FOLDER_OVERRIDE`, `WEATHERBENCH_MEAN_STD_PATH`,
  `MEMMAP_PATH_OVERRIDE`, `MEMMAP_META_PATH_OVERRIDE`,
  `CHECKPOINT_BASE_OVERRIDE`, `WEATHERPRED_CHECKPOINT_BASE`.

## Model and Strategy Keys

| Model key | Typical strategy (`training.litmodel`) |
| --- | --- |
| `SimVP` | `mutiout_f` |
| `WeatherGFT` | `multiout_double` |
| `WeatherGFTSingle` | `mutiout` |
| `PredFormer` | `mutiout_f` |
| `PredFormerGFT` | `mutiout_f` |
| `PredFormerGFT_HybridBlock` | `mutiout_f` |
| `PI-IAM4VP` | `mutiout_imvp_small_world` |
| `PredRNN` | `mutiout_predrnn` |
| `PredRNNv2` | `mutiout_predrnn` |

The `mutiout` spelling is legacy and intentional.

## Checkpoints

Native checkpoints are `.pt` files with a flat payload:

- `model`
- optional `normalize`
- optional `optimizer`, `scheduler`, `scaler`
- `epoch`, `global_step`, `metric`, `config`

See [`docs/checkpoint_migration.md`](docs/checkpoint_migration.md) for
loading native checkpoints and converting older Lightning `.ckpt` files.

## Data

WeatherBench data paths in committed configs default to the shared HSE cluster
root via `WEATHERBENCH_ROOT`. Override that in `.env` for another account or
storage layout. The v4 production configs expect packed memmaps, usually staged to node-local
`/tmp` by the matching Slurm scripts before training starts.

The memmap loader validates channel count, variable order, crop metadata and
time coverage on startup so a mismatched file fails early instead of training
on the wrong window.

## Acknowledgements

This work used the HSE cHARISMa HPC cluster. The original research builds on
WeatherBench, WeatherGFT, PredFormer, IAM4VP, SimVP and PredRNN.
