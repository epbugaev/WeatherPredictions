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
bash sh_files/launch_train.sh simvp_usa_v4
```

On the cluster, submit one of the Slurm scripts:

```bash
sbatch sh_files/train_simvp_usa_v4_2gpu.sh
sbatch sh_files/train_predformergft_usa_v4_2gpu.sh
```

The local-to-cluster workflow is documented in
[`docs/cluster_workflow.md`](docs/cluster_workflow.md). In short: commit the
branch locally, then run:

```bash
bash sh_files/remote_submit.sh sh_files/train_simvp_usa_v4_2gpu.sh
```

## Config Basics

Each YAML config has these blocks:

```yaml
experiment:
  name: SimVP-USA-v4-72h

model:
  type: SimVP
  params:
    in_shape: [12, 69, 32, 64]

data:
  dataset_version: v4
  cut: [[75, 107], [164, 228]]
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
  memmap_path: /home/fa.buzaev/era5_memmap/predformer_usa_2000_2004.dat

training:
  litmodel: mutiout_f
  lr: 1e-4
  max_epoch: 2000

logging:
  checkpoint_base: /home/fa.buzaev/WeatherPredictions/checkpoints/
```

Important details:

- `data.cut` is the spatial crop. USA configs use `[[75, 107], [164, 228]]`;
  the original South Atlantic crop is `[[36, 68], [125, 189]]`.
- `PredFormerGFT` and `PredFormerGFT_HybridBlock` also need
  `model.params.cut` to crop their static constants to the same window as the
  data.
- `v1`, `v3` and `v3_memmap` datasets return normalized tensors.
- `v4` returns raw memmap tensors; `trainer.py` applies `WeatherNormalize` on
  the batch, on GPU.
- Storage paths can be overridden without editing configs:
  `DATA_FOLDER_OVERRIDE`, `INPUT_FOLDER_OVERRIDE`, `MEMMAP_PATH_OVERRIDE`,
  `MEMMAP_META_PATH_OVERRIDE`, `CHECKPOINT_BASE_OVERRIDE`.

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

WeatherBench data paths in committed configs point to the HSE cluster. The
v4 production configs expect packed memmaps, usually staged to node-local
`/tmp` by the matching Slurm scripts before training starts.

The memmap loader validates channel count, variable order, crop metadata and
time coverage on startup so a mismatched file fails early instead of training
on the wrong window.

## Acknowledgements

This work used the HSE cHARISMa HPC cluster. The original research builds on
WeatherBench, WeatherGFT, PredFormer, IAM4VP, SimVP and PredRNN.
