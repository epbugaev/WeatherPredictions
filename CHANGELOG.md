# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
