# Migrating Lightning Checkpoints to the Native Format

This repository moved from PyTorch Lightning to a pure-PyTorch `Trainer`
(see `trainer.py`). Old `.ckpt` files saved by Lightning still load, but
they need a one-time conversion to the new flat format.

## What changes

A Lightning checkpoint stores model weights under `ckpt["state_dict"]`
with a `model.` prefix (because the `LightningModule` wrapped the real
`nn.Module` in `self.model`). The new `utils.checkpointing` reads and
writes a flat dict instead:

```python
{
    "model": state_dict,            # no "model." prefix
    "normalize": state_dict,        # optional WeatherNormalize buffers
    "optimizer": optimizer_state,   # optional
    "scheduler": scheduler_state,   # optional
    "scaler": amp_scaler_state,     # optional
    "epoch": int,
    "global_step": int,
    "metric": float,
    "config": dict,
}
```

## One-time conversion

Convert a single file:

```bash
python -m utils.checkpointing convert path/to/old.ckpt path/to/new.pt
```

Add `--keep-optimizer` if you want to resume training from the converted
checkpoint (otherwise only the weights are migrated):

```bash
python -m utils.checkpointing convert old.ckpt new.pt --keep-optimizer
```

Batch convert an entire directory tree:

```bash
find /home/ebugaev/checkpoints -name '*.ckpt' -print0 |
  xargs -0 -I {} bash -c '
    src="{}"; dst="${src%.ckpt}.pt";
    python -m utils.checkpointing convert "$src" "$dst"
  '
```

## Inference notebook

`Inference_and_plots.ipynb` was written for Lightning checkpoints and some
cells assume `ckpt["state_dict"]` with a `model.` prefix. Native `.pt`
checkpoints instead store weights in `ckpt["model"]`, so use a small helper
that supports both formats:

```python
def load_weatherpred_weights(model, path, strict=True):
    ckpt = torch.load(path, map_location="cpu")

    if "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = {
            key[len("model."):] if key.startswith("model.") else key: value
            for key, value in ckpt["state_dict"].items()
        }
    else:
        state = ckpt

    model.load_state_dict(state, strict=strict)
    return ckpt


ckpt = load_weatherpred_weights(model, path)
```

For v4 runs, native checkpoints may also contain `ckpt["normalize"]`. Load
those buffers into `utils.normalize.WeatherNormalize` if your inference path
feeds raw memmap values directly to the model. If the notebook already uses
dataset-normalized tensors, do not normalize a second time.

## Resuming training

The new `Trainer` does not yet wire a CLI flag for resume; if you need to
continue from a converted checkpoint, call `utils.checkpointing.load_checkpoint`
manually after constructing the model / optimiser / scheduler and pass
them into `Trainer.fit`.
