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

`Inference_and_plots.ipynb` already loads via `torch.load(path)` and
strips the `model.` prefix from each key by hand. After conversion the
prefix is no longer present, so the existing slicing code (`key[6:]`)
becomes a no-op — the notebook keeps working without edits. If you want
to simplify it, replace the manual loop with:

```python
ckpt = torch.load(path, map_location="cpu")
model.load_state_dict(ckpt["model"], strict=True)
```

## Resuming training

The new `Trainer` does not yet wire a CLI flag for resume; if you need to
continue from a converted checkpoint, call `utils.checkpointing.load_checkpoint`
manually after constructing the model / optimiser / scheduler and pass
them into `Trainer.fit`.
