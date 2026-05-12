"""Shared helper for logging prediction/truth/diff maps to an Experiment.

Replicates the matplotlib-heavy block that was duplicated across four
LitModels (mutiout, multiout_double, mutiout_imvp, mutiout_predrnn). The
helper is gated on ``is_main_process`` so non-main DDP ranks skip the work
entirely.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from utils.experiment import Experiment


def log_prediction_maps(
    experiment: Experiment,
    pred: torch.Tensor,
    truth: torch.Tensor,
    index_map: dict[str, int],
    vis_vars: tuple[str, ...],
    step: int | None = None,
) -> None:
    """Log prediction, ground-truth and difference maps for each variable.

    For every variable in ``vis_vars`` that exists in ``index_map``, three
    figures are sent to ``experiment``: ``{var}_prediction``, ``{var}_true``
    and ``{var}_diff``. The first batch element is used (``[0, var_idx]``),
    matching the legacy LitModels behaviour exactly.

    Args:
        experiment: Comet/Null experiment facade.
        pred: Prediction tensor of shape ``(B, C, H, W)`` (single timestep).
        truth: Ground-truth tensor with the same shape as ``pred``.
        index_map: Mapping ``variable_name -> channel index``.
        vis_vars: Variables to visualise (subset of ``index_map`` keys).
        step: Optional global step for Comet to align figures in time.
    """
    pred_cpu = pred.detach().cpu()
    truth_cpu = truth.detach().cpu()

    for var_name in vis_vars:
        if var_name not in index_map:
            continue
        var_idx = index_map[var_name]

        pred_map = pred_cpu[0, var_idx].numpy()
        true_map = truth_cpu[0, var_idx].numpy()
        diff_map = pred_map - true_map

        fig_pred, ax_pred = plt.subplots(figsize=(10, 8))
        im_pred = ax_pred.imshow(pred_map, cmap="viridis")
        plt.colorbar(im_pred, ax=ax_pred)
        ax_pred.set_title(f"{var_name} Prediction")
        experiment.log_figure(figure=fig_pred, name=f"{var_name}_prediction", step=step)
        plt.close(fig_pred)

        fig_true, ax_true = plt.subplots(figsize=(10, 8))
        im_true = ax_true.imshow(true_map, cmap="viridis")
        plt.colorbar(im_true, ax=ax_true)
        ax_true.set_title(f"{var_name} Ground Truth")
        experiment.log_figure(figure=fig_true, name=f"{var_name}_true", step=step)
        plt.close(fig_true)

        fig_diff, ax_diff = plt.subplots(figsize=(10, 8))
        im_diff = ax_diff.imshow(diff_map, cmap="RdBu_r")
        plt.colorbar(im_diff, ax=ax_diff)
        ax_diff.set_title(f"{var_name} Prediction - True")
        experiment.log_figure(figure=fig_diff, name=f"{var_name}_diff", step=step)
        plt.close(fig_diff)
