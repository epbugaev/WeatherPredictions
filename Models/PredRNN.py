"""PredRNN and PredRNN-V2 spatiotemporal predictors.

Faithful PyTorch port of the official ``thuml/predrnn-pytorch``
``core/models/predrnn.py`` and ``core/models/predrnn_v2.py`` (Wang et al.,
"PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning",
NeurIPS 2017 / TPAMI 2022, https://github.com/thuml/predrnn-pytorch),
adapted to this repository's contract:

* constructor ``PredRNN_Model(num_layers, num_hidden, configs: dict)`` where
  ``configs`` is a plain dict (not an ``argparse`` namespace);
* ``forward(frames_tensor, mask_true) -> (next_frames, loss)`` with
  channels-last ``(B, T, H, W, C)`` tensors, exactly as fed by
  :class:`training_strategies.predrnn.PredRNNStep`;
* the device is taken from the input tensor (no ``configs.device``);
* patch embedding (``reshape_patch``) is done inside ``forward`` so the model
  is self-contained (the upstream code patches in the data pipeline).

The upstream V2 t-SNE ``visualization`` hook is intentionally omitted: it is a
debug-only path with an external dependency and is unused for training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from Models.PredRNN_utils import SpatioTemporalLSTMCell, SpatioTemporalLSTMCellv2


def _reshape_patch(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Fold ``patch_size`` x ``patch_size`` spatial blocks into the channel axis.

    Args:
        frames: ``torch.Tensor`` of shape ``(B, T, H, W, C)`` (channels-last).
        patch_size: Spatial patch size; must divide ``H`` and ``W``.

    Returns:
        ``torch.Tensor`` of shape ``(B, T, H // p, W // p, p * p * C)``.
    """
    b, t, h, w, c = frames.shape
    x = frames.reshape(b, t, h // patch_size, patch_size, w // patch_size, patch_size, c)
    x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    return x.reshape(b, t, h // patch_size, w // patch_size, patch_size * patch_size * c)


def _reshape_patch_back(patch: torch.Tensor, patch_size: int, channels: int) -> torch.Tensor:
    """Inverse of :func:`_reshape_patch`.

    Args:
        patch: ``torch.Tensor`` of shape ``(B, T, H', W', p * p * C)``.
        patch_size: Spatial patch size used in the forward transform.
        channels: Original channel count ``C``.

    Returns:
        ``torch.Tensor`` of shape ``(B, T, H' * p, W' * p, C)``.
    """
    b, t, hp, wp, _ = patch.shape
    x = patch.reshape(b, t, hp, wp, patch_size, patch_size, channels)
    x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    return x.reshape(b, t, hp * patch_size, wp * patch_size, channels)


def _parse_configs(configs: dict) -> dict:
    """Derive PredRNN geometry from this repo's ``configs`` dict.

    Args:
        configs: Dict with keys ``in_shape`` ``(T, C, H, W)``, ``patch_size``,
            ``filter_size``, ``stride``, ``layer_norm``, ``pre_seq_length``,
            and optionally ``reverse_scheduled_sampling`` / ``decouple_beta``.

    Returns:
        Dict of resolved scalars used by both model variants.
    """
    in_shape = tuple(configs["in_shape"])
    _, img_channel, img_height, img_width = in_shape
    patch_size = configs["patch_size"]
    return {
        "img_channel": img_channel,
        "frame_channel": patch_size * patch_size * img_channel,
        "patch_size": patch_size,
        "cell_height": img_height // patch_size,
        "cell_width": img_width // patch_size,
        "filter_size": configs["filter_size"],
        "stride": configs["stride"],
        "layer_norm": configs["layer_norm"],
        "input_length": configs["pre_seq_length"],
        "reverse_scheduled_sampling": configs.get("reverse_scheduled_sampling", 0),
        "decouple_beta": configs.get("decouple_beta", 0.1),
    }


class PredRNN_Model(nn.Module):
    """PredRNN (v1) recurrent spatiotemporal predictor.

    Args:
        num_layers: Number of stacked :class:`SpatioTemporalLSTMCell` layers.
        num_hidden: Per-layer hidden channel counts, length ``num_layers``.
        configs: Geometry/config dict (see :func:`_parse_configs`).
    """

    def __init__(self, num_layers: int, num_hidden, configs: dict) -> None:
        super().__init__()
        cfg = _parse_configs(configs)
        self.img_channel = cfg["img_channel"]
        self.frame_channel = cfg["frame_channel"]
        self.patch_size = cfg["patch_size"]
        self.input_length = cfg["input_length"]
        self.reverse_scheduled_sampling = cfg["reverse_scheduled_sampling"]
        self.num_layers = num_layers
        self.num_hidden = list(num_hidden)

        cell_list = []
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(
                    in_channel,
                    self.num_hidden[i],
                    cfg["cell_height"],
                    cfg["cell_width"],
                    cfg["filter_size"],
                    cfg["stride"],
                    cfg["layer_norm"],
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(
            self.num_hidden[num_layers - 1],
            self.frame_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.mse_criterion = nn.MSELoss()

    def forward(
        self, frames_tensor: torch.Tensor, mask_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out predictions over the full clip.

        Args:
            frames_tensor: ``(B, T, H, W, C)`` channels-last input clip
                (the strategy concatenates context + target along time).
            mask_true: Scheduled-sampling mask, ``(1, T_pred, 1, 1, 1)``;
                all-zeros means fully autoregressive generation.

        Returns:
            Tuple ``(next_frames, loss)`` where ``next_frames`` is
            ``(B, T - 1, H, W, C)`` and ``loss`` is the MSE against the
            shifted input (the trainer strategy recomputes its own loss).
        """
        patches = _reshape_patch(frames_tensor, self.patch_size)
        frames = patches.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        device = frames.device
        batch, total_length = frames.shape[0], frames.shape[1]
        height, width = frames.shape[3], frames.shape[4]

        next_frames = []
        h_t, c_t = [], []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)
        memory = torch.zeros([batch, self.num_hidden[0], height, width], device=device)

        x_gen = frames[:, 0]
        for t in range(total_length - 1):
            if self.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.input_length:
                    net = frames[:, t]
                else:
                    net = (
                        mask_true[:, t - self.input_length] * frames[:, t]
                        + (1 - mask_true[:, t - self.input_length]) * x_gen
                    )

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = _reshape_patch_back(next_frames, self.patch_size, self.img_channel)
        loss = self.mse_criterion(next_frames, frames_tensor[:, 1:])
        return next_frames, loss


class PredRNNv2_Model(nn.Module):
    """PredRNN-V2 with memory decoupling (TPAMI 2022).

    Adds the shared 1x1 ``adapter`` and the cosine-similarity decoupling
    penalty between ``delta_c`` and ``delta_m`` on top of PredRNN v1.

    Args:
        num_layers: Number of stacked :class:`SpatioTemporalLSTMCellv2` layers.
        num_hidden: Per-layer hidden channel counts, length ``num_layers``.
        configs: Geometry/config dict (see :func:`_parse_configs`);
            ``decouple_beta`` weights the decoupling loss (default ``0.1``).
    """

    def __init__(self, num_layers: int, num_hidden, configs: dict) -> None:
        super().__init__()
        cfg = _parse_configs(configs)
        self.img_channel = cfg["img_channel"]
        self.frame_channel = cfg["frame_channel"]
        self.patch_size = cfg["patch_size"]
        self.input_length = cfg["input_length"]
        self.reverse_scheduled_sampling = cfg["reverse_scheduled_sampling"]
        self.decouple_beta = cfg["decouple_beta"]
        self.num_layers = num_layers
        self.num_hidden = list(num_hidden)

        cell_list = []
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCellv2(
                    in_channel,
                    self.num_hidden[i],
                    cfg["cell_height"],
                    cfg["cell_width"],
                    cfg["filter_size"],
                    cfg["stride"],
                    cfg["layer_norm"],
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(
            self.num_hidden[num_layers - 1],
            self.frame_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        adapter_num_hidden = self.num_hidden[0]
        self.adapter = nn.Conv2d(
            adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False
        )
        self.mse_criterion = nn.MSELoss()

    def forward(
        self, frames_tensor: torch.Tensor, mask_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out predictions and add the memory-decoupling penalty.

        Args:
            frames_tensor: ``(B, T, H, W, C)`` channels-last input clip.
            mask_true: Scheduled-sampling mask, ``(1, T_pred, 1, 1, 1)``.

        Returns:
            Tuple ``(next_frames, loss)`` where ``next_frames`` is
            ``(B, T - 1, H, W, C)`` and ``loss`` is MSE plus
            ``decouple_beta`` times the mean cosine-similarity decoupling
            term (the trainer strategy recomputes its own task loss).
        """
        patches = _reshape_patch(frames_tensor, self.patch_size)
        frames = patches.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        device = frames.device
        batch, total_length = frames.shape[0], frames.shape[1]
        height, width = frames.shape[3], frames.shape[4]

        next_frames = []
        h_t, c_t, delta_c_list, delta_m_list = [], [], [], []
        decouple_loss = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)
        memory = torch.zeros([batch, self.num_hidden[0], height, width], device=device)

        x_gen = frames[:, 0]
        for t in range(total_length - 1):
            if self.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.input_length:
                    net = frames[:, t]
                else:
                    net = (
                        mask_true[:, t - self.input_length] * frames[:, t]
                        + (1 - mask_true[:, t - self.input_length]) * x_gen
                    )

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](
                net, h_t[0], c_t[0], memory
            )
            delta_c_list[0] = F.normalize(
                self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2
            )
            delta_m_list[0] = F.normalize(
                self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2
            )
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory
                )
                delta_c_list[i] = F.normalize(
                    self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2
                )
                delta_m_list[i] = F.normalize(
                    self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2
                )

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
            for i in range(self.num_layers):
                decouple_loss.append(
                    torch.mean(
                        torch.abs(
                            torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2)
                        )
                    )
                )

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = _reshape_patch_back(next_frames, self.patch_size, self.img_channel)
        loss = (
            self.mse_criterion(next_frames, frames_tensor[:, 1:])
            + self.decouple_beta * decouple_loss
        )
        return next_frames, loss
