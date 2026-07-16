"""PredRNN and PredRNN-V2 spatiotemporal predictors.

Faithful PyTorch port of the official ``thuml/predrnn-pytorch``
``core/models/predrnn.py`` and ``core/models/predrnn_v2.py`` (Wang et al.,
"PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning",
NeurIPS 2017 / TPAMI 2022, https://github.com/thuml/predrnn-pytorch),
adapted to this repository's contract:

* constructor ``PredRNN_Model(num_layers, num_hidden, configs: dict)`` where
  ``configs`` is a plain dict (not an ``argparse`` namespace);
* ``forward(frames_tensor, mask_true) -> (next_frames, aux_losses)`` with
  channels-last ``(B, T, H, W, C)`` tensors, exactly as fed by
  :class:`training_strategies.predrnn.PredRNNStep`; ``aux_losses`` is a dict of
  already-weighted scalar penalties the strategy adds to its own task loss
  (upstream folds an MSE task term into the model's return — here the strategy
  owns the task loss, so the model reports auxiliary terms only);
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
from utils.physics_residual import PhysicsResidualMixin
from utils.static_input import StaticInputMixin


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

    def forward(
        self, frames_tensor: torch.Tensor, mask_true: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Roll out predictions over the full clip.

        Args:
            frames_tensor: ``(B, T, H, W, C)`` channels-last input clip
                (the strategy concatenates context + target along time).
            mask_true: Scheduled-sampling mask, ``(1, T_pred, 1, 1, 1)``;
                all-zeros means fully autoregressive generation.

        Returns:
            Tuple ``(next_frames, aux_losses)`` where ``next_frames`` is
            ``(B, T - 1, H, W, C)`` and ``aux_losses`` is empty: v1 has no
            auxiliary penalty. The task loss is owned by the training
            strategy, which is why no MSE is computed here.
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
        return next_frames, {}


class PredRNNv2_Model(StaticInputMixin, nn.Module):
    """PredRNN-V2 with memory decoupling (TPAMI 2022).

    Adds the shared 1x1 ``adapter`` and the cosine-similarity decoupling
    penalty between ``delta_c`` and ``delta_m`` on top of PredRNN v1.

    Args:
        num_layers: Number of stacked :class:`SpatioTemporalLSTMCellv2` layers.
        num_hidden: Per-layer hidden channel counts, length ``num_layers``.
        configs: Geometry/config dict (see :func:`_parse_configs`);
            ``decouple_beta`` weights the decoupling loss (default ``0.1``).
        static_input_fields: Optional list of static field names (exp 24);
            ``None``/empty disables the feature (bit-exact with the prior
            behavior).
        static_constants_path: Path to the constants NetCDF; required when
            ``static_input_fields`` is set.
        static_cut: Crop window ``[lat0, lat1, lon0, lon1]``; required when
            ``static_input_fields`` is set.
    """

    def __init__(
        self,
        num_layers: int,
        num_hidden,
        configs: dict,
        static_input_fields: list[str] | None = None,
        static_constants_path: str | None = None,
        static_cut: list[int] | None = None,
    ) -> None:
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

        _, _, img_height, img_width = tuple(configs["in_shape"])
        num_static = self.init_static_input(
            static_input_fields, static_constants_path, static_cut, img_height, img_width
        )
        # Кадры патчатся внутри forward, поэтому буфер статики хранится уже
        # пропатченным: (1, S, H, W) -> (1, S*p^2, H/p, W/p).
        self.static_channel = num_static * self.patch_size**2
        if num_static > 0:
            static_channels_last = self.static_input.permute(0, 2, 3, 1).unsqueeze(1)
            patched = _reshape_patch(static_channels_last, self.patch_size)
            self.static_input = patched[0].permute(0, 3, 1, 2).contiguous()

        cell_list = []
        for i in range(num_layers):
            in_channel = (
                self.frame_channel + self.static_channel if i == 0 else self.num_hidden[i - 1]
            )
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

    def _correct_step(self, x_gen: torch.Tensor, net: torch.Tensor) -> torch.Tensor:
        """Post-process one rollout step; the identity in the physics-free model.

        Extension point for :class:`PI_PredRNNv2_Model`, which corrects every
        step with a physics prior. Kept here so the physics subclass reuses the
        rollout instead of copying it.

        Args:
            x_gen: this step's cell output, ``(B, frame_channel, H', W')``.
            net: this step's input frame, same shape as ``x_gen``.

        Returns:
            ``torch.Tensor`` of the same shape — the step output to emit and to
            feed back into the next autoregressive step.
        """
        return x_gen

    def forward(
        self, frames_tensor: torch.Tensor, mask_true: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Roll out predictions and report the memory-decoupling penalty.

        Args:
            frames_tensor: ``(B, T, H, W, C)`` channels-last input clip.
            mask_true: Scheduled-sampling mask, ``(1, T_pred, 1, 1, 1)``.

        Returns:
            Tuple ``(next_frames, aux_losses)`` where ``next_frames`` is
            ``(B, T - 1, H, W, C)`` and ``aux_losses`` maps ``"decouple"`` to
            the already-weighted (``decouple_beta`` times) mean
            cosine-similarity decoupling term — a scalar the training strategy
            adds to its own task loss. No task loss is computed here: the
            strategy owns it (MAE), and an internal MSE would silently blend a
            second objective.
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
                self.append_static_input(net), h_t[0], c_t[0], memory
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

            x_gen = self._correct_step(self.conv_last(h_t[self.num_layers - 1]), net)
            next_frames.append(x_gen)
            for i in range(self.num_layers):
                decouple_loss.append(
                    torch.mean(
                        torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))
                    )
                )

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = _reshape_patch_back(next_frames, self.patch_size, self.img_channel)
        return next_frames, {"decouple": self.decouple_beta * decouple_loss}


class PI_PredRNNv2_Model(PhysicsResidualMixin, PredRNNv2_Model):
    """PredRNN-V2 with the shared physics residual corrector (exp20).

    Same physics path as PI-IAM4VP: at each autoregressive step the cell output
    is corrected by a head fed with the physics residual
    ``delta_phys = physics(prev) - prev``. The difference is *where* the rollout
    lives. PI-IAM4VP is driven step-by-step by the training strategy, which
    detaches the previous state between steps; PredRNN rolls out inside its own
    ``forward``, so this class detaches ``prev_state`` itself — otherwise every
    rollout step's physics kernel would stay in one autograd graph.

    Requires ``patch_size == 1``: the physics kernel consumes the raw state
    ``(B, 69, H, W)``, and patching folds space into the channel axis, which
    would destroy the channel layout (surface 0:4, then z/t/r/u/v blocks of 13)
    the kernel relies on.

    Args:
        num_layers: number of stacked :class:`SpatioTemporalLSTMCellv2` layers.
        num_hidden: per-layer hidden channel counts, length ``num_layers``.
        configs: geometry/config dict (see :func:`_parse_configs`).
        static_input_fields: Optional list of static field names (exp 24);
            ``None``/empty disables the feature (bit-exact with the prior
            behavior).
        static_constants_path: Path to the constants NetCDF; required when
            ``static_input_fields`` is set.
        static_cut: Crop window ``[lat0, lat1, lon0, lon1]``; required when
            ``static_input_fields`` is set.
        **physics_kwargs: physics-branch parameters, forwarded verbatim to
            :meth:`utils.physics_residual.PhysicsResidualMixin.init_physics_residual`.
    """

    def __init__(
        self,
        num_layers: int,
        num_hidden,
        configs: dict,
        static_input_fields: list[str] | None = None,
        static_constants_path: str | None = None,
        static_cut: list[int] | None = None,
        **physics_kwargs,
    ) -> None:
        super().__init__(
            num_layers,
            num_hidden,
            configs,
            static_input_fields=static_input_fields,
            static_constants_path=static_constants_path,
            static_cut=static_cut,
        )
        if self.patch_size != 1:
            raise ValueError(
                "PI-PredRNNv2 requires patch_size=1: the physics kernel consumes the raw "
                f"state (B, 69, H, W), but patch_size={self.patch_size} folds space into "
                "the channel axis and breaks the z/t/r/u/v channel layout."
            )
        _, img_channel, img_height, img_width = tuple(configs["in_shape"])
        self._physics_aux_steps: list[torch.Tensor] = []
        self.init_physics_residual(
            C_data=img_channel,
            H_data=img_height,
            W_data=img_width,
            downscaling_factor_all=4,
            **physics_kwargs,
        )

    def _correct_step(self, x_gen: torch.Tensor, net: torch.Tensor) -> torch.Tensor:
        """Correct one rollout step with the physics residual head.

        ``net`` is detached: the physics prior must not backpropagate through
        earlier autoregressive steps (parity with PI-IAM4VP, whose strategy
        stores ``prediction.detach()`` in the rollout history).

        Args:
            x_gen: this step's cell output, ``(B, 69, H, W)`` (normalized).
            net: this step's input frame, ``(B, 69, H, W)`` (normalized).

        Returns:
            ``torch.Tensor`` of shape ``(B, 69, H, W)`` — the corrected output.
            Side effect: appends this step's auxiliary penalty to
            ``_physics_aux_steps``.
        """
        corrected = self._apply_physics_residual(x_gen, net.detach())
        step_aux = self.physics_residual_aux_loss()
        if step_aux is not None:
            self._physics_aux_steps.append(step_aux)
        return corrected

    def forward(
        self, frames_tensor: torch.Tensor, mask_true: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Roll out predictions, correcting every step with the physics head.

        Args:
            frames_tensor: ``(B, T, H, W, C)`` channels-last input clip.
            mask_true: Scheduled-sampling mask, ``(1, T_pred, 1, 1, 1)``.

        Returns:
            Tuple ``(next_frames, aux_losses)`` where ``next_frames`` is
            ``(B, T - 1, H, W, C)`` and ``aux_losses`` carries the weighted
            ``"decouple"`` penalty plus ``"physics_aux"`` — the residual/diabatic
            L1 penalty **averaged over rollout steps**. The average is built here
            because the mixin overwrites its per-call aux loss on every step, so
            reading it once after the loop would keep only the last step's.
        """
        self._physics_aux_steps = []
        next_frames, aux_losses = super().forward(frames_tensor, mask_true)
        if self._physics_aux_steps:
            aux_losses["physics_aux"] = torch.mean(torch.stack(self._physics_aux_steps, dim=0))
        self._physics_aux_steps = []
        return next_frames, aux_losses
