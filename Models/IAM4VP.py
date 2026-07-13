import math

import torch
from torch import nn

from utils.physics_residual import PhysicsResidualMixin

from .IAM4VP_utils import Attention, CircularConvSC, ConvNeXt_block, ConvNeXt_bottle


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for scalar timesteps.

    Args:
        dim: Width of the produced embedding (number of output channels);
            must be even (the output concatenates ``dim // 2`` sin/cos pairs).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of scalar timesteps.

        Args:
            x: ``torch.Tensor`` of shape ``(B,)`` with the timestep values.

        Returns:
            ``torch.Tensor`` of shape ``(B, dim)`` — concatenated sin/cos features.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Time_MLP(nn.Module):
    """Timestep MLP: sinusoidal embedding followed by a GELU feed-forward block.

    Args:
        dim: Embedding width and output channel count.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.sinusoidaposemb = SinusoidalPosEmb(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project scalar timesteps to a learned embedding.

        Args:
            x: ``torch.Tensor`` of shape ``(B,)`` with the timestep values.

        Returns:
            ``torch.Tensor`` of shape ``(B, dim)``.
        """
        x = self.sinusoidaposemb(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


def stride_generator(N: int, reverse: bool = False) -> list[int]:
    """Build the alternating stride schedule for the conv encoder/decoder.

    Args:
        N: Number of conv layers (length of the returned schedule).
        reverse: If True, return the reversed schedule (used by the decoder).

    Returns:
        ``list[int]`` of length ``N`` alternating ``1, 2, 1, 2, ...``.
    """
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class Encoder(nn.Module):
    """Convolutional encoder producing a bottleneck latent plus decoder skips.

    Args:
        C_in: Input channel count.
        C_hid: Hidden channel count of every conv stage.
        N_S: Number of conv stages (must be 4; ``forward`` indexes ``enc[0..3]``).
    """

    def __init__(self, C_in: int, C_hid: int, N_S: int):
        super().__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            CircularConvSC(C_in, C_hid, stride=strides[0]),
            *[CircularConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one flattened batch-time stack of frames.

        Args:
            x: ``torch.Tensor`` of shape ``(B*T, C_in, H, W)``.

        Returns:
            Tuple ``(latent_3, enc1, latent_1, latent_2)`` of ``torch.Tensor``
            for the ``N_S=4`` stride pattern ``[1, 2, 1, 2]``: ``enc1``
            ``(B*T, C_hid, H, W)``, ``latent_1``/``latent_2``
            ``(B*T, C_hid, H/2, W/2)``, ``latent_3`` ``(B*T, C_hid, H/4, W/4)``.
        """
        enc1 = self.enc[0](x)
        latent = enc1

        latent_1 = self.enc[1](latent)
        latent_2 = self.enc[2](latent_1)
        latent_3 = self.enc[3](latent_2)

        return latent_3, enc1, latent_1, latent_2


class Decoder(nn.Module):
    """Convolutional decoder fusing encoder skips back to frame resolution.

    Args:
        C_hid: Hidden channel count.
        N_S: Number of transpose-conv stages (must be 4).
        T: Clip length; sets the ``readout`` conv input width (``64 * T``).
    """

    def __init__(self, C_hid: int, N_S: int, T: int):
        super().__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[CircularConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            CircularConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True),
        )
        self.readout = nn.Conv2d(64 * T, 64, 1)

    def forward(
        self,
        hid: torch.Tensor,
        enc1: torch.Tensor,
        latent_1: torch.Tensor,
        latent_2: torch.Tensor,
        latent_3: torch.Tensor,
        T: int = 10,
        H: int = 8,
        W: int = 16,
    ) -> torch.Tensor:
        """Decode bottleneck features back to frame resolution using skips.

        The ``T``/``H``/``W`` defaults are placeholders; callers pass the real
        clip length and full frame size.

        Args:
            hid: bottleneck features ``(B*T, C_hid, H/4, W/4)``.
            enc1: full-resolution encoder skip ``(B*T, C_hid, H, W)``.
            latent_1: mid-resolution skip ``(B*T, C_hid, H/2, W/2)``.
            latent_2: mid-resolution skip ``(B*T, C_hid, H/2, W/2)``.
            latent_3: bottleneck skip ``(B*T, C_hid, H/4, W/4)``.
            T: clip length; the ``B*T`` axis is folded so ``T`` frames become
                channels before the readout conv.
            H: full output height.
            W: full output width.

        Returns:
            ``torch.Tensor`` of shape ``(B, 64, H, W)`` — decoded features
            before the model's final readout. The ``readout`` conv hardcodes
            64 output channels, so the architecture requires ``C_hid == 64``.
        """
        hid = self.dec[0](hid + latent_3)
        hid = self.dec[1](hid + latent_2)
        hid = self.dec[2](hid + latent_1)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        ys = Y.shape
        Y = Y.reshape(int(ys[0] / T), int(ys[1] * T), H, W)
        Y = self.readout(Y)
        return Y


class Predictor(nn.Module):
    """Temporal predictor: a stack of time-conditioned ConvNeXt blocks.

    Args:
        channel_in: Input channel count (``T`` frames folded into channels).
        N_T: Number of ConvNeXt blocks after the leading bottleneck block.
    """

    def __init__(self, channel_in: int, N_T: int):
        super().__init__()

        self.N_T = N_T
        st_block = [ConvNeXt_bottle(dim=channel_in)]
        for _ in range(N_T):
            st_block.append(ConvNeXt_block(dim=channel_in))

        self.st_block = nn.Sequential(*st_block)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Run the time-conditioned ConvNeXt stack over folded features.

        Args:
            x: ``torch.Tensor`` of shape ``(B, T, C, H, W)``; ``T`` and ``C``
                are folded to ``T*C`` channels before the blocks and unfolded
                afterwards.
            time_emb: timestep embedding ``(B, D)`` injected into every block.

        Returns:
            ``torch.Tensor`` of shape ``(B, T // 2, C, H, W)``.
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        z = self.st_block[0](x, time_emb)
        for i in range(1, self.N_T):
            z = self.st_block[i](z, time_emb)

        y = z.reshape(B, int(T / 2), C, H, W)
        return y


class IAM4VP(PhysicsResidualMixin, nn.Module):
    """Iterative Auto-regressive Model for Video Prediction + опциональный physics-prior.

    На каждом шаге `t` модель видит `(x_raw, pred_list, t)` — начальное
    состояние, накопленный список prev-predictions и timestep-эмбеддинг.
    Backward сделан per-step снаружи (см. `IterativeManualStep` в стратегиях).

    Два взаимоисключающих физических пути:

    * **латентный (legacy, `use_physics=True`)** — архитектурно-специфичный:
      `HybridBlock` гоняется на prev-prediction, результат кодируется
      `self.lp_phys` и подмешивается в mask_token/skip с весом 0.1. Живёт здесь.
    * **residual (`use_physics_residual_corrector=True`)** — общий с PI-PredRNNv2:
      наследуется из :class:`utils.physics_residual.PhysicsResidualMixin`, работает
      в пространстве состояния `(B, C_data, H, W)` и об архитектуре ничего не знает.

    Args:
        T_data, C_data, H_data, W_data: длина клипа и shape кадра.
        hid_S, N_S, N_T: ширина скрытого слоя и глубины encoder/predictor/decoder
            (ширина Predictor'а определяется ``T_data * hid_S``).
        use_physics: если True, включён латентный legacy-путь (`AI + physics_correction`).
        **physics_kwargs: параметры residual-пути, см.
            :meth:`utils.physics_residual.PhysicsResidualMixin.init_physics_residual`.
    """

    def __init__(
        self,
        T_data: int = 6,
        C_data: int = 69,
        H_data: int = 32,
        W_data: int = 64,
        hid_S: int = 64,
        N_S: int = 4,
        N_T: int = 6,
        use_physics: bool = True,
        **physics_kwargs,
    ):
        super().__init__()
        self.time_mlp = Time_MLP(dim=hid_S)
        self.enc = Encoder(C_data, hid_S, N_S)
        self.hid = Predictor(T_data * hid_S, N_T)
        self.dec = Decoder(hid_S, N_S, T_data)
        self.attn = Attention(hid_S)
        self.readout = nn.Conv2d(hid_S, C_data, 1)
        self.mask_token = nn.Parameter(
            torch.zeros(T_data, hid_S, H_data // 4, W_data // 4)
        )  # for 1_4 and 5_6
        self.lp = Encoder(C_data, hid_S, N_S)
        self.lp_phys = Encoder(C_data, hid_S, N_S)

        self.skip_mask_token = nn.Parameter(torch.zeros(T_data, hid_S, H_data, W_data))
        self.embed_1_mask_token = nn.Parameter(torch.zeros(T_data, hid_S, H_data // 2, W_data // 2))
        self.embed_2_mask_token = nn.Parameter(torch.zeros(T_data, hid_S, H_data // 2, W_data // 2))

        self.use_physics = use_physics
        # Физический residual-путь строится последним: HybridBlock и голова
        # коррекции потребляют глобальный RNG, поэтому порядок конструирования
        # фиксирует инициализацию весов (побитовая совместимость с exp16).
        self.init_physics_residual(
            C_data=C_data,
            H_data=H_data,
            W_data=W_data,
            downscaling_factor_all=4,
            **physics_kwargs,
        )

    def forward(self, x_raw, y_raw=None, t=None):
        """Один шаг авторегрессивного прогноза + опциональный physics-correction.

        Сигнатура отличается от обычных `forward(self, x)`-моделей: используется
        вместе с `IterativeManualStep`, где трейнер делает per-timestep loop.

        Args:
            x_raw: начальное состояние `(B, T, C, H, W)`.
            y_raw: накопленный список prev-predictions от `IterativeManualStep`
                (передаётся как кортеж/список тензоров) или None.
            t: 1-D тензор `(B,)` — timestep-эмбеддинг (значение `(idx_time+1)*100`).

        Returns:
            Прогноз на текущий timestep, форма `(B, C, H, W)`.
        """
        if y_raw is None:
            y_raw = []
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)
        time_emb = self.time_mlp(t)

        embed, skip, embed_1, embed_2 = self.enc(x)
        mask_token = self.mask_token.repeat(B, 1, 1, 1, 1)

        skip_mask_token = self.skip_mask_token.repeat(B, 1, 1, 1, 1)
        embed_1_mask_token = self.embed_1_mask_token.repeat(B, 1, 1, 1, 1)
        embed_2_mask_token = self.embed_2_mask_token.repeat(B, 1, 1, 1, 1)

        use_legacy_latent_physics = self.use_physics and not self.use_physics_residual_corrector
        for idx, pred in enumerate(y_raw):
            embed2, skip_lp, embed_1_lp, embed_2_lp = self.lp(pred)

            if use_legacy_latent_physics:
                if idx == 0:
                    prev_pred = x_raw[:, -1]

                pred_phys = prev_pred[:, 4:, :, :]
                zquvtw = self.x_to_zquvtw(pred_phys)
                pred_phys = zquvtw

                for _ in range(3):
                    # Получаем физические эмбеддинги через hybrid_block
                    pred_phys, zquvtw = self.hybrid_block(
                        pred_phys, zquvtw
                    )  # Используем одинаковые данные для обоих входов

                # Возвращаем к исходному формату
                pred_phys = pred_phys.permute(0, 3, 1, 2)  # [B, C, H//4, W//4]

                # Масштабируем обратно до исходного размера
                pred_phys = torch.nn.functional.interpolate(pred_phys, size=(H, W), mode="bilinear")

                pred_to_hybrid = torch.cat([pred[:, :4, :, :], pred_phys], dim=1)

                embed2_phys, skip_lp_phys, embed_1_lp_phys, embed_2_lp_phys = self.lp_phys(
                    pred_to_hybrid
                )

                mask_token[:, idx, :, :, :] = embed2 + 0.1 * embed2_phys

                skip_mask_token[:, idx, :, :, :] = skip_lp + 0.1 * skip_lp_phys
                embed_1_mask_token[:, idx, :, :, :] = embed_1_lp + 0.1 * embed_1_lp_phys
                embed_2_mask_token[:, idx, :, :, :] = embed_2_lp + 0.1 * embed_2_lp_phys

                prev_pred = pred
            else:
                mask_token[:, idx, :, :, :] = embed2

                skip_mask_token[:, idx, :, :, :] = skip_lp
                embed_1_mask_token[:, idx, :, :, :] = embed_1_lp
                embed_2_mask_token[:, idx, :, :, :] = embed_2_lp

        _, C_, H_, W_ = embed.shape

        skip = skip + skip_mask_token.view(B * T, C_, H_ * 4, W_ * 4)
        embed_1 = embed_1 + embed_1_mask_token.view(B * T, C_, H_ * 2, W_ * 2)
        embed_2 = embed_2 + embed_2_mask_token.view(B * T, C_, H_ * 2, W_ * 2)

        z = embed.view(B, T, C_, H_, W_)
        z2 = mask_token
        z = torch.cat([z, z2], dim=1)
        hid = self.hid(z, time_emb)
        hid = hid.reshape(B * T, C_, H_, W_)

        Y = self.dec(hid, skip, embed_1, embed_2, embed, T=T, H=H, W=W)

        Y = self.attn(Y)
        y_nn = self.readout(Y)
        if self.use_physics_residual_corrector:
            prev_state = x_raw[:, -1] if len(y_raw) == 0 else y_raw[-1]
            return self._apply_physics_residual(y_nn, prev_state)
        self._last_residual_aux_loss = None
        self._last_residual_diagnostics = {}
        return y_nn
