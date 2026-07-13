import torch
from torch import nn

from Models.SimVP_utils import (
    ConvMixerSubBlock,
    ConvNeXtSubBlock,
    ConvSC,
    GASubBlock,
    HorNetSubBlock,
    MLPMixerSubBlock,
    MogaSubBlock,
    PoolFormerSubBlock,
    SwinSubBlock,
    TAUSubBlock,
    UniformerSubBlock,
    VANSubBlock,
    ViTSubBlock,
    gInception_ST,
)
from utils.physics_residual import PhysicsResidualMixin

PHYSICS_COUPLINGS = ("batched", "chained")


class SimVP_Model(nn.Module):
    r"""Видеопредиктор SimVP: encoder → mid → decoder с U-Net skip-connection’ами.

    Реализация `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_. Все таймстепы пропускаются через
    общий per-frame encoder/decoder, временные зависимости моделирует
    middle-блок над `(B, T·hid_S, H', W')` или `(B, T, hid_S, H', W')`.

    Args:
        in_shape: кортеж `(T, C, H, W)` — длина входного клипа и shape кадра.
        hid_S, hid_T: размерности скрытых пространств encoder’а и middle-блока.
        N_S, N_T: число слоёв в encoder’е/decoder’е и middle-блоке.
        model_type: тип middle-блока (`gSTA`, `IncepU`, `ConvNeXt`, …).
        mlp_ratio, drop, drop_path: типичные dropout/MLP-параметры.
        spatio_kernel_enc/dec: kernel-size для spatial-сверток.
        act_inplace: использовать ли in-place активации (для torch.compile = False).
    """

    def __init__(
        self,
        in_shape,
        hid_S=16,
        hid_T=256,
        N_S=4,
        N_T=4,
        model_type="gSTA",
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        act_inplace=True,
        **kwargs,
    ):
        super().__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2 ** (N_S / 2)), int(W / 2 ** (N_S / 2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        model_type = "gsta" if model_type is None else model_type.lower()
        if model_type == "incepu":
            self.hid = MidIncepNet(T * hid_S, hid_T, N_T)
        else:
            self.hid = MidMetaNet(
                T * hid_S,
                hid_T,
                N_T,
                input_resolution=(H, W),
                model_type=model_type,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
            )

    def forward(self, x_raw, **kwargs):
        """Прогон клипа `(B, T, C, H, W)` → клип той же формы.

        Args:
            x_raw: входной клип формы `(B, T, C, H, W)`.

        Returns:
            Прогноз формы `(B, T, C, H, W)`.
        """
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B * T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y


class PI_SimVP_Model(PhysicsResidualMixin, SimVP_Model):
    """SimVPv2 с физическим residual-корректором (exp21).

    Тот же физический путь, что у PI-IAM4VP и PI-PredRNNv2: прогноз бэкбона
    корректируется головой, которой на вход подаётся физическая невязка
    ``delta_phys = physics(prev) - prev``. Отличается **источник** ``prev``.

    SimVP — MIMO: все T кадров выдаются одним forward, рекуррентности нет, и
    «предыдущего состояния» для кадра t архитектура не предоставляет. Физядро же
    парное (``state(t) --dt--> state(t+1)``), поэтому ``prev`` конструируется, и
    способ его конструирования — предмет эксперимента, а не деталь реализации:

    * ``physics_coupling="batched"`` (основной режим лестницы): ``prev`` для
      кадра t — прогноз **бэкбона** на кадр t-1 (для кадра 0 — последний входной
      кадр). Все T кадров корректируются независимо, одним «плоским» батчем
      ``B*T`` (порезанным на чанки, см. :meth:`_apply_physics_chunked`), поэтому
      MIMO-характер модели сохраняется: авторегрессия не навязывается.
    * ``physics_coupling="chained"`` (контрольный арм S3c): ``prev`` — уже
      **скорректированный** кадр t-1, то есть семантика PI-IAM4VP 1:1, T
      последовательных вызовов. Разница арм ``S3c - S3`` измеряет цену
      авторегрессивной связки физики при идентичных уравнениях.

    ``prev`` детачится в обоих режимах — паритет с PI-IAM4VP (стратегия кладёт в
    историю ``prediction.detach()``): физприор не пробрасывает градиент сквозь
    предыдущие кадры, иначе T шагов физядра попали бы в один граф.

    Args:
        in_shape: ``(T, C, H, W)`` — длина клипа и shape кадра; физядро требует
            ``C=69`` и латент-сетку 8x16, то есть ``(H, W) = (32, 64)``.
        hid_S, hid_T, N_S, N_T, model_type, mlp_ratio, drop, drop_path,
            spatio_kernel_enc, spatio_kernel_dec, act_inplace: параметры бэкбона
            :class:`SimVP_Model`; дефолтный ``model_type="gSTA"`` — это v2.
        physics_coupling: ``"batched"`` или ``"chained"`` (см. выше).
        **physics_kwargs: параметры физветки, пробрасываются дословно в
            :meth:`utils.physics_residual.PhysicsResidualMixin.init_physics_residual`.
    """

    def __init__(
        self,
        in_shape,
        hid_S=16,
        hid_T=256,
        N_S=4,
        N_T=4,
        model_type="gSTA",
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        act_inplace=True,
        physics_coupling: str = "batched",
        physics_chunk_size: int = 256,
        **physics_kwargs,
    ) -> None:
        super().__init__(
            in_shape,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            model_type=model_type,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            spatio_kernel_enc=spatio_kernel_enc,
            spatio_kernel_dec=spatio_kernel_dec,
            act_inplace=act_inplace,
        )
        if physics_coupling not in PHYSICS_COUPLINGS:
            raise ValueError(
                f"Unknown physics_coupling {physics_coupling!r}; "
                f"expected one of {PHYSICS_COUPLINGS}"
            )
        if physics_chunk_size < 1:
            raise ValueError(f"physics_chunk_size must be >= 1, got {physics_chunk_size}")
        self.physics_coupling = physics_coupling
        self.physics_chunk_size = physics_chunk_size
        _, C_data, H_data, W_data = tuple(in_shape)
        self.init_physics_residual(
            C_data=C_data,
            H_data=H_data,
            W_data=W_data,
            downscaling_factor_all=4,
            **physics_kwargs,
        )

    def _apply_physics_chunked(self, y_nn: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """Apply the residual correction in chunks of at most ``physics_chunk_size``.

        Chunking exists purely to respect a CUDA grid limit, not for memory: the
        WENO derivative flattens the physics latent to ``(B*13*16, H)`` and calls
        ``reflection_pad1d``, whose CUDA kernel maps that flat batch onto a grid
        dimension capped at 65535. The ``batched`` coupling feeds ``B*T = 768``
        samples (batch 64 x 12 frames), i.e. 159 744 rows — which aborts with
        "CUDA error: invalid configuration argument" (smoke job 4175599). The
        ceiling is 315 samples per call; the default 256 stays under it.

        The split is exact for the physics passthrough arms: the residual
        corrector and the diabatic head are plain conv+GELU stacks, and
        ``physics_only_forward`` is a pure PDE integrator — every sample is
        independent. The legacy arm is the one exception: its ``PDE_kernel``
        carries BatchNorm, so the chunk size sets the BN batch statistics. Hence
        the exp21 configs pin ``physics_chunk_size: 64``, the very batch the
        physics saw in exp16 — the legacy arm stays comparable with exp16's R1.

        Args:
            y_nn: backbone prediction ``(N, C, H, W)`` (normalized).
            prev: previous state ``(N, C, H, W)`` (normalized, already detached).

        Returns:
            ``torch.Tensor`` ``(N, C, H, W)`` — corrected prediction. Side effect:
            sets the mixin's aux loss and diagnostics to the size-weighted mean
            over chunks (the mixin overwrites both on every call).
        """
        n_samples = y_nn.shape[0]
        if n_samples <= self.physics_chunk_size:
            return self._apply_physics_residual(y_nn, prev)

        corrected_chunks = []
        aux_terms = []
        diagnostic_terms: list[tuple[int, dict[str, torch.Tensor]]] = []
        for start in range(0, n_samples, self.physics_chunk_size):
            stop = min(start + self.physics_chunk_size, n_samples)
            corrected_chunks.append(
                self._apply_physics_residual(y_nn[start:stop], prev[start:stop])
            )
            weight = stop - start
            chunk_aux = self.physics_residual_aux_loss()
            if chunk_aux is not None:
                aux_terms.append(weight * chunk_aux)
            diagnostic_terms.append((weight, self.physics_residual_diagnostics()))

        if aux_terms:
            self._last_residual_aux_loss = torch.stack(aux_terms).sum() / n_samples
        self._last_residual_diagnostics = {
            key: torch.stack([w * diag[key] for w, diag in diagnostic_terms]).sum() / n_samples
            for key in diagnostic_terms[0][1]
        }
        return torch.cat(corrected_chunks, dim=0)

    def _correct_batched(self, y_nn: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        """Correct every frame against the backbone's own previous frame, MIMO-style.

        Args:
            y_nn: backbone prediction ``(B, T, C, H, W)`` (normalized).
            x_raw: input clip ``(B, T_in, C, H, W)`` (normalized).

        Returns:
            ``torch.Tensor`` ``(B, T, C, H, W)`` — corrected prediction.
        """
        B, T, C, H, W = y_nn.shape
        prev = torch.cat([x_raw[:, -1:], y_nn[:, :-1]], dim=1).detach()
        corrected = self._apply_physics_chunked(
            y_nn.reshape(B * T, C, H, W), prev.reshape(B * T, C, H, W)
        )
        return corrected.view(B, T, C, H, W)

    def _correct_chained(self, y_nn: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        """Correct frames one by one, feeding each corrected frame forward.

        The aux penalty is averaged over the T calls: the mixin overwrites it on
        every ``_apply_physics_residual``, so without averaging only the last
        frame's penalty would reach the optimizer. The diagnostics are those of
        the last frame.

        Args:
            y_nn: backbone prediction ``(B, T, C, H, W)`` (normalized).
            x_raw: input clip ``(B, T_in, C, H, W)`` (normalized).

        Returns:
            ``torch.Tensor`` ``(B, T, C, H, W)`` — corrected prediction. Side
            effect: sets the mixin's aux loss to the mean over frames.
        """
        prev = x_raw[:, -1].detach()
        corrected_frames = []
        step_aux_losses = []
        for frame in range(y_nn.shape[1]):
            corrected = self._apply_physics_chunked(y_nn[:, frame], prev)
            step_aux = self.physics_residual_aux_loss()
            if step_aux is not None:
                step_aux_losses.append(step_aux)
            corrected_frames.append(corrected)
            prev = corrected.detach()
        if step_aux_losses:
            self._last_residual_aux_loss = torch.mean(torch.stack(step_aux_losses, dim=0))
        return torch.stack(corrected_frames, dim=1)

    def forward(self, x_raw: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run the SimVPv2 backbone, then correct its clip with the physics head.

        Args:
            x_raw: input clip ``(B, T, C, H, W)``, normalized.

        Returns:
            ``torch.Tensor`` ``(B, T, C, H, W)`` — the corrected forecast clip.
        """
        y_nn = super().forward(x_raw)
        if not self.use_physics_residual_corrector:
            return y_nn
        if self.physics_coupling == "batched":
            return self._correct_batched(y_nn, x_raw)
        return self._correct_chained(y_nn, x_raw)


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse:
        return list(reversed(samplings[:N]))
    else:
        return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super().__init__()
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0], act_inplace=act_inplace),
            *[
                ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s, act_inplace=act_inplace)
                for s in samplings[1:]
            ],
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super().__init__()
        self.dec = nn.Sequential(
            *[
                ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s, act_inplace=act_inplace)
                for s in samplings[:-1]
            ],
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1], act_inplace=act_inplace),
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(
        self,
        channel_in,
        channel_hid,
        N2,
        incep_ker: tuple[int, ...] = (3, 5, 7, 11),
        groups=8,
        **kwargs,
    ):
        super().__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [
            gInception_ST(
                channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups
            )
        ]
        for _ in range(1, N2 - 1):
            enc_layers.append(
                gInception_ST(
                    channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups
                )
            )
        enc_layers.append(
            gInception_ST(
                channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups
            )
        )
        dec_layers = [
            gInception_ST(
                channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups
            )
        ]
        for _ in range(1, N2 - 1):
            dec_layers.append(
                gInception_ST(
                    2 * channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups,
                )
            )
        dec_layers.append(
            gInception_ST(
                2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups
            )
        )

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2 - 1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(
        self,
        in_channels,
        out_channels,
        input_resolution=None,
        model_type=None,
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
        layer_i=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else "gsta"

        if model_type == "gsta":
            self.block = GASubBlock(
                in_channels,
                kernel_size=21,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                act_layer=nn.GELU,
            )
        elif model_type == "convmixer":
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == "convnext":
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path
            )
        elif model_type == "hornet":
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type in ["mlp", "mlpmixer"]:
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path
            )
        elif model_type in ["moga", "moganet"]:
            self.block = MogaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path
            )
        elif model_type == "poolformer":
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path
            )
        elif model_type == "swin":
            self.block = SwinSubBlock(
                in_channels,
                input_resolution,
                layer_i=layer_i,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
            )
        elif model_type == "uniformer":
            block_type = "MHSA" if in_channels == out_channels and layer_i > 0 else "Conv"
            self.block = UniformerSubBlock(
                in_channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                block_type=block_type,
            )
        elif model_type == "van":
            self.block = VANSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU
            )
        elif model_type == "vit":
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path
            )
        elif model_type == "tau":
            self.block = TAUSubBlock(
                in_channels,
                kernel_size=21,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                act_layer=nn.GELU,
            )
        else:
            raise ValueError(f"Invalid model_type {model_type!r} in MetaBlock")

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(
        self,
        channel_in,
        channel_hid,
        N2,
        input_resolution=None,
        model_type=None,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.1,
    ):
        super().__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)
        ]

        # downsample
        enc_layers = [
            MetaBlock(
                channel_in,
                channel_hid,
                input_resolution,
                model_type,
                mlp_ratio,
                drop,
                drop_path=dpr[0],
                layer_i=0,
            )
        ]
        # middle layers
        for i in range(1, N2 - 1):
            enc_layers.append(
                MetaBlock(
                    channel_hid,
                    channel_hid,
                    input_resolution,
                    model_type,
                    mlp_ratio,
                    drop,
                    drop_path=dpr[i],
                    layer_i=i,
                )
            )
        # upsample
        enc_layers.append(
            MetaBlock(
                channel_hid,
                channel_in,
                input_resolution,
                model_type,
                mlp_ratio,
                drop,
                drop_path=drop_path,
                layer_i=N2 - 1,
            )
        )
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y
