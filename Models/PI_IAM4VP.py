import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from .imvp_modules import Attention, CircularConvSC, ConvNeXt_block, ConvNeXt_bottle
from .PredFormerGFT_HybridBlock import HybridBlock


PRESSURE_LEVELS_HPA: tuple[int, ...] = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Time_MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sinusoidaposemb = SinusoidalPosEmb(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.sinusoidaposemb(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super().__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            CircularConvSC(C_in, C_hid, stride=strides[0]),
            *[CircularConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
        )

    def forward(self, x):  # B*10, 2, 32, 64
        enc1 = self.enc[0](x)
        latent = enc1

        latent_1 = self.enc[1](latent)
        latent_2 = self.enc[2](latent_1)
        latent_3 = self.enc[3](latent_2)

        return latent_3, enc1, latent_1, latent_2


class LP(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super().__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            CircularConvSC(C_in, C_hid, stride=strides[0]),
            *[CircularConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
        )

    def forward(self, x):  # B*10, 2, 32, 64
        enc1 = self.enc[0](x)
        latent = enc1

        latent_1 = self.enc[1](latent)
        latent_2 = self.enc[2](latent_1)
        latent_3 = self.enc[3](latent_2)

        return latent_3, enc1, latent_1, latent_2


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, T):
        super().__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[CircularConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            CircularConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True),
        )
        self.readout = nn.Conv2d(64 * T, 64, 1)

    def forward(self, hid, enc1, latent_1, latent_2, latent_3, T=10, H=8, W=16):

        hid = self.dec[0](hid + latent_3)
        hid = self.dec[1](hid + latent_2)
        hid = self.dec[2](hid + latent_1)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        ys = Y.shape
        Y = Y.reshape(int(ys[0] / T), int(ys[1] * T), H, W)
        Y = self.readout(Y)
        return Y


class Predictor(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T):
        super().__init__()

        self.N_T = N_T
        st_block = [ConvNeXt_bottle(dim=channel_in)]
        for i in range(0, N_T):
            st_block.append(ConvNeXt_block(dim=channel_in))

        self.st_block = nn.Sequential(*st_block)

    def forward(self, x, time_emb):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        z = self.st_block[0](x, time_emb)
        for i in range(1, self.N_T):
            z = self.st_block[i](z, time_emb)

        y = z.reshape(B, int(T / 2), C, H, W)
        return y


class PhysicsTendencyResidualCorrector(nn.Module):
    """Small zero-start residual head for physics-derived tendency features.

    This module treats the WeatherGFT/HybridBlock branch as a feature generator,
    not as a trusted forecast. With zero initialisation the final convolution
    emits exactly zero at step 0, so enabling the experiment starts from the
    plain IAM4VP prediction and learns only if the features help.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        if zero_init:
            final = self.net[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class IAM4VP(nn.Module):
    """Iterative Auto-regressive Model for Video Prediction + опциональный physics-prior.

    На каждом шаге `t` модель видит `(x_raw, pred_list, t)` — начальное
    состояние, накопленный список prev-predictions и timestep-эмбеддинг.
    Backward сделан per-step снаружи (см. `IterativeManualStep` в стратегиях).

    При `use_physics=True` параллельный путь через `HybridBlock` добавляет
    в выход physics-step из `Models.PredFormerGFT_HybridBlock`.

    Args:
        T_data, C_data, H_data, W_data: длина клипа и shape кадра.
        hid_S, hid_T, N_S, N_T: размерности и глубины encoder/predictor/decoder.
        use_physics: если True, выход — `AI + physics_correction`.
    """

    def __init__(
        self,
        T_data=6,
        C_data=69,
        H_data=32,
        W_data=64,
        hid_S=64,
        hid_T=256,
        N_S=4,
        N_T=6,
        use_physics=True,
        use_physics_residual_corrector=False,
        physics_residual_hidden_channels=128,
        physics_residual_apply_to="upper_air_only",
        physics_residual_zero_init=True,
        physics_residual_lambda_l1=0.0,
        physics_feature_mode="tendency",
        physics_residual_shuffle="none",
        physics_residual_hybrid_steps=3,
        physics_residual_input_space="normalized",
        physics_residual_humidity_mode="as_is",
        freeze_iam4vp_for_residual_warmup=False,
        residual_warmup_epochs=0,
    ):
        super().__init__()
        self.C_data = C_data
        self.time_mlp = Time_MLP(dim=hid_S)
        self.enc = Encoder(C_data, hid_S, N_S)
        self.hid = Predictor(T_data * hid_S, hid_T, N_T)
        self.dec = Decoder(hid_S, C_data, N_S, T_data)
        self.attn = Attention(hid_S)
        self.readout = nn.Conv2d(hid_S, C_data, 1)
        self.mask_token = nn.Parameter(
            torch.zeros(T_data, hid_S, H_data // 4, W_data // 4)
        )  # for 1_4 and 5_6
        self.lp = LP(C_data, hid_S, N_S)
        self.lp_phys = LP(C_data, hid_S, N_S)
        self.hybrid_block = HybridBlock(
            dim=C_data - 4,
            zquvtw_channel=13,
            depth=3,
            block_dt=1200,
            inverse_time=False,
            physics_part_coef=0.5,
        )

        self.skip_mask_token = nn.Parameter(torch.zeros(T_data, hid_S, H_data, W_data))
        self.embed_1_mask_token = nn.Parameter(torch.zeros(T_data, hid_S, H_data // 2, W_data // 2))
        self.embed_2_mask_token = nn.Parameter(torch.zeros(T_data, hid_S, H_data // 2, W_data // 2))
        self.downscaling_factor_all = 4

        self.use_physics = use_physics
        self.use_physics_residual_corrector = use_physics_residual_corrector
        self.physics_residual_apply_to = physics_residual_apply_to
        self.physics_residual_lambda_l1 = float(physics_residual_lambda_l1)
        self.physics_feature_mode = physics_feature_mode
        self.physics_residual_shuffle = physics_residual_shuffle
        self.physics_residual_hybrid_steps = int(physics_residual_hybrid_steps)
        self.physics_residual_input_space = physics_residual_input_space
        self.physics_residual_humidity_mode = physics_residual_humidity_mode
        self.freeze_iam4vp_for_residual_warmup = freeze_iam4vp_for_residual_warmup
        self.residual_warmup_epochs = int(residual_warmup_epochs)
        self._last_residual_aux_loss: torch.Tensor | None = None
        self._last_residual_diagnostics: dict[str, torch.Tensor] = {}
        self.register_buffer(
            "physics_data_mean",
            torch.zeros(1, C_data, 1, 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "physics_data_std",
            torch.ones(1, C_data, 1, 1, dtype=torch.float32),
            persistent=False,
        )
        self._physics_normalization_ready = False
        self.register_buffer(
            "physics_pressure_pa",
            torch.tensor(PRESSURE_LEVELS_HPA, dtype=torch.float32).view(1, 13, 1, 1) * 100.0,
            persistent=False,
        )

        valid_apply_to = {"upper_air_only", "all_channels"}
        if self.physics_residual_apply_to not in valid_apply_to:
            raise ValueError(
                "physics_residual_apply_to must be one of "
                f"{sorted(valid_apply_to)}, got {self.physics_residual_apply_to!r}"
            )
        valid_feature_modes = {"tendency", "prior_and_tendency", "no_physics"}
        if self.physics_feature_mode not in valid_feature_modes:
            raise ValueError(
                "physics_feature_mode must be one of "
                f"{sorted(valid_feature_modes)}, got {self.physics_feature_mode!r}"
            )
        valid_shuffle_modes = {"none", "batch"}
        if self.physics_residual_shuffle not in valid_shuffle_modes:
            raise ValueError(
                "physics_residual_shuffle must be one of "
                f"{sorted(valid_shuffle_modes)}, got {self.physics_residual_shuffle!r}"
            )
        valid_input_spaces = {"normalized", "physical"}
        if self.physics_residual_input_space not in valid_input_spaces:
            raise ValueError(
                "physics_residual_input_space must be one of "
                f"{sorted(valid_input_spaces)}, got {self.physics_residual_input_space!r}"
            )
        valid_humidity_modes = {"as_is", "relative_to_specific"}
        if self.physics_residual_humidity_mode not in valid_humidity_modes:
            raise ValueError(
                "physics_residual_humidity_mode must be one of "
                f"{sorted(valid_humidity_modes)}, got {self.physics_residual_humidity_mode!r}"
            )
        if (
            self.physics_residual_humidity_mode == "relative_to_specific"
            and self.physics_residual_input_space != "physical"
        ):
            raise ValueError(
                "physics_residual_humidity_mode='relative_to_specific' requires "
                "physics_residual_input_space='physical'"
            )

        self.surface_channels = 4
        self.upper_air_channels = C_data - self.surface_channels
        if self.use_physics_residual_corrector:
            corrected_channels = (
                self.upper_air_channels
                if self.physics_residual_apply_to == "upper_air_only"
                else C_data
            )
            feature_blocks = 3
            if self.physics_feature_mode == "tendency":
                feature_blocks += 1
            elif self.physics_feature_mode == "prior_and_tendency":
                feature_blocks += 2
            self.physics_residual_corrector = PhysicsTendencyResidualCorrector(
                in_channels=corrected_channels * feature_blocks,
                out_channels=corrected_channels,
                hidden_channels=physics_residual_hidden_channels,
                zero_init=physics_residual_zero_init,
            )
            if self.physics_residual_humidity_mode == "relative_to_specific":
                warnings.warn(
                    "PI-IAM4VP residual-corrector mode treats the inherited "
                    "HybridBlock as a tendency feature generator. The physics "
                    "branch denormalizes to physical units, converts relative "
                    "humidity r -> specific humidity q before HybridBlock, and "
                    "converts q -> r before returning to model-normalized space.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "PI-IAM4VP residual-corrector mode treats the WeatherGFT "
                    "HybridBlock as a tendency feature generator, not as a trusted "
                    "forecast. WeatherBench channel group 30:43 is relative "
                    "humidity (r); the inherited HybridBlock equations name this "
                    "block q. Interpret humidity tendencies as learned features "
                    "unless a relative->specific humidity conversion is enabled.",
                    UserWarning,
                    stacklevel=2,
                )
            print(
                "[PI-IAM4VP residual] channel layout: "
                "surface=0:4, z=4:17, t=17:30, r=30:43, u=43:56, v=56:69"
            )
            print(
                "[PI-IAM4VP residual] physics input_space="
                f"{self.physics_residual_input_space}, humidity_mode="
                f"{self.physics_residual_humidity_mode}"
            )
        else:
            self.physics_residual_corrector = None

    def set_physics_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store per-channel stats for physical-unit residual physics.

        The trainer still normalizes batches before IAM4VP sees them. These
        buffers are used only inside the residual physics branch to temporarily
        denormalize the previous state before running HybridBlock features.
        """
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
            raise ValueError(
                "set_physics_normalization expects matching 1-D mean/std; "
                f"got {tuple(mean.shape)} and {tuple(std.shape)}"
            )
        if mean.numel() != self.C_data:
            raise ValueError(
                f"Expected {self.C_data} normalization channels, got {mean.numel()}"
            )
        device = next(self.parameters()).device
        self.physics_data_mean = mean.view(1, -1, 1, 1).to(device=device)
        self.physics_data_std = std.view(1, -1, 1, 1).to(device=device)
        self._physics_normalization_ready = True

    def _require_physics_normalization(self) -> None:
        if (
            not self._physics_normalization_ready
            or self.physics_data_mean.numel() != self.C_data
            or self.physics_data_std.numel() != self.C_data
        ):
            raise RuntimeError(
                "physics_residual_input_space='physical' requires dataset mean/std. "
                "Call IAM4VP.set_physics_normalization(...) before training."
            )

    def _denormalize_state(self, x: torch.Tensor) -> torch.Tensor:
        self._require_physics_normalization()
        mean = self.physics_data_mean.to(device=x.device, dtype=x.dtype)
        std = self.physics_data_std.to(device=x.device, dtype=x.dtype)
        return x * std + mean

    def _normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        self._require_physics_normalization()
        mean = self.physics_data_mean.to(device=x.device, dtype=x.dtype)
        std = self.physics_data_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    @staticmethod
    def _avoid_small_abs(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        sign = torch.sign(x)
        sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
        return torch.where(torch.abs(x) < threshold, sign * threshold, x)

    def _saturation_specific_humidity(self, t_kelvin: torch.Tensor) -> torch.Tensor:
        """Magnus saturation specific humidity q_s(T, p) in kg/kg."""
        pressure = self.physics_pressure_pa.to(device=t_kelvin.device, dtype=t_kelvin.dtype)
        pressure = pressure.expand_as(t_kelvin)
        t_c = t_kelvin - 273.15
        exponent = 17.67 * t_c / self._avoid_small_abs(t_c + 243.5)
        # Keeps pathological early predictions from producing inf before the
        # residual head has learned; ERA5 temperatures sit comfortably inside.
        exponent = torch.clamp(exponent, min=-20.0, max=20.0)
        e_s = 611.2 * torch.exp(exponent)
        denom = self._avoid_small_abs(pressure - 0.378 * e_s)
        return torch.clamp(0.622 * e_s / denom, min=1e-8)

    def _relative_to_specific_humidity(
        self,
        r_percent: torch.Tensor,
        t_kelvin: torch.Tensor,
    ) -> torch.Tensor:
        return (r_percent / 100.0) * self._saturation_specific_humidity(t_kelvin)

    def _specific_to_relative_humidity(
        self,
        q: torch.Tensor,
        t_kelvin: torch.Tensor,
    ) -> torch.Tensor:
        return 100.0 * q / self._saturation_specific_humidity(t_kelvin)

    def x_to_zquvtw(self, x):
        """
        Преобразует входные данные x в формат zquvtw, пригодный для обработки через hybrid_block.

        Args:
            x: Входной тензор формы [B, C, H, W], где C - число каналов (обычно 65)

        Returns:
            zquvtw: Тензор формы [B, H//4, W//4, C] - пространственно понижающее преобразование и перестановка осей
        """
        # x имеет форму [B, C, H, W]
        B, C, H, W = x.shape

        # Понижающая дискретизация для уменьшения размера пространственных координат
        zquvtw = torch.nn.functional.interpolate(
            x,
            size=(H // self.downscaling_factor_all, W // self.downscaling_factor_all),
            mode="bilinear",
        )

        # Перестановка осей для формата [B, H, W, C], который ожидает HybridBlock
        zquvtw = zquvtw.permute(0, 2, 3, 1)  # [B, H//4, W//4, C]

        return zquvtw

    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(x.float() * x.float()))

    def _physics_prior_from_state(self, prev_state: torch.Tensor) -> torch.Tensor:
        """Run the inherited HybridBlock on the previous state.

        By default this keeps the legacy normalized-space behavior used by the
        old PI-IAM4VP/WeatherGFT-style code. When
        ``physics_residual_input_space='physical'`` it temporarily
        denormalizes, converts r<->q around the HybridBlock, then returns to
        the model's normalized 69-channel layout before building the tendency.
        """
        _, _, H, W = prev_state.shape
        latent_h = H // self.downscaling_factor_all
        latent_w = W // self.downscaling_factor_all
        if (latent_h, latent_w) != (8, 16):
            raise ValueError(
                "PI-IAM4VP HybridBlock currently has hardcoded derivative "
                f"geometry for an 8x16 latent grid, got {latent_h}x{latent_w}. "
                "Pass 32x64 crops or update PredFormerGFT_HybridBlock geometry."
            )

        if self.physics_residual_input_space == "physical":
            prev_physical = self._denormalize_state(prev_state)
            z = prev_physical[:, 4:17]
            t = prev_physical[:, 17:30]
            humidity = prev_physical[:, 30:43]
            u = prev_physical[:, 43:56]
            v = prev_physical[:, 56:69]
            if self.physics_residual_humidity_mode == "relative_to_specific":
                humidity = self._relative_to_specific_humidity(humidity, t)
            hybrid_input = torch.cat([z, t, humidity, u, v], dim=1)
        else:
            prev_physical = None
            hybrid_input = prev_state[:, self.surface_channels :, :, :]

        pred_phys = self.x_to_zquvtw(hybrid_input)
        zquvtw = pred_phys
        for _ in range(self.physics_residual_hybrid_steps):
            pred_phys, zquvtw = self.hybrid_block(pred_phys, zquvtw)

        pred_phys = pred_phys.permute(0, 3, 1, 2)
        pred_phys = F.interpolate(pred_phys, size=(H, W), mode="bilinear")
        if self.physics_residual_input_space == "physical":
            z_new, t_new, humidity_new, u_new, v_new = pred_phys.chunk(5, dim=1)
            if self.physics_residual_humidity_mode == "relative_to_specific":
                humidity_new = self._specific_to_relative_humidity(humidity_new, t_new)
            prior_physical = torch.cat(
                [
                    prev_physical[:, : self.surface_channels],
                    z_new,
                    t_new,
                    humidity_new,
                    u_new,
                    v_new,
                ],
                dim=1,
            )
            return self._normalize_state(prior_physical)
        return torch.cat([prev_state[:, : self.surface_channels, :, :], pred_phys], dim=1)

    def _residual_slice(self, x: torch.Tensor) -> torch.Tensor:
        if self.physics_residual_apply_to == "upper_air_only":
            return x[:, self.surface_channels :, :, :]
        return x

    def _apply_physics_residual(
        self,
        y_nn: torch.Tensor,
        prev_state: torch.Tensor,
    ) -> torch.Tensor:
        if self.physics_residual_corrector is None:
            self._last_residual_aux_loss = None
            self._last_residual_diagnostics = {}
            return y_nn

        if self.physics_feature_mode == "no_physics":
            y_phys = prev_state
        else:
            y_phys = self._physics_prior_from_state(prev_state)
            if self.physics_residual_shuffle == "batch" and y_phys.shape[0] > 1:
                y_phys = torch.roll(y_phys, shifts=1, dims=0)

        delta_phys = y_phys - prev_state
        y_nn_part = self._residual_slice(y_nn)
        prev_part = self._residual_slice(prev_state)
        parts = [y_nn_part, prev_part, y_nn_part - prev_part]

        if self.physics_feature_mode == "tendency":
            parts.append(self._residual_slice(delta_phys))
        elif self.physics_feature_mode == "prior_and_tendency":
            parts.extend([self._residual_slice(y_phys), self._residual_slice(delta_phys)])

        features = torch.cat(parts, dim=1)
        correction = self.physics_residual_corrector(features)

        if self.physics_residual_apply_to == "upper_air_only":
            y_hat = torch.cat(
                [
                    y_nn[:, : self.surface_channels, :, :],
                    y_nn[:, self.surface_channels :, :, :] + correction,
                ],
                dim=1,
            )
            full_correction = torch.cat(
                [torch.zeros_like(y_nn[:, : self.surface_channels, :, :]), correction],
                dim=1,
            )
            correction_for_cosine = correction
            tendency_for_cosine = delta_phys[:, self.surface_channels :, :, :]
        else:
            y_hat = y_nn + correction
            full_correction = correction
            correction_for_cosine = correction
            tendency_for_cosine = delta_phys

        self._last_residual_aux_loss = (
            self.physics_residual_lambda_l1 * full_correction.abs().mean()
        )

        correction_flat = correction_for_cosine.detach().flatten(1).float()
        tendency_flat = tendency_for_cosine.detach().flatten(1).float()
        cosine = F.cosine_similarity(correction_flat, tendency_flat, dim=1, eps=1e-8).mean()
        correction_rms = self._rms(full_correction.detach())
        y_nn_rms = self._rms(y_nn.detach())
        tendency_rms = self._rms(delta_phys.detach())
        self._last_residual_diagnostics = {
            "physics_residual_correction_rms": correction_rms,
            "physics_residual_correction_to_prediction_ratio": correction_rms
            / (y_nn_rms + 1e-8),
            "physics_residual_tendency_rms": tendency_rms,
            "physics_residual_correction_to_tendency_cosine": cosine,
            "physics_residual_pi_minus_iam4vp_rms": self._rms((y_hat - y_nn).detach()),
        }
        return y_hat

    def physics_residual_aux_loss(self) -> torch.Tensor | None:
        return self._last_residual_aux_loss

    def physics_residual_diagnostics(self) -> dict[str, torch.Tensor]:
        return dict(self._last_residual_diagnostics)

    def set_residual_warmup(self, active: bool) -> None:
        """Optionally freeze IAM4VP while training residual/physics modules.

        The HybridBlock is kept trainable together with the residual head because
        it contains learned convolutions; freezing it at random initialisation
        would turn the physics features into mostly arbitrary tensors.
        """
        if not self.use_physics_residual_corrector:
            return
        trainable_prefixes = ("physics_residual_corrector", "hybrid_block")
        for name, param in self.named_parameters():
            param.requires_grad = (not active) or name.startswith(trainable_prefixes)

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

                for j in range(3):
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
