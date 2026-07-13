"""Architecture-agnostic physics residual path shared by the PI-* models.

The residual path works purely in the 69-channel state space
``(B, C_data, H, W)``: it denormalizes the previous state, evolves it with the
``HybridBlock`` primitive-equation kernel, builds a physics tendency feature
``delta_phys = physics(prev) - prev`` and lets a learned head correct the raw
network prediction with it. Nothing in here touches the internals of the host
model, so the same path serves both PI-IAM4VP (rollout driven by the strategy,
see ``training_strategies/iterative_manual.py``) and PI-PredRNNv2 (rollout
inside the model's own ``forward``).

``PhysicsResidualMixin`` is a **plain class, deliberately not an**
``nn.Module``: the host model already is one, and the mixin relies on the
host's ``register_buffer`` / ``named_parameters`` / attribute registration.
Mix it in *before* ``nn.Module`` in the host's bases, e.g.
``class IAM4VP(PhysicsResidualMixin, nn.Module)``, and call
:meth:`PhysicsResidualMixin.init_physics_residual` from the host ``__init__``
after ``super().__init__()``.

The mixin owns these attributes on the host (contract, do not rename — the
training strategies duck-type on them): ``C_data``, ``surface_channels``,
``upper_air_channels``, ``downscaling_factor_all``, ``hybrid_block``,
``physics_residual_corrector``, ``diabatic_head``,
``freeze_iam4vp_for_residual_warmup``, ``residual_warmup_epochs``.
"""

from __future__ import annotations

import warnings

import h5netcdf
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.physics_hybrid import (
    HybridBlock,
    PhysicsTendencyResidualCorrector,
    relative_to_specific_humidity,
    specific_to_relative_humidity,
)

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


class PhysicsResidualMixin:
    """Physics-prior residual correction, independent of the host architecture.

    Not an ``nn.Module``: the host model provides ``register_buffer``,
    ``parameters()`` and submodule registration. See the module docstring for
    the attribute contract and the expected base-class order.
    """

    def init_physics_residual(
        self,
        *,
        C_data: int,
        H_data: int,
        W_data: int,
        downscaling_factor_all: int,
        use_physics_residual_corrector: bool = False,
        physics_residual_hidden_channels: int = 128,
        physics_residual_apply_to: str = "upper_air_only",
        physics_residual_zero_init: bool = True,
        physics_residual_lambda_l1: float = 0.0,
        physics_feature_mode: str = "tendency",
        physics_residual_shuffle: str = "none",
        physics_residual_hybrid_steps: int = 3,
        physics_residual_hybrid_mode: str | None = None,
        physics_residual_input_space: str = "normalized",
        physics_residual_humidity_mode: str = "as_is",
        physics_residual_tendency_clip: float = 0.0,
        physics_w_diagnostic: str = "plain",
        physics_horizon_seconds: float = 3600.0,
        physics_lat_start_deg: float = -70.0,
        physics_dlat_deg: float = 20.0,
        physics_dlon_deg: float = 22.5,
        physics_coriolis_formulation: str = "spherical",
        physics_t_t_formulation: str = "adiabatic_omega",
        physics_use_universal_R: bool = False,
        physics_tendency_limiter: str = "physical_clip",
        physics_tendency_caps: dict[str, float] | None = None,
        physics_tendency_on_latent: bool = True,
        physics_prior_detach: bool = False,
        physics_advection_form: str = "flux",
        physics_metric_terms: bool = False,
        physics_spherical_divergence: bool = False,
        physics_rayleigh_friction: bool = False,
        physics_vertical_scheme: str = "stencil",
        physics_omega_free: tuple[str, ...] | list[str] = (),
        physics_latent_heating_coupling: bool = False,
        physics_clim_sources_path: str | None = None,
        physics_level_mask: dict[str, float] | None = None,
        physics_level_mask_learnable: bool = False,
        physics_ekman_K_profile: tuple[float, ...] | None = None,
        physics_humidity_evolution: str = "as_is",
        use_diabatic_term: bool = False,
        diabatic_hidden_channels: int = 64,
        diabatic_lambda_l1: float = 0.0,
        diabatic_constants_path: str | None = None,
        diabatic_cut: list[int] | None = None,
        diabatic_apply_to: str = "all_upper_air",
        freeze_iam4vp_for_residual_warmup: bool = False,
        residual_warmup_epochs: int = 0,
    ) -> None:
        """Build the physics branch on the host model (call from its ``__init__``).

        Registers the normalization/pressure buffers, validates every mode
        fail-fast, constructs the ``HybridBlock`` physics kernel, the residual
        corrector head and (optionally) the learned diabatic head ``Q_theta``,
        and installs the gradient sanitizer hook on the hybrid parameters.

        Call this **after** the host has built its own submodules: the hybrid
        block and the corrector consume the global RNG, so their construction
        order fixes the initialisation of every weight.

        Args:
            C_data: state channels (69: surface 0:4, then z/t/r/u/v blocks of 13).
            H_data: state height in pixels.
            W_data: state width in pixels.
            downscaling_factor_all: fine->latent spatial factor the physics
                kernel runs on; ``H_data // factor`` must be 8 (see
                :meth:`_physics_prior_from_state`).
            use_physics_residual_corrector: master switch of the residual path.
                When False no corrector is built and
                :meth:`_apply_physics_residual` is the identity.
            physics_residual_hidden_channels: width of the corrector head.
            physics_residual_apply_to: ``'upper_air_only'`` (correct channels
                4:69) or ``'all_channels'``.
            physics_residual_zero_init: zero-init the corrector output conv, so
                training starts from the unmodified network prediction.
            physics_residual_lambda_l1: L1 weight of the correction in the aux loss.
            physics_feature_mode: ``'tendency'``, ``'prior_and_tendency'`` or
                ``'no_physics'`` (control arm: head of the same capacity, but the
                physics feature is identically zero).
            physics_residual_shuffle: ``'none'`` or ``'batch'`` (shuffle control:
                rolls the physics prior across the batch).
            physics_residual_hybrid_steps: kernel calls per model step.
            physics_residual_hybrid_mode: ``'legacy_normalized'``,
                ``'stable_physical'`` or ``'stable_physical_v2'`` (pure
                primitive-equation passthrough). ``None`` derives it from
                ``physics_residual_input_space``.
            physics_residual_input_space: ``'normalized'`` or ``'physical'``;
                forced by the hybrid mode.
            physics_residual_humidity_mode: ``'as_is'`` or
                ``'relative_to_specific'`` (r -> q around the kernel).
            physics_residual_tendency_clip: symmetric clip of the normalized
                physics tendency; ``0`` disables it.
            physics_w_diagnostic: ``'plain'`` or ``'mass_consistent'`` vertical
                velocity diagnostic.
            physics_horizon_seconds: model step horizon, split across
                ``depth * hybrid_steps`` kernel calls.
            physics_lat_start_deg: latitude of the first (southern) grid row.
            physics_dlat_deg: latitude step of the data grid, degrees.
            physics_dlon_deg: longitude step of the data grid, degrees.
            physics_coriolis_formulation: ``'spherical'``, ``'beta_plane'`` or
                ``'constant'``.
            physics_t_t_formulation: ``'adiabatic_omega'`` or ``'legacy_paper'``.
            physics_use_universal_R: legacy universal gas constant instead of R_d.
            physics_tendency_limiter: ``'physical_clip'`` or ``'scale_diff'``.
            physics_tendency_caps: per-block caps for the limiter; ``None``
                picks the defaults for the resolved humidity mode.
            physics_tendency_on_latent: upsample only the latent physics
                increment (True) instead of the whole evolved state.
            physics_prior_detach: cut the task-loss gradient into the kernel.
            physics_advection_form: ``'flux'`` or ``'advective'``.
            physics_metric_terms: enable spherical metric terms in the momentum
                equations.
            physics_spherical_divergence: spherical divergence in the continuity
                equation.
            physics_rayleigh_friction: boundary-layer Rayleigh friction.
            physics_vertical_scheme: ``'stencil'`` or ``'lagrange3'``.
            physics_omega_free: blocks evolved with a free (diagnosed) omega.
            physics_latent_heating_coupling: couple latent heating to T.
            physics_clim_sources_path: ``.npz`` with the climatological Q1/Q2.
            physics_level_mask: per-block level masks for the physics tendency.
            physics_level_mask_learnable: make that mask learnable.
            physics_ekman_K_profile: per-level Ekman diffusivity profile.
            physics_humidity_evolution: humidity evolution mode of the kernel.
            use_diabatic_term: add the learned diabatic source ``Q_theta``.
            diabatic_hidden_channels: width of the ``Q_theta`` head.
            diabatic_lambda_l1: L1 weight of ``Q_theta`` in the aux loss.
            diabatic_constants_path: NetCDF with orography / lsm / lat2d.
            diabatic_cut: ``[lat0, lat1, lon0, lon1]`` crop of those constants.
            diabatic_apply_to: ``'all_upper_air'``, ``'t_and_q'`` or ``'t_only'``.
            freeze_iam4vp_for_residual_warmup: freeze the host backbone while the
                residual head warms up (read by the training strategy).
            residual_warmup_epochs: length of that warmup, in epochs.

        Raises:
            ValueError: on any invalid or mutually incompatible mode.

        Side effects:
            Registers buffers and submodules on the host; emits ``UserWarning``
            about the humidity-channel semantics and the legacy latent grid.
        """
        self.C_data = C_data
        if physics_w_diagnostic not in ("plain", "mass_consistent"):
            raise ValueError(
                "physics_w_diagnostic must be 'plain' or 'mass_consistent', "
                f"got {physics_w_diagnostic!r}"
            )
        self.physics_w_diagnostic = physics_w_diagnostic
        # exp 16 (порт exp 14/15): opt-in термы hybrid-ядра; валидация значений —
        # в PDE_kernel (fail-fast на конструкторе ниже). Дефолты бит-в-бит.
        self.physics_advection_form = physics_advection_form
        self.physics_metric_terms = physics_metric_terms
        self.physics_spherical_divergence = physics_spherical_divergence
        self.physics_rayleigh_friction = physics_rayleigh_friction
        self.physics_vertical_scheme = physics_vertical_scheme
        self.physics_omega_free = tuple(physics_omega_free)
        self.physics_latent_heating_coupling = physics_latent_heating_coupling
        self.physics_clim_sources_path = physics_clim_sources_path
        self.physics_level_mask = physics_level_mask
        self.physics_level_mask_learnable = physics_level_mask_learnable
        self.physics_ekman_K_profile = physics_ekman_K_profile
        self.physics_humidity_evolution = physics_humidity_evolution

        self.downscaling_factor_all = downscaling_factor_all

        self.use_physics_residual_corrector = use_physics_residual_corrector
        self.physics_residual_apply_to = physics_residual_apply_to
        self.physics_residual_lambda_l1 = float(physics_residual_lambda_l1)
        self.physics_feature_mode = physics_feature_mode
        self.physics_residual_shuffle = physics_residual_shuffle
        self.physics_residual_hybrid_steps = int(physics_residual_hybrid_steps)
        if physics_residual_hybrid_mode is None:
            physics_residual_hybrid_mode = (
                "stable_physical"
                if physics_residual_input_space == "physical"
                else "legacy_normalized"
            )
        self.physics_residual_hybrid_mode = physics_residual_hybrid_mode
        self.physics_residual_input_space = physics_residual_input_space
        self.physics_residual_humidity_mode = physics_residual_humidity_mode
        self.physics_residual_tendency_clip = float(physics_residual_tendency_clip or 0.0)
        self.physics_horizon_seconds = float(physics_horizon_seconds)
        if self.physics_horizon_seconds <= 0:
            raise ValueError(
                f"physics_horizon_seconds must be > 0, got {self.physics_horizon_seconds!r}"
            )
        self.physics_lat_start_deg = float(physics_lat_start_deg)
        self.physics_dlat_deg = float(physics_dlat_deg)
        self.physics_dlon_deg = float(physics_dlon_deg)
        self.physics_coriolis_formulation = physics_coriolis_formulation
        self.physics_t_t_formulation = physics_t_t_formulation
        self.physics_use_universal_R = bool(physics_use_universal_R)
        self.physics_tendency_limiter = physics_tendency_limiter
        self.physics_tendency_on_latent = bool(physics_tendency_on_latent)
        self.physics_prior_detach = bool(physics_prior_detach)
        self.freeze_iam4vp_for_residual_warmup = freeze_iam4vp_for_residual_warmup
        self.residual_warmup_epochs = int(residual_warmup_epochs)
        self._last_residual_aux_loss: torch.Tensor | None = None
        self._last_residual_diagnostics: dict[str, torch.Tensor] = {}
        self._last_physics_nonfinite_ratio: torch.Tensor | None = None
        self._last_physics_tendency_clip_ratio: torch.Tensor | None = None
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
        valid_hybrid_modes = {"legacy_normalized", "stable_physical", "stable_physical_v2"}
        if self.physics_residual_hybrid_mode not in valid_hybrid_modes:
            raise ValueError(
                "physics_residual_hybrid_mode must be one of "
                f"{sorted(valid_hybrid_modes)}, got {self.physics_residual_hybrid_mode!r}"
            )
        # stable_physical_v2 (defect-B fix): same physical-unit plumbing as
        # stable_physical (denormalize, r<->q, sanitize, tendency clip), but the
        # HybridBlock runs as a pure primitive-equation integrator on the clean
        # physical state — no variable_norm conv contamination, no norm_*, no
        # router. Set the passthrough flag consumed by _hybrid_block_forward and
        # the HybridBlock construction below.
        self._hybrid_physical_passthrough = (
            self.physics_residual_hybrid_mode == "stable_physical_v2"
        )
        if self.physics_residual_hybrid_mode == "legacy_normalized":
            self.physics_residual_input_space = "normalized"
            self.physics_residual_humidity_mode = "as_is"
        elif self.physics_residual_hybrid_mode in ("stable_physical", "stable_physical_v2"):
            self.physics_residual_input_space = "physical"
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

        # Build the physics prior AFTER the humidity/hybrid mode is resolved: the
        # q-cap and geometry depend on it. grid_h = H_data // 4 (patch factor 4),
        # block_dt splits the 1-hour horizon across depth*hybrid_steps kernel calls.
        if physics_tendency_caps is None:
            physics_tendency_caps = {"z": 500.0, "t": 5.0, "q": 5.0, "u": 10.0, "v": 10.0}
            if self.physics_residual_humidity_mode == "relative_to_specific":
                # Kernel then sees specific humidity q (kg/kg); cap in kg/kg.
                physics_tendency_caps["q"] = 2e-3
        self.physics_tendency_caps = physics_tendency_caps
        hybrid_depth = 3
        physics_block_dt = self.physics_horizon_seconds / (
            hybrid_depth * max(1, self.physics_residual_hybrid_steps)
        )
        self.hybrid_block = HybridBlock(
            dim=self.upper_air_channels,
            zquvtw_channel=13,
            depth=hybrid_depth,
            block_dt=physics_block_dt,
            inverse_time=False,
            physics_part_coef=0.5,
            w_diagnostic=self.physics_w_diagnostic,
            lat_start_deg=self.physics_lat_start_deg,
            dlat_deg=self.physics_dlat_deg,
            dlon_deg=self.physics_dlon_deg,
            grid_h=H_data // self.downscaling_factor_all,
            coriolis_formulation=self.physics_coriolis_formulation,
            t_t_formulation=self.physics_t_t_formulation,
            use_universal_R=self.physics_use_universal_R,
            tendency_limiter=self.physics_tendency_limiter,
            tendency_caps=self.physics_tendency_caps,
            physical_passthrough=self._hybrid_physical_passthrough,
            advection_form=self.physics_advection_form,
            metric_terms=self.physics_metric_terms,
            spherical_divergence=self.physics_spherical_divergence,
            rayleigh_friction=self.physics_rayleigh_friction,
            vertical_scheme=self.physics_vertical_scheme,
            omega_free=self.physics_omega_free,
            latent_heating_coupling=self.physics_latent_heating_coupling,
            clim_sources_path=self.physics_clim_sources_path,
            physics_level_mask=self.physics_level_mask,
            physics_level_mask_learnable=self.physics_level_mask_learnable,
            ekman_K_profile=self.physics_ekman_K_profile,
            humidity_evolution=self.physics_humidity_evolution,
        )
        # Optimizer-poisoning guard. When the hybrid forward produces масked
        # nonfinite values (physics_residual_nonfinite_ratio > 0), the raw
        # NaN/inf activations still reach the BatchNorm/Conv backward inside
        # HybridBlock, so those parameters receive NaN gradients even though
        # the loss itself is finite; one optimizer.step() then poisons the
        # weights permanently and the physics branch silently degrades to its
        # sanitize fallback. Sanitizing parameter gradients at the boundary
        # keeps healthy steps bit-identical (nan_to_num of a finite tensor is
        # the identity) and only zeroes the poisoned entries. The hook runs
        # before DDP gradient accumulation, so all ranks reduce sanitized
        # gradients. ``_hybrid_nonfinite_grad_count`` counts affected
        # parameters since the last diagnostics read (observability; the
        # count of backward N surfaces in the diagnostics of forward N+1).
        self._hybrid_nonfinite_grad_count: torch.Tensor | float = 0.0
        for hybrid_param in self.hybrid_block.parameters():
            hybrid_param.register_hook(self._sanitize_hybrid_param_grad)

        legacy_grid = (
            self.physics_lat_start_deg == -70.0
            and self.physics_dlat_deg == 20.0
            and self.physics_dlon_deg == 22.5
        )
        if use_physics_residual_corrector and physics_feature_mode != "no_physics" and legacy_grid:
            warnings.warn(
                "Residual physics runs on the legacy fake-global latent grid "
                "(lat -70..70, dlon 22.5): Coriolis and pixel spacing will not "
                "match a regional crop. Pass physics_lat_start_deg / "
                "physics_dlat_deg / physics_dlon_deg for the data cut "
                "(expected only for the legacy_hybrid regression arm).",
                UserWarning,
                stacklevel=2,
            )

        self.use_diabatic_term = bool(use_diabatic_term)
        self.diabatic_lambda_l1 = float(diabatic_lambda_l1)
        self.diabatic_head = None
        valid_diabatic_apply_to = {"all_upper_air", "t_and_q", "t_only"}
        if diabatic_apply_to not in valid_diabatic_apply_to:
            raise ValueError(
                "diabatic_apply_to must be one of "
                f"{sorted(valid_diabatic_apply_to)}, got {diabatic_apply_to!r}"
            )
        self.diabatic_apply_to = diabatic_apply_to
        if self.use_diabatic_term and not self.use_physics_residual_corrector:
            raise ValueError("use_diabatic_term=True requires use_physics_residual_corrector=True")
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
            if self.use_diabatic_term:
                geo = self._load_static_geo(diabatic_constants_path, diabatic_cut, H_data, W_data)
                self.register_buffer("diabatic_geo", geo)
                self.diabatic_head = PhysicsTendencyResidualCorrector(
                    in_channels=corrected_channels + geo.shape[1],
                    out_channels=corrected_channels,
                    hidden_channels=diabatic_hidden_channels,
                    zero_init=True,
                )
                surface_offset = (
                    0
                    if self.physics_residual_apply_to == "upper_air_only"
                    else self.surface_channels
                )
                self.register_buffer(
                    "diabatic_channel_mask",
                    self._build_diabatic_mask(
                        self.diabatic_apply_to, corrected_channels, surface_offset
                    ),
                )
            if self.physics_residual_humidity_mode == "relative_to_specific":
                warnings.warn(
                    "Physics residual-corrector mode treats the inherited "
                    "HybridBlock as a tendency feature generator. The physics "
                    "branch denormalizes to physical units, converts relative "
                    "humidity r -> specific humidity q before HybridBlock, and "
                    "converts q -> r before returning to model-normalized space.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "Physics residual-corrector mode treats the WeatherGFT "
                    "HybridBlock as a tendency feature generator, not as a trusted "
                    "forecast. WeatherBench channel group 30:43 is relative "
                    "humidity (r); the inherited HybridBlock equations name this "
                    "block q. Interpret humidity tendencies as learned features "
                    "unless a relative->specific humidity conversion is enabled.",
                    UserWarning,
                    stacklevel=2,
                )
            warnings.warn(
                "Physics residual channel layout: surface=0:4, z=4:17, "
                "t=17:30, r=30:43, u=43:56, v=56:69; physics hybrid_mode="
                f"{self.physics_residual_hybrid_mode}, input_space="
                f"{self.physics_residual_input_space}, humidity_mode="
                f"{self.physics_residual_humidity_mode}, tendency_clip="
                f"{self.physics_residual_tendency_clip:g}",
                UserWarning,
                stacklevel=2,
            )
        else:
            self.physics_residual_corrector = None

    def set_physics_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store per-channel stats for physical-unit residual physics.

        The trainer still normalizes batches before the model sees them. These
        buffers are used only inside the residual physics branch to temporarily
        denormalize the previous state before running HybridBlock features.

        Args:
            mean: 1-D ``torch.Tensor`` of shape ``(C_data,)``.
            std: 1-D ``torch.Tensor`` of shape ``(C_data,)``.

        Raises:
            ValueError: if the shapes mismatch or do not cover ``C_data``.
        """
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
            raise ValueError(
                "set_physics_normalization expects matching 1-D mean/std; "
                f"got {tuple(mean.shape)} and {tuple(std.shape)}"
            )
        if mean.numel() != self.C_data:
            raise ValueError(f"Expected {self.C_data} normalization channels, got {mean.numel()}")
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
                "Call model.set_physics_normalization(...) before training."
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
    def _nonfinite_ratio(x: torch.Tensor) -> torch.Tensor:
        return (~torch.isfinite(x)).float().mean()

    @staticmethod
    def _finite_or_fallback(x: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isfinite(x), x, fallback.expand_as(x))

    @staticmethod
    def _finite_clamp(
        x: torch.Tensor,
        min_value: float,
        max_value: float,
        fallback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if fallback is None:
            fallback = torch.zeros_like(x)
        x = torch.where(torch.isfinite(x), x, fallback.expand_as(x))
        return torch.clamp(x, min=min_value, max=max_value)

    def _sanitize_hybrid_param_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Zero nonfinite entries of a HybridBlock parameter gradient.

        Registered as a ``Tensor.register_hook`` on every ``hybrid_block``
        parameter (see ``init_physics_residual``). Finite gradients pass through
        unchanged; nonfinite entries are zeroed so one poisoned backward cannot
        turn the physics-branch weights into NaN via ``optimizer.step()``. Counts
        affected parameters into ``_hybrid_nonfinite_grad_count`` (a lazily
        created on-device scalar, read and reset by
        ``_apply_physics_residual`` diagnostics) without a host sync.

        Args:
            grad: raw gradient of one hybrid parameter, any shape.

        Returns:
            torch.Tensor of the same shape with nonfinite entries replaced by 0.
        """
        nonfinite_any = (~torch.isfinite(grad)).any()
        count = self._hybrid_nonfinite_grad_count
        if not isinstance(count, torch.Tensor) or count.device != grad.device:
            count = torch.zeros((), device=grad.device)
        self._hybrid_nonfinite_grad_count = count + nonfinite_any.float()
        return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    def _sanitize_physical_parts(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        humidity: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        *,
        humidity_is_specific: bool,
        fallback_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if fallback_parts is None:
            fallback_parts = (
                torch.zeros_like(z),
                torch.full_like(t, 273.15),
                torch.zeros_like(humidity),
                torch.zeros_like(u),
                torch.zeros_like(v),
            )
        fallback_z, fallback_t, fallback_humidity, fallback_u, fallback_v = fallback_parts
        z = self._finite_clamp(z, -10000.0, 250000.0, fallback_z)
        t = self._finite_clamp(t, 150.0, 350.0, fallback_t)
        if humidity_is_specific:
            humidity = self._finite_clamp(humidity, 0.0, 0.08, fallback_humidity)
        else:
            humidity = self._finite_clamp(humidity, 0.0, 150.0, fallback_humidity)
        u = self._finite_clamp(u, -150.0, 150.0, fallback_u)
        v = self._finite_clamp(v, -150.0, 150.0, fallback_v)
        return z, t, humidity, u, v

    def _sanitize_hybrid_latent_physical(
        self,
        x: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        x_cf = x.permute(0, 3, 1, 2)
        fallback_cf = fallback.permute(0, 3, 1, 2)
        z, t, humidity, u, v = x_cf.chunk(5, dim=1)
        fallback_parts = fallback_cf.chunk(5, dim=1)
        z, t, humidity, u, v = self._sanitize_physical_parts(
            z,
            t,
            humidity,
            u,
            v,
            humidity_is_specific=(self.physics_residual_humidity_mode == "relative_to_specific"),
            fallback_parts=fallback_parts,
        )
        return torch.cat([z, t, humidity, u, v], dim=1).permute(0, 2, 3, 1)

    def _clip_normalized_tendency(
        self,
        prior: torch.Tensor,
        prev_state: torch.Tensor,
    ) -> torch.Tensor:
        clip = self.physics_residual_tendency_clip
        if clip <= 0:
            self._last_physics_tendency_clip_ratio = torch.zeros((), device=prior.device)
            return prior
        tendency = prior - prev_state
        clipped = torch.clamp(tendency, min=-clip, max=clip)
        self._last_physics_tendency_clip_ratio = (clipped != tendency).float().mean().detach()
        return prev_state + clipped

    def _hybrid_block_forward(
        self,
        pred_phys: torch.Tensor,
        zquvtw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one HybridBlock step, forcing BatchNorm to eval in stable_physical.

        In ``stable_physical`` mode every ``nn.BatchNorm`` inside
        ``self.hybrid_block`` is forced to ``eval()`` for the call. Consequently
        their running stats stay frozen at initialisation (running mean 0,
        running var 1) forever, and BatchNorm acts purely as the learnable affine
        ``gamma * x + beta``. This invariant is load-bearing: the hybrid working
        space is in physical units, and eval-mode BN leaves those units intact.
        Any new code path that runs the block in train mode would let the running
        stats adapt to physical-unit activations and silently change the semantics
        of the physics prior.

        In ``stable_physical_v2`` (defect-B fix) the block runs as a pure
        primitive-equation integrator via ``physics_only_forward``: no
        BatchNorm, conv or router touches the state, so no eval-forcing is
        needed and both returned tensors are the clean physics-evolved state.

        Args:
            pred_phys: x-path latent ``(B, h, w, C)`` on the coarse grid.
            zquvtw: physics-path latent ``(B, h, w, C)`` on the coarse grid.

        Returns:
            Tuple ``(x, zquvtw)``, each ``torch.Tensor`` of shape ``(B, h, w, C)``.
        """
        if self._hybrid_physical_passthrough:
            evolved = self.hybrid_block.physics_only_forward(zquvtw)
            return evolved, evolved
        if self.physics_residual_hybrid_mode != "stable_physical":
            return self.hybrid_block(pred_phys, zquvtw)

        batch_norm_states = []
        for module in self.hybrid_block.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                batch_norm_states.append((module, module.training))
                module.eval()
        try:
            return self.hybrid_block(pred_phys, zquvtw)
        finally:
            for module, was_training in batch_norm_states:
                module.train(was_training)

    def _hybrid_bn_gamma_drift(self) -> torch.Tensor:
        """Mean ``|gamma - 1|`` over all BatchNorm affine weights in the block.

        Diagnostic only (reads parameters, runs no forward): tracks how far the
        HybridBlock BatchNorm affine scale has drifted from its identity
        initialisation. Cheap drift observability for the frozen-prior invariant.

        Returns:
            Scalar ``torch.Tensor`` (detached); zero if the block exposes no
            affine BatchNorm weight.
        """
        drifts = [
            (module.weight.detach() - 1.0).abs().mean()
            for module in self.hybrid_block.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.weight is not None
        ]
        if not drifts:
            return torch.zeros((), device=self.hybrid_block.router_weight.device)
        return torch.stack(drifts).mean()

    def x_to_zquvtw(self, x: torch.Tensor) -> torch.Tensor:
        """
        Преобразует входные данные x в формат zquvtw, пригодный для обработки через hybrid_block.

        Args:
            x: Входной тензор формы [B, C, H, W], где C - число каналов (обычно 65)

        Returns:
            zquvtw: Тензор формы [B, H//4, W//4, C] — пространственно понижающее
                преобразование и перестановка осей.
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

    @staticmethod
    def _load_static_geo(
        path: str | None,
        cut: list[int] | None,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Load + crop static geography (orography, |lat|, lsm) as (1, 3, H, W).

        E9'-geo diabatic inputs: standardized orography, |lat|/90, land-sea mask.
        Cropped to the region window ``cut=[lat0, lat1, lon0, lon1]`` on the
        native constants grid. These are the geographic drivers exp 05 flagged
        as the physics failure modes (mountains, tropics f->0).
        """
        if path is None:
            raise ValueError("use_diabatic_term=True requires diabatic_constants_path")
        if cut is None:
            cut = [75, 107, 164, 228]
        la0, la1, lo0, lo1 = cut
        with h5netcdf.File(path, "r") as f:
            orog = np.asarray(f.variables["orography"], dtype=np.float32)[la0:la1, lo0:lo1]
            lsm = np.asarray(f.variables["lsm"], dtype=np.float32)[la0:la1, lo0:lo1]
            lat2d = np.asarray(f.variables["lat2d"], dtype=np.float32)[la0:la1, lo0:lo1]
        if orog.shape != (H, W):
            raise ValueError(f"geo crop {orog.shape} != ({H},{W}); check diabatic_cut {cut}")
        orog_n = (orog - orog.mean()) / (orog.std() + 1e-6)
        abslat_n = np.abs(lat2d) / 90.0
        geo = np.stack([orog_n, abslat_n, lsm], axis=0)[None]
        return torch.from_numpy(geo).float()

    def _physics_prior_from_state(self, prev_state: torch.Tensor) -> torch.Tensor:
        """Run the inherited HybridBlock on the previous state.

        ``legacy_normalized`` keeps the old PI-IAM4VP residual behavior:
        normalized upper-air channels are sent directly to HybridBlock.
        ``stable_physical`` temporarily denormalizes, converts r<->q around the
        HybridBlock, clamps nonphysical values, then returns to normalized
        69-channel space before building the tendency.

        When ``physics_tendency_on_latent`` is True the evolved upper-air state is
        ``hybrid_input + up(pred_latent - state_latent)``: only the latent-grid
        physics increment is upsampled, so the prior differs from the input by
        physics alone rather than by the coarse<->fine resampling residual of the
        whole state. When False the legacy path upsamples the evolved latent state
        directly.

        Args:
            prev_state: normalized state ``torch.Tensor`` of shape ``(B, C, H, W)``
                with ``C = 69`` (surface 0:4, then z/t/r/u/v blocks of 13).

        Returns:
            ``torch.Tensor`` of shape ``(B, C, H, W)`` — the physics prior in the
            same normalized space as ``prev_state``.
        """
        _, _, H, W = prev_state.shape
        latent_h = H // self.downscaling_factor_all
        latent_w = W // self.downscaling_factor_all
        self._last_physics_nonfinite_ratio = torch.zeros((), device=prev_state.device)
        self._last_physics_tendency_clip_ratio = torch.zeros((), device=prev_state.device)
        if (latent_h, latent_w) != (8, 16):
            raise ValueError(
                "Physics HybridBlock has hardcoded derivative "
                f"geometry for an 8x16 latent grid, got {latent_h}x{latent_w}. "
                "Pass 32x64 crops or update utils.physics_hybrid geometry."
            )

        if self.physics_residual_hybrid_mode in ("stable_physical", "stable_physical_v2"):
            prev_physical = self._denormalize_state(prev_state)
            mean = self.physics_data_mean.to(device=prev_state.device, dtype=prev_state.dtype)
            prev_physical = self._finite_or_fallback(prev_physical, mean)
            z = prev_physical[:, 4:17]
            t = prev_physical[:, 17:30]
            humidity = prev_physical[:, 30:43]
            u = prev_physical[:, 43:56]
            v = prev_physical[:, 56:69]
            z, t, humidity, u, v = self._sanitize_physical_parts(
                z,
                t,
                humidity,
                u,
                v,
                humidity_is_specific=False,
            )
            if self.physics_residual_humidity_mode == "relative_to_specific":
                humidity = relative_to_specific_humidity(humidity, t, self.physics_pressure_pa)
                humidity = self._finite_clamp(humidity, 0.0, 0.08)
            hybrid_input = torch.cat([z, t, humidity, u, v], dim=1)
        else:
            prev_physical = None
            hybrid_input = prev_state[:, self.surface_channels :, :, :]
            hybrid_input = self._finite_or_fallback(hybrid_input, torch.zeros_like(hybrid_input))

        state_coarse = self.x_to_zquvtw(hybrid_input)
        pred_phys = state_coarse
        zquvtw = state_coarse
        for _ in range(self.physics_residual_hybrid_steps):
            pred_phys, zquvtw = self._hybrid_block_forward(pred_phys, zquvtw)
            self._last_physics_nonfinite_ratio = torch.maximum(
                self._last_physics_nonfinite_ratio,
                self._nonfinite_ratio(pred_phys).detach(),
            )
            if self.physics_residual_hybrid_mode in ("stable_physical", "stable_physical_v2"):
                # F14: reuse the once-computed coarse state as the sanitize fallback.
                pred_phys = self._sanitize_hybrid_latent_physical(pred_phys, state_coarse)
                zquvtw = self._sanitize_hybrid_latent_physical(zquvtw, state_coarse)
            else:
                pred_phys = self._finite_or_fallback(pred_phys, zquvtw)
                zquvtw = self._finite_or_fallback(zquvtw, pred_phys)

        if self.physics_tendency_on_latent:
            # F4: upsample only the latent physics increment and add it to the
            # working-space input, so the prior carries physics alone rather than
            # the coarse<->fine resampling residual of the whole state.
            evolved_upper = (pred_phys - state_coarse).permute(0, 3, 1, 2)
            evolved_upper = F.interpolate(evolved_upper, size=(H, W), mode="bilinear")
            evolved_upper = hybrid_input + evolved_upper
        else:
            evolved_upper = pred_phys.permute(0, 3, 1, 2)
            evolved_upper = F.interpolate(evolved_upper, size=(H, W), mode="bilinear")
        if self.physics_residual_hybrid_mode in ("stable_physical", "stable_physical_v2"):
            z_new, t_new, humidity_new, u_new, v_new = evolved_upper.chunk(5, dim=1)
            fallback_parts = (
                prev_physical[:, 4:17],
                prev_physical[:, 17:30],
                (
                    relative_to_specific_humidity(
                        prev_physical[:, 30:43],
                        prev_physical[:, 17:30],
                        self.physics_pressure_pa,
                    )
                    if self.physics_residual_humidity_mode == "relative_to_specific"
                    else prev_physical[:, 30:43]
                ),
                prev_physical[:, 43:56],
                prev_physical[:, 56:69],
            )
            z_new, t_new, humidity_new, u_new, v_new = self._sanitize_physical_parts(
                z_new,
                t_new,
                humidity_new,
                u_new,
                v_new,
                humidity_is_specific=(
                    self.physics_residual_humidity_mode == "relative_to_specific"
                ),
                fallback_parts=fallback_parts,
            )
            if self.physics_residual_humidity_mode == "relative_to_specific":
                humidity_new = specific_to_relative_humidity(
                    humidity_new, t_new, self.physics_pressure_pa
                )
                humidity_new = self._finite_clamp(
                    humidity_new,
                    0.0,
                    150.0,
                    prev_physical[:, 30:43],
                )
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
            prior = self._normalize_state(prior_physical)
            prior = self._finite_or_fallback(prior, prev_state)
            return self._clip_normalized_tendency(prior, prev_state)
        prior = torch.cat([prev_state[:, : self.surface_channels, :, :], evolved_upper], dim=1)
        return self._finite_or_fallback(prior, prev_state)

    @staticmethod
    def _build_diabatic_mask(
        apply_to: str,
        corrected_channels: int,
        surface_offset: int,
    ) -> torch.Tensor:
        """Channel mask restricting the diabatic head output (E9'/exp 10).

        Args:
            apply_to: ``'all_upper_air'`` (mask of ones), ``'t_and_q'``
                (nonzero only on the T block and the humidity block; the data
                humidity channel is relative humidity r, which the inherited
                HybridBlock equations name q) or ``'t_only'`` (nonzero only on
                the T block — exp16-long R3q: isolates the humidity tax of the
                learned diabatic source, see exp16 README §8.5-г).
            corrected_channels: number of channels the residual heads emit
                (65 for ``upper_air_only``, 69 for ``all_channels``).
            surface_offset: 0 when the corrected slice starts at upper-air
                channels, 4 when surface channels are included.

        Returns:
            torch.Tensor of shape ``(1, corrected_channels, 1, 1)`` with
            values in {0.0, 1.0}.
        """
        mask = torch.ones(1, corrected_channels, 1, 1, dtype=torch.float32)
        if apply_to == "t_and_q":
            mask = torch.zeros(1, corrected_channels, 1, 1, dtype=torch.float32)
            mask[:, surface_offset + 13 : surface_offset + 39] = 1.0
        elif apply_to == "t_only":
            mask = torch.zeros(1, corrected_channels, 1, 1, dtype=torch.float32)
            mask[:, surface_offset + 13 : surface_offset + 26] = 1.0
        return mask

    def _residual_slice(self, x: torch.Tensor) -> torch.Tensor:
        if self.physics_residual_apply_to == "upper_air_only":
            return x[:, self.surface_channels :, :, :]
        return x

    def _apply_physics_residual(
        self,
        y_nn: torch.Tensor,
        prev_state: torch.Tensor,
    ) -> torch.Tensor:
        """Add the learned physics-feature correction (and Q_theta) to y_nn.

        Semantics note: ``delta_phys = y_phys - prev_state`` is a **per-step
        state delta in normalized units** (sigma_x over the 1-hour
        autoregression step), not an SI tendency d/dt — no /dt factor is ever
        applied on the NN side, and the corrector output lives in the same
        units, so the additive convention is closed. Diagnostic keys keep the
        historical "tendency" naming for Comet panel continuity.

        Args:
            y_nn: raw network prediction ``(B, C, H, W)`` (normalized).
            prev_state: previous state ``(B, C, H, W)`` (normalized; the caller
                detaches it — the rollout strategy for PI-IAM4VP, the model's own
                loop for PI-PredRNNv2).

        Returns:
            torch.Tensor ``(B, C, H, W)`` — corrected prediction. Side
            effects: sets ``_last_residual_aux_loss`` and
            ``_last_residual_diagnostics``.
        """
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
        if self.physics_prior_detach:
            # F5: freeze the physics feature generator — no task-loss gradient into
            # the HybridBlock convs / BN affine / router through the corrector path.
            y_phys = y_phys.detach()

        delta_phys = self._finite_or_fallback(
            y_phys - prev_state,
            torch.zeros_like(prev_state),
        )
        y_nn_part = self._residual_slice(y_nn)
        prev_part = self._residual_slice(prev_state)
        parts = [y_nn_part, prev_part, y_nn_part - prev_part]

        if self.physics_feature_mode == "tendency":
            parts.append(self._residual_slice(delta_phys))
        elif self.physics_feature_mode == "prior_and_tendency":
            parts.extend([self._residual_slice(y_phys), self._residual_slice(delta_phys)])

        features = torch.cat(parts, dim=1)
        features = self._finite_or_fallback(features, torch.zeros_like(features))
        correction = self.physics_residual_corrector(features)
        correction = self._finite_or_fallback(correction, torch.zeros_like(correction))

        diabatic_term = None
        if self.diabatic_head is not None:
            st = self._residual_slice(prev_state)
            geo = self.diabatic_geo.to(dtype=st.dtype, device=st.device)
            geo = geo.expand(st.shape[0], -1, -1, -1)
            diabatic_term = self.diabatic_head(torch.cat([st, geo], dim=1))
            diabatic_term = self._finite_or_fallback(diabatic_term, torch.zeros_like(diabatic_term))
            diabatic_term = diabatic_term * self.diabatic_channel_mask.to(
                dtype=diabatic_term.dtype, device=diabatic_term.device
            )
            correction = correction + diabatic_term

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
        if diabatic_term is not None and self.diabatic_lambda_l1 > 0:
            self._last_residual_aux_loss = (
                self._last_residual_aux_loss + self.diabatic_lambda_l1 * diabatic_term.abs().mean()
            )

        correction_flat = correction_for_cosine.detach().flatten(1).float()
        tendency_flat = tendency_for_cosine.detach().flatten(1).float()
        cosine = F.cosine_similarity(correction_flat, tendency_flat, dim=1, eps=1e-8).mean()
        correction_rms = self._rms(full_correction.detach())
        y_nn_rms = self._rms(y_nn.detach())
        tendency_rms = self._rms(delta_phys.detach())
        nonfinite_ratio = (
            self._last_physics_nonfinite_ratio
            if self._last_physics_nonfinite_ratio is not None
            else torch.zeros((), device=y_nn.device)
        )
        tendency_clip_ratio = (
            self._last_physics_tendency_clip_ratio
            if self._last_physics_tendency_clip_ratio is not None
            else torch.zeros((), device=y_nn.device)
        )
        nonfinite_grad_count = self._hybrid_nonfinite_grad_count
        if not isinstance(nonfinite_grad_count, torch.Tensor):
            nonfinite_grad_count = torch.zeros((), device=y_nn.device)
        self._hybrid_nonfinite_grad_count = 0.0
        self._last_residual_diagnostics = {
            "physics_residual_correction_rms": correction_rms,
            "physics_residual_correction_to_prediction_ratio": correction_rms / (y_nn_rms + 1e-8),
            "physics_residual_tendency_rms": tendency_rms,
            "physics_residual_correction_to_tendency_cosine": cosine,
            "physics_residual_pi_minus_iam4vp_rms": self._rms((y_hat - y_nn).detach()),
            "physics_residual_nonfinite_ratio": nonfinite_ratio.detach(),
            "physics_residual_tendency_clip_ratio": tendency_clip_ratio.detach(),
            "physics_hybrid_nonfinite_grad_params": nonfinite_grad_count.detach(),
            "physics_diabatic_rms": (
                self._rms(diabatic_term.detach())
                if diabatic_term is not None
                else torch.zeros((), device=y_nn.device)
            ),
        }
        # Edge-contamination observability (audit claim on 8x16 WENO stencil
        # coverage): RMS of delta_phys on the one-latent-cell border ring vs
        # the interior. Ratio >> 1 flags boundary artefacts leaking into the
        # physics features on the crop.
        _, _, delta_h, delta_w = delta_phys.shape
        ring = self.downscaling_factor_all
        if delta_h > 2 * ring and delta_w > 2 * ring:
            delta_det = delta_phys.detach().float()
            total_sq = delta_det.square().sum()
            interior = delta_det[..., ring:-ring, ring:-ring]
            interior_sq = interior.square().sum()
            edge_count = delta_det.numel() - interior.numel()
            edge_rms = torch.sqrt((total_sq - interior_sq) / max(edge_count, 1))
            interior_rms = torch.sqrt(interior_sq / interior.numel())
            self._last_residual_diagnostics["physics_residual_delta_edge_interior_ratio"] = (
                edge_rms / (interior_rms + 1e-8)
            )
        if diabatic_term is not None:
            diabatic_det = diabatic_term.detach()
            block_offset = (
                0 if self.physics_residual_apply_to == "upper_air_only" else self.surface_channels
            )
            for block_name, block_index in (("z", 0), ("t", 1), ("r", 2), ("u", 3), ("v", 4)):
                start = block_offset + 13 * block_index
                self._last_residual_diagnostics[f"physics_diabatic_rms_{block_name}"] = self._rms(
                    diabatic_det[:, start : start + 13]
                )
        if self.physics_feature_mode != "no_physics":
            self._last_residual_diagnostics["physics_router_weight_abs"] = (
                self.hybrid_block.router_weight.detach().abs().mean()
            )
            self._last_residual_diagnostics["physics_hybrid_bn_gamma_drift"] = (
                self._hybrid_bn_gamma_drift()
            )
        return y_hat

    def physics_residual_aux_loss(self) -> torch.Tensor | None:
        """Return the aux penalty of the last ``_apply_physics_residual`` call.

        Returns:
            Scalar ``torch.Tensor`` (L1 of the correction, plus the diabatic L1
            when enabled) or ``None`` if the residual path is off.
        """
        return self._last_residual_aux_loss

    def physics_residual_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return a copy of the diagnostics of the last residual application.

        Returns:
            Mapping from Comet metric name to a scalar ``torch.Tensor``.
        """
        return dict(self._last_residual_diagnostics)

    def set_residual_warmup(self, active: bool) -> None:
        """Optionally freeze the host backbone while training residual/physics modules.

        The HybridBlock is kept trainable together with the residual head because
        it contains learned convolutions; freezing it at random initialisation
        would turn the physics features into mostly arbitrary tensors.

        Args:
            active: True freezes every parameter outside
                ``physics_residual_corrector`` / ``hybrid_block``; False unfreezes
                the whole model.
        """
        if not self.use_physics_residual_corrector:
            return
        trainable_prefixes = ("physics_residual_corrector", "hybrid_block")
        for name, param in self.named_parameters():
            param.requires_grad = (not active) or name.startswith(trainable_prefixes)
