"""Snapshot всех текущих физических интеграторов в репо — frozen baseline.

Этот модуль НЕ предназначен для нового кода. Это zero-change снимок физики
из 5 разных файлов на момент 2026-05-14, сделанный перед миграцией на
:mod:`utils.physics`. Каждая копия живёт в своём подпространстве имён,
повторяя оригинал построчно (включая module-globals, комментарии и баги).

Назначение:
  * фиксация старого поведения для регрессии;
  * сравнение «old vs new» в бейзлайнах на ERA5 1.4° глобус 2000-2010;
  * возможность откатить семантику, если новая `utils.physics` сломает что-то.

Source files (на момент snapshot, branch `refactoring`, commit `daa3fb7`):
  * Models/WeatherGFT.py
  * Models/WeatherGFTSingle.py
  * Models/PredFormerGFT.py
  * Models/PredFormerGFT_HybridBlock.py
  * Models/dev/WeatherGFT_3.py
  * Models/dev/PredFormerGFTSmallWorld.py

Каждая копия выставлена под dataclass-неймспейсом (см. словарь `SOURCES`),
поэтому импорт не имеет побочных эффектов (никаких module-globals, всё через
factory-функции).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# =============================================================================
# Helpers: построение module-global констант под заданный latents_size
# =============================================================================


@dataclass(frozen=True)
class _GridSnapshot:
    """Срез module-global констант из исходного файла (под фикс. latents_size)."""

    latents_size: tuple[int, int]
    radius: float
    latitudes: torch.Tensor  # (H,)
    c_lats: torch.Tensor  # (1, 1, H, 1)
    pixel_x: torch.Tensor  # (1, 1, H, 1)
    pixel_y: float
    pressure: torch.Tensor  # (1, 13, 1, 1)
    pixel_z: torch.Tensor  # (1, 13, 1, 1)
    M_z: torch.Tensor  # (13, 13)


def _build_grid_weathergft_style(latents_size: tuple[int, int]) -> _GridSnapshot:
    """Точное воспроизведение module-global блока из Models/WeatherGFT.py:14-37.

    Используется во всех файлах WeatherGFT*, dev/WeatherGFT_3.py, dev/WeatherGFT_2,
    dev/WeatherGFT.py (с разными значениями `latents_size`).
    """
    radius = 6371.0 * 1000
    num_lat = latents_size[0] + 2
    lat_t = torch.arange(start=0, end=num_lat)
    # def lat(j, num_lat): return 90 - j * 180 / (num_lat - 1)  (WeatherGFT.py:11-12)
    latitudes = (90.0 - lat_t.float() * 180.0 / float(num_lat - 1))[1:-1]
    latitudes = latitudes / 180.0 * torch.pi

    c_lats = 2 * torch.pi * radius * torch.cos(latitudes)
    c_lats = c_lats.reshape([1, 1, latents_size[0], 1])
    pixel_x = c_lats / latents_size[1]
    pixel_y = float(torch.pi * radius / (latents_size[0] + 1))

    pressure = torch.tensor(
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    ).reshape([1, 13, 1, 1])
    pixel_z = torch.tensor([50, 50, 50, 50, 50, 75, 100, 100, 100, 125, 112, 75, 75]).reshape(
        [1, 13, 1, 1]
    )

    pressure_level_num = pixel_z.shape[1]
    M_z = torch.zeros(pressure_level_num, pressure_level_num)
    for M_z_i in range(pressure_level_num):
        for M_z_j in range(pressure_level_num):
            if M_z_i <= M_z_j:
                M_z[M_z_i, M_z_j] = pixel_z[0, M_z_j, 0, 0]

    return _GridSnapshot(
        latents_size=latents_size,
        radius=radius,
        latitudes=latitudes,
        c_lats=c_lats,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        pressure=pressure,
        pixel_z=pixel_z,
        M_z=M_z,
    )


def _build_grid_predformergft_style(latents_size: tuple[int, int]) -> _GridSnapshot:
    """Воспроизведение module-global блока из Models/PredFormerGFT.py:16-40.

    Отличие от WeatherGFT-стиля: используется `torch.linspace(-90, 90, num_lat)`
    вместо `90 - j * 180/(num_lat-1)`. Для num_lat = 10 (latents_size[0]=8)
    результат численно тот же (равномерное разбиение), но переопределение
    функции `lat` внутри файла на L21-22 фактически перетирает jit-script-
    версию с L12-13. См. PredFormerGFT.py:21-23.
    """
    radius = 6371.0 * 1000
    num_lat = latents_size[0] + 2
    # PredFormerGFT.py:21-23 переопределяет lat:
    #   def lat(lat_t, num_lat): return torch.linspace(-90, 90, steps=num_lat)
    latitudes = torch.linspace(-90.0, 90.0, steps=num_lat)[1:-1]
    latitudes = latitudes / 180.0 * torch.pi

    c_lats = 2 * torch.pi * radius * torch.cos(latitudes)
    c_lats = c_lats.reshape([1, 1, latents_size[0], 1])
    pixel_x = c_lats / latents_size[1]
    pixel_y = float(torch.pi * radius / (latents_size[0] + 1))

    pressure = torch.tensor(
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    ).reshape([1, 13, 1, 1])
    pixel_z = torch.tensor([50, 50, 50, 50, 50, 75, 100, 100, 100, 125, 112, 75, 75]).reshape(
        [1, 13, 1, 1]
    )

    pressure_level_num = pixel_z.shape[1]
    M_z = torch.zeros(pressure_level_num, pressure_level_num)
    for M_z_i in range(pressure_level_num):
        for M_z_j in range(pressure_level_num):
            if M_z_i <= M_z_j:
                M_z[M_z_i, M_z_j] = pixel_z[0, M_z_j, 0, 0]

    return _GridSnapshot(
        latents_size=latents_size,
        radius=radius,
        latitudes=latitudes,
        c_lats=c_lats,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        pressure=pressure,
        pixel_z=pixel_z,
        M_z=M_z,
    )


# =============================================================================
# WeatherGFT family: FD-4 central differences + Euler, periodic boundaries
# =============================================================================


def make_weathergft_ops(latents_size: tuple[int, int] = (8, 16)):
    """Точная копия d_x/d_y/d_z/integral_z из Models/WeatherGFT.py:40-110.

    Возвращает namespace c функциями, замкнутыми на конкретный latents_size.

    Source file: Models/WeatherGFT.py (default latents_size=[8, 16]).
    Sibling file: Models/WeatherGFTSingle.py (latents_size=[32, 64]).
    Sibling file: Models/dev/WeatherGFT.py (latents_size=[8, 16]).
    Sibling file: Models/dev/WeatherGFT_2.py (similar).
    Sibling file: Models/dev/WeatherGFT_3.py (latents_size=[32, 64]).

    Args:
        latents_size: (H, W) сетки для precomputed `pixel_x`/`pixel_y`/`M_z`.

    Returns:
        Namespace с `integral_z`, `d_x`, `d_y`, `d_z` и raw-доступом к гриду
        через атрибут `grid` (объект `_GridSnapshot`).
    """
    grid = _build_grid_weathergft_style(latents_size)
    pixel_x = grid.pixel_x
    pixel_y = grid.pixel_y
    pixel_z = grid.pixel_z
    M_z = grid.M_z

    def integral_z(input_tensor: torch.Tensor) -> torch.Tensor:
        # Pressure-direction integral (WeatherGFT.py:40-46)
        B, pressure_level_num, H, W = input_tensor.shape
        input_tensor = input_tensor.reshape(B, pressure_level_num, H * W)
        output = M_z.to(input_tensor.dtype).to(input_tensor.device) @ input_tensor
        output = output.reshape(B, pressure_level_num, H, W)
        return output

    def d_x(input_tensor: torch.Tensor) -> torch.Tensor:
        # Latitude-direction differential (WeatherGFT.py:49-68)
        # Stencil [1, -8, 0, 8, -1] / 12 along last axis (W). Periodic.
        B, C, H, W = input_tensor.shape
        conv_kernel = torch.zeros(
            [1, 1, 1, 5],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
            requires_grad=False,
        )
        conv_kernel[0, 0, 0, 0] = 1
        conv_kernel[0, 0, 0, 1] = -8
        conv_kernel[0, 0, 0, 3] = 8
        conv_kernel[0, 0, 0, 4] = -1

        input_tensor = torch.cat(
            (input_tensor[:, :, :, -2:], input_tensor, input_tensor[:, :, :, :2]),
            dim=3,
        )
        _, _, H_, W_ = input_tensor.shape

        input_tensor = input_tensor.reshape(B * C, 1, H_, W_)
        output_x = F.conv2d(input_tensor, conv_kernel) / 12
        output_x = output_x.reshape(B, C, H, W)
        output_x = output_x / pixel_x.to(output_x.dtype).to(output_x.device)

        return output_x

    def d_y(input_tensor: torch.Tensor) -> torch.Tensor:
        # Longitude-direction differential (WeatherGFT.py:71-90)
        # Stencil [-1, 8, 0, -8, 1] / 12 along axis H. Periodic.
        B, C, H, W = input_tensor.shape
        conv_kernel = torch.zeros(
            [1, 1, 5, 1],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
            requires_grad=False,
        )
        conv_kernel[0, 0, 0] = -1
        conv_kernel[0, 0, 1] = 8
        conv_kernel[0, 0, 3] = -8
        conv_kernel[0, 0, 4] = 1

        input_tensor = torch.cat(
            (input_tensor[:, :, :2], input_tensor, input_tensor[:, :, -2:]), dim=2
        )
        _, _, H_, W_ = input_tensor.shape

        input_tensor = input_tensor.reshape(B * C, 1, H_, W_)
        output_y = F.conv2d(input_tensor, conv_kernel) / 12
        output_y = output_y.reshape(B, C, H, W)
        output_y = output_y / pixel_y

        return output_y

    def d_z(input_tensor: torch.Tensor) -> torch.Tensor:
        # Pressure-direction differential (WeatherGFT.py:93-110)
        # Stencil [-1, 8, 0, -8, 1] / 12 along pressure axis via conv3d. Periodic.
        conv_kernel = torch.zeros(
            [1, 1, 5, 1, 1],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
            requires_grad=False,
        )
        conv_kernel[0, 0, 0] = -1
        conv_kernel[0, 0, 1] = 8
        conv_kernel[0, 0, 3] = -8
        conv_kernel[0, 0, 4] = 1

        input_tensor = torch.cat((input_tensor[:, :2], input_tensor, input_tensor[:, -2:]), dim=1)
        input_tensor = input_tensor.unsqueeze(1)  # B, 1, C, H, W
        output_z = F.conv3d(input_tensor, conv_kernel) / 12
        output_z = output_z.squeeze(1)
        output_z = output_z / pixel_z.to(output_z.dtype).to(output_z.device)

        return output_z

    namespace = type("WeatherGFTOps", (), {})()
    namespace.grid = grid
    namespace.integral_z = integral_z
    namespace.d_x = d_x
    namespace.d_y = d_y
    namespace.d_z = d_z
    return namespace


# =============================================================================
# PredFormerGFT family: WENO-5 (Jiang-Shu 1996) + AMR pseudo-refinement
# =============================================================================


def make_predformergft_ops(latents_size: tuple[int, int] = (8, 16)):
    """Точная копия weno5_flux/weno_derivative/d_x_weno/d_y_weno/d_z/laplacian/AMR
    из Models/PredFormerGFT.py:52-200.

    Source files:
      * Models/PredFormerGFT.py (latents_size=[8, 16]).
      * Models/PredFormerGFT_HybridBlock.py (latents_size=[32, 64]).
      * Models/dev/PredFormerGFTSmallWorld.py (latents_size=[32, 64]).

    Стенсиль WENO-5 Jiang-Shu (1996) с оптимальными весами d=(0.1, 0.6, 0.3).
    Boundary default='reflect' (НЕ periodic, как в FD).

    Args:
        latents_size: (H, W) сетки.

    Returns:
        Namespace с `integral_z`, `weno5_flux`, `weno_derivative`, `d_x_weno`,
        `d_y_weno`, `d_x` (= d_x_weno), `d_y` (= d_y_weno), `d_z` (FD-4!),
        `laplacian_tensor`, `adaptive_mesh_refinement`, `compute_derivative_with_amr`,
        `grid`.
    """
    grid = _build_grid_predformergft_style(latents_size)
    pixel_x = grid.pixel_x
    pixel_y = grid.pixel_y
    pixel_z = grid.pixel_z
    M_z = grid.M_z

    def integral_z(input_tensor: torch.Tensor) -> torch.Tensor:
        # PredFormerGFT.py:42-48
        B, pressure_level_num, H, W = input_tensor.shape
        input_tensor = input_tensor.reshape(B, pressure_level_num, H * W)
        output = M_z.to(input_tensor.dtype).to(input_tensor.device) @ input_tensor
        output = output.reshape(B, pressure_level_num, H, W)
        return output

    def weno5_flux(
        u: torch.Tensor, epsilon: float = 1e-6, boundary: str = "periodic"
    ) -> torch.Tensor:
        # PredFormerGFT.py:52-96
        if boundary == "periodic":
            u_m2 = torch.roll(u, shifts=2, dims=-1)
            u_m1 = torch.roll(u, shifts=1, dims=-1)
            u_0 = u
            u_p1 = torch.roll(u, shifts=-1, dims=-1)
            u_p2 = torch.roll(u, shifts=-2, dims=-1)
        elif boundary == "reflect":
            u_pad = F.pad(u, pad=(2, 2), mode="reflect")
            u_m2 = u_pad[..., 0:-4]
            u_m1 = u_pad[..., 1:-3]
            u_0 = u_pad[..., 2:-2]
            u_p1 = u_pad[..., 3:-1]
            u_p2 = u_pad[..., 4:]
        else:
            raise ValueError("Unknown boundary condition")

        f1 = (2 * u_m2 - 7 * u_m1 + 11 * u_0) / 6.0
        f2 = (-u_m1 + 5 * u_0 + 2 * u_p1) / 6.0
        f3 = (2 * u_0 + 5 * u_p1 - u_p2) / 6.0

        beta1 = (13 / 12.0) * (u_m2 - 2 * u_m1 + u_0) ** 2 + (1 / 4.0) * (
            u_m2 - 4 * u_m1 + 3 * u_0
        ) ** 2
        beta2 = (13 / 12.0) * (u_m1 - 2 * u_0 + u_p1) ** 2 + (1 / 4.0) * (u_m1 - u_p1) ** 2
        beta3 = (13 / 12.0) * (u_0 - 2 * u_p1 + u_p2) ** 2 + (1 / 4.0) * (
            3 * u_0 - 4 * u_p1 + u_p2
        ) ** 2

        d1, d2, d3 = 0.1, 0.6, 0.3
        alpha1 = d1 / (epsilon + beta1) ** 2
        alpha2 = d2 / (epsilon + beta2) ** 2
        alpha3 = d3 / (epsilon + beta3) ** 2

        alpha_sum = alpha1 + alpha2 + alpha3
        omega1 = alpha1 / alpha_sum
        omega2 = alpha2 / alpha_sum
        omega3 = alpha3 / alpha_sum

        flux_iphalf = omega1 * f1 + omega2 * f2 + omega3 * f3
        return flux_iphalf

    def weno_derivative(
        u: torch.Tensor,
        dx: torch.Tensor | float,
        epsilon: float = 1e-6,
        boundary: str = "periodic",
    ) -> torch.Tensor:
        # PredFormerGFT.py:98-120
        flux_iphalf = weno5_flux(u, epsilon=epsilon, boundary=boundary).to(u.device)
        if boundary == "periodic":
            flux_imhalf = torch.roll(flux_iphalf, shifts=1, dims=-1).to(u.device)
        elif boundary == "reflect":
            flux_imhalf = flux_iphalf.clone()
            flux_imhalf[..., 1:] = flux_iphalf[..., :-1]
            flux_imhalf[..., 0] = flux_iphalf[..., 0]
        else:
            raise ValueError("Unknown boundary condition")

        if not isinstance(dx, torch.Tensor):
            dx = torch.tensor(dx, dtype=u.dtype, device=u.device)
        if dx.dim() == 1:
            dx = dx.unsqueeze(-1).to(u.device)
        return (flux_iphalf - flux_imhalf) / dx

    def d_x_weno(input_tensor: torch.Tensor, boundary: str = "reflect") -> torch.Tensor:
        # PredFormerGFT.py:122-132
        B, C, H, W = input_tensor.shape
        input_flat = input_tensor.reshape(B * C * H, W)
        dx_flat = pixel_x.expand(B, C, H, 1).reshape(B * C * H)
        derivative_flat = weno_derivative(input_flat, dx_flat, boundary=boundary)
        derivative = derivative_flat.reshape(B, C, H, W)
        return derivative

    def d_y_weno(input_tensor: torch.Tensor, boundary: str = "reflect") -> torch.Tensor:
        # PredFormerGFT.py:134-145
        B, C, H, W = input_tensor.shape
        input_perm = input_tensor.permute(0, 1, 3, 2)
        input_flat = input_perm.reshape(B * C * W, H)
        derivative_flat = weno_derivative(input_flat, pixel_y, boundary=boundary)
        derivative_perm = derivative_flat.reshape(B, C, W, H)
        derivative = derivative_perm.permute(0, 1, 3, 2)
        return derivative

    def d_z(input_tensor: torch.Tensor) -> torch.Tensor:
        # PredFormerGFT.py:151-166 (FD-4, не WENO! Это документированно.)
        conv_kernel = torch.zeros(
            [1, 1, 5, 1, 1],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
            requires_grad=False,
        )
        conv_kernel[0, 0, 0] = -1
        conv_kernel[0, 0, 1] = 8
        conv_kernel[0, 0, 3] = -8
        conv_kernel[0, 0, 4] = 1

        input_tensor = torch.cat((input_tensor[:, :2], input_tensor, input_tensor[:, -2:]), dim=1)
        input_tensor = input_tensor.unsqueeze(1)
        output_z = F.conv3d(input_tensor, conv_kernel) / 12
        output_z = output_z.squeeze(1)
        output_z = output_z / pixel_z.to(output_z.dtype).to(output_z.device)
        return output_z

    def laplacian_tensor(u: torch.Tensor) -> torch.Tensor:
        # PredFormerGFT.py:168-171
        d2u_dx2 = d_x_weno(d_x_weno(u))
        d2u_dy2 = d_y_weno(d_y_weno(u))
        return d2u_dx2 + d2u_dy2

    def adaptive_mesh_refinement(
        field: torch.Tensor, grad_threshold: float = 1e-3, upscale_factor: int = 2
    ) -> tuple[torch.Tensor, bool]:
        # PredFormerGFT.py:175-185
        grad_field = torch.sqrt(d_x_weno(field) ** 2 + d_y_weno(field) ** 2)
        if grad_field.max() > grad_threshold:
            refined_field = F.interpolate(
                field, scale_factor=upscale_factor, mode="bilinear", align_corners=True
            )
            return refined_field, True
        else:
            return field, False

    def compute_derivative_with_amr(
        field: torch.Tensor,
        derivative_fn: Callable[..., torch.Tensor],
        grad_threshold: float = 1e-3,
        upscale_factor: int = 2,
        boundary: str = "reflect",
    ) -> torch.Tensor:
        # PredFormerGFT.py:187-200
        refined_field, refined = adaptive_mesh_refinement(field, grad_threshold, upscale_factor)
        if refined:
            refined_deriv = derivative_fn(refined_field, boundary=boundary)
            deriv = F.interpolate(
                refined_deriv,
                scale_factor=1 / upscale_factor,
                mode="bilinear",
                align_corners=True,
            )
            return deriv
        else:
            return derivative_fn(field, boundary=boundary)

    namespace = type("PredFormerGFTOps", (), {})()
    namespace.grid = grid
    namespace.integral_z = integral_z
    namespace.weno5_flux = weno5_flux
    namespace.weno_derivative = weno_derivative
    namespace.d_x_weno = d_x_weno
    namespace.d_y_weno = d_y_weno
    namespace.d_x = d_x_weno
    namespace.d_y = d_y_weno
    namespace.d_z = d_z
    namespace.laplacian_tensor = laplacian_tensor
    namespace.adaptive_mesh_refinement = adaptive_mesh_refinement
    namespace.compute_derivative_with_amr = compute_derivative_with_amr
    return namespace


# =============================================================================
# Source registry
# =============================================================================

SOURCES: dict[str, str] = {
    "WeatherGFT.py": "FD-4 central differences, Euler, constant Coriolis f=7.29e-5, "
    "periodic boundaries. latents_size=[8, 16].",
    "WeatherGFTSingle.py": "Same as WeatherGFT.py but latents_size=[32, 64].",
    "PredFormerGFT.py": "WENO-5 (Jiang-Shu), AMR pseudo-refinement, beta-plane "
    "Coriolis f=f0+beta*y, Euler. latents_size=[8, 16].",
    "PredFormerGFT_HybridBlock.py": "Same as PredFormerGFT.py but latents_size=[32, 64].",
    "dev/WeatherGFT_3.py": "FD-4 + RK4 + spherical Coriolis f=2*Omega*sin(phi) + "
    "turbulent mixing + radiative cooling + convective precip. latents_size=[32, 64].",
    "dev/PredFormerGFTSmallWorld.py": "Copy of PredFormerGFT_HybridBlock physics.",
}


__all__ = [
    "make_weathergft_ops",
    "make_predformergft_ops",
    "SOURCES",
]
