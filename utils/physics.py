"""Единый параметризованный модуль физических интеграторов для WeatherPredictions.

Объединяет FD-4 и WENO-5 семейства, которые раньше жили в нескольких
экспериментальных физических реализациях. В отличие от старых копий:

  * Сетка параметризуется через :class:`Grid` (любые H, W; не hardcoded
    module-globals; широты — через ``lat_range_deg``).
  * Буферы регистрируются через ``nn.Module.register_buffer`` — корректно
    переезжают на GPU и попадают в ``state_dict``.
  * Чистый :class:`PurePDEKernel` без обучаемых Conv2d/BatchNorm — для
    бейзлайнов и для residual-метрик «физика vs ERA5».
  * Граничные условия (``periodic`` / ``reflect`` / ``replicate``)
    выставляются явно для каждой оси.
  * Все физические константы — атрибуты класса, видны в ``repr``.

**Изменения семантики по сравнению со старой физикой**:

  * Coriolis: везде ``f = 2·Ω·sin(...)`` (в старой физике было ``f = Ω``
    без множителя 2 в constant- и beta_plane-вариантах).
  * Гидростатика: дефолт ``R_d = 287 Дж/(кг·К)`` (per-mass), не универсальная
    ``R = 8.314``. Старое поведение восстанавливается флагом
    ``use_universal_R=True``.
  * Температурная tendency: дефолт ``adiabatic_omega``
    ``dT/dt = R_d·T·ω/(c_p·p)``, не сломанная ``(Q − z_z·w)/c_p`` с
    ``Q = −L·z_z·w``. Старая формула — ``t_t_formulation='legacy_paper'``.
  * Magnus saturation работает в SI: ``e_s`` в Па, ``p`` в Па; экспонента
    ограничивается поэлементным ``clamp(±20)``, а не batch-global remap.
  * Знаковая конвенция вертикали (аудит 13, 2026-07-08): ``d_z`` возвращает
    −∂/∂p, ``get_w`` — вверх-положительную w = −ω [гПа/с]; адиабата берёт
    ``ω = −100·w``, гидростатика интегрирует ``+R·T_t/p`` от уровня к
    поверхности, конденсация триггерится на подъёме (ω < 0) с реальным
    давлением уровней. См. ``docs/equations.md`` и
    ``docs/experiments/13_sign_convention_fix/``.
  * Boundary ``'periodic'`` теперь настоящее периодическое замыкание (на
    оси W — единственной физически цикличной); cat-pad-поведение, ранее
    ошибочно звавшееся ``'periodic'``, переименовано в ``'replicate'``.

Новые модели должны использовать только этот модуль.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Grid
# =============================================================================


@dataclass
class GridConfig:
    """Параметры сетки для построения coordinate-буферов.

    Args:
        H: число точек по широте (latents_size[0]).
        W: число точек по долготе (latents_size[1]).
        pressure_levels: давления в гПа, по умолчанию ERA5-13 уровней.
        pixel_z_values: межуровневые расстояния по давлению в гПа.
        radius: радиус Земли в метрах.
        lat_range_deg: (low, high) широт в градусах. Сетка строится как
            ``linspace(low, high, H+2)[1:-1]`` (без крайних точек). Дефолт
            ``(-90, 90)`` соответствует глобальной сетке без полюсов. Для
            региональных кропов (например USA ≈ 24°N..56°N — см.
            ``utils.regions.USA_CROP``) передавайте свои значения, чтобы
            ``pixel_x``, ``pixel_y`` и Coriolis-буферы соответствовали
            реальным широтам данных.
    """

    H: int
    W: int
    pressure_levels: tuple[int, ...] = (
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
    pixel_z_values: tuple[int, ...] = (50, 50, 50, 50, 50, 75, 100, 100, 100, 125, 112, 75, 75)
    radius: float = 6371.0 * 1000.0
    lat_range_deg: tuple[float, float] = (-90.0, 90.0)


class Grid(nn.Module):
    """Coordinate-буферы как nn.Module.

    Регистрирует `pixel_x`, `pixel_y`, `pixel_z`, `M_z`, `pressure`, `latitudes`.

    Registered buffers (попадают в state_dict и переезжают через `.to(device)`):
        * `latitudes`: (H,) — широты в радианах, без полюсов.
        * `pixel_x`: (1, 1, H, 1) — широта-зависимый шаг по долготе в метрах.
        * `pixel_y`: (1,) — равномерный шаг по широте в метрах (скаляр-тензор).
        * `pressure`: (1, P, 1, 1) — давления в Па (×100 от гПа).
        * `pixel_z`: (1, P, 1, 1) — Δp между уровнями в гПа.
        * `M_z`: (P, P) — нижнетреугольная для вертикального интегрирования.
        * `f_constant`: (1,) — `2·Ω·sin(45°)` ≈ 1.03e-4 (mid-latitude скаляр).
        * `f_beta_plane`: (1, 1, H, 1) — `2·Ω·sin(45°) + β·R·φ` (beta-plane
          вокруг 45°). NB: f0 — это `2·Ω·sin(45°)`, а не `Ω`; см. CHANGELOG.
        * `f_spherical`: (1, 1, H, 1) — `2·Ω·sin(φ)` (полное сферическое).
    """

    def __init__(self, config: GridConfig):
        super().__init__()
        self.config = config
        H, W = config.H, config.W

        lat_low, lat_high = config.lat_range_deg
        if not (-90.0 <= lat_low < lat_high <= 90.0):
            raise ValueError(
                f"Invalid lat_range_deg {config.lat_range_deg!r}: expected "
                "(low, high) with -90 ≤ low < high ≤ 90."
            )
        # linspace(low, high, H+2)[1:-1] — равномерная сетка широт в указанном
        # диапазоне без крайних точек.
        latitudes_deg = torch.linspace(lat_low, lat_high, steps=H + 2)[1:-1]

        latitudes = latitudes_deg / 180.0 * torch.pi  # (H,)
        c_lats = 2 * torch.pi * config.radius * torch.cos(latitudes)
        pixel_x = (c_lats / W).reshape(1, 1, H, 1)
        # pixel_y соответствует длине одного шага широты в метрах:
        # (high - low) [deg] / (H + 1) * π/180 * R.
        lat_span_rad = (lat_high - lat_low) / 180.0 * torch.pi
        pixel_y = torch.tensor([lat_span_rad * config.radius / (H + 1)])

        pressure_hpa = torch.tensor(config.pressure_levels, dtype=torch.float32)
        pressure = (pressure_hpa * 100.0).reshape(1, -1, 1, 1)  # Па
        pixel_z = torch.tensor(config.pixel_z_values, dtype=torch.float32).reshape(1, -1, 1, 1)

        P = pixel_z.shape[1]
        M_z = torch.zeros(P, P)
        for i in range(P):
            for j in range(P):
                if i <= j:
                    M_z[i, j] = pixel_z[0, j, 0, 0]

        # Coriolis variants. Все используют один и тот же `omega`; во всех
        # вариантах множитель 2 присутствует (f = 2Ω·sin(φ) с разными
        # приближениями по широте). См. docstring класса.
        omega = 7.2921e-5  # рад/с
        f_mid_latitude = 2 * omega * torch.sin(torch.tensor(torch.pi / 4))  # f0 = 2Ω·sin(45°)
        f_constant = f_mid_latitude.reshape(1)
        # Beta-plane: f ≈ f0 + β·y, где f0 = 2Ω·sin(45°), y = R·φ.
        y_coords = config.radius * latitudes
        f_beta_plane = (f_mid_latitude + 1.6e-11 * y_coords).reshape(1, 1, H, 1)
        # Spherical: f = 2Ω sin φ
        f_spherical = (2 * omega * torch.sin(latitudes)).reshape(1, 1, H, 1)

        self.register_buffer("latitudes", latitudes)
        self.register_buffer("pixel_x", pixel_x)
        self.register_buffer("pixel_y", pixel_y)
        self.register_buffer("pressure", pressure)
        self.register_buffer("pixel_z", pixel_z)
        self.register_buffer("M_z", M_z)
        self.register_buffer("f_constant", f_constant)
        self.register_buffer("f_beta_plane", f_beta_plane)
        self.register_buffer("f_spherical", f_spherical)

    def extra_repr(self) -> str:
        return (
            f"H={self.config.H}, W={self.config.W}, "
            f"P={len(self.config.pressure_levels)}, "
            f"lat_range_deg={self.config.lat_range_deg!r}"
        )


# =============================================================================
# Differential operators
# =============================================================================


def integral_z(field: torch.Tensor, M_z: torch.Tensor) -> torch.Tensor:
    """Вертикальное кумулятивное интегрирование через нижнетреугольную M_z.

    Args:
        field: тензор формы ``(B, P, H, W)``.
        M_z: матрица интегрирования ``(P, P)``.

    Returns:
        Тензор той же формы ``(B, P, H, W)``.
    """
    B, P, H, W = field.shape
    flat = field.reshape(B, P, H * W)
    out = M_z.to(flat.dtype) @ flat
    return out.reshape(B, P, H, W)


BoundaryMode = Literal["periodic", "reflect", "replicate"]


class FiniteDifference(nn.Module):
    """Центральные разности 4-го порядка `[1, -8, 0, 8, -1] / 12`.

    Параметры:
        grid: объект :class:`Grid` с буферами `pixel_x`, `pixel_y`, `pixel_z`.
        boundary_x: один из 'periodic' / 'reflect' / 'replicate' для оси W (lon).
            Дефолт 'periodic' — долгота циклична.
        boundary_y: то же для оси H (lat). Дефолт 'replicate' — широта не
            циклична, а 'reflect' через полюс физически бессмыслен.
        boundary_z: то же для оси P (pressure). Дефолт 'replicate'; 'periodic'
            запрещён (давление не цикличное).

    Семантика опций:
        * 'periodic' — настоящая периодика: cat(last2, …, first2) на циклической
          оси (lon). Для lat и pressure возможно, но физически интерпретируется
          как замыкание через полюс/верх → используйте осознанно.
        * 'reflect' — F.pad(..., mode='reflect'): отражение **через** границу,
          не дублируя крайнюю точку.
        * 'replicate' — cat((:2, …, -2:)): дублирование первых/последних строк.
          В старой физике это поведение ошибочно называлось 'periodic'.
    """

    def __init__(
        self,
        grid: Grid,
        boundary_x: BoundaryMode = "periodic",
        boundary_y: BoundaryMode = "replicate",
        boundary_z: BoundaryMode = "replicate",
    ):
        super().__init__()
        if boundary_z == "periodic":
            raise ValueError(
                "boundary_z='periodic' physically meaningless: pressure axis is not cyclic. "
                "Use 'replicate' or 'reflect'."
            )
        self.grid = grid
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y
        self.boundary_z = boundary_z

    @staticmethod
    def _pad_2d(field: torch.Tensor, dim: int, boundary: BoundaryMode) -> torch.Tensor:
        """Добавить по 2 ячейки с каждой стороны вдоль `dim` (2 или 3).

        Args:
            field: тензор формы ``(B, C, H, W)``.
            dim: 2 (lat) или 3 (lon).
            boundary: см. :class:`FiniteDifference`.

        Returns:
            Поле, расширенное по 2 точки с каждой стороны.
        """
        if dim not in (2, 3):
            raise ValueError(f"Unsupported dim={dim}")
        if boundary == "periodic":
            # Настоящее периодическое замыкание: первые 2 точки добавляются
            # в конец, последние 2 — в начало.
            if dim == 3:
                return torch.cat((field[:, :, :, -2:], field, field[:, :, :, :2]), dim=3)
            return torch.cat((field[:, :, -2:], field, field[:, :, :2]), dim=2)
        if boundary == "reflect":
            # F.pad: (left, right, top, bottom).
            if dim == 3:
                return F.pad(field, pad=(2, 2, 0, 0), mode="reflect")
            return F.pad(field, pad=(0, 0, 2, 2), mode="reflect")
        if boundary == "replicate":
            # Дублирование крайних строк (раньше ошибочно звалось periodic).
            if dim == 3:
                return torch.cat((field[:, :, :, :2], field, field[:, :, :, -2:]), dim=3)
            return torch.cat((field[:, :, :2], field, field[:, :, -2:]), dim=2)
        raise ValueError(f"Unknown boundary {boundary!r}")

    def d_x(self, field: torch.Tensor) -> torch.Tensor:
        """∂f/∂λ (along W, axis=3) — FD-4 / pixel_x(lat)."""
        B, C, H, W = field.shape
        kernel = torch.zeros([1, 1, 1, 5], device=field.device, dtype=field.dtype)
        kernel[0, 0, 0, 0] = 1
        kernel[0, 0, 0, 1] = -8
        kernel[0, 0, 0, 3] = 8
        kernel[0, 0, 0, 4] = -1

        padded = self._pad_2d(field, dim=3, boundary=self.boundary_x)
        _, _, H_, W_ = padded.shape
        out = F.conv2d(padded.reshape(B * C, 1, H_, W_), kernel) / 12.0
        out = out.reshape(B, C, H, W)
        return out / self.grid.pixel_x

    def d_y(self, field: torch.Tensor) -> torch.Tensor:
        """∂f/∂y (along H, axis=2) — FD-4 / pixel_y.

        NB: стенсиль ``[-1, 8, 0, -8, 1]/12`` — отрицательный стандартный FD-4,
        потому что в нашем массиве индекс ``j`` растёт **к югу** (поле lat
        упорядочено high→low), а y-координата (метров) растёт к северу. Этот
        знак сохранён байт-в-байт из оригинальной физики, чтобы новая семья
        давала те же численные значения на тех же входах.
        """
        B, C, H, W = field.shape
        kernel = torch.zeros([1, 1, 5, 1], device=field.device, dtype=field.dtype)
        kernel[0, 0, 0] = -1
        kernel[0, 0, 1] = 8
        kernel[0, 0, 3] = -8
        kernel[0, 0, 4] = 1

        padded = self._pad_2d(field, dim=2, boundary=self.boundary_y)
        _, _, H_, W_ = padded.shape
        out = F.conv2d(padded.reshape(B * C, 1, H_, W_), kernel) / 12.0
        out = out.reshape(B, C, H, W)
        return out / self.grid.pixel_y

    def d_z(self, field: torch.Tensor) -> torch.Tensor:
        """∂f/∂p (along P, axis=1) — FD-4 / pixel_z.

        boundary_z='periodic' запрещён в __init__; здесь поддерживаются
        'replicate' (старое cat-pad поведение) и 'reflect'.
        """
        kernel = torch.zeros([1, 1, 5, 1, 1], device=field.device, dtype=field.dtype)
        kernel[0, 0, 0] = -1
        kernel[0, 0, 1] = 8
        kernel[0, 0, 3] = -8
        kernel[0, 0, 4] = 1

        if self.boundary_z == "replicate":
            padded = torch.cat((field[:, :2], field, field[:, -2:]), dim=1)
        elif self.boundary_z == "reflect":
            padded = F.pad(field, pad=(0, 0, 0, 0, 2, 2), mode="reflect")
        else:
            raise ValueError(f"Unknown boundary_z {self.boundary_z!r}")

        out = F.conv3d(padded.unsqueeze(1), kernel) / 12.0
        out = out.squeeze(1)
        return out / self.grid.pixel_z


class WENO5(nn.Module):
    """5-й порядок Jiang-Shu (1996) WENO-производная вдоль W и H.

    `d_z` использует FD-4, как в legacy hybrid physics kernels.

    Параметры:
        grid: :class:`Grid`.
        epsilon: стабилизатор знаменателя ω-весов.
        boundary_x: 'periodic' / 'reflect' / 'replicate' для оси W (lon).
            Дефолт 'periodic'.
        boundary_y: то же для оси H (lat). Дефолт 'replicate'.
        boundary_z: 'replicate' или 'reflect' для оси P (pressure); 'periodic'
            запрещён (давление не цикличное). Дефолт 'replicate'.
    """

    def __init__(
        self,
        grid: Grid,
        epsilon: float = 1e-6,
        boundary_x: BoundaryMode = "periodic",
        boundary_y: BoundaryMode = "replicate",
        boundary_z: BoundaryMode = "replicate",
    ):
        super().__init__()
        if boundary_z == "periodic":
            raise ValueError(
                "boundary_z='periodic' physically meaningless: pressure axis is not cyclic."
            )
        self.grid = grid
        self.epsilon = epsilon
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y
        self.boundary_z = boundary_z

    def _weno5_flux(self, u: torch.Tensor, boundary: BoundaryMode) -> torch.Tensor:
        eps = self.epsilon
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
        elif boundary == "replicate":
            u_pad = F.pad(u, pad=(2, 2), mode="replicate")
            u_m2 = u_pad[..., 0:-4]
            u_m1 = u_pad[..., 1:-3]
            u_0 = u_pad[..., 2:-2]
            u_p1 = u_pad[..., 3:-1]
            u_p2 = u_pad[..., 4:]
        else:
            raise ValueError(f"Unknown boundary {boundary!r}")

        f1 = (2 * u_m2 - 7 * u_m1 + 11 * u_0) / 6.0
        f2 = (-u_m1 + 5 * u_0 + 2 * u_p1) / 6.0
        f3 = (2 * u_0 + 5 * u_p1 - u_p2) / 6.0

        beta1 = (13 / 12.0) * (u_m2 - 2 * u_m1 + u_0) ** 2 + 0.25 * (u_m2 - 4 * u_m1 + 3 * u_0) ** 2
        beta2 = (13 / 12.0) * (u_m1 - 2 * u_0 + u_p1) ** 2 + 0.25 * (u_m1 - u_p1) ** 2
        beta3 = (13 / 12.0) * (u_0 - 2 * u_p1 + u_p2) ** 2 + 0.25 * (3 * u_0 - 4 * u_p1 + u_p2) ** 2

        a1 = 0.1 / (eps + beta1) ** 2
        a2 = 0.6 / (eps + beta2) ** 2
        a3 = 0.3 / (eps + beta3) ** 2
        asum = a1 + a2 + a3

        return (a1 * f1 + a2 * f2 + a3 * f3) / asum

    def _weno_derivative(
        self, u: torch.Tensor, dx: torch.Tensor, boundary: BoundaryMode
    ) -> torch.Tensor:
        flux_iphalf = self._weno5_flux(u, boundary)
        if boundary == "periodic":
            flux_imhalf = torch.roll(flux_iphalf, shifts=1, dims=-1)
        else:
            # reflect/replicate: на левой границе используем сам flux_iphalf[0]
            # (one-sided), затем сдвинутый ряд. Та же асимметрия, что в оригинале;
            # см. docs/physics.md.
            flux_imhalf = flux_iphalf.clone()
            flux_imhalf[..., 1:] = flux_iphalf[..., :-1]
            flux_imhalf[..., 0] = flux_iphalf[..., 0]

        if dx.dim() == 1:
            dx = dx.unsqueeze(-1)
        return (flux_iphalf - flux_imhalf) / dx

    def d_x(self, field: torch.Tensor) -> torch.Tensor:
        B, C, H, W = field.shape
        flat = field.reshape(B * C * H, W)
        dx_flat = self.grid.pixel_x.expand(B, C, H, 1).reshape(B * C * H)
        deriv = self._weno_derivative(flat, dx_flat, self.boundary_x)
        return deriv.reshape(B, C, H, W)

    def d_y(self, field: torch.Tensor) -> torch.Tensor:
        B, C, H, W = field.shape
        perm = field.permute(0, 1, 3, 2)
        flat = perm.reshape(B * C * W, H)
        dy = self.grid.pixel_y
        deriv = self._weno_derivative(flat, dy, self.boundary_y)
        return deriv.reshape(B, C, W, H).permute(0, 1, 3, 2)

    def d_z(self, field: torch.Tensor) -> torch.Tensor:
        """FD-4 по давлению; 'periodic' по pressure запрещён в __init__."""
        kernel = torch.zeros([1, 1, 5, 1, 1], device=field.device, dtype=field.dtype)
        kernel[0, 0, 0] = -1
        kernel[0, 0, 1] = 8
        kernel[0, 0, 3] = -8
        kernel[0, 0, 4] = 1

        if self.boundary_z == "replicate":
            padded = torch.cat((field[:, :2], field, field[:, -2:]), dim=1)
        elif self.boundary_z == "reflect":
            padded = F.pad(field, pad=(0, 0, 0, 0, 2, 2), mode="reflect")
        else:
            raise ValueError(f"Unknown boundary_z {self.boundary_z!r}")

        out = F.conv3d(padded.unsqueeze(1), kernel) / 12.0
        return out.squeeze(1) / self.grid.pixel_z


# =============================================================================
# Pure PDE kernel (без обучаемых слоёв; для бейзлайнов и метрик)
# =============================================================================


@dataclass
class PhysicsConstants:
    """Физические константы, разделяемые всеми PDE_kernel-вариантами.

    Дефолты численно совпадают с оригинальным PDE_kernel из WeatherGFT.py.
    """

    L: float = 2.5e6  # Дж/кг, скрытая теплота парообразования
    R: float = 8.314  # Дж/(моль·К), универсальная (sic — как в оригинале)
    R_d: float = 287.0  # Дж/(кг·К), сухой воздух
    R_v: float = 461.5  # Дж/(кг·К), водяной пар
    c_p: float = 1005.0  # Дж/(кг·К), теплоёмкость
    diff_ratio: float = 0.05  # scale_diff coefficient


class PurePDEKernel(nn.Module):
    """PDE_kernel без обучаемых слоёв — чистая физика для бейзлайнов.

    На вход подаётся уже распакованный набор (z, t, q, u, v) каждого формой
    ``(B, P, H, W)`` (P=13). На выходе — обновлённые (z, t, q, u, v) после
    одного Euler-шага длины ``block_dt`` секунд.

    Не использует Conv2d/BatchNorm/`scale_diff`/`detach` (в отличие от
    оригинального :class:`Models.WeatherGFT.PDE_kernel`). Тендеции
    интегрируются напрямую, чтобы можно было честно мерить residual PDE.

    Параметры:
        grid: :class:`Grid`.
        stencil: ``'fd4'`` — центральные разности; ``'weno5'`` — Jiang-Shu.
        coriolis: ``'constant'`` (f = 2Ω·sin(45°) ≈ 1.03e-4),
            ``'beta_plane'`` (f0 = 2Ω·sin(45°), плюс β·R·φ),
            ``'spherical'`` (f = 2Ω·sin(φ)).
        block_dt: шаг Euler-интегратора в секундах.
        time_scheme: ``'euler'`` или ``'rk4'``.
        boundary_x: для FD/WENO по оси W (lon). Дефолт 'periodic' — долгота
            циклична.
        boundary_y: то же для оси H (lat). Дефолт 'replicate' — широта не
            циклична (через полюс данные не замкнуты).
        boundary_z: то же для оси P (pressure). 'periodic' запрещён.
            Дефолт 'replicate'.
        consts: :class:`PhysicsConstants`.
        use_universal_R: если True — использовать ``R=8.314 Дж/(моль·К)`` в
            гидростатике вместо ``R_d=287 Дж/(кг·К)``. Дефолт False → R_d.
            Это opt-in для регрессии старой физики, где использовалась
            универсальная газовая постоянная; такое поведение даёт численную
            ошибку ~34× в z-tendency.
        t_t_formulation: формула температурной тенденции.

            * ``'adiabatic_omega'`` (default, физически корректно):
              ``dT/dt = R_d·T·ω/(c_p·p) − u·t_x − v·t_y − w·t_z``,
              где ``ω`` = `dp/dt` в Pa/s, получается из ``w·100`` (legacy
              ``w`` имеет единицы hPa/s, см. continuity).
            * ``'legacy_paper'``: ``(Q − z_z·w)/c_p − adv`` с
              ``Q = −L·z_z·w`` — дословно eq. (17)-(18) WeatherGFT paper.
              Известно даёт overflow в первом substep (см.
              ``tools/diagnose_first_substep.py``). Доступно для регрессии.
    """

    def __init__(
        self,
        grid: Grid,
        stencil: Literal["fd4", "weno5"] = "fd4",
        coriolis: Literal["constant", "beta_plane", "spherical"] = "constant",
        block_dt: float = 300.0,
        time_scheme: Literal["euler", "rk4", "ssp_rk3", "semi_implicit"] = "euler",
        boundary_x: BoundaryMode = "periodic",
        boundary_y: BoundaryMode = "replicate",
        boundary_z: BoundaryMode = "replicate",
        consts: PhysicsConstants | None = None,
        use_universal_R: bool = False,
        t_t_formulation: Literal["adiabatic_omega", "legacy_paper"] = "adiabatic_omega",
        hyperdiffusion: bool = False,
        hyperdiff_tau_hours: float = 6.0,
        polar_filter: bool = False,
        polar_filter_lat_deg: float = 60.0,
        advection_form: Literal["advective", "flux"] = "advective",
        w_diagnostic: Literal["plain", "mass_consistent"] = "plain",
    ):
        super().__init__()
        self.grid = grid
        self.stencil = stencil
        self.coriolis = coriolis
        self.block_dt = block_dt
        self.time_scheme = time_scheme
        self.consts = consts if consts is not None else PhysicsConstants()
        self.use_universal_R = use_universal_R
        if t_t_formulation not in ("adiabatic_omega", "legacy_paper"):
            raise ValueError(f"Unknown t_t_formulation {t_t_formulation!r}")
        self.t_t_formulation = t_t_formulation

        if stencil == "fd4":
            self.diff = FiniteDifference(
                grid, boundary_x=boundary_x, boundary_y=boundary_y, boundary_z=boundary_z
            )
        elif stencil == "weno5":
            self.diff = WENO5(
                grid, boundary_x=boundary_x, boundary_y=boundary_y, boundary_z=boundary_z
            )
        else:
            raise ValueError(f"Unknown stencil {stencil!r}")

        # ∇⁴-гипердиффузия — НЕЯВНЫЙ зональный спектральный фильтр
        # (анти-алиасинг, scale-selective замена scale_diff).
        #
        # БАГ-фикс (B1): прежняя реализация — ЯВНЫЙ −K4·∇⁴ с единым K4,
        # калиброванным на самой КРУПНОЙ ячейке. Т.к. собств. значение ∇⁴
        # ∝ 1/Δx⁴, на полюсе (Δx ~×10 мельче) явный множитель
        # 1−K4·k⁴·dt ≈ −172 → усиление 2Δx-моды ×172/шаг вместо демпфирования
        # (явная схема нестабильна за пределом K4·k⁴·dt<2). Замена: неявный
        # спектр. фильтр rfft по долготе, множитель
        #   1/(1 + (dt/τ)·((2−2cos kx)/4)²)  ∈ (0,1]
        # — БЕЗУСЛОВНО устойчив при любых Δx/dt; нормировка на Найквист (÷4)
        # делает его Δx-независимым (одинаков на всех широтах), 2Δx-мода
        # гаснет с e-folding≈τ, крупный масштаб (kx→0) не тронут.
        self.hyperdiffusion = hyperdiffusion
        self.hyperdiff_tau_hours = hyperdiff_tau_hours
        if hyperdiffusion:
            W = self.grid.config.W
            tau_s = hyperdiff_tau_hours * 3600.0
            kx = torch.fft.rfftfreq(W) * 2.0 * math.pi  # [0, π]
            sx = (2.0 - 2.0 * torch.cos(kx)) / 4.0  # ∈[0,1], =1 на Найквисте
            hd_inv = 1.0 / (1.0 + (self.block_dt / tau_s) * sx**2)
            self.register_buffer("hd_zonal_inv", hd_inv.reshape(1, 1, 1, -1))
        else:
            self.hd_zonal_inv = None

        # Полярный Фурье-фильтр: убрать зональные m, чьё эфф. разрешение тоньше,
        # чем на широте φ0 — снимает полюсную CFL регулярной lat-lon сетки.
        # Маска (1,1,H,n_freq) фиксирована (зависит только от сетки+φ0),
        # считается один раз и регистрируется буфером (едет на device/.double()).
        self.polar_filter = polar_filter
        self.polar_filter_lat_deg = polar_filter_lat_deg
        if polar_filter:
            W = self.grid.config.W
            n_freq = W // 2 + 1
            lat = self.grid.latitudes  # (H,) рад
            cphi = torch.cos(lat).clamp_min(1e-6)
            phi0 = math.radians(polar_filter_lat_deg)
            c0 = math.cos(phi0)
            # poleward |φ|>φ0: m_cut = (W/2)·cosφ/cosφ0; экваториальнее — все m.
            mcut = torch.clamp(torch.round((W / 2.0) * cphi / c0), min=1.0)
            keep_all = lat.abs() <= phi0
            mcut = torch.where(keep_all, torch.full_like(mcut, float(n_freq)), mcut)
            m_idx = torch.arange(n_freq, dtype=torch.float32)
            mask = (m_idx.reshape(1, 1, 1, n_freq) <= mcut.reshape(1, 1, -1, 1)).float()
            self.register_buffer("polar_mask", mask)  # (1,1,H,n_freq)
        else:
            self.polar_mask = None

        # E5: консервативная форма адвекции + mass-consistent диагностический w.
        if advection_form not in ("advective", "flux"):
            raise ValueError(f"Unknown advection_form {advection_form!r}")
        if w_diagnostic not in ("plain", "mass_consistent"):
            raise ValueError(f"Unknown w_diagnostic {w_diagnostic!r}")
        self.advection_form = advection_form
        self.w_diagnostic = w_diagnostic

        # E6: semi-implicit. Crank–Nicolson на ЛИНЕЙНОЙ быстрой подсистеме
        # (PGF ↔ дивергенция ↔ гидростатика) — источник взрыва <1ч; остальное
        # (адвекция/Кориолис/источники) явно. Линейная связка сводится к
        # Гельмгольцу для z; решается спектрально на reference-операторе
        # (постоянные коэффициенты, дважды-периодич. приближение — стандарт SI;
        # разница с истинным L уходит в явный остаток).
        if time_scheme == "semi_implicit":
            self._si_setup()

    def _si_setup(self) -> None:
        """Вертикально-модовый semi-implicit reference-оператор.

        Линейная быстрая связка z_t|_L = −A·δ, где A (P×P) — РЕАЛЬНЫЙ
        линейный оператор модели (continuity→adiabatic→hydrostatic) при
        reference T̄=250 K, построенный по-столбцово (без ручной алгебры).
        Диагонализуем A = V·diag(λ)·V⁻¹ (вертикальные нормальные моды);
        исключение u,v даёт по каждой моде m скалярный Гельмгольц
        (I − a²λ_m∇²). λ_m ≈ c_m² (эквивалентные глубины); λ_m≤0 → α=0
        (непропагирующая мода, явно).
        """
        H, W = self.grid.config.H, self.grid.config.W
        P = len(self.grid.config.pressure_levels)
        t_ref = 250.0
        p_col = self.grid.pressure.reshape(P)  # Па
        # A[:, j] = −(z_t-отклик на δ = e_j): по-столбцовый зонд линейной цепи.
        a_mat = torch.zeros(P, P)
        for j in range(P):
            d = torch.zeros(1, P, 1, 1)
            d[0, j, 0, 0] = 1.0
            # Зеркалит фактическую цепочку rhs: ω = −100·w и z_zt = +R/p·t_t
            # (знак-фиксы аудита 13). Оба флипа гасятся — оператор A численно
            # совпадает с прежним.
            w = integral_z(-d, self.grid.M_z)
            t_t = (
                self.consts.R_d
                * t_ref
                * (-100.0 * w)
                / (self.consts.c_p * p_col.reshape(1, P, 1, 1))
            )
            p_hpa = p_col.reshape(1, P, 1, 1) / 100.0
            z_t = integral_z(self.R_eff / p_hpa * t_t, self.grid.M_z)
            a_mat[:, j] = -z_t.reshape(P)
        # A (cumsum-integral_z) НЕ самосопряжён → дефектен (нет
        # well-conditioned модального базиса; pinv даёт V·Vinv≠I). Стандартный
        # фикс: масс-взвешенная симметризация во вертикальном Δp-скаляр-произв.
        # S=diag(pixel_z): M=S^½·A·S^{-½} (подобие, λ сохраняются), берём
        # симметричную часть Msym (асимметрия = reference-приближение, уходит
        # в явный остаток). eigh(Msym) → λ вещ., Q ортонормирован (Q⁻¹=Qᵀ) ⇒
        # V=S^{-½}Q, Vinv=Qᵀ·S^½, V·Vinv=I ТОЧНО (roundtrip-exact).
        pz = self.grid.pixel_z.reshape(P).clamp_min(1e-6)
        s_sqrt = torch.sqrt(pz)
        s_isqrt = 1.0 / s_sqrt
        m_mat = (s_sqrt.reshape(P, 1) * a_mat) * s_isqrt.reshape(1, P)
        m_sym = 0.5 * (m_mat + m_mat.T)
        lam, q_mat = torch.linalg.eigh(m_sym)
        # B2-фикс: НЕ clamp_min(0) — это зануляло БЫСТРЕЙШИЕ гравитационные
        # моды (внешняя c≈309 м/с попадала в λ<0 из-за знака
        # несамосопряжённой cumsum-вертикали) → _si_solve ≈ identity (no-op).
        # Физически c_m² = |λ| > 0 (эквивалентные глубины); знак — артефакт
        # симметризации. Берём |λ|, оставляем ВСЕ моды (вкл. внешнюю).
        lam = lam.abs()
        v_mat = s_isqrt.reshape(P, 1) * q_mat  # (P,P) modal→physical
        v_inv = q_mat.T * s_sqrt.reshape(1, P)  # (P,P) physical→modal
        self.register_buffer("si_A", a_mat)  # (P,P) z_t|_L = −A·δ
        self.register_buffer("si_V", v_mat)
        self.register_buffer("si_Vinv", v_inv)
        # B4/полюс-фикс: ЗОНАЛЬНО-неявный reference. Прежде Гельмгольц брал
        # ПОСТОЯННЫЙ Δx_ref=mean(pixel_x) → не «видел» полюсные мелко-Δx
        # быстрые моды (именно они нарушают CFL) и расходился с физическим
        # _laplacian в RHS (52% mismatch). Теперь rfft ТОЛЬКО по долготе,
        # символ ∂²/∂x² = −sx(j)/Δx(j)² ЗАВИСИТ ОТ ШИРОТЫ j (на полюсе Δx
        # мал → sx велик → сильное неявное демпфирование там, где нужно).
        # Меридиональная/бароклинная часть гравиволны → явный остаток (Δy
        # однороден и крупен — не стиффовое направление). a=dt/2.
        a_cn2 = (0.5 * self.block_dt) ** 2
        dx_lat = self.grid.pixel_x.reshape(H)  # (H,) м, λ-зависимый
        kx = torch.fft.rfftfreq(W) * 2.0 * math.pi  # (W//2+1,)
        sx = (2.0 - 2.0 * torch.cos(kx)).reshape(1, -1) / (
            dx_lat.reshape(H, 1) ** 2
        )  # (H, W//2+1) = −символ ∂²/∂x² ≥ 0
        self.register_buffer("si_sx", sx)  # (H, W//2+1) для _si_xlap (B4-консист.)
        helm = 1.0 / (1.0 + a_cn2 * lam.reshape(P, 1, 1) * sx.reshape(1, H, -1))
        self.register_buffer("si_helm_inv", helm)  # (P, H, W//2+1)

    def _si_xlap(self, field: torch.Tensor) -> torch.Tensor:
        """∂²/∂x² через ТОТ ЖЕ зональный спектральный символ, что и в
        _si_solve (B4-консистентность CN: один ∇² по обе стороны)."""
        spec = torch.fft.rfft(field, dim=-1) * (-self.si_sx)
        return torch.fft.irfft(spec, n=field.shape[-1], dim=-1)

    def _si_solve(self, rhs: torch.Tensor) -> torch.Tensor:
        """Решить (I − a²A·∂²ₓ) z = rhs: модальная проекция + per-mode
        зональный (rfft по долготе) Гельмгольц с λ-зависимым Δx."""
        rhs_m = torch.einsum("pq,bqhw->bphw", self.si_Vinv, rhs)
        spec = torch.fft.rfft(rhs_m, dim=-1) * self.si_helm_inv
        z_m = torch.fft.irfft(spec, n=rhs.shape[-1], dim=-1)
        return torch.einsum("pq,bqhw->bphw", self.si_V, z_m)

    # ----- buffer accessors -----

    @property
    def f_field(self) -> torch.Tensor:
        if self.coriolis == "constant":
            return self.grid.f_constant
        elif self.coriolis == "beta_plane":
            return self.grid.f_beta_plane
        elif self.coriolis == "spherical":
            return self.grid.f_spherical
        else:
            raise ValueError(f"Unknown coriolis {self.coriolis!r}")

    @property
    def R_eff(self) -> float:
        """Газовая постоянная в гидростатике. По умолчанию R_d=287 (per-mass)."""
        return self.consts.R if self.use_universal_R else self.consts.R_d

    # ----- RHS computations (no autograd-breaking) -----

    def _horiz_adv(self, field: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Горизонтальный перенос поля X потоком (u, v).

        ``advective`` — материальная производная −(u·∂ₓX + v·∂_yX)
        (неконсервативна при ∇·V≠0).
        ``flux`` (E5) — консервативная дивергентная форма −∇·(VX) =
        −[∂ₓ(uX)+∂_y(vX)]. Отличается от advective на X·∇·V (для
        бездивергентного потока совпадают); ∑X сохраняется ТОЧНО на
        периодической сетке (телескопирование потоков) при любом V —
        стандартная flux-форма динамического ядра.
        """
        if self.advection_form == "advective":
            return -(u * self.diff.d_x(field) + v * self.diff.d_y(field))
        return -(self.diff.d_x(u * field) + self.diff.d_y(v * field))

    def get_w(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Диагностическая вертикальная скорость w из континуити.

        ВАЖНО (конвенция, аудит 13): ``integral_z``/``M_z`` суммируют от
        уровня К ПОВЕРХНОСТИ, поэтому возвращается w = −∫_p^ps div dp' =
        −ω(p) [гПа/с] — вверх-положительная величина, а не ω. Потребители,
        которым нужна ω (адиабата, конденсация), берут ω = −100·w [Па/с].

        ``mass_consistent`` (E5): вычесть p-взвешенное колоночное среднее
        дивергенции, чтобы ∫(∂ₓu+∂_yv) dp ≈ 0 на каждый столб — нет
        накопления дрейфа массы по вертикали.
        """
        div = self.diff.d_x(u) + self.diff.d_y(v)
        if self.w_diagnostic == "mass_consistent":
            pz = self.grid.pixel_z  # (1,P,1,1) hPa
            div_bar = (div * pz).sum(dim=1, keepdim=True) / pz.sum()
            div = div - div_bar
        return integral_z(-div, self.grid.M_z)

    def get_uv_dt(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        z_x: torch.Tensor,
        z_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tendency для (u, v) — momentum (горизонт. адвекция через _horiz_adv:
        advective или flux-форма по self.advection_form)."""
        u_z = self.diff.d_z(u)
        v_z = self.diff.d_z(v)
        f = self.f_field
        u_t = self._horiz_adv(u, u, v) - w * u_z + f * v - z_x
        v_t = self._horiz_adv(v, u, v) - w * v_z - f * u - z_y
        return u_t, v_t

    def get_t_t(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        t: torch.Tensor,
        z_z: torch.Tensor,
    ) -> torch.Tensor:
        """Tendency температуры.

        Две формулы:

        * ``adiabatic_omega`` (default, физически корректно):
          ``dT/dt|_adia = R_d·T·ω/(c_p·p)``, где ``ω`` — pressure velocity
          в Pa/s. `get_w()` возвращает вверх-положительную ``w = −ω`` в hPa/s
          (интеграл идёт от уровня к поверхности), поэтому ``ω = −100·w``
          (знак-фикс B1, аудит 13). Минус знаки advection как обычно.
        * ``legacy_paper`` (eq. (17)-(18) WeatherGFT NeurIPS 2024):
          ``(Q − z_z·w)/c_p`` с ``Q = −L·z_z·w``. Документально точная,
          но даёт overflow в первый substep на ERA5.
        """
        t_z = self.diff.d_z(t)
        adv = self._horiz_adv(t, u, v) - w * t_z
        if self.t_t_formulation == "adiabatic_omega":
            omega_pa = -100.0 * w  # w = −ω в hPa/s → ω в Pa/s
            t_t_adia = self.consts.R_d * t * omega_pa / (self.consts.c_p * self.grid.pressure)
            return t_t_adia + adv
        # legacy_paper:
        Q = -self.consts.L * z_z * w
        return (Q - z_z * w) / self.consts.c_p + adv

    def get_z_t(self, t_t: torch.Tensor) -> torch.Tensor:
        """Tendency геопотенциала через гидростатику.

        Φ_t(p) = Φ_t(p_s) + ∫_p^{p_s} (R_eff/p')·T_t dp' — `integral_z`/`M_z`
        суммируют именно от уровня к поверхности, подынтегральное
        ПОЛОЖИТЕЛЬНОЕ (прогрев столба поднимает геопотенциал над
        поверхностью; прежний минус — знак-баг B2 аудита 13, давал −Φ_t).
        Переменная интегрирования p и делитель p должны быть в ОДНИХ
        единицах: `M_z`/`pixel_z` в **hPa**, поэтому делитель тоже в hPa
        (`grid.pressure` зарегистрирован в Pa → /100); иначе z_t был бы
        ×100 меньше корректного.
        """
        pressure_hpa = self.grid.pressure / 100.0
        z_zt = self.R_eff / pressure_hpa * t_t
        return integral_z(z_zt, self.grid.M_z)

    @staticmethod
    def _avoid_inf(tensor: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        """Clamp |tensor| ≥ threshold с сохранением знака; нули → +threshold.

        Используется в знаменателях Magnus / уравнения состояния, чтобы не
        делить на 0. В оригинале WeatherGFT была двухшаговая «защита» сначала
        нулей на 0.1, потом всего <1 на ±1 — но второй шаг перетирал первый,
        нули на выходе становились +1, а не +0.1. Здесь схлопнули в один where:
        нули и значения с |x|<threshold заменяются на `sign(x)·threshold`
        (для x==0 sign=0, поэтому отдельная ветка явно ставит +threshold).
        """
        sign = torch.sign(tensor)
        # x==0 → +1 (sign==0 → нужен «искусственный» знак); иначе sign(x).
        sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
        return torch.where(torch.abs(tensor) < threshold, sign * threshold, tensor)

    def _get_qs(self, p_pa: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Saturation specific humidity через Magnus formula.

        Args:
            p_pa: давление в Па (НЕ гПа), согласовано с :attr:`Grid.pressure`.
            T: температура в K.

        Returns:
            q_s: безразмерное (кг/кг), та же форма что и p_pa·T.

        Note:
            Экспонента ограничивается ПОЭЛЕМЕНТНЫМ ``clamp(±20)`` — числовой
            guard, batch-независимый (как в
            ``physics_hybrid.saturation_specific_humidity``). Прежний
            batch-global remap ``_scale_tensor(arg, -3.47, 3.01)`` из
            оригинала WeatherGFT связывал сэмплы батча и искажал q_s даже
            для типичных T (баг №4 exp 01, здесь оставался неисправленным).
            `e_s` получается в Па (6.112 hPa × 100 = 611.2 Pa).
        """
        t_c = T - 273.15
        arg = 17.67 * t_c / self._avoid_inf(t_c + 243.5)
        e_s = 6.112 * torch.exp(torch.clamp(arg, min=-20.0, max=20.0)) * 100  # Па
        return 0.622 * e_s / self._avoid_inf(p_pa - 0.378 * e_s)

    def get_q_dt(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        w: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Tendency влажности: адвекция + крупномасштабная конденсация.

        Конденсация — схема насыщенной псевдоадиабаты (Haltiner & Williams):
        ``dq/dt|cond = δ·F(T, q_s)·ω/p``, δ=1 при подъёме (ω<0) и q >= q_s.
        Давление — реальные уровни (:attr:`Grid.pressure`, Па), а не
        реконструкция ``p = ρRT`` из ``z_z`` (та давала p < 0: конвенция
        d_z = −∂/∂p, шкала гПа и универсальная R — баг B3 аудита 13).
        ω = −100·w, т.к. ``get_w`` возвращает вверх-положительную w = −ω.

        Args:
            u, v: горизонтальный ветер ``(B, P, H, W)``, м/с.
            t: температура, K. w: скорость из ``get_w``, гПа/с.
            q: удельная влажность, кг/кг.

        Returns:
            ``torch.Tensor`` ``(B, P, H, W)`` — dq/dt, (кг/кг)/с.
        """
        q_z = self.diff.d_z(q)

        q_s = torch.maximum(self._get_qs(self.grid.pressure, t), torch.full_like(q, 1e-6))
        omega_pa = -100.0 * w
        delta = ((omega_pa < 0) & (q >= q_s)).to(q.dtype)
        R_moist = (1 + 0.608 * q) * self.consts.R_d
        F_factor = (
            (self.consts.L * R_moist - self.consts.c_p * self.consts.R_v * t)
            / self._avoid_inf(self.consts.c_p * self.consts.R_v * t * t + self.consts.L**2 * q_s)
            * q_s
            * t
        )
        return self._horiz_adv(q, u, v) - w * q_z + delta * F_factor * omega_pa / self.grid.pressure

    # ----- Time stepping -----

    def _apply_hyperdiffusion(self, field: torch.Tensor) -> torch.Tensor:
        """Неявная зональная ∇⁴-гипердиффузия: rfft по долготе × hd_zonal_inv.

        Множитель 1/(1+(dt/τ)·((2−2cos kx)/4)²) ∈ (0,1] — безусловно
        устойчив (любые Δx/dt), Δx-независим (одинаков на всех широтах),
        2Δx гаснет с e-folding≈τ, m=0/крупный масштаб не тронут.
        """
        spec = torch.fft.rfft(field, dim=-1)
        spec = spec * self.hd_zonal_inv
        return torch.fft.irfft(spec, n=field.shape[-1], dim=-1)

    def _apply_polar_filter(self, field: torch.Tensor) -> torch.Tensor:
        """Зональный спектральный фильтр: rFFT по долготе, обрезать m>m_cut(φ).

        m=0 (зональное среднее) всегда сохраняется. Долгота физически
        периодична → rFFT точен; маска зависит только от широты.
        """
        spec = torch.fft.rfft(field, dim=-1)
        spec = spec * self.polar_mask  # complex × real-маска (broadcast по B,P)
        return torch.fft.irfft(spec, n=field.shape[-1], dim=-1)

    def _finalize(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Post-step хук: неявная гипердиффузия (E2) + полярный фильтр (E3)
        к прогностическим полям.

        Едино для euler/rk4/ssp_rk3/semi_implicit (DRY). Оба — безусловно
        устойчивые спектральные операторы. Тенденции (`*_t`) и `w` не
        фильтруются — это диагностика, не состояние.
        """
        for k in ("u", "v", "t", "q", "z"):
            if self.hyperdiffusion:
                out[k] = self._apply_hyperdiffusion(out[k])
            if self.polar_filter:
                out[k] = self._apply_polar_filter(out[k])
        return out

    def rhs(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        q: torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Полный RHS: словарь с tendency для всех 5 переменных + диагностика w.

        Гипердиффузия здесь НЕ применяется (B1-фикс): она реализована как
        безусловно-устойчивый НЕЯВНЫЙ спектральный фильтр в ``_finalize``
        (post-step), а не явный ``−K4·∇⁴`` в tendency (тот усиливал 2Δx
        на полюсе ×172/шаг — explicit-нестабильность).

        Returns:
            dict с ключами ``u_t``, ``v_t``, ``t_t``, ``q_t``, ``z_t``, ``w``,
            каждый формы ``(B, P, H, W)``.
        """
        w = self.get_w(u, v)
        z_x = self.diff.d_x(z)
        z_y = self.diff.d_y(z)
        z_z = self.diff.d_z(z)
        u_t, v_t = self.get_uv_dt(u, v, w, z_x, z_y)
        t_t = self.get_t_t(u, v, w, t, z_z)
        z_t = self.get_z_t(t_t)
        q_t = self.get_q_dt(u, v, t, w, q)
        return {"u_t": u_t, "v_t": v_t, "t_t": t_t, "q_t": q_t, "z_t": z_t, "w": w}

    def step(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        q: torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Один шаг по времени длины `block_dt`. Возвращает новые поля + диагностику."""
        if self.time_scheme == "euler":
            rhs = self.rhs(u, v, t, q, z)
            dt = self.block_dt
            return self._finalize(
                {
                    "u": u + dt * rhs["u_t"],
                    "v": v + dt * rhs["v_t"],
                    "t": t + dt * rhs["t_t"],
                    "q": q + dt * rhs["q_t"],
                    "z": z + dt * rhs["z_t"],
                    "w": rhs["w"],
                    "u_t": rhs["u_t"],
                    "v_t": rhs["v_t"],
                    "t_t": rhs["t_t"],
                    "q_t": rhs["q_t"],
                    "z_t": rhs["z_t"],
                }
            )
        elif self.time_scheme == "rk4":
            dt = self.block_dt
            k1 = self.rhs(u, v, t, q, z)
            k2 = self.rhs(
                u + 0.5 * dt * k1["u_t"],
                v + 0.5 * dt * k1["v_t"],
                t + 0.5 * dt * k1["t_t"],
                q + 0.5 * dt * k1["q_t"],
                z + 0.5 * dt * k1["z_t"],
            )
            k3 = self.rhs(
                u + 0.5 * dt * k2["u_t"],
                v + 0.5 * dt * k2["v_t"],
                t + 0.5 * dt * k2["t_t"],
                q + 0.5 * dt * k2["q_t"],
                z + 0.5 * dt * k2["z_t"],
            )
            k4 = self.rhs(
                u + dt * k3["u_t"],
                v + dt * k3["v_t"],
                t + dt * k3["t_t"],
                q + dt * k3["q_t"],
                z + dt * k3["z_t"],
            )

            def comb(key: str) -> torch.Tensor:
                return (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key]) / 6.0

            # NB: u_t/v_t/.../z_t — это **combined** RK4-tendency, не raw k1.
            # w_k1 — диагностическая вертикальная скорость на старте шага (из k1),
            # отличается от семантики Euler-ветки, где `w` = current-state w.
            return self._finalize(
                {
                    "u": u + dt * comb("u_t"),
                    "v": v + dt * comb("v_t"),
                    "t": t + dt * comb("t_t"),
                    "q": q + dt * comb("q_t"),
                    "z": z + dt * comb("z_t"),
                    "w_k1": k1["w"],
                    "u_t": comb("u_t"),
                    "v_t": comb("v_t"),
                    "t_t": comb("t_t"),
                    "q_t": comb("q_t"),
                    "z_t": comb("z_t"),
                }
            )
        elif self.time_scheme == "ssp_rk3":
            # SSP-RK3 (Shu–Osher, 3-stage strong-stability-preserving).
            # Родной интегратор для WENO-5; область устойчивости включает
            # отрезок мнимой оси (в отличие от Forward Euler). Стадии схемы:
            # s1 = sⁿ + dt·L(sⁿ);  s2 = ¾·sⁿ + ¼·s1 + ¼·dt·L(s1);
            # sⁿ⁺¹ = ⅓·sⁿ + ⅔·s2 + ⅔·dt·L(s2).
            dt = self.block_dt
            keys = ("u", "v", "t", "q", "z")
            s0 = {"u": u, "v": v, "t": t, "q": q, "z": z}
            k1 = self.rhs(u, v, t, q, z)
            s1 = {k: s0[k] + dt * k1[f"{k}_t"] for k in keys}
            k2 = self.rhs(s1["u"], s1["v"], s1["t"], s1["q"], s1["z"])
            s2 = {k: 0.75 * s0[k] + 0.25 * s1[k] + 0.25 * dt * k2[f"{k}_t"] for k in keys}
            k3 = self.rhs(s2["u"], s2["v"], s2["t"], s2["q"], s2["z"])
            s3 = {k: (s0[k] + 2.0 * s2[k] + 2.0 * dt * k3[f"{k}_t"]) / 3.0 for k in keys}
            # *_t — эффективный шаговый tendency (X⁺ − Xⁿ)/dt (для лога/residual);
            # w — диагностика начала шага (k1), как в euler-ветке.
            eff = {f"{k}_t": (s3[k] - s0[k]) / dt for k in keys}
            return self._finalize(
                {
                    "u": s3["u"],
                    "v": s3["v"],
                    "t": s3["t"],
                    "q": s3["q"],
                    "z": s3["z"],
                    "w": k1["w"],
                    **eff,
                }
            )
        elif self.time_scheme == "semi_implicit":
            # ЗОНАЛЬНО-неявный Crank–Nicolson. Линейная БЫСТРАЯ
            # (стиффовая на полюсе) подсистема — только зональная:
            #   u_t|_L = −∂ₓz ,  z_t|_L = −A·(∂ₓu)
            # неявно (CN); меридиональная PGF (−∂_yz в v_t), −A·∂_yv,
            # адвекция/Кориолис/источники/бароклин-остаток — явно (Δy
            # однороден/крупен → не стиффовое направление). Исключение u из
            # z-уравнения → зональный Гельмгольц для zⁿ⁺¹:
            #   (I − a²A·∂²ₓ) zⁿ⁺¹ = zⁿ + dt·N_z − aA(∂ₓu*+∂ₓuⁿ) + a²A·∂²ₓzⁿ
            # a=dt/2; ∂²ₓ — λ-зависимый зональный символ (_si_solve/_si_xlap,
            # один и тот же → CN-консистентно, B4).
            dt = self.block_dt
            a = 0.5 * dt
            r = self.rhs(u, v, t, q, z)
            z_x = self.diff.d_x(z)

            def a_op(field: torch.Tensor) -> torch.Tensor:  # вертикальный A·field
                return torch.einsum("pq,bqhw->bphw", self.si_A, field)

            dux_n = self.diff.d_x(u)
            # N = full − L_zonal: убираем зональный PGF из u_t и −A·∂ₓu из z_t;
            # v_t (вкл. меридион. PGF) и −A·∂_yv остаются в r явно.
            n_u = r["u_t"] + z_x
            n_v = r["v_t"]
            n_z = r["z_t"] + a_op(dux_n)
            n_t = r["t_t"]
            n_q = r["q_t"]
            u_star = u + dt * n_u
            v_star = v + dt * n_v
            dux_star = self.diff.d_x(u_star)
            helm_rhs = z + dt * n_z - a * a_op(dux_star + dux_n) + (a * a) * a_op(self._si_xlap(z))
            z_new = self._si_solve(helm_rhs)
            z_xn = self.diff.d_x(z_new)
            u_new = u_star - a * (z_xn + z_x)
            v_new = v_star
            t_new = t + dt * n_t
            q_new = q + dt * n_q
            return self._finalize(
                {
                    "u": u_new,
                    "v": v_new,
                    "t": t_new,
                    "q": q_new,
                    "z": z_new,
                    "w": r["w"],
                    "u_t": (u_new - u) / dt,
                    "v_t": (v_new - v) / dt,
                    "t_t": n_t,
                    "q_t": n_q,
                    "z_t": (z_new - z) / dt,
                }
            )
        else:
            raise ValueError(f"Unknown time_scheme {self.time_scheme!r}")


# =============================================================================
# Physics-consistency metrics
# =============================================================================


def pde_residual(
    kernel: PurePDEKernel,
    state_now: dict[str, torch.Tensor],
    state_next: dict[str, torch.Tensor],
    dt_seconds: float,
) -> dict[str, torch.Tensor]:
    """Residual PDE на двух последовательных снимках ERA5.

    Для каждой переменной X ∈ {u, v, t, q, z}:
        R_X = (X_next - X_now) / dt - RHS_X(state_now)

    Если физика хорошо описывает атмосферу — R_X должен быть мал по сравнению
    с самой tendency. На практике для упрощённой PDE-формы (как в этом репо)
    residual нетривиален; абсолютное значение — мера «модельной ошибки».

    Args:
        kernel: PurePDEKernel.
        state_now: dict с ключами 'u', 'v', 't', 'q', 'z', все ``(B, P, H, W)``.
        state_next: то же на момент `t + dt`.
        dt_seconds: шаг между снимками.

    Returns:
        dict residual’ов для каждой переменной, тех же форм.
    """
    rhs = kernel.rhs(state_now["u"], state_now["v"], state_now["t"], state_now["q"], state_now["z"])
    return {
        "u": (state_next["u"] - state_now["u"]) / dt_seconds - rhs["u_t"],
        "v": (state_next["v"] - state_now["v"]) / dt_seconds - rhs["v_t"],
        "t": (state_next["t"] - state_now["t"]) / dt_seconds - rhs["t_t"],
        "q": (state_next["q"] - state_now["q"]) / dt_seconds - rhs["q_t"],
        "z": (state_next["z"] - state_now["z"]) / dt_seconds - rhs["z_t"],
    }


def mass_divergence(kernel: PurePDEKernel, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """∇·v = ∂u/∂x + ∂v/∂y (per pressure level).

    Для несжимаемой жидкости должно быть мало (но в гидростатике точно не 0).

    Returns:
        Тензор ``(B, P, H, W)``.
    """
    return kernel.diff.d_x(u) + kernel.diff.d_y(v)


def kinetic_energy_density(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Кинетическая энергия на единицу массы: 0.5 * (u² + v²)."""
    return 0.5 * (u * u + v * v)


def potential_vorticity_proxy(
    kernel: PurePDEKernel, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Прокси потенциальной вортичности: η = ∂v/∂x − ∂u/∂y + f.

    Это абсолютная вортичность 2D-проекции; настоящая PV требует ∂θ/∂p,
    тут нет потенциальной температуры θ — упрощение.
    """
    zeta = kernel.diff.d_x(v) - kernel.diff.d_y(u)
    return zeta + kernel.f_field


def geostrophic_residual(
    kernel: PurePDEKernel, u: torch.Tensor, v: torch.Tensor, z: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Невязка геострофического баланса.

    Геострофический баланс: f·v = ∂z/∂x, f·u = −∂z/∂y (в Cartesian-приближении).
    На средних широтах атмосфера ~ геострофична — residual должен быть малым
    по сравнению с f·v и f·u по абсолютной величине.

    Returns:
        dict с ключами ``u_residual`` (f·u + ∂z/∂y) и ``v_residual`` (f·v − ∂z/∂x).
    """
    f = kernel.f_field
    z_x = kernel.diff.d_x(z)
    z_y = kernel.diff.d_y(z)
    return {
        "u_residual": f * u + z_y,
        "v_residual": f * v - z_x,
    }


def cfl_number(
    kernel: PurePDEKernel, u: torch.Tensor, v: torch.Tensor, dt: float
) -> dict[str, torch.Tensor]:
    """CFL = |c|·dt/dx по каждому из horizontal-axes.

    Для устойчивости явной схемы должен быть ≤ 1 (для FD-4 ~ 0.7; для WENO-5 ~ 1.4).

    Returns:
        dict с тензорами ``cfl_x``, ``cfl_y`` (та же форма, что у u/v).
    """
    cfl_x = torch.abs(u) * dt / kernel.grid.pixel_x
    cfl_y = torch.abs(v) * dt / kernel.grid.pixel_y
    return {"cfl_x": cfl_x, "cfl_y": cfl_y}


__all__ = [
    "GridConfig",
    "Grid",
    "integral_z",
    "FiniteDifference",
    "WENO5",
    "PhysicsConstants",
    "PurePDEKernel",
    "pde_residual",
    "mass_divergence",
    "kinetic_energy_density",
    "potential_vorticity_proxy",
    "geostrophic_residual",
    "cfl_number",
]
