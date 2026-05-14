"""Единый параметризованный модуль физических интеграторов для WeatherPredictions.

Объединяет FD-4 и WENO-5 семьи из 5 дублирующихся файлов
(Models/WeatherGFT*.py, Models/PredFormerGFT*.py, Models/dev/WeatherGFT_3.py).
В отличие от старых копий:

  * Сетка параметризуется через `Grid` (любые H, W; не hardcoded module-globals).
  * Буферы регистрируются через `nn.Module.register_buffer` — корректно
    переезжают на GPU и попадают в `state_dict`.
  * Чистый `PurePDEKernel` без обучаемых Conv2d/BatchNorm — для бейзлайнов
    и для residual-метрик «физика vs ERA5».
  * Граничные условия (`periodic` / `reflect`) выставляются явно.
  * Все физические константы — атрибуты класса, видны в `repr`.

Что НЕ изменено по сравнению со старой физикой (чтобы сохранить численную
эквивалентность baseline-у):
  * Стенсили FD-4 и WENO-5 — байт-в-байт из оригинала.
  * Гидростатика `z_t = integral_z(-R / p * t_t)` использует `R=8.314` как
    в оригинале (см. open question в docs/physics.md о R vs R_d).
  * Constant-Coriolis `f=7.29e-5` сохранён как один из режимов (см. также
    `coriolis='beta_plane'` и `coriolis='spherical'`).
  * Magnus-формула для q_s с `scale_tensor(-3.47, 3.01)` clipping’ом.

Этот модуль не импортируется production-моделями автоматически — рефакторинг
`Models/*GFT*.py` для использования `utils.physics` оставлен как отдельный
коммит, чтобы избежать риска поломки уже обученных чекпоинтов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
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
        lat_scheme: 'linear_minus90_90' — torch.linspace(-90, 90, H+2)[1:-1]
            (как в PredFormerGFT.py); 'arange' — старая формула из
            WeatherGFT.py (90 - j * 180/(H+1)). На равномерной сетке оба дают
            идентичные значения; разница только при неравномерном lat_t.
    """

    H: int
    W: int
    pressure_levels: tuple[int, ...] = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    pixel_z_values: tuple[int, ...] = (50, 50, 50, 50, 50, 75, 100, 100, 100, 125, 112, 75, 75)
    radius: float = 6371.0 * 1000.0
    lat_scheme: Literal["linear_minus90_90", "arange"] = "linear_minus90_90"


class Grid(nn.Module):
    """Coordinate-буферы как nn.Module: `pixel_x`, `pixel_y`, `pixel_z`, `M_z`, `pressure`, `latitudes`.

    Registered buffers (попадают в state_dict и переезжают через `.to(device)`):
        * `latitudes`: (H,) — широты в радианах, без полюсов.
        * `pixel_x`: (1, 1, H, 1) — широта-зависимый шаг по долготе в метрах.
        * `pixel_y`: (1,) — равномерный шаг по широте в метрах (скаляр-тензор).
        * `pressure`: (1, P, 1, 1) — давления в Па (×100 от гПа).
        * `pixel_z`: (1, P, 1, 1) — Δp между уровнями в гПа.
        * `M_z`: (P, P) — нижнетреугольная для вертикального интегрирования.
        * `f_constant`: (1,) — `2 * Ω * sin(45°)` ≈ 1.03e-4 (на случай константного Coriolis).
        * `f_beta_plane`: (1, 1, H, 1) — `f0 + β·R·φ` (beta-plane).
        * `f_spherical`: (1, 1, H, 1) — `2 * Ω * sin(φ)` (корректное сферическое).
    """

    def __init__(self, config: GridConfig):
        super().__init__()
        self.config = config
        H, W = config.H, config.W

        if config.lat_scheme == "linear_minus90_90":
            latitudes_deg = torch.linspace(-90.0, 90.0, steps=H + 2)[1:-1]
        else:  # 'arange'
            lat_t = torch.arange(start=0, end=H + 2).float()
            latitudes_deg = 90.0 - lat_t * 180.0 / float(H + 2 - 1)
            latitudes_deg = latitudes_deg[1:-1]

        latitudes = latitudes_deg / 180.0 * torch.pi  # (H,)
        c_lats = 2 * torch.pi * config.radius * torch.cos(latitudes)
        pixel_x = (c_lats / W).reshape(1, 1, H, 1)
        pixel_y = torch.tensor([torch.pi * config.radius / (H + 1)])

        pressure_hpa = torch.tensor(config.pressure_levels, dtype=torch.float32)
        pressure = (pressure_hpa * 100.0).reshape(1, -1, 1, 1)  # Па
        pixel_z = torch.tensor(config.pixel_z_values, dtype=torch.float32).reshape(1, -1, 1, 1)

        P = pixel_z.shape[1]
        M_z = torch.zeros(P, P)
        for i in range(P):
            for j in range(P):
                if i <= j:
                    M_z[i, j] = pixel_z[0, j, 0, 0]

        # Coriolis variants
        omega = 7.2921e-5  # рад/с
        f_constant = torch.tensor([2 * omega * torch.sin(torch.tensor(torch.pi / 4))])  # f на 45°
        # Beta-plane: y = R * φ
        y_coords = config.radius * latitudes
        f_beta_plane = (7.29e-5 + 1.6e-11 * y_coords).reshape(1, 1, H, 1)
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
        return f"H={self.config.H}, W={self.config.W}, P={len(self.config.pressure_levels)}, lat_scheme={self.config.lat_scheme!r}"


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


class FiniteDifference(nn.Module):
    """Центральные разности 4-го порядка `[1, -8, 0, 8, -1] / 12`.

    Параметры:
        grid: объект :class:`Grid` с буферами `pixel_x`, `pixel_y`, `pixel_z`.
        boundary_x: `'periodic'` (cat last2 + first2) или `'reflect'` (F.pad).
        boundary_y: то же для axis=H.
        boundary_z: то же для axis=P.

    Не имеет обучаемых параметров.
    """

    def __init__(
        self,
        grid: Grid,
        boundary_x: Literal["periodic", "reflect"] = "periodic",
        boundary_y: Literal["periodic", "reflect"] = "periodic",
        boundary_z: Literal["periodic", "reflect"] = "periodic",
    ):
        super().__init__()
        self.grid = grid
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y
        self.boundary_z = boundary_z

    @staticmethod
    def _pad_2d(field: torch.Tensor, dim: int, boundary: str) -> torch.Tensor:
        """Добавить по 2 ячейки с каждой стороны вдоль `dim` (2 или 3)."""
        if boundary == "periodic":
            # Для periodic по lon (dim=3) cat -2: и :2; по lat (dim=2) cat :2 и -2:
            # см. WeatherGFT.py:58-60 (dim=3) и L80-82 (dim=2).
            if dim == 3:
                return torch.cat((field[:, :, :, -2:], field, field[:, :, :, :2]), dim=3)
            elif dim == 2:
                return torch.cat((field[:, :, :2], field, field[:, :, -2:]), dim=2)
            else:
                raise ValueError(f"Unsupported dim={dim}")
        elif boundary == "reflect":
            # F.pad берёт пары снизу-вверх по dim. (left, right, top, bottom).
            if dim == 3:
                return F.pad(field, pad=(2, 2, 0, 0), mode="reflect")
            elif dim == 2:
                return F.pad(field, pad=(0, 0, 2, 2), mode="reflect")
            else:
                raise ValueError(f"Unsupported dim={dim}")
        else:
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
        """∂f/∂φ (along H, axis=2) — FD-4 / pixel_y."""
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
        """∂f/∂p (along P, axis=1) — FD-4 / pixel_z."""
        kernel = torch.zeros([1, 1, 5, 1, 1], device=field.device, dtype=field.dtype)
        kernel[0, 0, 0] = -1
        kernel[0, 0, 1] = 8
        kernel[0, 0, 3] = -8
        kernel[0, 0, 4] = 1

        if self.boundary_z == "periodic":
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

    `d_z` использует FD-4 (как в оригинале PredFormerGFT.py:151-166).

    Параметры:
        grid: :class:`Grid`.
        epsilon: стабилизатор знаменателя ω-весов.
        boundary: `'periodic'` или `'reflect'`.
    """

    def __init__(
        self,
        grid: Grid,
        epsilon: float = 1e-6,
        boundary: Literal["periodic", "reflect"] = "reflect",
        boundary_z: Literal["periodic", "reflect"] = "periodic",
    ):
        super().__init__()
        self.grid = grid
        self.epsilon = epsilon
        self.boundary = boundary
        self.boundary_z = boundary_z

    def _weno5_flux(self, u: torch.Tensor) -> torch.Tensor:
        eps = self.epsilon
        if self.boundary == "periodic":
            u_m2 = torch.roll(u, shifts=2, dims=-1)
            u_m1 = torch.roll(u, shifts=1, dims=-1)
            u_0 = u
            u_p1 = torch.roll(u, shifts=-1, dims=-1)
            u_p2 = torch.roll(u, shifts=-2, dims=-1)
        elif self.boundary == "reflect":
            u_pad = F.pad(u, pad=(2, 2), mode="reflect")
            u_m2 = u_pad[..., 0:-4]
            u_m1 = u_pad[..., 1:-3]
            u_0 = u_pad[..., 2:-2]
            u_p1 = u_pad[..., 3:-1]
            u_p2 = u_pad[..., 4:]
        else:
            raise ValueError(f"Unknown boundary {self.boundary!r}")

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

    def _weno_derivative(self, u: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        flux_iphalf = self._weno5_flux(u)
        if self.boundary == "periodic":
            flux_imhalf = torch.roll(flux_iphalf, shifts=1, dims=-1)
        else:  # reflect
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
        deriv = self._weno_derivative(flat, dx_flat)
        return deriv.reshape(B, C, H, W)

    def d_y(self, field: torch.Tensor) -> torch.Tensor:
        B, C, H, W = field.shape
        perm = field.permute(0, 1, 3, 2)
        flat = perm.reshape(B * C * W, H)
        dy = self.grid.pixel_y
        deriv = self._weno_derivative(flat, dy)
        return deriv.reshape(B, C, W, H).permute(0, 1, 3, 2)

    def d_z(self, field: torch.Tensor) -> torch.Tensor:
        """FD-4 по давлению (как в оригинале PredFormerGFT.py)."""
        kernel = torch.zeros([1, 1, 5, 1, 1], device=field.device, dtype=field.dtype)
        kernel[0, 0, 0] = -1
        kernel[0, 0, 1] = 8
        kernel[0, 0, 3] = -8
        kernel[0, 0, 4] = 1

        if self.boundary_z == "periodic":
            padded = torch.cat((field[:, :2], field, field[:, -2:]), dim=1)
        else:
            padded = F.pad(field, pad=(0, 0, 0, 0, 2, 2), mode="reflect")

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

    L: float = 2.5e6       # Дж/кг, скрытая теплота парообразования
    R: float = 8.314       # Дж/(моль·К), универсальная (sic — как в оригинале)
    R_d: float = 287.0     # Дж/(кг·К), сухой воздух
    R_v: float = 461.5     # Дж/(кг·К), водяной пар
    c_p: float = 1005.0    # Дж/(кг·К), теплоёмкость
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
        coriolis: ``'constant'`` (f=7.29e-5), ``'beta_plane'`` (f0+β·y),
            ``'spherical'`` (2Ω sin φ).
        block_dt: шаг Euler-интегратора в секундах.
        time_scheme: ``'euler'`` или ``'rk4'``.
        boundary_horiz: для FD/WENO по horizontal axes.
        boundary_z: вертикальная.
        consts: :class:`PhysicsConstants`.
        use_R_d_in_hydrostatic: если True — заменить ``R=8.314`` на ``R_d=287``
            в гидростатике (фикс one из open questions в docs/physics.md).
    """

    def __init__(
        self,
        grid: Grid,
        stencil: Literal["fd4", "weno5"] = "fd4",
        coriolis: Literal["constant", "beta_plane", "spherical"] = "constant",
        block_dt: float = 300.0,
        time_scheme: Literal["euler", "rk4"] = "euler",
        boundary_horiz: Literal["periodic", "reflect"] = "periodic",
        boundary_z: Literal["periodic", "reflect"] = "periodic",
        consts: PhysicsConstants | None = None,
        use_R_d_in_hydrostatic: bool = False,
    ):
        super().__init__()
        self.grid = grid
        self.stencil = stencil
        self.coriolis = coriolis
        self.block_dt = block_dt
        self.time_scheme = time_scheme
        self.consts = consts if consts is not None else PhysicsConstants()
        self.use_R_d_in_hydrostatic = use_R_d_in_hydrostatic

        if stencil == "fd4":
            self.diff = FiniteDifference(grid, boundary_x=boundary_horiz, boundary_y=boundary_horiz, boundary_z=boundary_z)
        elif stencil == "weno5":
            self.diff = WENO5(grid, boundary=boundary_horiz, boundary_z=boundary_z)
        else:
            raise ValueError(f"Unknown stencil {stencil!r}")

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
        return self.consts.R_d if self.use_R_d_in_hydrostatic else self.consts.R

    # ----- RHS computations (no autograd-breaking) -----

    def get_w(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Диагностическая вертикальная скорость w из континуити."""
        w_z = -(self.diff.d_x(u) + self.diff.d_y(v))
        return integral_z(w_z, self.grid.M_z)

    def get_uv_dt(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        z_x: torch.Tensor,
        z_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tendency для (u, v) — momentum equations (неконсервативная форма)."""
        u_x = self.diff.d_x(u)
        u_y = self.diff.d_y(u)
        u_z = self.diff.d_z(u)
        v_x = self.diff.d_x(v)
        v_y = self.diff.d_y(v)
        v_z = self.diff.d_z(v)

        f = self.f_field
        u_t = -u * u_x - v * u_y - w * u_z + f * v - z_x
        v_t = -u * v_x - v * v_y - w * v_z - f * u - z_y
        return u_t, v_t

    def get_t_t(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        t: torch.Tensor,
        z_z: torch.Tensor,
    ) -> torch.Tensor:
        """Tendency температуры (как в WeatherGFT.py:195-202)."""
        t_x = self.diff.d_x(t)
        t_y = self.diff.d_y(t)
        t_z = self.diff.d_z(t)
        # NB: формула из оригинала, Q = -L · z_z · w → (Q - z_z·w)/c_p
        # подозрительная (open question в docs/physics.md), но сохранена как есть.
        Q = -self.consts.L * z_z * w
        return (Q - z_z * w) / self.consts.c_p - u * t_x - v * t_y - w * t_z

    def get_z_t(self, t_t: torch.Tensor) -> torch.Tensor:
        """Tendency геопотенциала через гидростатику."""
        z_zt = -self.R_eff / self.grid.pressure * t_t
        return integral_z(z_zt, self.grid.M_z)

    @staticmethod
    def _avoid_inf(tensor: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        tensor = torch.where(torch.abs(tensor) == 0.0, torch.full_like(tensor, 0.1), tensor)
        return torch.where(torch.abs(tensor) < threshold, torch.sign(tensor) * threshold, tensor)

    @staticmethod
    def _scale_tensor(tensor: torch.Tensor, a: float, b: float) -> torch.Tensor:
        mn = tensor.min().detach()
        mx = tensor.max().detach()
        scaled = (tensor - mn) / (mx - mn + 1e-12)
        return scaled * (b - a) + a

    def _get_qs(self, p: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Saturation specific humidity (Magnus formula с clipping’ом как в оригинале)."""
        t_c = T - 273.15
        arg = 17.67 * t_c / self._avoid_inf(t_c + 243.5)
        e_s = 6.112 * torch.exp(self._scale_tensor(arg, -3.47, 3.01)) * 100
        return 0.622 * e_s / self._avoid_inf(p - 0.378 * e_s)

    def get_q_dt(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        w: torch.Tensor,
        q: torch.Tensor,
        z_x: torch.Tensor,
        z_y: torch.Tensor,
        z_z: torch.Tensor,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        """Tendency влажности (Magnus + упрощённый Kuo)."""
        q_x = self.diff.d_x(q)
        q_y = self.diff.d_y(q)
        q_z = self.diff.d_z(q)

        rho = -1.0 / self._avoid_inf(z_z)
        p = rho * self.consts.R * t
        q_s = torch.maximum(self._get_qs(p, t), torch.full_like(q, 1e-6))

        p_t = z_t + u * z_x + v * z_y + w * z_z
        delta = ((p_t < 0) & (q >= q_s)).float()
        R_moist = (1 + 0.608 * q) * self.consts.R_d
        F_factor = (
            (self.consts.L * R_moist - self.consts.c_p * self.consts.R_v * t)
            / self._avoid_inf(self.consts.c_p * self.consts.R_v * t * t + self.consts.L ** 2 * q_s)
            * q_s * t
        )
        return -(u * q_x + v * q_y + w * q_z) + p_t * delta * F_factor / self._avoid_inf(self.consts.R * t)

    # ----- Time stepping -----

    def rhs(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        q: torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Полный RHS: словарь с tendency для всех 5 переменных + диагностика w.

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
        q_t = self.get_q_dt(u, v, t, w, q, z_x, z_y, z_z, z_t)
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
            return {
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
            comb = lambda k: (k1[k] + 2 * k2[k] + 2 * k3[k] + k4[k]) / 6.0
            return {
                "u": u + dt * comb("u_t"),
                "v": v + dt * comb("v_t"),
                "t": t + dt * comb("t_t"),
                "q": q + dt * comb("q_t"),
                "z": z + dt * comb("z_t"),
                "w": k1["w"],
                "u_t": comb("u_t"),
                "v_t": comb("v_t"),
                "t_t": comb("t_t"),
                "q_t": comb("q_t"),
                "z_t": comb("z_t"),
            }
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


def cfl_number(kernel: PurePDEKernel, u: torch.Tensor, v: torch.Tensor, dt: float) -> dict[str, torch.Tensor]:
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
