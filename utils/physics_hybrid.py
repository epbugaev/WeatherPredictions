"""Обучаемая физика для PI-моделей: ядро ``HybridBlock`` и residual-head.

Модуль реализует примитивные уравнения на латент-сетке: WENO-5 производные
(``weno5_flux``/``weno_derivative``/``d_x_weno``/``d_y_weno``), вертикальную
производную и интегрирование по давлению (``d_z``/``integral_z``), ядро явного
шага ``PDE_kernel`` и его стек ``PDE_block``. ``HybridBlock`` — адаптивный
роутер между физической PDE-эволюцией и conv-путём. Дополнительно здесь живут
модель-независимые компоненты physics-informed надстройки: residual-head
``PhysicsTendencyResidualCorrector`` и термодинамические хелперы конверсии
влажности (Magnus, ``saturation_specific_humidity`` и парные r<->q функции).

Отличие от ``utils/physics.py``: тот модуль — необучаемая диагностическая
библиотека (FD/WENO-анализ, PDE-невязки, check-physics); здесь — обучаемые
``nn.Module`` и хелперы, встраиваемые в PI-модели (сейчас ``Models/IAM4VP.py``,
далее PI-PredRNN/PI-SimVP).
"""

import torch
import torch.nn.functional as F
from torch import nn

# ===== Исходные расчёты параметров дискретизации =====
# USA-crop: вход 32×64 → patch=4 → latents 8×16.
latents_size = [8, 16]
radius = 6371.0 * 1000
num_lat = latents_size[0] + 2
# Равномерное распределение широт от -90 до 90 градусов; края (-90/+90) обрезаются как полюсы.
latitudes = torch.linspace(-90, 90, steps=num_lat)[1:-1]
latitudes = latitudes / 180 * torch.pi  # перевод в радианы

c_lats = 2 * torch.pi * radius * torch.cos(latitudes)
c_lats = c_lats.reshape([1, 1, latents_size[0], 1])

pixel_x = c_lats / latents_size[1]  # горизонтальное расстояние (ось x)
pixel_y = torch.pi * radius / (latents_size[0] + 1)  # вертикальное расстояние (ось y)

pressure = torch.tensor([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]).reshape(
    [1, 13, 1, 1]
)
pixel_z = torch.tensor([50, 50, 50, 50, 50, 75, 100, 100, 100, 125, 112, 75, 75]).reshape(
    [1, 13, 1, 1]
)

pressure_level_num = pixel_z.shape[1]
M_z = torch.zeros(pressure_level_num, pressure_level_num)
for M_z_i in range(pressure_level_num):
    for M_z_j in range(pressure_level_num):
        if M_z_i <= M_z_j:
            M_z[M_z_i, M_z_j] = pixel_z[0, M_z_j, 0, 0]


def integral_z(input_tensor):
    # Вертикальное интегрирование по давлению
    B, pressure_level_num, H, W = input_tensor.shape
    input_tensor = input_tensor.reshape(B, pressure_level_num, H * W)
    output = M_z.to(input_tensor.dtype).to(input_tensor.device) @ input_tensor
    output = output.reshape(B, pressure_level_num, H, W)
    return output


# ===== Реализация WENO 5-го порядка для вычисления производных =====


def weno5_flux(u, epsilon=1e-6, boundary="periodic"):
    """
    Вычисляет численный поток на интерфейсах ячеек по схеме WENO 5-го порядка.
    Аргументы:
      u        : тензор, по последней оси которого вычисляются срезы (форма (..., W))
      epsilon  : малое число для предотвращения деления на ноль.
      boundary : тип граничных условий: "periodic" или "reflect".
    """
    if boundary == "periodic":
        u_m2 = torch.roll(u, shifts=2, dims=-1)  # u_{i-2}
        u_m1 = torch.roll(u, shifts=1, dims=-1)  # u_{i-1}
        u_0 = u  # u_i
        u_p1 = torch.roll(u, shifts=-1, dims=-1)  # u_{i+1}
        u_p2 = torch.roll(u, shifts=-2, dims=-1)  # u_{i+2}
    elif boundary == "reflect":
        # Отражающее дополнение: pad=(2,2) по последней оси
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


def weno_derivative(u, dx, epsilon=1e-6, boundary="periodic"):
    """
    Вычисляет первую производную функции u по последней оси с использованием WENO 5-го порядка.
    Аргументы:
      u        : тензор с формой (N, W), по последней оси которого считается производная.
      dx       : шаг по координате (скаляр или 0D/1D тензор).
      boundary : тип граничных условий ("periodic" или "reflect").
    """
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


def d_x_weno(input_tensor, boundary="reflect", dx=None):
    """Производная по оси 3 (ширина, долгота) схемой WENO 5-го порядка.

    Args:
        input_tensor: ``torch.Tensor`` формы ``(B, C, H, W)``.
        boundary: режим границ ``"periodic"`` или ``"reflect"``.
        dx: тензор шага по долготе формы ``(1, 1, H, 1)``. При ``None`` берётся
            module-level ``pixel_x`` (глобальная сетка, backward-compat).

    Returns:
        ``torch.Tensor`` формы ``(B, C, H, W)`` — производная по ширине.
    """
    if dx is None:
        dx = pixel_x
    B, C, H, W = input_tensor.shape
    input_flat = input_tensor.reshape(B * C * H, W)
    dx_flat = dx.expand(B, C, H, 1).reshape(B * C * H)
    derivative_flat = weno_derivative(input_flat, dx_flat, boundary=boundary)
    derivative = derivative_flat.reshape(B, C, H, W)
    return derivative


def d_y_weno(input_tensor, boundary="reflect", dy=None):
    """Производная по оси 2 (высота, широта) схемой WENO 5-го порядка.

    Args:
        input_tensor: ``torch.Tensor`` формы ``(B, C, H, W)``.
        boundary: режим границ ``"periodic"`` или ``"reflect"``.
        dy: скалярный тензор шага по широте. При ``None`` берётся module-level
            ``pixel_y`` (глобальная сетка, backward-compat).

    Returns:
        ``torch.Tensor`` формы ``(B, C, H, W)`` — производная по высоте.
    """
    if dy is None:
        dy = pixel_y
    B, C, H, W = input_tensor.shape
    input_perm = input_tensor.permute(0, 1, 3, 2)
    input_flat = input_perm.reshape(B * C * W, H)
    derivative_flat = weno_derivative(input_flat, dy, boundary=boundary)
    derivative_perm = derivative_flat.reshape(B, C, W, H)
    derivative = derivative_perm.permute(0, 1, 3, 2)
    return derivative


def d_z(input_tensor):
    # Вертикальная производная по давлению; края реплицируются через concat
    conv_kernel = torch.zeros(
        [1, 1, 5, 1, 1], device=input_tensor.device, dtype=input_tensor.dtype, requires_grad=False
    )
    conv_kernel[0, 0, 0] = -1
    conv_kernel[0, 0, 1] = 8
    conv_kernel[0, 0, 3] = -8
    conv_kernel[0, 0, 4] = 1

    input_tensor = torch.cat((input_tensor[:, :2], input_tensor, input_tensor[:, -2:]), dim=1)
    input_tensor = input_tensor.unsqueeze(1)  # [B, 1, C, H, W]
    output_z = F.conv3d(input_tensor, conv_kernel) / 12
    output_z = output_z.squeeze(1)
    output_z = output_z / pixel_z.to(output_z.dtype).to(output_z.device)
    return output_z


# ===== Пространственная производная на нативной латент-сетке =====


def compute_spatial_derivative(field, derivative_fn, boundary="reflect"):
    """Производная ``field`` по ``derivative_fn`` (d_x/d_y) на исходной сетке.

    Ранее здесь был AMR: при ``grad.max() > 1e-3`` поле апскейлилось
    ``F.interpolate(scale_factor=2)`` перед взятием производной. Но d_x_weno
    /d_y_weno масштабируют результат на module-level ``pixel_x``/``pixel_y``,
    форма которых жёстко привязана к ``latents_size`` (для USA-кропа —
    8×16). На уточнённой сетке (16×32) ``pixel_x.expand(B, C, H, 1)`` падал
    с ``expanded size (16) must match existing size (8)`` (job 3998966).
    Срабатывание AMR data-dependent: на v3/нормированных smoke-данных порог
    не достигался, на v4 (raw memmap) — достигался. AMR убран: производная
    всегда считается на нативном латенте, где ``pixel_*`` согласованы.

    Args:
        field: ``torch.Tensor`` ``(B, C, H, W)`` на латент-сетке.
        derivative_fn: связанный метод ``_d_x``/``_d_y`` ядра (WENO-5).
        boundary: режим границ, пробрасывается в ``derivative_fn``.

    Returns:
        ``torch.Tensor`` той же формы, что ``field``.
    """
    return derivative_fn(field, boundary=boundary)


class PDE_kernel(nn.Module):
    """Инлайн-ядро примитивных уравнений (физический prior PI-IAM4VP).

    Эволюционирует состояние ``zquvtw`` (порядок каналов z, t, q, u, v) на
    латент-сетке одним явным шагом за вызов. Геометрия кропа (широты, шаги
    ``pixel_x``/``pixel_y``, параметр Кориолиса ``f_field``) строится в
    ``__init__`` под конкретное окно и хранится в буферах — DDP-совместимо и
    не зависит от module-level глобалей. Формулировки термодинамики,
    гидростатики, Кориолиса и способ ограничения приращений — переключаемы
    явными флагами конструктора (по умолчанию — исправленная физика).
    """

    def __init__(
        self,
        in_dim: int,
        physics_part_coef: float | None,
        variable_dim: int = 13,
        block_dt: float = 300,
        inverse_time: bool = False,
        norm: bool = False,
        eddy_viscosity: float = 0.0,
        beta: float = 1.6e-11,
        f0: float = 1.0313e-4,
        w_diagnostic: str = "plain",
        lat_start_deg: float = -70.0,
        dlat_deg: float = 20.0,
        dlon_deg: float = 22.5,
        grid_h: int = 8,
        coriolis_formulation: str = "spherical",
        t_t_formulation: str = "adiabatic_omega",
        use_universal_R: bool = False,
        tendency_limiter: str = "physical_clip",
        tendency_caps: dict[str, float] | None = None,
    ):
        """Инициализирует ядро с crop-aware геометрией и переключателями физики.

        Args:
            in_dim: число каналов x-пути (Conv2d ``variable_norm``/``variable_innorm``).
            physics_part_coef: вес физической ветки; ``None`` → обучаемая матрица.
            variable_dim: число уровней давления на переменную (13).
            block_dt: физический шаг интегрирования (с); знак задаёт ``inverse_time``.
            inverse_time: если True, ``block_dt`` берётся со знаком минус.
            norm: флаг нормировки (совместимость; в арифметике не используется).
            eddy_viscosity: коэффициент вихревой вязкости (лапласиан u/v).
            beta: β (с⁻¹·м⁻¹) для ``coriolis_formulation='beta_plane'``.
            f0: базовый Кориолис (с⁻¹) для ``beta_plane``; дефолт 2Ω·sin45°=1.0313e-4.
            w_diagnostic: ``'plain'`` или ``'mass_consistent'`` — диагностика ω.
            lat_start_deg: широта (град) центра южной (первой) строки латента.
            dlat_deg: шаг по широте между строками латента (град).
            dlon_deg: шаг по долготе между столбцами латента (град).
            grid_h: число строк по широте (H) латент-сетки.
            coriolis_formulation: ``'spherical'`` (f=2Ω·sinφ) или ``'beta_plane'``
                (f0+β·R·φ, для регрессии старого поведения).
            t_t_formulation: ``'adiabatic_omega'`` (R_d·T·ω/(c_p·p)) или
                ``'legacy_paper'`` (старая Q=−L·z_z·w, байт-в-байт).
            use_universal_R: если True — R=8.314 (молярная) в гидростатике вместо
                R_d=287 (масс-удельная).
            tendency_limiter: ``'physical_clip'`` (поэлементный кап приращения) или
                ``'scale_diff'`` (легаси min-max нормировка приращения по батчу).
            tendency_caps: поэлементные капы приращения на вызов ядра по переменной;
                ``None`` → ``{'z':500,'t':5,'q':5,'u':10,'v':10}`` (единицы:
                м²/с², K, ед. влажности, м/с, м/с).

        Raises:
            ValueError: при недопустимом значении любого строкового флага
                (``w_diagnostic``/``coriolis_formulation``/``t_t_formulation``/
                ``tendency_limiter``).
        """
        super().__init__()
        if w_diagnostic not in ("plain", "mass_consistent"):
            raise ValueError(
                f"Unknown w_diagnostic {w_diagnostic!r}; expected 'plain' or 'mass_consistent'"
            )
        if coriolis_formulation not in ("spherical", "beta_plane"):
            raise ValueError(
                f"Unknown coriolis_formulation {coriolis_formulation!r}; "
                "expected 'spherical' or 'beta_plane'"
            )
        if t_t_formulation not in ("adiabatic_omega", "legacy_paper"):
            raise ValueError(
                f"Unknown t_t_formulation {t_t_formulation!r}; "
                "expected 'adiabatic_omega' or 'legacy_paper'"
            )
        if tendency_limiter not in ("physical_clip", "scale_diff"):
            raise ValueError(
                f"Unknown tendency_limiter {tendency_limiter!r}; "
                "expected 'physical_clip' or 'scale_diff'"
            )

        self.norm = norm
        self.eddy_viscosity = eddy_viscosity
        self.w_diagnostic = w_diagnostic
        self.coriolis_formulation = coriolis_formulation
        self.t_t_formulation = t_t_formulation
        self.use_universal_R = use_universal_R
        self.tendency_limiter = tendency_limiter
        self.tendency_caps = (
            {"z": 500.0, "t": 5.0, "q": 5.0, "u": 10.0, "v": 10.0}
            if tendency_caps is None
            else dict(tendency_caps)
        )

        self.f0 = f0
        self.beta = beta
        self.grid_h = grid_h

        # ===== Crop-aware геометрия (буферы → DDP-совместимо, .to(device) авто) =====
        deg2rad = torch.pi / 180.0
        lat_rad = (lat_start_deg + dlat_deg * torch.arange(grid_h, dtype=torch.float32)) * deg2rad
        pixel_x = (radius * torch.cos(lat_rad) * (dlon_deg * deg2rad)).reshape(1, 1, grid_h, 1)
        pixel_y = torch.tensor(radius * dlat_deg * deg2rad, dtype=torch.float32)
        if coriolis_formulation == "spherical":
            f_field = 2.0 * 7.2921e-5 * torch.sin(lat_rad)
        else:  # beta_plane: старая f0 + β·y, y = R·lat (для регрессии)
            f_field = self.f0 + self.beta * (radius * lat_rad)
        # Буферы имеют форму [1, 1, H, 1] для вещания с полями [B, C, H, W].
        self.register_buffer("latitudes", lat_rad)
        self.register_buffer("pixel_x", pixel_x)
        self.register_buffer("pixel_y", pixel_y)
        self.register_buffer("f_field", f_field.reshape(1, 1, grid_h, 1))

        self.variable_norm = nn.Conv2d(
            in_channels=in_dim, out_channels=variable_dim * 5, kernel_size=3, stride=1, padding=1
        )
        if physics_part_coef is not None:
            self.physics_part_coef = physics_part_coef
        else:  # Activate learnable matrix for these coefs: shape C x W x H
            self.physics_part_coef = nn.Parameter(
                0.5 * torch.ones(1, variable_dim * 5, 32, 64), requires_grad=True
            )  # 32 and 64 is for H/W grid

        self.L = 2.5e6
        self.R = 8.314
        self.c_p = 1005
        self.R_v = 461.5
        self.R_d = 287
        self.diff_ratio = 0.05
        self.block_dt = block_dt
        if inverse_time:
            self.block_dt = -self.block_dt

        self.norm_z = nn.BatchNorm2d(variable_dim)
        self.norm_q = nn.BatchNorm2d(variable_dim)
        self.norm_u = nn.BatchNorm2d(variable_dim)
        self.norm_v = nn.BatchNorm2d(variable_dim)
        self.norm_t = nn.BatchNorm2d(variable_dim)

        self.variable_innorm = nn.Conv2d(
            in_channels=variable_dim * 5, out_channels=in_dim, kernel_size=3, stride=1, padding=1
        )
        self.block_norm = nn.BatchNorm2d(in_dim)

    def scale_tensor(self, tensor, a, b):
        min_val = tensor.min().detach()
        max_val = tensor.max().detach()
        denom = torch.clamp(max_val - min_val, min=1e-6)
        scaled_tensor = (tensor - min_val) / denom
        return scaled_tensor * (b - a) + a

    def scale_diff(self, diff_x, x):
        x_min, x_mean, x_max = x.min().detach(), x.mean().detach(), x.max().detach()
        diff_min = (x_min - x_mean) * self.diff_ratio
        diff_max = (x_max - x_mean) * self.diff_ratio
        return self.scale_tensor(diff_x, diff_min, diff_max)

    def _limit_increment(
        self, raw_increment: torch.Tensor, state: torch.Tensor, var_key: str
    ) -> torch.Tensor:
        """Ограничивает физическое приращение поля за один вызов ядра.

        В режиме ``physical_clip`` — поэлементный ``clamp`` в ``±tendency_caps[var_key]``:
        приращение пропорционально ``block_dt`` для ненасыщенных тенденций и не
        связывает сэмплы батча. В режиме ``scale_diff`` — легаси min-max нормировка
        (инвариантна к положительному масштабу, ``block_dt`` по модулю не влияет).

        Args:
            raw_increment: ``torch.Tensor`` формы ``(B, C, H, W)`` = tendency·block_dt.
            state: текущее поле ``(B, C, H, W)`` (нужно ``scale_diff`` для диапазона).
            var_key: ключ переменной для капа, одно из ``{'z','t','q','u','v'}``.

        Returns:
            ``torch.Tensor`` той же формы, отвязанный от графа (``.detach()``) —
            приращение для аддитивной эволюции поля (frozen-prior).
        """
        if self.tendency_limiter == "scale_diff":
            return self.scale_diff(raw_increment, state).detach()
        cap = self.tendency_caps[var_key]
        return torch.clamp(raw_increment, -cap, cap).detach()

    def avoid_inf(self, tensor, threshold=1.0):
        sign = torch.sign(tensor)
        sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
        return torch.where(torch.abs(tensor) < threshold, sign * threshold, tensor)

    def _d_x(self, field: torch.Tensor, boundary: str = "reflect") -> torch.Tensor:
        """Производная по долготе (ось x) на геометрии инстанса (``self.pixel_x``)."""
        return d_x_weno(field, boundary=boundary, dx=self.pixel_x)

    def _d_y(self, field: torch.Tensor, boundary: str = "reflect") -> torch.Tensor:
        """Производная по широте (ось y) на геометрии инстанса (``self.pixel_y``)."""
        return d_y_weno(field, boundary=boundary, dy=self.pixel_y)

    def _laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """∂²/∂x²+∂²/∂y² на геометрии инстанса (для вихревой вязкости)."""
        return self._d_x(self._d_x(field)) + self._d_y(self._d_y(field))

    def share_z_dxyz(self, z):
        self.z_x = self._d_x(z)
        self.z_y = self._d_y(z)
        self.z_z = d_z(z)

    ############################# u, v #############################
    def get_uv_dt(self, u, v, w):
        # Консервативное представление нелинейных членов (адвекция) на геометрии инстанса
        adv_u = (
            compute_spatial_derivative(u * u, self._d_x)
            + compute_spatial_derivative(u * v, self._d_y)
            + d_z(u * w)
        )
        adv_v = (
            compute_spatial_derivative(u * v, self._d_x)
            + compute_spatial_derivative(v * v, self._d_y)
            + d_z(v * w)
        )

        # Используем f_field (вариация по широте)
        self.u_t = -adv_u + self.f_field * v - self.z_x
        self.v_t = -adv_v - self.f_field * u - self.z_y

        # Параметризация субрешеточной турбулентности через вихревую вязкость
        if self.eddy_viscosity > 0:
            lap_u = self._laplacian(u)
            lap_v = self._laplacian(v)
            self.u_t += self.eddy_viscosity * lap_u
            self.v_t += self.eddy_viscosity * lap_v

        return self.u_t, self.v_t

    def uv_evolution(self, u, v, w):
        u_t, v_t = self.get_uv_dt(u, v, w)
        u = u + self._limit_increment(u_t * self.block_dt, u, "u")
        v = v + self._limit_increment(v_t * self.block_dt, v, "v")
        return u, v

    ################################################################

    ############################# t #############################
    def get_t_t(self, u, v, w, t):
        t_x = self._d_x(t)
        t_y = self._d_y(t)
        t_z = d_z(t)
        if self.t_t_formulation == "adiabatic_omega":
            # Адиабата dT/dt = R_d·T·ω/(c_p·p); ω=100·w [Pa/с] (w из get_w в гПа/с), p в Pa.
            omega_pa = 100.0 * w
            pressure_pa = pressure.to(t.dtype).to(t.device) * 100.0
            t_t_adia = self.R_d * t * omega_pa / (self.c_p * pressure_pa)
            self.t_t = t_t_adia - u * t_x - v * t_y - w * t_z
        else:  # legacy_paper: старая формула Q=−L·z_z·w (байт-в-байт)
            Q = -self.L * self.z_z * w
            self.t_t = (Q - self.z_z * w) / self.c_p - u * t_x - v * t_y - w * t_z
        return self.t_t

    def t_evolution(self, u, v, w, t):
        t_t = self.get_t_t(u, v, w, t)
        return t + self._limit_increment(t_t * self.block_dt, t, "t")

    ################################################################

    ############################# z #############################
    def get_z_zt(self):
        # Гидростатика: R_d=287 (масс-удельная) по умолчанию; use_universal_R → R=8.314 (молярная).
        r_eff = self.R if self.use_universal_R else self.R_d
        return -r_eff / pressure.to(self.t_t.dtype).to(self.t_t.device) * self.t_t

    def get_z_t(self):
        z_zt = self.get_z_zt()
        self.z_t = integral_z(z_zt)
        return self.z_t

    def z_evolution(self, z):
        z_t = self.get_z_t()
        return z + self._limit_increment(z_t * self.block_dt, z, "z")

    ################################################################

    ############################# w #############################
    def get_w(self, u, v):
        self.u_x = self._d_x(u)
        self.v_y = self._d_y(v)
        div = self.u_x + self.v_y
        if getattr(self, "w_diagnostic", "plain") == "mass_consistent":
            # p-weighted column-mean divergence removed so int(div) dp ~ 0 per column
            pz = pixel_z.reshape(1, -1, 1, 1).to(div.dtype).to(div.device)
            div_bar = (div * pz).sum(dim=1, keepdim=True) / pz.sum()
            div = div - div_bar
        w_z = -div
        return integral_z(w_z).detach()

    ################################################################

    ############################# q #############################
    def get_q_dt(self, u, v, t, w, q):
        def get_qs(p, T):
            # Magnus: экспонента ограничивается ПОЭЛЕМЕНТНЫМ clamp — числовой guard,
            # batch-независимый. Ранее scale_tensor делал batch-global min/max remap:
            # связывал сэмплы батча и искажал q_s даже для in-range температур.
            t_c = T - 273.15
            exponent = torch.clamp(17.67 * t_c / self.avoid_inf(t_c + 243.5), min=-3.47, max=3.01)
            e_s = 6.112 * torch.exp(exponent) * 100
            return 0.622 * e_s / self.avoid_inf(p - 0.378 * e_s)

        def get_delta(p_t, q, q_s):
            cond = torch.logical_and(p_t < 0, torch.ge(q, q_s))
            return torch.where(cond, torch.ones_like(p_t), torch.zeros_like(p_t))

        def get_F(T, q, q_s):
            R_ = (1 + 0.608 * q) * self.R_d
            F_ = (self.L * R_ - self.c_p * self.R_v * T) / self.avoid_inf(
                self.c_p * self.R_v * T * T + self.L * self.L * q_s
            )
            return F_ * q_s * T

        q_x = self._d_x(q)
        q_y = self._d_y(q)
        q_z = d_z(q)

        rho = -1 / self.avoid_inf(self.z_z)
        p = rho * self.R * t

        q_s = get_qs(p, t).detach()
        q_s = torch.maximum(q_s, torch.ones_like(q_s) * 1e-6)
        delta = get_delta(self.z_t + u * self.z_x + v * self.z_y + w * self.z_z, q, q_s).detach()
        F_ = get_F(t, q, q_s).detach()

        q_t = -(u * q_x + v * q_y + w * q_z) + (
            self.z_t + u * self.z_x + v * self.z_y + w * self.z_z
        ) * delta * F_ / self.avoid_inf(self.R * t)
        return q_t

    def q_evolution(self, u, v, t, w, q):
        q_t = self.get_q_dt(u, v, t, w, q)
        return q + self._limit_increment(q_t * self.block_dt, q, "q")

    ################################################################

    def forward(self, x: torch.Tensor, zquvtw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Один явный PDE-шаг поверх латента x-пути и физического состояния.

        Args:
            x: ``torch.Tensor`` формы ``(B, D, H, W)`` — латент x-пути
                (channels-first, ``D = 5 * zquvtw_channel``).
            zquvtw: ``torch.Tensor`` той же формы — физическое состояние в
                layout'е ``[z, t, q, u, v]`` по ``zquvtw_channel`` уровней
                на переменную.

        Returns:
            Кортеж ``(x, zquvtw_new)`` тех же форм: латент с residual-skip
            после обратной нормировки и новое нормированное состояние.
        """
        skip = x

        ################################################################
        zquvtw_old = (1 - self.physics_part_coef) * self.variable_norm(
            x
        ) + self.physics_part_coef * zquvtw
        z_old, t_old, q_old, u_old, v_old = zquvtw_old.chunk(5, dim=1)

        w_old = self.get_w(u_old, v_old)
        self.share_z_dxyz(z_old)

        u_new, v_new = self.uv_evolution(u_old, v_old, w_old)
        t_new = self.t_evolution(u_old, v_old, w_old, t_old)
        z_new = self.z_evolution(z_old)
        q_new = self.q_evolution(u_old, v_old, t_old, w_old, q_old)

        z_new = self.norm_z(z_new)
        q_new = self.norm_q(q_new)
        u_new = self.norm_u(u_new)
        v_new = self.norm_v(v_new)
        t_new = self.norm_t(t_new)

        zquvtw_new = torch.cat([z_new, t_new, q_new, u_new, v_new], dim=1)

        x = self.variable_innorm(zquvtw_new) + skip
        ################################################################

        x = self.block_norm(x)
        return x, zquvtw_new


class PDE_block(nn.Module):
    """Стек из ``depth`` идентично сконфигурированных ядер ``PDE_kernel``."""

    def __init__(
        self,
        in_dim: int,
        variable_dim: int,
        physics_part_coef: float | None,
        depth: int = 3,
        block_dt: float = 300,
        inverse_time: bool = False,
        w_diagnostic: str = "plain",
        lat_start_deg: float = -70.0,
        dlat_deg: float = 20.0,
        dlon_deg: float = 22.5,
        grid_h: int = 8,
        coriolis_formulation: str = "spherical",
        t_t_formulation: str = "adiabatic_omega",
        use_universal_R: bool = False,
        tendency_limiter: str = "physical_clip",
        tendency_caps: dict[str, float] | None = None,
    ):
        """Собирает стек ядер, прокидывая геометрию и флаги физики в каждое.

        Args:
            in_dim: число каналов x-пути.
            variable_dim: число уровней давления на переменную.
            physics_part_coef: вес физической ветки (или ``None`` для обучаемой).
            depth: число ядер в стеке.
            block_dt: физический шаг интегрирования (с).
            inverse_time: инверсия знака шага.
            w_diagnostic: диагностика ω (``'plain'``/``'mass_consistent'``).
            lat_start_deg: широта южной строки латента (град).
            dlat_deg: шаг по широте (град).
            dlon_deg: шаг по долготе (град).
            grid_h: число строк по широте (H).
            coriolis_formulation: ``'spherical'``/``'beta_plane'``.
            t_t_formulation: ``'adiabatic_omega'``/``'legacy_paper'``.
            use_universal_R: R=8.314 вместо R_d в гидростатике.
            tendency_limiter: ``'physical_clip'``/``'scale_diff'``.
            tendency_caps: поэлементные капы приращения по переменной.
        """
        super().__init__()
        self.PDE_kernels = nn.ModuleList([])
        for _ in range(depth):
            self.PDE_kernels.append(
                PDE_kernel(
                    in_dim,
                    variable_dim=variable_dim,
                    block_dt=block_dt,
                    inverse_time=inverse_time,
                    physics_part_coef=physics_part_coef,
                    w_diagnostic=w_diagnostic,
                    lat_start_deg=lat_start_deg,
                    dlat_deg=dlat_deg,
                    dlon_deg=dlon_deg,
                    grid_h=grid_h,
                    coriolis_formulation=coriolis_formulation,
                    t_t_formulation=t_t_formulation,
                    use_universal_R=use_universal_R,
                    tendency_limiter=tendency_limiter,
                    tendency_caps=tendency_caps,
                )
            )

    def forward(self, x: torch.Tensor, zquvtw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Стек ``PDE_kernel``-шагов с residual-skip по обоим путям.

        Args:
            x: ``torch.Tensor`` формы ``(B, H, W, D)`` — латент x-пути
                (channels-last).
            zquvtw: ``torch.Tensor`` той же формы — физическое состояние
                ``[z, t, q, u, v]`` (channels-last).

        Returns:
            Кортеж ``(x + skip_x, zquvtw + skip_zquvtw)`` форм ``(B, H, W, D)``.
        """
        skip_x, skip_zquvtw = x, zquvtw
        x, zquvtw = x.permute(0, 3, 1, 2), zquvtw.permute(0, 3, 1, 2)  # [B, D, H, W]
        for PDE_kernel in self.PDE_kernels:
            x, zquvtw = PDE_kernel(x, zquvtw)
        x, zquvtw = x.permute(0, 2, 3, 1), zquvtw.permute(0, 2, 3, 1)
        return x + skip_x, zquvtw + skip_zquvtw  # x [B, H, W, D]


class HybridBlock(nn.Module):
    """Адаптивный роутер между PDE-эволюцией (физика) и conv-путём (AI)."""

    def __init__(
        self,
        dim: int,
        zquvtw_channel: int,
        depth: int,
        block_dt: float,
        inverse_time: bool,
        physics_part_coef: float | None,
        w_diagnostic: str = "plain",
        lat_start_deg: float = -70.0,
        dlat_deg: float = 20.0,
        dlon_deg: float = 22.5,
        grid_h: int = 8,
        coriolis_formulation: str = "spherical",
        t_t_formulation: str = "adiabatic_omega",
        use_universal_R: bool = False,
        tendency_limiter: str = "physical_clip",
        tendency_caps: dict[str, float] | None = None,
    ):
        """Строит стек ``PDE_block`` и обучаемый роутер-вес.

        Args:
            dim: число каналов x-пути (и роутер-веса).
            zquvtw_channel: число уровней давления на переменную (variable_dim).
            depth: число ядер в ``PDE_block``.
            block_dt: физический шаг интегрирования (с).
            inverse_time: инверсия знака шага.
            physics_part_coef: вес физической ветки (или ``None`` для обучаемой).
            w_diagnostic: диагностика ω (``'plain'``/``'mass_consistent'``).
            lat_start_deg: широта южной строки латента (град).
            dlat_deg: шаг по широте (град).
            dlon_deg: шаг по долготе (град).
            grid_h: число строк по широте (H).
            coriolis_formulation: ``'spherical'``/``'beta_plane'``.
            t_t_formulation: ``'adiabatic_omega'``/``'legacy_paper'``.
            use_universal_R: R=8.314 вместо R_d в гидростатике.
            tendency_limiter: ``'physical_clip'``/``'scale_diff'``.
            tendency_caps: поэлементные капы приращения по переменной.
        """
        super().__init__()

        self.pde_block = PDE_block(
            dim,
            zquvtw_channel,
            depth=depth,
            block_dt=block_dt,
            inverse_time=inverse_time,
            physics_part_coef=physics_part_coef,
            w_diagnostic=w_diagnostic,
            lat_start_deg=lat_start_deg,
            dlat_deg=dlat_deg,
            dlon_deg=dlon_deg,
            grid_h=grid_h,
            coriolis_formulation=coriolis_formulation,
            t_t_formulation=t_t_formulation,
            use_universal_R=use_universal_R,
            tendency_limiter=tendency_limiter,
            tendency_caps=tendency_caps,
        )
        self.router_weight = nn.Parameter(torch.zeros(1, 1, 1, dim), requires_grad=True)

    def forward(self, x, zquvtw=None):
        feat_pde, zquvtw = self.pde_block(x, zquvtw)

        # Adaptive Router. zquvtw = PDE-эволюция (физический путь);
        # feat_pde = conv-heavy x-путь (AI). Имена весов сопоставлены содержимому.
        weight_physics = 0.5 * torch.ones_like(x) + self.router_weight
        weight_ai = 0.5 * torch.ones_like(x) - self.router_weight
        x = weight_physics * zquvtw + weight_ai * feat_pde
        return x, zquvtw


# ===== Термодинамика и residual-head для PI-моделей =====


def avoid_small_abs(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Отводит |x| от нуля с сохранением знака (sign(0) трактуется как +1).

    Args:
        x: произвольный ``torch.Tensor``.
        threshold: минимально допустимый модуль значения.

    Returns:
        ``torch.Tensor`` той же формы: ``x`` там, где ``|x| >= threshold``,
        иначе ``sign(x) * threshold``.
    """
    sign = torch.sign(x)
    sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
    return torch.where(torch.abs(x) < threshold, sign * threshold, x)


def saturation_specific_humidity(
    t_kelvin: torch.Tensor,
    pressure_pa: torch.Tensor,
) -> torch.Tensor:
    """Насыщенная удельная влажность q_s(T, p) по формуле Магнуса, кг/кг.

    Args:
        t_kelvin: температура в Кельвинах, форма ``(B, L, H, W)``.
        pressure_pa: давление уровней в Па, broadcast-совместимо с ``t_kelvin``
            (обычно ``(1, L, 1, 1)``).

    Returns:
        ``torch.Tensor`` формы ``t_kelvin`` со значениями ``>= 1e-8``.
    """
    pressure = pressure_pa.to(device=t_kelvin.device, dtype=t_kelvin.dtype)
    pressure = pressure.expand_as(t_kelvin)
    t_c = t_kelvin - 273.15
    exponent = 17.67 * t_c / avoid_small_abs(t_c + 243.5)
    # Keeps pathological early predictions from producing inf before the
    # residual head has learned; ERA5 temperatures sit comfortably inside.
    exponent = torch.clamp(exponent, min=-20.0, max=20.0)
    e_s = 611.2 * torch.exp(exponent)
    denom = avoid_small_abs(pressure - 0.378 * e_s)
    return torch.clamp(0.622 * e_s / denom, min=1e-8)


def relative_to_specific_humidity(
    r_percent: torch.Tensor,
    t_kelvin: torch.Tensor,
    pressure_pa: torch.Tensor,
) -> torch.Tensor:
    """Конверсия относительной влажности (%) в удельную (кг/кг).

    Args:
        r_percent: относительная влажность в процентах, ``(B, L, H, W)``.
        t_kelvin: температура в Кельвинах той же формы.
        pressure_pa: давление уровней в Па, broadcast-совместимо.

    Returns:
        Удельная влажность q, ``torch.Tensor`` формы ``r_percent``.
    """
    return (r_percent / 100.0) * saturation_specific_humidity(t_kelvin, pressure_pa)


def specific_to_relative_humidity(
    q: torch.Tensor,
    t_kelvin: torch.Tensor,
    pressure_pa: torch.Tensor,
) -> torch.Tensor:
    """Конверсия удельной влажности (кг/кг) в относительную (%).

    Args:
        q: удельная влажность, ``(B, L, H, W)``.
        t_kelvin: температура в Кельвинах той же формы.
        pressure_pa: давление уровней в Па, broadcast-совместимо.

    Returns:
        Относительная влажность в процентах, ``torch.Tensor`` формы ``q``.
    """
    return 100.0 * q / saturation_specific_humidity(t_kelvin, pressure_pa)


class PhysicsTendencyResidualCorrector(nn.Module):
    """Small zero-start residual head for physics-derived tendency features.

    This module treats the ``HybridBlock`` branch as a feature generator, not
    as a trusted forecast. With zero initialisation the final convolution
    emits exactly zero at step 0, so enabling the experiment starts from the
    plain backbone prediction and learns only if the features help. Shared by
    the PI-model family (currently PI-IAM4VP).
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
        """Map stacked physics/prediction features to an additive correction.

        Args:
            features: ``torch.Tensor`` of shape ``(B, in_channels, H, W)`` — the
                concatenated prediction/prior/tendency feature blocks.

        Returns:
            ``torch.Tensor`` of shape ``(B, out_channels, H, W)``. With
            ``zero_init=True`` this is exactly zero until the head is trained.
        """
        return self.net(features)
