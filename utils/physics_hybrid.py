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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.physics import _lagrange3_dz_matrix

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
    # Вертикальная производная; края реплицируются через concat.
    # ВАЖНО (конвенция, аудит 13): стенсиль [-1, 8, 0, -8, 1] — это МИНУС
    # стандартного FD-4 по оси индекса, а индекс растёт с давлением, поэтому
    # d_z возвращает −∂/∂p (эквивалентно ∂/∂ζ при ζ=−p). Все адвективные
    # члены самосогласованы с w = −ω из get_w (двойные инверсии гасятся).
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
        physical_passthrough: bool = False,
        advection_form: str = "flux",
        metric_terms: bool = False,
        spherical_divergence: bool = False,
        rayleigh_friction: bool = False,
        vertical_scheme: str = "stencil",
        omega_free: tuple[str, ...] = (),
        latent_heating_coupling: bool = False,
        clim_sources_path: str | None = None,
        clim_sources_prefix: str = "C15_now__",
        physics_level_mask: dict[str, float] | None = None,
        physics_level_mask_learnable: bool = False,
        ekman_K_profile: tuple[float, ...] | None = None,
        humidity_evolution: str = "as_is",
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
            physical_passthrough: если True, ``physics_only_forward`` интегрирует
                ЧИСТОЕ физическое состояние примитивными уравнениями, минуя
                conv-микс (``variable_norm``), ``norm_*``, ``variable_innorm`` и
                роутер (fix дефекта B, режим ``stable_physical_v2`` модели). На
                стандартный ``forward`` не влияет; существующие армы бит-в-бит.
            physics_level_mask: жёсткая маска уровня (exp 19, B1) — ключи ⊆
                ``{'u','v','t','q'}``, значение = порог давления в гПа; физика
                активна там, где ``pressure <= порог``, ниже приращение
                обнуляется (см. ``_mask_increment``). ``z`` не маскируется
                (диагностический столбовой интеграл). ``None`` (дефолт) —
                маска выключена, поведение бит-в-бит.
            physics_level_mask_learnable: обучаемый гейт уровня (exp 19, B1L).
                Если True, вместо жёсткой маски используется параметр
                ``level_gate_logit`` формы ``(4, 13)`` (строки ``u,v,t,q``),
                α=sigmoid(logit), инициализированный ``+4.0``/``−4.0`` по
                значениям жёсткой маски из ``physics_level_mask``. Требует
                заданного ``physics_level_mask`` (источник инициализации).
                Дефолт False — параметр не создаётся, поведение бит-в-бит.
            ekman_K_profile: 13 коэффициентов вихревой вязкости K(p) (Па²/с)
                для Экмановского трения (exp 19) — вертикальная диффузия
                импульса ``∂/∂p(K ∂u/∂p)``, добавляется к ``u_t``/``v_t`` в
                ``get_uv_dt``. ``None`` (дефолт) — член выключен, поведение
                бит-в-бит.
            humidity_evolution: ``'as_is'`` (дефолт, бит-в-бит) или
                ``'cc_bridge'`` — мост Клаузиуса–Клапейрона (exp 19): к
                q-тенденции добавляется ``q·(L/(R_v·T²))·t_t``, чтобы
                наблюдаемая относительная влажность ``r=q/q_s`` сохранялась
                под изменением температуры (``self.t_t`` — полный кэш из
                ``t_evolution``, доступен благодаря порядку вызовов в
                ``_evolve_fields``).

        Raises:
            ValueError: при недопустимом значении любого строкового флага
                (``w_diagnostic``/``coriolis_formulation``/``t_t_formulation``/
                ``tendency_limiter``/``humidity_evolution``), если
                ``physics_level_mask`` содержит ключи вне
                ``{'u','v','t','q'}``, если
                ``physics_level_mask_learnable=True`` без ``physics_level_mask``,
                или если ``ekman_K_profile`` задан с длиной ≠ ``variable_dim``.
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
        if humidity_evolution not in ("as_is", "cc_bridge"):
            raise ValueError(
                f"Unknown humidity_evolution {humidity_evolution!r}; "
                "expected 'as_is' or 'cc_bridge'"
            )
        self.humidity_evolution = humidity_evolution

        self.norm = norm
        self.eddy_viscosity = eddy_viscosity
        self.w_diagnostic = w_diagnostic
        self.coriolis_formulation = coriolis_formulation
        self.t_t_formulation = t_t_formulation
        self.use_universal_R = use_universal_R
        self.tendency_limiter = tendency_limiter
        self.physical_passthrough = physical_passthrough
        if advection_form not in ("flux", "advective"):
            raise ValueError(f"Unknown advection_form {advection_form!r}")
        if vertical_scheme not in ("stencil", "lagrange3"):
            raise ValueError(f"Unknown vertical_scheme {vertical_scheme!r}")
        self.advection_form = advection_form
        self.metric_terms = metric_terms
        self.spherical_divergence = spherical_divergence
        self.rayleigh_friction = rayleigh_friction
        self.vertical_scheme = vertical_scheme
        if not set(omega_free) <= {"u", "v", "t", "q"}:
            raise ValueError(f"omega_free must be a subset of u,v,t,q; got {omega_free!r}")
        self.omega_free = tuple(omega_free)
        if physics_level_mask is not None and not set(physics_level_mask) <= {"u", "v", "t", "q"}:
            raise ValueError(
                f"physics_level_mask keys must be subset of u,v,t,q; got {physics_level_mask!r}"
            )
        self.physics_level_mask = physics_level_mask
        if physics_level_mask_learnable and physics_level_mask is None:
            raise ValueError("physics_level_mask_learnable=True requires physics_level_mask init")
        self.physics_level_mask_learnable = physics_level_mask_learnable
        self.latent_heating_coupling = latent_heating_coupling
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

        # ===== exp 14 (порт из PurePDEKernel): кривизна сферы, трение
        # Хелда–Суареса, Лагранжева вертикальная производная. Дефолты
        # выключены — существующие армы бит-в-бит. =====
        if metric_terms or spherical_divergence:
            self.register_buffer(
                "tan_phi_over_a", (torch.tan(lat_rad) / radius).reshape(1, 1, grid_h, 1)
            )
        else:
            self.tan_phi_over_a = None
        if rayleigh_friction:
            # k_v(σ) = k_f·max(0, (σ−σ_b)/(1−σ_b)), σ = p/p₀, p₀ = 1000 гПа,
            # k_f = 1/сут, σ_b = 0.7 (Held & Suarez 1994).
            sigma = pressure.to(torch.float32) / 1000.0
            k_v = (1.0 / 86400.0) * torch.clamp((sigma - 0.7) / (1.0 - 0.7), min=0.0)
            self.register_buffer("rayleigh_k", k_v)
        else:
            self.rayleigh_k = None
        if ekman_K_profile is not None:
            if len(ekman_K_profile) != variable_dim:
                raise ValueError(
                    f"ekman_K_profile must have {variable_dim} entries; got {len(ekman_K_profile)}"
                )
            self.register_buffer(
                "ekman_K",
                torch.tensor(ekman_K_profile, dtype=torch.float32).reshape(1, -1, 1, 1),
            )
        else:
            self.ekman_K = None
        if vertical_scheme == "lagrange3":
            self.register_buffer(
                "dz_lagrange", _lagrange3_dz_matrix(pressure.reshape(-1).to(torch.float32))
            )
        else:
            self.dz_lagrange = None

        # ===== exp 15: климатологические Q₁/Q₂-источники (annual-приближение
        # S2_map: месячный индекс в модельном forward недоступен — календаря в
        # API нет; ограничение фиксируется в отчёте exp 16). Буферы
        # ``<prefix>annual_{t,q,z}`` (P, H_data) → усреднение широтных строк
        # до grid_h → (1, P, grid_h, 1), тенденции [ед./с]. =====
        if clim_sources_path is not None:
            clim_arrays = dict(np.load(clim_sources_path))
            for clim_var in ("t", "q", "z"):
                annual = torch.from_numpy(clim_arrays[f"{clim_sources_prefix}annual_{clim_var}"])
                pooled = F.adaptive_avg_pool1d(annual.unsqueeze(0), grid_h).squeeze(0)
                self.register_buffer(
                    f"clim_src_{clim_var}", pooled.reshape(1, -1, grid_h, 1).to(torch.float32)
                )
        else:
            self.clim_src_t = None
            self.clim_src_q = None
            self.clim_src_z = None

        if physics_level_mask is not None:
            for mask_var in ("u", "v", "t", "q"):
                threshold = physics_level_mask.get(mask_var)
                if threshold is None:
                    mask = torch.ones_like(pressure, dtype=torch.float32)
                else:
                    mask = (pressure <= threshold).to(torch.float32)
                self.register_buffer(f"level_mask_{mask_var}", mask)

            if physics_level_mask_learnable:
                mask_rows = torch.stack(
                    [getattr(self, f"level_mask_{v}").reshape(-1) for v in ("u", "v", "t", "q")]
                )
                self.level_gate_logit = nn.Parameter(torch.where(mask_rows > 0.5, 4.0, -4.0))

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

    def _mask_increment(self, increment: torch.Tensor, var: str) -> torch.Tensor:
        """Обнуляет/масштабирует физическое приращение переменной по уровню.

        Маска — буфер ``level_mask_<var>`` формы ``(1, 13, 1, 1)`` из порогов
        ``physics_level_mask``; при выключенной маске приращение возвращается
        без изменений (бит-в-бит). При ``physics_level_mask_learnable`` маска —
        обучаемый гейт α=sigmoid(logit) (``level_gate_logit``) вместо жёсткой
        буфер-маски. z не маскируется (диагностический интеграл).

        Args:
            increment: приращение поля ``(B, 13, H, W)`` (выход ``_limit_increment``).
            var: ключ переменной, одно из ``{'u','v','t','q'}``.

        Returns:
            ``torch.Tensor`` той же формы — приращение после маски уровня.
        """
        if self.physics_level_mask is None:
            return increment
        if self.physics_level_mask_learnable:
            row = ("u", "v", "t", "q").index(var)
            alpha = torch.sigmoid(self.level_gate_logit[row]).reshape(1, -1, 1, 1)
            return increment * alpha.to(increment.dtype)
        return increment * getattr(self, f"level_mask_{var}").to(increment.dtype)

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

    def _d_z(self, field: torch.Tensor) -> torch.Tensor:
        """Вертикальная производная d_z = −∂/∂p [1/гПа] по ``self.vertical_scheme``.

        ``'stencil'`` (дефолт) — легаси модульная ``d_z`` бит-в-бит;
        ``'lagrange3'`` — точная 3-точечная производная Лагранжа на
        нерегулярной p-сетке (матрица уже содержит знак −∂/∂p).
        """
        if self.vertical_scheme == "lagrange3":
            return torch.einsum("pq,bqhw->bphw", self.dz_lagrange, field)
        return d_z(field)

    def _laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """∂²/∂x²+∂²/∂y² на геометрии инстанса (для вихревой вязкости)."""
        return self._d_x(self._d_x(field)) + self._d_y(self._d_y(field))

    def share_z_dxyz(self, z):
        self.z_x = self._d_x(z)
        self.z_y = self._d_y(z)
        self.z_z = self._d_z(z)

    ############################# u, v #############################
    def get_uv_dt(self, u, v, w):
        # omega_free (exp 15): исключить ω-зависимый вертикальный член из
        # выбранных уравнений движения; при выключенном флаге w_u/w_v — тот же
        # объект w (дефолт бит-в-бит).
        w_u = torch.zeros_like(w) if "u" in self.omega_free else w
        w_v = torch.zeros_like(w) if "v" in self.omega_free else w
        if self.advection_form == "advective":
            # Адвективная форма (exp 14, C_best): −(u·∂ₓ + v·∂ᵧ + w·∂_z);
            # flux-форма X·∇·V контаминирует перенос (измерено хуже, exp 14 §3.4).
            adv_u = u * self._d_x(u) + v * self._d_y(u) + w_u * self._d_z(u)
            adv_v = u * self._d_x(v) + v * self._d_y(v) + w_v * self._d_z(v)
        else:
            # Консервативное представление нелинейных членов (легаси, байт-в-байт)
            adv_u = (
                compute_spatial_derivative(u * u, self._d_x)
                + compute_spatial_derivative(u * v, self._d_y)
                + self._d_z(u * w_u)
            )
            adv_v = (
                compute_spatial_derivative(u * v, self._d_x)
                + compute_spatial_derivative(v * v, self._d_y)
                + self._d_z(v * w_v)
            )

        # Используем f_field (вариация по широте)
        self.u_t = -adv_u + self.f_field * v - self.z_x
        self.v_t = -adv_v - self.f_field * u - self.z_y
        if self.metric_terms:
            # Кривизна сферы: +u·v·tanφ/a (u_t), −u²·tanφ/a (v_t) —
            # Holton & Hakim, ур. (2.19)–(2.20).
            self.u_t = self.u_t + self.tan_phi_over_a * u * v
            self.v_t = self.v_t - self.tan_phi_over_a * u * u
        if self.rayleigh_friction:
            self.u_t = self.u_t - self.rayleigh_k * u
            self.v_t = self.v_t - self.rayleigh_k * v

        if self.ekman_K is not None:
            # Экмановское трение: ∂/∂p(K ∂/∂p u). _d_z = −∂/∂p, поэтому двойное
            # применение восстанавливает знак: _d_z(K·_d_z(u)) = ∂/∂p(K ∂u/∂p).
            self.u_t = self.u_t + self._d_z(self.ekman_K * self._d_z(u))
            self.v_t = self.v_t + self._d_z(self.ekman_K * self._d_z(v))

        # Параметризация субрешеточной турбулентности через вихревую вязкость
        if self.eddy_viscosity > 0:
            lap_u = self._laplacian(u)
            lap_v = self._laplacian(v)
            self.u_t += self.eddy_viscosity * lap_u
            self.v_t += self.eddy_viscosity * lap_v

        return self.u_t, self.v_t

    def uv_evolution(self, u, v, w):
        u_t, v_t = self.get_uv_dt(u, v, w)
        u = u + self._mask_increment(self._limit_increment(u_t * self.block_dt, u, "u"), "u")
        v = v + self._mask_increment(self._limit_increment(v_t * self.block_dt, v, "v"), "v")
        return u, v

    ################################################################

    ############################# t #############################
    def get_t_t(self, u, v, w, t):
        t_x = self._d_x(t)
        t_y = self._d_y(t)
        t_z = self._d_z(t)
        if self.t_t_formulation == "adiabatic_omega":
            # Адиабата dT/dt = R_d·T·ω/(c_p·p), p в Pa. Знаковая конвенция ядра:
            # get_w возвращает вверх-положительную w = −ω [гПа/с] (интеграл M_z
            # идёт от уровня к поверхности), поэтому ω = −100·w. См.
            # docs/experiments/13_sign_convention_fix и docs/equations.md.
            omega_pa = -100.0 * w
            pressure_pa = pressure.to(t.dtype).to(t.device) * 100.0
            t_t_adia = self.R_d * t * omega_pa / (self.c_p * pressure_pa)
            self.t_t = t_t_adia - u * t_x - v * t_y - w * t_z
        else:  # legacy_paper: старая формула Q=−L·z_z·w (байт-в-байт)
            Q = -self.L * self.z_z * w
            self.t_t = (Q - self.z_z * w) / self.c_p - u * t_x - v * t_y - w * t_z
        if self.latent_heating_coupling:
            # Энергетическая связка Q₁/Q₂ (Yanai et al. 1973): скрытое тепло
            # конденсации −(L/c_p)·(dq/dt|cond) входит в ПОЛНЫЙ t_t (и в
            # столбовой z-интеграл через кэш self.t_t).
            cond = self._condensation_source(t, self._q_for_latent, w)
            self.t_t = self.t_t - (self.L / self.c_p) * cond
        return self.t_t

    def t_evolution(self, u, v, w, t):
        t_t = self.get_t_t(u, v, w, t)  # полный (+latent): кэш self.t_t идёт в z-интеграл
        if "t" in self.omega_free:
            # Зеркало physics.py rhs (exp 15): убрать адиабату и вернуть w·t_z
            # из ВЫХОДНОЙ тенденции; self.t_t (для z) остаётся полным.
            omega_pa = -100.0 * w
            pressure_pa = pressure.to(t.dtype).to(t.device) * 100.0
            adia = self.R_d * t * omega_pa / (self.c_p * pressure_pa)
            t_t = t_t - adia + w * self._d_z(t)
        if self.clim_src_t is not None:
            t_t = t_t + self.clim_src_t
        return t + self._mask_increment(self._limit_increment(t_t * self.block_dt, t, "t"), "t")

    ################################################################

    ############################# z #############################
    def get_z_zt(self):
        # Гидростатика: R_d=287 (масс-удельная) по умолчанию; use_universal_R → R=8.314 (молярная).
        # Знак: integral_z суммирует ОТ уровня К поверхности (∫_p^ps), а
        # Φ_t(p) = Φ_t(ps) + ∫_p^ps R·T_t/p' dp' — подынтегральное ПОЛОЖИТЕЛЬНОЕ
        # (прогрев столба поднимает геопотенциал выше поверхности). Минус здесь
        # (знак-баг B2 аудита 13) давал z_t = −Φ_t.
        r_eff = self.R if self.use_universal_R else self.R_d
        return r_eff / pressure.to(self.t_t.dtype).to(self.t_t.device) * self.t_t

    def get_z_t(self):
        z_zt = self.get_z_zt()
        self.z_t = integral_z(z_zt)
        return self.z_t

    def z_evolution(self, z):
        z_t = self.get_z_t()
        if self.clim_src_z is not None:
            z_t = z_t + self.clim_src_z
        return z + self._limit_increment(z_t * self.block_dt, z, "z")

    ################################################################

    ############################# w #############################
    def get_w(self, u, v):
        # Диагностика вертикальной скорости из континуити ∂ω/∂p = −div.
        # ВАЖНО (конвенция, аудит 13): integral_z суммирует ОТ уровня К
        # поверхности, поэтому возвращается w = −∫_p^ps div dp' = −ω(p)
        # [гПа/с] — вверх-положительная величина. Потребители, которым нужна
        # настоящая ω (адиабата, конденсация), берут ω = −100·w [Па/с].
        self.u_x = self._d_x(u)
        self.v_y = self._d_y(v)
        div = self.u_x + self.v_y
        if self.spherical_divergence:
            # ∇·V на сфере: меридиональный метрический член −v·tanφ/a (exp 14)
            div = div - v * self.tan_phi_over_a
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
        """Tendency влажности: адвекция + крупномасштабная конденсация.

        Конденсация — стандартная схема насыщенной псевдоадиабаты
        (Haltiner & Williams): dq/dt|cond = δ·F(T, q_s)·ω/p, где δ=1 при
        подъёме (ω<0) и q >= q_s, иначе 0; F > 0, так что подъём сушит.
        Давление — реальные уровни (буфер ``pressure``, гПа → Па), а не
        реконструкция p=ρRT из z_z (та давала p<0 из-за конвенции d_z=−∂/∂p,
        шкалы гПа и универсальной R — баг B3 аудита 13). ω = −100·w, т.к.
        ``get_w`` возвращает вверх-положительную w = −ω [гПа/с].

        Args:
            u, v: горизонтальный ветер, ``(B, P, H, W)``, м/с.
            t: температура, K. w: вертикальная скорость из ``get_w``, гПа/с.
            q: удельная влажность, кг/кг.

        Returns:
            ``torch.Tensor`` ``(B, P, H, W)`` — dq/dt, (кг/кг)/с.
        """

        q_x = self._d_x(q)
        q_y = self._d_y(q)
        q_z = self._d_z(q)

        cond = self._condensation_source(t, q, w)
        if "q" in self.omega_free:
            # omega_free('q') (exp 15): убрать вертикальную ω-адвекцию, но
            # конденсация остаётся (зеркало physics.py get_q_dt).
            q_t = -(u * q_x + v * q_y) + cond
        else:
            q_t = -(u * q_x + v * q_y + w * q_z) + cond
        if self.clim_src_q is not None:
            q_t = q_t + self.clim_src_q
        return q_t

    def _saturation_specific_humidity(self, t: torch.Tensor) -> torch.Tensor:
        """q_s(T,p) по Магнусу (СИ), detached. Общий для конденсации и CC-моста.

        Args:
            t: температура ``(B, 13, H, W)``, K.

        Returns:
            ``torch.Tensor`` той же формы — насыщающая удельная влажность, кг/кг.
        """
        pressure_pa = pressure.to(t.dtype).to(t.device) * 100.0
        t_c = t - 273.15
        exponent = torch.clamp(17.67 * t_c / self.avoid_inf(t_c + 243.5), min=-20.0, max=20.0)
        e_s = 6.112 * torch.exp(exponent) * 100
        q_s = (0.622 * e_s / self.avoid_inf(pressure_pa - 0.378 * e_s)).detach()
        return torch.maximum(q_s, torch.ones_like(q_s) * 1e-6)

    def _condensation_source(self, t, q, w):
        """Конденсационный сток q: δ·F(T,q_s)·ω/p ≤ 0 (насыщенный подъём).

        Вынесен из :meth:`get_q_dt`, чтобы ``latent_heating_coupling`` мог
        добавить согласованное скрытое тепло −(L/c_p)·(dq/dt|cond) в t_t
        (Yanai et al. 1973: Q₂-сток обязан греть Q₁). Арифметика — бит-в-бит
        прежний хвост get_q_dt.

        Args:
            t: температура ``(B, P, H, W)``, K. q: влажность, кг/кг.
            w: вертикальная скорость из ``get_w``, гПа/с.

        Returns:
            ``torch.Tensor`` ``(B, P, H, W)`` — dq/dt|cond, (кг/кг)/с.
        """
        pressure_pa = pressure.to(t.dtype).to(t.device) * 100.0
        omega_pa = -100.0 * w
        q_s = self._saturation_specific_humidity(t)
        delta = torch.logical_and(omega_pa < 0, torch.ge(q, q_s)).to(q.dtype).detach()
        R_ = (1 + 0.608 * q) * self.R_d
        F_ = (self.L * R_ - self.c_p * self.R_v * t) / self.avoid_inf(
            self.c_p * self.R_v * t * t + self.L * self.L * q_s
        )
        F_ = (F_ * q_s * t).detach()
        return delta * F_ * omega_pa / pressure_pa

    def q_evolution(self, u, v, t, w, q):
        q_t = self.get_q_dt(u, v, t, w, q)
        if self.humidity_evolution == "cc_bridge":
            # Мост Клаузиуса–Клапейрона: наблюдаемая величина — r=q/q_s; чтобы r
            # эволюционировала под изменением T, к q-тенденции добавляется
            # q·(L/(R_v T²))·t_t (dq_s/dT-связка). self.t_t — полный кэш из t_evolution.
            cc_term = q * (self.L / (self.R_v * t * t)) * self.t_t
            q_t = q_t + cc_term
        return q + self._mask_increment(self._limit_increment(q_t * self.block_dt, q, "q"), "q")

    ################################################################

    def _evolve_fields(
        self,
        z_old: torch.Tensor,
        t_old: torch.Tensor,
        q_old: torch.Tensor,
        u_old: torch.Tensor,
        v_old: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Один шаг примитивных уравнений над (z, t, q, u, v); без norm/conv/router.

        Разделяемое ядро физики: диагностирует ω, кэширует производные z и
        интегрирует u, v, t, z, q на ``block_dt``. Одинаково вызывается штатным
        ``forward`` (после conv-микса, с последующим ``norm_*``) и
        ``physics_only_forward`` (над чистым состоянием, без ``norm_*``).

        Args:
            z_old, t_old, q_old, u_old, v_old: поля ``(B, variable_dim, H, W)``.

        Returns:
            Кортеж ``(z_new, t_new, q_new, u_new, v_new)`` тех же форм.
        """
        w_old = self.get_w(u_old, v_old)
        self.share_z_dxyz(z_old)
        # Кэш q для согласованного скрытого тепла в t_t (latent_heating_coupling).
        self._q_for_latent = q_old

        u_new, v_new = self.uv_evolution(u_old, v_old, w_old)
        t_new = self.t_evolution(u_old, v_old, w_old, t_old)
        z_new = self.z_evolution(z_old)
        q_new = self.q_evolution(u_old, v_old, t_old, w_old, q_old)
        return z_new, t_new, q_new, u_new, v_new

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

        z_new, t_new, q_new, u_new, v_new = self._evolve_fields(z_old, t_old, q_old, u_old, v_old)

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

    def physics_only_forward(self, zquvtw: torch.Tensor) -> torch.Tensor:
        """Чистый шаг примитивных уравнений над физическим состоянием (fix дефекта B).

        В отличие от ``forward``: НЕ подмешивает ``variable_norm(x)`` (случайный
        Conv2d, контаминировавший состояние z-масштабом), НЕ применяет ``norm_*``,
        ``variable_innorm``, ``block_norm`` и роутер. Интегрирует ровно то чистое
        физическое состояние, что подано. Используется режимом
        ``stable_physical_v2`` модели.

        Args:
            zquvtw: ``torch.Tensor`` формы ``(B, D, H, W)`` — физическое состояние
                ``[z, t, q, u, v]`` (channels-first, ``D = 5 * variable_dim``).

        Returns:
            ``torch.Tensor`` той же формы — эволюционированное физическое состояние.
        """
        z_old, t_old, q_old, u_old, v_old = zquvtw.chunk(5, dim=1)
        z_new, t_new, q_new, u_new, v_new = self._evolve_fields(z_old, t_old, q_old, u_old, v_old)
        return torch.cat([z_new, t_new, q_new, u_new, v_new], dim=1)


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
        physical_passthrough: bool = False,
        advection_form: str = "flux",
        metric_terms: bool = False,
        spherical_divergence: bool = False,
        rayleigh_friction: bool = False,
        vertical_scheme: str = "stencil",
        omega_free: tuple[str, ...] = (),
        latent_heating_coupling: bool = False,
        clim_sources_path: str | None = None,
        clim_sources_prefix: str = "C15_now__",
        physics_level_mask: dict[str, float] | None = None,
        physics_level_mask_learnable: bool = False,
        ekman_K_profile: tuple[float, ...] | None = None,
        humidity_evolution: str = "as_is",
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
            physical_passthrough: прокидывается в каждое ядро; включает
                ``physics_only_forward`` (чистая физика, fix дефекта B).
            physics_level_mask: прокидывается в каждое ``PDE_kernel`` (exp 19,
                B1) — см. ``PDE_kernel.__init__``. ``None`` (дефолт) — маска
                выключена, поведение бит-в-бит.
            physics_level_mask_learnable: прокидывается в каждое ``PDE_kernel``
                (exp 19, B1L) — см. ``PDE_kernel.__init__``. Дефолт False.
            ekman_K_profile: прокидывается в каждое ``PDE_kernel`` (exp 19,
                B2) — см. ``PDE_kernel.__init__``. ``None`` (дефолт) — член
                выключен, поведение бит-в-бит.
            humidity_evolution: прокидывается в каждое ``PDE_kernel`` (exp 19,
                B3) — см. ``PDE_kernel.__init__``. ``'as_is'`` (дефолт) —
                поведение бит-в-бит.
        """
        super().__init__()
        self.physical_passthrough = physical_passthrough
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
                    physical_passthrough=physical_passthrough,
                    advection_form=advection_form,
                    metric_terms=metric_terms,
                    spherical_divergence=spherical_divergence,
                    rayleigh_friction=rayleigh_friction,
                    vertical_scheme=vertical_scheme,
                    omega_free=omega_free,
                    latent_heating_coupling=latent_heating_coupling,
                    clim_sources_path=clim_sources_path,
                    clim_sources_prefix=clim_sources_prefix,
                    physics_level_mask=physics_level_mask,
                    physics_level_mask_learnable=physics_level_mask_learnable,
                    ekman_K_profile=ekman_K_profile,
                    humidity_evolution=humidity_evolution,
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

    def physics_only_forward(self, zquvtw: torch.Tensor) -> torch.Tensor:
        """Чистая цепочка физ-шагов над состоянием; без residual-skip (fix дефекта B).

        В отличие от ``forward``: НЕ добавляет skip-удвоение входа к выходу
        (``zquvtw + skip_zquvtw`` буквально удваивало физическое состояние по
        глубине блока). Возвращает эволюционированное состояние как есть.

        Args:
            zquvtw: ``torch.Tensor`` формы ``(B, H, W, D)`` — физическое состояние
                ``[z, t, q, u, v]`` (channels-last).

        Returns:
            ``torch.Tensor`` формы ``(B, H, W, D)`` — эволюционированное состояние.
        """
        zquvtw = zquvtw.permute(0, 3, 1, 2)  # [B, D, H, W]
        for pde_kernel in self.PDE_kernels:
            zquvtw = pde_kernel.physics_only_forward(zquvtw)
        return zquvtw.permute(0, 2, 3, 1)  # [B, H, W, D]


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
        physical_passthrough: bool = False,
        advection_form: str = "flux",
        metric_terms: bool = False,
        spherical_divergence: bool = False,
        rayleigh_friction: bool = False,
        vertical_scheme: str = "stencil",
        omega_free: tuple[str, ...] = (),
        latent_heating_coupling: bool = False,
        clim_sources_path: str | None = None,
        clim_sources_prefix: str = "C15_now__",
        physics_level_mask: dict[str, float] | None = None,
        physics_level_mask_learnable: bool = False,
        ekman_K_profile: tuple[float, ...] | None = None,
        humidity_evolution: str = "as_is",
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
            physical_passthrough: если True, ``physics_only_forward`` даёт чистую
                физику без роутера/conv-контаминации (fix дефекта B, режим
                ``stable_physical_v2`` модели). Штатный ``forward`` бит-в-бит.
            physics_level_mask: прокидывается в ``PDE_block`` → каждое
                ``PDE_kernel`` (exp 19, B1) — см. ``PDE_kernel.__init__``.
                ``None`` (дефолт) — маска выключена, поведение бит-в-бит.
            physics_level_mask_learnable: прокидывается в ``PDE_block`` →
                каждое ``PDE_kernel`` (exp 19, B1L) — см.
                ``PDE_kernel.__init__``. Дефолт False.
            ekman_K_profile: прокидывается в ``PDE_block`` → каждое
                ``PDE_kernel`` (exp 19, B2) — см. ``PDE_kernel.__init__``.
                ``None`` (дефолт) — член выключен, поведение бит-в-бит.
            humidity_evolution: прокидывается в ``PDE_block`` → каждое
                ``PDE_kernel`` (exp 19, B3) — см. ``PDE_kernel.__init__``.
                ``'as_is'`` (дефолт) — поведение бит-в-бит.
        """
        super().__init__()

        self.physical_passthrough = physical_passthrough
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
            physical_passthrough=physical_passthrough,
            advection_form=advection_form,
            metric_terms=metric_terms,
            spherical_divergence=spherical_divergence,
            rayleigh_friction=rayleigh_friction,
            vertical_scheme=vertical_scheme,
            omega_free=omega_free,
            latent_heating_coupling=latent_heating_coupling,
            clim_sources_path=clim_sources_path,
            clim_sources_prefix=clim_sources_prefix,
            physics_level_mask=physics_level_mask,
            physics_level_mask_learnable=physics_level_mask_learnable,
            ekman_K_profile=ekman_K_profile,
            humidity_evolution=humidity_evolution,
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

    def physics_only_forward(self, zquvtw: torch.Tensor) -> torch.Tensor:
        """Чистый физический путь для ``stable_physical_v2``: без роутера и conv.

        Args:
            zquvtw: ``torch.Tensor`` формы ``(B, H, W, D)`` — чистое физическое
                состояние ``[z, t, q, u, v]`` (channels-last).

        Returns:
            ``torch.Tensor`` формы ``(B, H, W, D)`` — эволюционированное состояние
                (``router_weight`` не участвует).
        """
        return self.pde_block.physics_only_forward(zquvtw)


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
