import torch
import numpy as np
import xarray as xr
import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

# ===== Исходные расчёты параметров дискретизации =====
latents_size = [8, 16]  # patch size = 4, input size [128, 256], latents size = [128/4, 256/4]
radius = 6371.0 * 1000
num_lat = latents_size[0] + 2
lat_t = torch.arange(start=0, end=num_lat)
# Функция для равномерного распределения широт от -90 до 90 градусов:
def lat(lat_t, num_lat):
    return torch.linspace(-90, 90, steps=num_lat)
latitudes = lat(lat_t, num_lat)[1:-1]
latitudes = latitudes / 180 * torch.pi  # перевод в радианы

c_lats = 2 * torch.pi * radius * torch.cos(latitudes)
c_lats = c_lats.reshape([1, 1, latents_size[0], 1])

pixel_x = c_lats / latents_size[1]  # горизонтальное расстояние (ось x)
pixel_y = torch.pi * radius / (latents_size[0] + 1)  # вертикальное расстояние (ось y)

pressure = torch.tensor([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]).reshape([1, 13, 1, 1])
pixel_z = torch.tensor([50, 50, 50, 50, 50, 75, 100, 100, 100, 125, 112, 75, 75]).reshape([1, 13, 1, 1])

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
        u_0  = u                                # u_i
        u_p1 = torch.roll(u, shifts=-1, dims=-1)  # u_{i+1}
        u_p2 = torch.roll(u, shifts=-2, dims=-1)  # u_{i+2}
    elif boundary == "reflect":
        # Отражающее дополнение: pad=(2,2) по последней оси
        u_pad = F.pad(u, pad=(2, 2), mode="reflect")
        u_m2 = u_pad[..., 0:-4]
        u_m1 = u_pad[..., 1:-3]
        u_0  = u_pad[..., 2:-2]
        u_p1 = u_pad[..., 3:-1]
        u_p2 = u_pad[..., 4:]
    else:
        raise ValueError("Unknown boundary condition")
        
    f1 = (2 * u_m2 - 7 * u_m1 + 11 * u_0) / 6.0
    f2 = (-u_m1 + 5 * u_0 + 2 * u_p1) / 6.0
    f3 = (2 * u_0 + 5 * u_p1 - u_p2) / 6.0

    beta1 = (13/12.0) * (u_m2 - 2*u_m1 + u_0)**2 + (1/4.0) * (u_m2 - 4*u_m1 + 3*u_0)**2
    beta2 = (13/12.0) * (u_m1 - 2*u_0 + u_p1)**2 + (1/4.0) * (u_m1 - u_p1)**2
    beta3 = (13/12.0) * (u_0 - 2*u_p1 + u_p2)**2 + (1/4.0) * (3*u_0 - 4*u_p1 + u_p2)**2

    d1, d2, d3 = 0.1, 0.6, 0.3
    alpha1 = d1 / (epsilon + beta1)**2
    alpha2 = d2 / (epsilon + beta2)**2
    alpha3 = d3 / (epsilon + beta3)**2

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

def d_x_weno(input_tensor, boundary="reflect"):
    """
    Вычисляет производную по оси 3 (ширина) с использованием WENO 5-го порядка.
    Результат масштабируется с помощью pixel_x.
    """
    B, C, H, W = input_tensor.shape
    input_flat = input_tensor.reshape(B * C * H, W)
    dx_flat = pixel_x.expand(B, C, H, 1).reshape(B * C * H)
    derivative_flat = weno_derivative(input_flat, dx_flat, boundary=boundary)
    derivative = derivative_flat.reshape(B, C, H, W)
    return derivative

def d_y_weno(input_tensor, boundary="reflect"):
    """
    Вычисляет производную по оси 2 (высота) с использованием WENO 5-го порядка.
    Результат масштабируется с помощью pixel_y.
    """
    B, C, H, W = input_tensor.shape
    input_perm = input_tensor.permute(0, 1, 3, 2)
    input_flat = input_perm.reshape(B * C * W, H)
    derivative_flat = weno_derivative(input_flat, pixel_y, boundary=boundary)
    derivative_perm = derivative_flat.reshape(B, C, W, H)
    derivative = derivative_perm.permute(0, 1, 3, 2)
    return derivative

# Используем функции d_x_weno и d_y_weno напрямую
d_x = d_x_weno
d_y = d_y_weno

def d_z(input_tensor):
    # Вертикальная производная по давлению без изменений (используется периодическая логика через concat)
    conv_kernel = torch.zeros([1, 1, 5, 1, 1], device=input_tensor.device, dtype=input_tensor.dtype, requires_grad=False)
    conv_kernel[0, 0, 0] = -1
    conv_kernel[0, 0, 1] = 8
    conv_kernel[0, 0, 3] = -8
    conv_kernel[0, 0, 4] = 1

    input_tensor = torch.cat((input_tensor[:, :2],
                              input_tensor,
                              input_tensor[:, -2:]), dim=1)
    input_tensor = input_tensor.unsqueeze(1)  # [B, 1, C, H, W]
    output_z = F.conv3d(input_tensor, conv_kernel) / 12
    output_z = output_z.squeeze(1)
    output_z = output_z / pixel_z.to(output_z.dtype).to(output_z.device)
    return output_z

def laplacian_tensor(u):
    d2u_dx2 = d_x(d_x(u))
    d2u_dy2 = d_y(d_y(u))
    return d2u_dx2 + d2u_dy2

# ===== Функции для адаптивного уточнения сетки (AMR) =====

def adaptive_mesh_refinement(field, grad_threshold=1e-3, upscale_factor=2):
    """
    Если максимальное значение градиента (вычисляемого с помощью d_x и d_y) превышает grad_threshold,
    возвращает уточнённое поле с увеличенным разрешением (с использованием F.interpolate).
    """
    grad_field = torch.sqrt(d_x(field)**2 + d_y(field)**2)
    if grad_field.max() > grad_threshold:
        refined_field = F.interpolate(field, scale_factor=upscale_factor, mode='bilinear', align_corners=True)
        return refined_field, True
    else:
        return field, False

def compute_derivative_with_amr(field, derivative_fn, grad_threshold=1e-3, upscale_factor=2, boundary="reflect"):
    """
    Вычисляет производную поля с использованием AMR.
    Если поле имеет сильные градиенты (макс. значение > grad_threshold), оно уточняется,
    затем вычисляется производная на уточнённом поле, и результат обратно интерполируется до исходного разрешения.
    """
    refined_field, refined = adaptive_mesh_refinement(field, grad_threshold, upscale_factor)
    if refined:
        refined_deriv = derivative_fn(refined_field, boundary=boundary)
        deriv = F.interpolate(refined_deriv, scale_factor=1/upscale_factor, mode='bilinear', align_corners=True)
        return deriv
    else:
        return derivative_fn(field, boundary=boundary)

# ===== Класс PDE_kernel с учётом бета-подхода, улучшенных граничных условий и AMR =====
#
# Коэффициент Кориолиса определяется как:
#     f = f0 + beta * y,
# где y = R * lat (меридиональное расстояние в метрах).
#
class PDE_kernel(nn.Module):
    def __init__(self, in_dim, physics_part_coef, variable_dim=13, block_dt=300, inverse_time=False, norm=False, eddy_viscosity=0.0,
                 beta=1.6e-11, f0=7.29e-5):
        """
        eddy_viscosity: коэффициент вихревой вязкости для субрешеточной турбулентности.
        beta: коэффициент бета (с^-1 м^-1) для вариации f по меридионали.
        f0: базовое значение коэффициента Кориолиса.
        """
        super().__init__()
        self.norm = norm
        self.eddy_viscosity = eddy_viscosity
        
        self.f0 = f0
        self.beta = beta
        # Вычисляем меридиональное расстояние y = R * lat для каждого пикселя по широте.
        y_coords = radius * latitudes  # в метрах
        # f_field имеет форму [1, 1, H, 1] для вещания с полями [B, C, H, W]
        f_field = self.f0 + self.beta * y_coords
        self.register_buffer("f_field", f_field.reshape(1, 1, -1, 1))

        self.variable_norm = nn.Conv2d(in_channels=in_dim, out_channels=variable_dim*5, kernel_size=3, stride=1, padding=1)
        if physics_part_coef is not None:
            self.physics_part_coef = physics_part_coef
        else: # Activate learnable matrix for these coefs: shape C x W x H
            self.physics_part_coef = nn.Parameter(0.5 * torch.ones(1, variable_dim*5, 32, 64), requires_grad=True) # 32 and 64 is for H/W grid
        
        
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

        self.variable_innorm = nn.Conv2d(in_channels=variable_dim*5, out_channels=in_dim, kernel_size=3, stride=1, padding=1)
        self.block_norm = nn.BatchNorm2d(in_dim)

    def scale_tensor(self, tensor, a, b):
        min_val = tensor.min().detach()
        max_val = tensor.max().detach()
        scaled_tensor = (tensor - min_val) / (max_val - min_val)
        return scaled_tensor * (b - a) + a
    
    def scale_diff(self, diff_x, x):
        x_min, x_mean, x_max = x.min().detach(), x.mean().detach(), x.max().detach()
        diff_min = (x_min - x_mean) * self.diff_ratio
        diff_max = (x_max - x_mean) * self.diff_ratio
        return self.scale_tensor(diff_x, diff_min, diff_max)
    
    def avoid_inf(self, tensor, threshold=1.0):
        tensor = torch.where(torch.abs(tensor) == 0.0, torch.ones_like(tensor) * 0.1, tensor)
        return torch.where(torch.abs(tensor) < threshold, torch.sign(tensor) * threshold, tensor)

    def share_z_dxyz(self, z):
        self.z_x = d_x(z)
        self.z_y = d_y(z)
        self.z_z = d_z(z)

    ############################# u, v #############################
    def get_uv_dt(self, u, v, w):
        # Консервативное представление нелинейных членов с применением AMR для уточнения
        adv_u = compute_derivative_with_amr(u * u, d_x) \
              + compute_derivative_with_amr(u * v, d_y) \
              + d_z(u * w)  # вертикальная производная без AMR
        adv_v = compute_derivative_with_amr(u * v, d_x) \
              + compute_derivative_with_amr(v * v, d_y) \
              + d_z(v * w)
        
        # Используем f_field (вариация по широте)
        self.u_t = -adv_u + self.f_field * v - self.z_x
        self.v_t = -adv_v - self.f_field * u - self.z_y

        # Параметризация субрешеточной турбулентности через вихревую вязкость
        if self.eddy_viscosity > 0:
            lap_u = laplacian_tensor(u)
            lap_v = laplacian_tensor(v)
            self.u_t += self.eddy_viscosity * lap_u
            self.v_t += self.eddy_viscosity * lap_v

        return self.u_t, self.v_t
    
    def uv_evolution(self, u, v, w):
        u_t, v_t = self.get_uv_dt(u, v, w)
        u = u + self.scale_diff(u_t * self.block_dt, u).detach()
        v = v + self.scale_diff(v_t * self.block_dt, v).detach()
        return u, v
    ################################################################
    
    ############################# t #############################
    def get_t_t(self, u, v, w, t):
        t_x = d_x(t)
        t_y = d_y(t)
        t_z = d_z(t)
        Q = -self.L * self.z_z * w
        self.t_t = (Q - self.z_z * w) / self.c_p - u * t_x - v * t_y - w * t_z
        return self.t_t
    
    def t_evolution(self, u, v, w, t):
        t_t = self.get_t_t(u, v, w, t)
        return t + self.scale_diff(t_t * self.block_dt, t).detach()
    ################################################################

    ############################# z #############################
    def get_z_zt(self):    
        return -self.R / pressure.to(self.t_t.dtype).to(self.t_t.device) * self.t_t
    
    def get_z_t(self):
        z_zt = self.get_z_zt()
        self.z_t = integral_z(z_zt)
        return self.z_t
    
    def z_evolution(self, z):
        z_t = self.get_z_t()
        return z + self.scale_diff(z_t * self.block_dt, z).detach()
    ################################################################

    ############################# w #############################
    def get_w(self, u, v):
        self.u_x = d_x(u)
        self.v_y = d_y(v)
        w_z = - self.u_x - self.v_y
        return integral_z(w_z).detach()
    ################################################################

    ############################# q #############################
    def get_q_dt(self, u, v, t, w, q):
        def get_qs(p, T):
            t_c = T - 273.15
            e_s = 6.112 * torch.exp(self.scale_tensor(17.67 * t_c / self.avoid_inf(t_c + 243.5), -3.47, 3.01)) * 100
            return 0.622 * e_s / self.avoid_inf(p - 0.378 * e_s)

        def get_delta(p_t, q, q_s):
            cond = torch.logical_and(p_t < 0, torch.ge(q, q_s))
            return torch.where(cond, torch.ones_like(p_t), torch.zeros_like(p_t))

        def get_F(T, q, q_s):
            R_ = (1 + 0.608 * q) * self.R_d
            F_ = (self.L * R_ - self.c_p * self.R_v * T) / self.avoid_inf(self.c_p * self.R_v * T * T + self.L * self.L * q_s)
            return F_ * q_s * T

        q_x = d_x(q)
        q_y = d_y(q)
        q_z = d_z(q)

        rho = -1 / self.avoid_inf(self.z_z)
        p = rho * self.R * t

        q_s = get_qs(p, t).detach()
        q_s = torch.maximum(q_s, torch.ones_like(q_s) * 1e-6)
        delta = get_delta(self.z_t + u * self.z_x + v * self.z_y + w * self.z_z, q, q_s).detach()
        F_ = get_F(t, q, q_s).detach()

        q_t = -(u * q_x + v * q_y + w * q_z) + (self.z_t + u * self.z_x + v * self.z_y + w * self.z_z) * delta * F_ / self.avoid_inf(self.R * t)
        return q_t
    
    def q_evolution(self, u, v, t, w, q):
        q_t = self.get_q_dt(u, v, t, w, q)
        return q + self.scale_diff(q_t * self.block_dt, q).detach()
    ################################################################


    def forward(self, x, zquvtw):
        # x [B, D, H, W]
        skip = x

        ################################################################
        zquvtw_old = (1 - self.physics_part_coef)*self.variable_norm(x) + self.physics_part_coef*zquvtw
        z_old, t_old, q_old, u_old, v_old= zquvtw_old.chunk(5, dim=1)

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
    def __init__(self, in_dim, variable_dim, physics_part_coef, depth=3, block_dt=300, inverse_time=False):
        super().__init__()
        self.PDE_kernels = nn.ModuleList([])
        for _ in range(depth):
            self.PDE_kernels.append(PDE_kernel(in_dim, variable_dim=variable_dim, block_dt=block_dt, inverse_time=inverse_time, physics_part_coef=physics_part_coef))
    
    def forward(self, x, zquvtw):
        # x [B, H, W, D]
        skip_x, skip_zquvtw = x, zquvtw
        x, zquvtw = x.permute(0,3,1,2), zquvtw.permute(0,3,1,2)  # [B, D, H, W]
        for PDE_kernel in self.PDE_kernels:
            x, zquvtw = PDE_kernel(x, zquvtw)
        x, zquvtw = x.permute(0,2,3,1), zquvtw.permute(0,2,3,1)
        return x+skip_x, zquvtw+skip_zquvtw # x [B, H, W, D]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
     
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class SwiGLU(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class GatedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)       
            
    def forward(self, x):
        for attn, ff, drop_path1,drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)
    
class PredFormerLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super(PredFormerLayer, self).__init__()

        self.temporal_transformer_first = GatedTransformer(dim, depth, heads, dim_head, 
                                                      mlp_dim, dropout, attn_dropout, drop_path)
        self.space_transformer = GatedTransformer(dim, depth, heads, dim_head, 
                                             mlp_dim, dropout, attn_dropout, drop_path)
        self.temporal_transformer_second = GatedTransformer(dim, depth, heads, dim_head, 
                                                       mlp_dim, dropout, attn_dropout, drop_path)

    def forward(self, x):
        b, t, n, _ = x.shape        
        x_t, x_ori = x, x 
        
        # t branch (first temporal)
        x_t = rearrange(x_t, 'b t n d -> b n t d')
        x_t = rearrange(x_t, 'b n t d -> (b n) t d')
        x_t = self.temporal_transformer_first(x_t)
        
        # s branch (space)
        x_ts = rearrange(x_t, '(b n) t d -> b n t d', b=b)
        x_ts = rearrange(x_ts, 'b n t d -> b t n d')
        x_ts = rearrange(x_ts, 'b t n d -> (b t) n d') 
        x_ts = self.space_transformer(x_ts)
        
        # t branch (second temporal)
        x_tst = rearrange(x_ts, '(b t) n d -> b t n d', b=b)
        x_tst = rearrange(x_tst, 'b t n d -> b n t d')
        x_tst = rearrange(x_tst, 'b n t d -> (b n) t d')
        x_tst = self.temporal_transformer_second(x_tst)

        # ts output branch     
        x_tst = rearrange(x_tst, '(b n) t d -> b n t d', b=b)
        x_tst = rearrange(x_tst, 'b n t d -> b t n d', b=b) 
  
        # add residual connection, we only add this for human3.6m  
        # x_tst += x_ori
        
        return x_tst

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

      
class PredFormer_Model(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
        self.image_height = model_config['height']
        self.image_width = model_config['width']
        self.patch_size = model_config['patch_size']
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = model_config['pre_seq']
        self.dim = model_config['dim']
        self.num_channels = model_config['num_channels']
        self.num_classes = self.num_channels
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']  # Ensure this is defined
        self.depth = model_config['depth']  # Ensure this is defined
        self.path_to_constants = model_config['path_to_constants']
        self.ds = xr.open_dataset(self.path_to_constants)
        self.orography_mask = torch.Tensor(np.array(self.ds.orography)) # shape = [H, W]
        self.soil_mask = torch.Tensor(np.array(self.ds.slt)) # shape = [H, W]
        self.lsm_mask = torch.Tensor(np.array(self.ds.lsm)) # shape = [H, W]
        self.static_masks = torch.stack([self.orography_mask, self.soil_mask, self.lsm_mask], dim=0).unsqueeze(0) # [1, 3, H, W]
        self.num_masks = 3 # Определяем количество масок
        self.downscaling_factor_all = 4  # Default downscaling factor for GFT
        self.gft_weight = 0.1  # Вес физических эмбеддингов при добавлении к основным данным

        self.static_masks = self.static_masks[..., 128 - 92:128 - 60, 256 - 131:256 - 67]

        assert self.image_height % self.patch_size == 0, 'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, 'Image width must be divisible by the patch size.'
        self.patch_dim = self.num_channels * self.patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
        )
        
        self.mask_patch_dim = self.num_masks * self.patch_size ** 2
        self.rearrange_masks = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        self.mask_embedding = nn.Linear(self.mask_patch_dim, self.dim)

        self.pos_embedding = nn.Parameter(sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),
                                               requires_grad=False).view(1, self.num_frames_in, self.num_patches, self.dim)

        self.blocks = nn.ModuleList([
            PredFormerLayer(self.dim, self.depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout, self.attn_dropout, self.drop_path)
            for i in range(self.Ndepth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_channels * self.patch_size ** 2)
            ) 
        
        # Создаем HybridBlock для GFT
        self._init_gft_block()
        
    def _init_gft_block(self):
        """Инициализирует HybridBlock для GFT с оптимальными параметрами"""
        # Размерность входных данных (без первых 4 каналов)
        input_dim = 65  # Обычно 65 каналов для физических переменных
        
        self.hybrid_block = HybridBlock(
            dim=input_dim,             # Размерность входного канала
            zquvtw_channel=13,         # 13 вертикальных уровней
            depth=3,                   # Глубина блока PDE
            block_dt=300,              # Временной шаг для PDE в секундах
            inverse_time=False,        # Не инвертировать временную эволюцию
            physics_part_coef=0.5      # Равный вес для AI и физики
        )
        
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
            size=(H//self.downscaling_factor_all, W//self.downscaling_factor_all), 
            mode='bilinear'
        )
        
        # Перестановка осей для формата [B, H, W, C], который ожидает HybridBlock
        zquvtw = zquvtw.permute(0, 2, 3, 1)  # [B, H//4, W//4, C]
        
        return zquvtw
                
     
    def forward(self, x):
        B, T, C, H, W = x.shape
        assert C == self.num_channels

        x_gft_list = []
        for i in range(T - 1):
            # Применяем GFT с помощью hybrid_block
            x_patch = x[:, i, 4:, :, :]  # Берем только каналы с индекса 4, получая Z, Q, U, V, T
            # x_patch = x_patch.permute(0, 2, 3, 1)  # [B, H, W, C]
            
            # Получаем входные данные для hybrid_block
            zquvtw = self.x_to_zquvtw(x_patch)
            x_patch = x_patch.permute(0, 2, 3, 1)

            print('shapes: x_patch', x_patch.shape, 'zqutw', zquvtw.shape)
            for j in range(12):
                # Получаем физические эмбеддинги через hybrid_block
                x_patch, zquvtw = self.hybrid_block(x_patch, zquvtw)  # Используем одинаковые данные для обоих входов
            
            x_gft = x_patch
            # Возвращаем к исходному формату
            x_gft = x_gft.permute(0, 3, 1, 2)  # [B, C, H//4, W//4]
            
            # Масштабируем обратно до исходного размера
            x_gft = torch.nn.functional.interpolate(x_gft, size=(H, W), mode='bilinear')
            
            x_gft_list.append(x_gft)

        # Создаем новый список с нулевым тензором в начале и элементами из оригинального списка, исключая последний
        zero_tensor = torch.zeros_like(x[:, 0, 4:, :, :], device=x.device)
        x_gft_list = [zero_tensor] + x_gft_list # [B, 0, C-4, H, W] + [B, T-1, C-4, H, W]
        # Объединяем результаты в один тензор
        x_gft = torch.stack(x_gft_list, dim=1)  # [B, T, C-4, H, W]
        
        # Добавляем физические эмбеддинги к исходным данным
        x_with_physics = x.clone()
        
        # Проверяем соответствие размерностей
        channels_physics = x[:, :, 4:, :, :].shape[2]

        if x_gft.shape[2] == channels_physics:
            x_with_physics[:, :, 4:, :, :] = x[:, :, 4:, :, :] + self.gft_weight * x_gft
        else:
            # Если размерности не совпадают, выполняем дополнительное преобразование
            
            if x_gft.shape[2] < channels_physics:
                # Дополняем нулями
                padding = torch.zeros(B, T, channels_physics - x_gft.shape[2], H, W, device=x.device)
                x_gft = torch.cat([x_gft, padding], dim=2)
            else:
                # Обрезаем лишние каналы
                x_gft = x_gft[:, :, :channels_physics, :, :]
            
            x_with_physics[:, :, 4:, :, :] = x[:, :, 4:, :, :] + self.gft_weight * x_gft
            
        x = x_with_physics

        mask_patches = self.rearrange_masks(self.static_masks.to(x.device)) # [1, num_patches, 3*ps*ps]
        mask_embed = self.mask_embedding(mask_patches) # [1, num_patches, dim]
        mask_embed = mask_embed.unsqueeze(1) # [1, 1, num_patches, dim]
        mask_embed = mask_embed.to(x.device) 

        # Patch Embedding для входа x
        x_embed = self.to_patch_embedding(x) # [B, T, num_patches, dim]
        
        x_combined = x_embed + mask_embed       

        # Position Embedding
        x_combined += self.pos_embedding.to(x.device)
        
        # PredFormer Encoder
        for idx, blk in enumerate(self.blocks):
            x_combined = blk(x_combined)
        # MLP head        
        x = self.mlp_head(x_combined.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        
        return x


class HybridBlock(nn.Module):
    def __init__(self, dim, zquvtw_channel, depth, block_dt, inverse_time, physics_part_coef):
        super().__init__()
        
        self.pde_block = PDE_block(dim, zquvtw_channel, depth=depth, block_dt=block_dt, inverse_time=inverse_time, physics_part_coef=physics_part_coef)
        self.router_weight = nn.Parameter(torch.zeros(1, 1, 1, dim), requires_grad=True)

    def forward(self, x, zquvtw=None):

        feat_pde, zquvtw = self.pde_block(x, zquvtw)
        
        # Adaptive Router
        weight_AI = 0.5*torch.ones_like(x)+self.router_weight
        weight_Physics = 0.5*torch.ones_like(x)-self.router_weight
        x = weight_AI*zquvtw + weight_Physics*feat_pde
        return x, zquvtw



# model_config = {
#     # image h w c
#     'height': 128,
#     'width': 256,
#     'num_channels': 69,
#     # video length in and out
#     'pre_seq': 12,
#     'after_seq': 12,
#     # patch size
#     'patch_size': 8,
#     'dim': 256, 
#     'heads': 8,
#     'dim_head': 32,
#     # dropout
#     'dropout': 0.0,
#     'attn_dropout': 0.0,
#     'drop_path': 0.0,
#     'scale_dim': 4,
#     # depth
#     'depth': 1,
#     'Ndepth': 24,
#     'path_to_constants': '/home/user/mamba_x_predformer/PredFormer/constants_1.40625deg.nc',
# }

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# print("Start testing PredFormer model")
# model = PredFormer_Model(model_config).to(device)
# # Count and print the total number of parameters in the model
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total parameters: {total_params:,}")
# print(f"Trainable parameters: {trainable_params:,}")
# print(f"Model size: {total_params * 4 / (1024 * 1024):.2f} MB")

# print("Model loaded successfully")
# x = torch.rand(1, 12, 69, 128, 256).to(device)
# print("Input tensor created successfully")
# output = model(x)
# print("Output tensor created successfully")
# print(output.shape)  # [B, T, C, H, W]
