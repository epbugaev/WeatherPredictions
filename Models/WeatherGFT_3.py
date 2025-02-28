import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import math
import types  # For method type binding in the utility functions
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper


@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

latents_size = [32, 64] # patch size = 4, input size [128, 256], latents size = [128/4, 256/4]

radius = 6371.0 * 1000
num_lat = latents_size[0] + 2
lat_t = torch.arange(start=0, end=num_lat)
latitudes = lat(lat_t, num_lat)[1:-1]
latitudes = latitudes/180*torch.pi

c_lats = 2*torch.pi*radius*torch.cos(latitudes)
c_lats = c_lats.reshape([1, 1, latents_size[0], 1])

pixel_x = c_lats/latents_size[1] # The actual distance each pixel corresponds to in the horizontal direction
pixel_y = torch.pi*radius/(latents_size[0]+1) # The actual distance each pixel corresponds to in the vertical direction

pressure = torch.tensor([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]).reshape([1, 13, 1, 1])
pixel_z = torch.tensor([50, 50, 50, 50, 50, 75, 100, 100, 100, 125, 112, 75, 75]).reshape([1, 13, 1, 1]) # The difference between adjacent pressure levels, which will be used to calculate the p-direction integral


pressure_level_num = pixel_z.shape[1]
M_z = torch.zeros(pressure_level_num, pressure_level_num)
for M_z_i in range(pressure_level_num):
    for M_z_j in range(pressure_level_num):
        if M_z_i <= M_z_j:
            M_z[M_z_i, M_z_j] = pixel_z[0, M_z_j, 0, 0]


def integral_z(input_tensor):
    # Pressure-direction integral
    B, pressure_level_num, H, W = input_tensor.shape
    input_tensor = input_tensor.reshape(B, pressure_level_num, H*W)
    output = M_z.to(input_tensor.dtype).to(input_tensor.device) @ input_tensor
    output = output.reshape(B, pressure_level_num, H, W)
    return output


def d_x(input_tensor):
    # Latitude-direction differential
    B, C, H, W = input_tensor.shape
    conv_kernel = torch.zeros([1,1,1,5], device=input_tensor.device, dtype=input_tensor.dtype, requires_grad=False)
    conv_kernel[0,0,0,0] = 1
    conv_kernel[0,0,0,1] = -8
    conv_kernel[0,0,0,3] = 8
    conv_kernel[0,0,0,4] = -1

    input_tensor = torch.cat((input_tensor[:,:,:,-2:], 
                              input_tensor,
                              input_tensor[:,:,:,:2]), dim=3)
    _, _, H_, W_ = input_tensor.shape
    
    input_tensor = input_tensor.reshape(B*C, 1, H_, W_)
    output_x = F.conv2d(input_tensor, conv_kernel)/12
    output_x = output_x.reshape(B, C, H, W)
    output_x = output_x/pixel_x.to(output_x.dtype).to(output_x.device)
    
    return output_x


def d_y(input_tensor):
    # longitude-direction differential
    B, C, H, W = input_tensor.shape
    conv_kernel = torch.zeros([1,1,5,1], device=input_tensor.device, dtype=input_tensor.dtype, requires_grad=False)
    conv_kernel[0,0,0] = -1
    conv_kernel[0,0,1] = 8
    conv_kernel[0,0,3] = -8
    conv_kernel[0,0,4] = 1

    input_tensor = torch.cat((input_tensor[:,:,:2], 
                              input_tensor,
                              input_tensor[:,:,-2:]), dim=2)
    _, _, H_, W_ = input_tensor.shape
    
    input_tensor = input_tensor.reshape(B*C, 1, H_, W_)
    output_y = F.conv2d(input_tensor, conv_kernel)/12
    output_y = output_y.reshape(B, C, H, W)
    output_y = output_y/pixel_y
    
    return output_y


def d_z(input_tensor):
    # Pressure-direction differential
    conv_kernel = torch.zeros([1,1,5,1,1], device=input_tensor.device, dtype=input_tensor.dtype, requires_grad=False)
    conv_kernel[0,0,0] = -1
    conv_kernel[0,0,1] = 8
    conv_kernel[0,0,3] = -8
    conv_kernel[0,0,4] = 1

    input_tensor = torch.cat((input_tensor[:,:2], 
                              input_tensor,
                              input_tensor[:,-2:]), dim=1)
    
    input_tensor = input_tensor.unsqueeze(1) # B, 1, C, H, W
    output_z = F.conv3d(input_tensor, conv_kernel)/12
    output_z = output_z.squeeze(1)
    output_z = output_z/pixel_z.to(output_z.dtype).to(output_z.device)
    
    return output_z


class PDE_kernel(nn.Module):
    def __init__(self, in_dim, physics_part_coef, variable_dim=13, block_dt=300, inverse_time=False):
        super().__init__()
        self.in_dim = in_dim
        self.variable_dim = variable_dim

        self.variable_norm = nn.Conv2d(in_channels=in_dim, out_channels=variable_dim*5, kernel_size=3, stride=1, padding=1)
        if physics_part_coef is not None:
            self.physics_part_coef = physics_part_coef
        else: # Activate learnable matrix for these coefs: shape C x W x H
            self.physics_part_coef = nn.Parameter(0.5 * torch.ones(1, variable_dim*5, 32, 64), requires_grad=True) # 32 and 64 is for H/W grid
        
        # Coriolis parameter is now computed based on latitude - no longer a constant
        self.omega = 7.29e-5  # Earth's angular velocity in rad/s
        
        self.L = 2.5e6        # Latent heat of vaporization (J/kg)
        self.R = 8.314        # Universal gas constant (J/(mol·K))
        self.c_p = 1005       # Specific heat capacity of air at constant pressure (J/(kg·K))
        self.R_v = 461.5      # Gas constant for water vapor (J/(kg·K))
        self.R_d = 287        # Gas constant for dry air (J/(kg·K))
        self.diff_ratio = 0.05
        self.block_dt = block_dt
        if inverse_time:
            self.block_dt = - self.block_dt
            
        # Parameters for turbulent mixing
        self.k_h = 15.0       # Horizontal eddy diffusivity (m²/s)
        self.k_v = 0.1        # Vertical eddy diffusivity (m²/s)
        
        # Parameters for radiative cooling
        self.sigma = 5.67e-8  # Stefan-Boltzmann constant (W/(m²·K⁴))
        self.emissivity = 0.7 # Atmospheric emissivity
        
        # Parameters for precipitation
        self.precip_threshold = 0.8  # Relative humidity threshold for precipitation

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
        scaled_tensor = scaled_tensor * (b - a) + a
        return scaled_tensor
    

    def scale_diff(self, diff_x, x):
        x_min, x_mean, x_max = x.min().detach(), x.mean().detach(), x.max().detach()
        diff_min = (x_min-x_mean) * self.diff_ratio
        diff_max = (x_max-x_mean) * self.diff_ratio
        diff_x = self.scale_tensor(diff_x, diff_min, diff_max)
        return diff_x
    
    def avoid_inf(self, tensor, threshold=1.0):
        tensor = torch.where(torch.abs(tensor) == 0.0, torch.ones_like(tensor)*0.1, tensor)
        tensor = torch.where(torch.abs(tensor) < threshold, torch.sign(tensor) * threshold, tensor)
        return tensor

    def get_coriolis_parameter(self, lat):
        # Compute the Coriolis parameter based on latitude
        # f = 2*Ω*sin(φ) where Ω is Earth's angular velocity and φ is latitude
        return 2 * self.omega * torch.sin(lat)

    def share_z_dxyz(self, z):
        self.z_x = d_x(z)
        self.z_y = d_y(z)
        self.z_z = d_z(z)

    def apply_turbulent_mixing(self, field):
        # Apply horizontal and vertical turbulent mixing (diffusion)
        field_xx = d_x(d_x(field))
        field_yy = d_y(d_y(field))
        field_zz = d_z(d_z(field))
        
        horizontal_mixing = self.k_h * (field_xx + field_yy)
        vertical_mixing = self.k_v * field_zz
        
        return horizontal_mixing + vertical_mixing

    def calculate_radiative_cooling(self, t):
        # Simple radiative cooling parameterization
        # Approximate cooling rate based on Stefan-Boltzmann law
        # More realistic models would use band models or line-by-line calculations
        
        # Convert to actual temperature in Kelvin assuming t is temperature anomaly
        t_abs = t + 273.15
        
        # Net cooling rate (W/m²)
        cooling_rate = self.emissivity * self.sigma * t_abs**4
        
        # Convert to temperature tendency (K/s)
        dt_rad = -cooling_rate / (self.c_p * pressure.to(t.device) * 100 / (self.R_d * t_abs))
        
        return dt_rad

    def calculate_convective_precipitation(self, q, q_s):
        # Simple parameterization for convective precipitation
        # Based on relative humidity exceeding a threshold
        rel_humidity = q / self.avoid_inf(q_s)
        precip_mask = (rel_humidity > self.precip_threshold).float()
        
        # Calculate precipitation rate (kg/m²/s)
        precip_rate = precip_mask * (q - self.precip_threshold * q_s) / self.block_dt
        
        # Calculate associated latent heating (K/s)
        latent_heating = precip_rate * self.L / self.c_p
        
        return precip_rate, latent_heating

    ############################# u v #############################
    def get_uv_dt(self, u, v, w):
        u_x = self.u_x
        u_y = d_y(u)
        u_z = d_z(u)

        v_x = d_x(v)
        v_y = self.v_y
        v_z = d_z(v)
        
        # Get latitude-dependent Coriolis parameter
        B, C, H, W = u.shape
        lat_grid = latitudes.to(u.device).reshape(1, 1, H, 1).expand(B, C, H, W)
        f_coriolis = self.get_coriolis_parameter(lat_grid)

        # Apply turbulent mixing to momentum equations
        u_mixing = self.apply_turbulent_mixing(u)
        v_mixing = self.apply_turbulent_mixing(v)

        self.u_t = -u*u_x - v*u_y - w*u_z + f_coriolis*v - self.z_x + u_mixing
        self.v_t = -u*v_x - v*v_y - w*v_z - f_coriolis*u - self.z_y + v_mixing
        return self.u_t, self.v_t
    
    # Modified RK4 integration for u and v
    def uv_evolution(self, u, v, w):
        # First, store original values
        u_orig, v_orig = u.clone(), v.clone()
        
        # Stage 1
        k1_u, k1_v = self.get_uv_dt(u, v, w)
        u1 = u + 0.5 * self.block_dt * k1_u
        v1 = v + 0.5 * self.block_dt * k1_v
        
        # Stage 2
        k2_u, k2_v = self.get_uv_dt(u1, v1, w)
        u2 = u + 0.5 * self.block_dt * k2_u
        v2 = v + 0.5 * self.block_dt * k2_v
        
        # Stage 3
        k3_u, k3_v = self.get_uv_dt(u2, v2, w)
        u3 = u + self.block_dt * k3_u
        v3 = v + self.block_dt * k3_v
        
        # Stage 4
        k4_u, k4_v = self.get_uv_dt(u3, v3, w)
        
        # Combine stages with appropriate weights
        u_tendency = (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6.0
        v_tendency = (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6.0
        
        # Apply scaled tendency
        u_new = u_orig + self.scale_diff(u_tendency * self.block_dt, u_orig).detach()
        v_new = v_orig + self.scale_diff(v_tendency * self.block_dt, v_orig).detach()
        
        return u_new, v_new
    ################################################################
    
    ############################# t #############################
    def get_t_t(self, u, v, w, t, q, q_s=None):
        t_x = d_x(t)
        t_y = d_y(t)
        t_z = d_z(t)

        # Basic adiabatic heating/cooling
        Q = -self.L*self.z_z*w
        
        # Add turbulent mixing contribution to temperature
        t_mixing = self.apply_turbulent_mixing(t)
        
        # Add radiative cooling
        radiative_cooling = self.calculate_radiative_cooling(t)
        
        # Calculate saturation specific humidity if not provided
        if q_s is None:
            # Calculate pressure and density
            rho = - 1/self.avoid_inf(self.z_z)
            p = rho*self.R*t
            
            # Calculate saturation specific humidity
            t_celsius = t - 273.15
            e_s = 6.112 * torch.exp(self.scale_tensor(17.67 * t_celsius / self.avoid_inf(t_celsius + 243.5), -3.47, 3.01)) * 100
            q_s = 0.622 * e_s / self.avoid_inf(p - 0.378 * e_s)
            q_s = torch.maximum(q_s, torch.ones_like(q_s)*1e-6)
        
        # Add latent heating from precipitation
        _, latent_heating = self.calculate_convective_precipitation(q, q_s)
        
        # Combine all contributions to temperature tendency
        self.t_t = (Q-self.z_z*w)/self.c_p - u*t_x - v*t_y - w*t_z + t_mixing + radiative_cooling + latent_heating
        
        return self.t_t, q_s
    
    # Modified RK4 integration for temperature
    def t_evolution(self, u, v, w, t, q):
        # First, store original value
        t_orig = t.clone()
        
        # Get initial saturation specific humidity
        rho = - 1/self.avoid_inf(self.z_z)
        p = rho*self.R*t
        t_celsius = t - 273.15
        e_s = 6.112 * torch.exp(self.scale_tensor(17.67 * t_celsius / self.avoid_inf(t_celsius + 243.5), -3.47, 3.01)) * 100
        q_s = 0.622 * e_s / self.avoid_inf(p - 0.378 * e_s)
        q_s = torch.maximum(q_s, torch.ones_like(q_s)*1e-6)
        
        # Stage 1
        k1_t, _ = self.get_t_t(u, v, w, t, q, q_s)
        t1 = t + 0.5 * self.block_dt * k1_t
        
        # Stage 2
        k2_t, _ = self.get_t_t(u, v, w, t1, q, q_s)
        t2 = t + 0.5 * self.block_dt * k2_t
        
        # Stage 3
        k3_t, _ = self.get_t_t(u, v, w, t2, q, q_s)
        t3 = t + self.block_dt * k3_t
        
        # Stage 4
        k4_t, _ = self.get_t_t(u, v, w, t3, q, q_s)
        
        # Combine stages with appropriate weights
        t_tendency = (k1_t + 2*k2_t + 2*k3_t + k4_t) / 6.0
        
        # Apply scaled tendency
        t_new = t_orig + self.scale_diff(t_tendency * self.block_dt, t_orig).detach()
        
        return t_new
    ################################################################

    ############################# z #############################
    def get_z_zt(self):    
        z_zt = -self.R/pressure.to(self.t_t.dtype).to(self.t_t.device)*self.t_t
        return z_zt
    
    def get_z_t(self):
        z_zt = self.get_z_zt()
        self.z_t = integral_z(z_zt)
        return self.z_t
    
    # Modified RK4 integration for geopotential height
    def z_evolution(self, z, u, v, w, t, q):
        # The initial temperature tendency is already calculated in forward()
        # so we don't need to recalculate initial t_t and z_t
        
        # First, store original value
        z_orig = z.clone()
        
        # For stage 1, use the pre-computed z_t
        k1_z = self.z_t
        z1 = z + 0.5 * self.block_dt * k1_z
        
        # For the subsequent stages, we need to update t_t and z_t
        # based on the intermediate state
        
        # Stage 2
        # Calculate temperature tendency for this stage
        t_t2, _ = self.get_t_t(u, v, w, t, q)
        self.t_t = t_t2
        # Calculate z tendency based on updated temperature tendency
        k2_z = self.get_z_t()
        z2 = z + 0.5 * self.block_dt * k2_z
        
        # Stage 3
        t_t3, _ = self.get_t_t(u, v, w, t, q)
        self.t_t = t_t3
        k3_z = self.get_z_t()
        z3 = z + self.block_dt * k3_z
        
        # Stage 4
        t_t4, _ = self.get_t_t(u, v, w, t, q)
        self.t_t = t_t4
        k4_z = self.get_z_t()
        
        # Combine stages with appropriate weights
        z_tendency = (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6.0
        
        # Apply scaled tendency
        z_new = z_orig + self.scale_diff(z_tendency * self.block_dt, z_orig).detach()
        
        return z_new
    ################################################################

    ############################# w #############################
    def get_w(self, u, v):
        self.u_x = d_x(u)
        self.v_y = d_y(v)
        w_z = - self.u_x - self.v_y
        w = integral_z(w_z).detach()
        return w
    ################################################################


    ############################# q #############################
    def get_q_dt(self, u, v, t, w, q):
        def get_qs(p, T):
            t = T - 273.15
            e_s = 6.112 * torch.exp(self.scale_tensor(17.67 * t / self.avoid_inf(t + 243.5), -3.47, 3.01)) * 100
            q_s = 0.622 * e_s / self.avoid_inf(p - 0.378 * e_s)
            return q_s

        def get_delta(p_t, q, q_s):
            cond = torch.logical_and(p_t < 0, torch.ge(q, q_s))
            return torch.where(cond, torch.ones_like(p_t), torch.zeros_like(p_t))

        def get_F(T, q, q_s):
            R = (1 + 0.608 * q) * self.R_d
            F_ = (self.L * R - self.c_p * self.R_v * T) / self.avoid_inf(self.c_p * self.R_v * T * T + self.L * self.L * q_s)
            F_ = F_ * q_s * T
            return F_

        q_x = d_x(q)
        q_y = d_y(q)
        q_z = d_z(q)

        # Apply turbulent mixing to moisture
        q_mixing = self.apply_turbulent_mixing(q)

        rho = - 1/self.avoid_inf(self.z_z)
        p = rho*self.R*t

        q_s = get_qs(p, t).detach()
        q_s = torch.maximum(q_s, torch.ones_like(q_s)*1e-6)
        delta = get_delta(self.z_t + u*self.z_x + v*self.z_y + w*self.z_z, q, q_s).detach()
        F_ = get_F(t, q, q_s).detach()
        
        # Get precipitation rate to account for moisture sink
        precip_rate, _ = self.calculate_convective_precipitation(q, q_s)

        q_t = -(u*q_x + v*q_y + w*q_z) + (self.z_t + u*self.z_x + v*self.z_y + w*self.z_z) * delta * F_ / self.avoid_inf(self.R*t) + q_mixing - precip_rate
        return q_t, q_s
    
    # Modified RK4 integration for specific humidity
    def q_evolution(self, u, v, t, w, q):
        # First, store original value
        q_orig = q.clone()
        
        # Stage 1 - use the pre-computed z_t
        k1_q, q_s = self.get_q_dt(u, v, t, w, q)
        q1 = q + 0.5 * self.block_dt * k1_q
        
        # Stage 2
        # Note: we're reusing the same z_t for all stages since q calculations
        # don't affect z_t directly in this simplified model
        k2_q, _ = self.get_q_dt(u, v, t, w, q1)
        q2 = q + 0.5 * self.block_dt * k2_q
        
        # Stage 3
        k3_q, _ = self.get_q_dt(u, v, t, w, q2)
        q3 = q + self.block_dt * k3_q
        
        # Stage 4
        k4_q, _ = self.get_q_dt(u, v, t, w, q3)
        
        # Combine stages with appropriate weights
        q_tendency = (k1_q + 2*k2_q + 2*k3_q + k4_q) / 6.0
        
        # Apply scaled tendency
        q_new = q_orig + self.scale_diff(q_tendency * self.block_dt, q_orig).detach()
        
        # Ensure specific humidity remains non-negative
        q_new = torch.maximum(q_new, torch.zeros_like(q_new))
        
        # Ensure saturation is not exceeded significantly
        q_new = torch.minimum(q_new, 1.1 * q_s)
        
        return q_new
    ################################################################


    def forward(self, x, zquvtw):
        # x [B, D, H, W]
        skip = x

        ################################################################
        # Use physics part coefficient to blend model and physics
        zquvtw_old = (1 - self.physics_part_coef)*self.variable_norm(x) + self.physics_part_coef*zquvtw
        z_old, t_old, q_old, u_old, v_old= zquvtw_old.chunk(5, dim=1)

        # First compute vertical velocity from continuity equation
        w_old = self.get_w(u_old, v_old)
        
        # Compute spatial gradients of geopotential
        self.share_z_dxyz(z_old)
        
        # Calculate t_t and z_t tendencies first, needed by multiple evolution steps
        self.t_t, _ = self.get_t_t(u_old, v_old, w_old, t_old, q_old)
        self.z_t = self.get_z_t()

        # Now apply RK4 integration to evolve the system
        u_new, v_new = self.uv_evolution(u_old, v_old, w_old)
        t_new = self.t_evolution(u_old, v_old, w_old, t_old, q_old)
        q_new = self.q_evolution(u_old, v_old, t_old, w_old, q_old)
        
        # Update geopotential using the updated temperature
        z_new = self.z_evolution(z_old, u_old, v_old, w_old, t_new, q_new)

        # Apply normalization
        z_new = self.norm_z(z_new)
        q_new = self.norm_q(q_new)
        u_new = self.norm_u(u_new)
        v_new = self.norm_v(v_new)
        t_new = self.norm_t(t_new)

        zquvtw_new = torch.cat([z_new, t_new, q_new, u_new, v_new], dim=1)

        # Combine with skip connection
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


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement[0], self.displacement[1]), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size[0] * window_size[1], window_size[0] * window_size[1])
    displacement_ = [window_size[0]-displacement[0], window_size[1]-displacement[1]]

    if upper_lower:
        mask[displacement_[0] * window_size[1]:, :displacement_[0] * window_size[1]] = float('-inf')
        mask[:displacement_[0] * window_size[1], displacement_[0] * window_size[1]:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size[0], h2=window_size[0])
        mask[:, displacement_[1]:, :, :displacement_[1]] = float('-inf')
        mask[:, :displacement_[1], :, displacement_[1]:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size[0]) for y in range(window_size[1])]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = [window_size[0] // 2, window_size[1] // 2]
            displacement_ = [-window_size[0] // 2, -window_size[1] // 2]
            self.cyclic_shift = CyclicShift(displacement_)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)
            nn.init.trunc_normal_(self.upper_lower_mask, std=.02)
            nn.init.trunc_normal_(self.left_right_mask, std=.02)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            relative_indices = get_relative_distances(window_size)
            relative_indices[:,:,0] += (window_size[0] - 1)
            relative_indices[:,:,1] += (window_size[1] - 1)
            self.relative_indices = relative_indices
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size[0] - 1, 2 * window_size[1] - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size[0] * window_size[1], window_size[0] * window_size[1]))
        nn.init.trunc_normal_(self.pos_embedding, std=.02)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads # 1, 180, 360, 96, 3

        qkv = self.to_qkv(x).chunk(3, dim=-1) #3[1, 180, 360, 96]
        nw_h = n_h // self.window_size[0] # 36
        nw_w = n_w // self.window_size[1] # 36

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size[0], w_w=self.window_size[1]), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale #[1, 3, 1296, 50, 50]

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1) # [1, 3, 1296, 50, 50]

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size[0], w_w=self.window_size[1], nw_h=nw_h, nw_w=nw_w) #[1, 180, 360, 96]
        out = self.to_out(out) #[1, 180, 360, 96]

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class AttentionGatingRouter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.att_key = nn.Linear(dim, dim)
        self.pde_key = nn.Linear(dim, dim)
    
    def forward(self, x, feat_att, feat_pde):
        q = self.query(x)
        k_att = self.att_key(feat_att)
        k_pde = self.pde_key(feat_pde)
        
        # Вычисляем скоры внимания
        att_score = torch.sum(q * k_att, dim=-1, keepdim=True) / np.sqrt(q.size(-1))
        pde_score = torch.sum(q * k_pde, dim=-1, keepdim=True) / np.sqrt(q.size(-1))
        
        # Нормализуем скоры через softmax
        scores = torch.cat([att_score, pde_score], dim=-1)
        weights = F.softmax(scores, dim=-1)
        
        return weights[..., 0:1] * feat_att + weights[..., 1:2] * feat_pde


class HybridBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding, use_pde, zquvtw_channel, depth, block_dt, inverse_time, physics_part_coef):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.use_pde = use_pde
        if use_pde:
            self.pde_block = PDE_block(dim, zquvtw_channel, depth=depth, block_dt=block_dt, inverse_time=inverse_time, physics_part_coef=physics_part_coef)
            # Добавляем роутер на основе внимания
            self.attention_router = AttentionGatingRouter(dim)
            self.router_weight = nn.Parameter(torch.zeros(1, 1, 1, dim), requires_grad=True)

        self.router_MLP = Residual(PreNorm(dim, MLP(dim, hidden_dim=mlp_dim)))

    def forward(self, x, zquvtw=None):
        if self.use_pde:
            # AI & Physics
            feat_att = self.attention_block(x)
            feat_pde, zquvtw = self.pde_block(x, zquvtw)
            
            # Используем AttentionGatingRouter вместо простых весов
            x = self.attention_router(x, feat_att, feat_pde)
            
            x = self.router_MLP(x)
            return x, zquvtw
        else:
            x = self.attention_block(x)
            x = self.router_MLP(x)
            return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x

class PatchRemaining(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        return x
    
class PatchExpanding(nn.Module):
    def __init__(self, in_channels, out_channels, upscaling_factor):
        super().__init__()
        self.upscaling_factor = upscaling_factor
        self.expand = nn.Linear(in_channels, out_channels*upscaling_factor**2)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.expand(x)
        b, h, w, c = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', 
                      p1=self.upscaling_factor, p2=self.upscaling_factor, 
                      c = c//self.upscaling_factor**2)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, scaling_factors, num_heads, head_dim, window_size,
                 relative_pos_embedding, physics_part_coef, use_pde=False, zquvtw_channel=None, depth=3, block_dt=300, inverse_time=False):
        super().__init__()
        self.use_pde = use_pde
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        if scaling_factors < 1:
            self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                                downscaling_factor=int(1/scaling_factors))
        elif scaling_factors == 1:
            self.patch_partition = PatchRemaining(in_channels=in_channels, out_channels=hidden_dimension)
        elif scaling_factors > 1:
            self.patch_partition = PatchExpanding(in_channels=in_channels, out_channels=hidden_dimension,
                                                  upscaling_factor=scaling_factors)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                HybridBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            physics_part_coef=physics_part_coef,
                            shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding, 
                            use_pde=use_pde, zquvtw_channel=zquvtw_channel, depth=depth, block_dt=block_dt, inverse_time=inverse_time),
                HybridBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            physics_part_coef=physics_part_coef,
                            shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding, 
                            use_pde=use_pde, zquvtw_channel=zquvtw_channel, depth=depth, block_dt=block_dt, inverse_time=inverse_time),
            ]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Изменение: используем xavier/kaiming в зависимости от активации
            if hasattr(m, 'activation') and isinstance(m.activation, nn.GELU):
                # Для GELU лучше работает Kaiming
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                # Xavier хорошо работает для линейных слоев без активации
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # Для сверточных слоев тоже применяем Kaiming
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, zquvtw=None):
        # print("***************************")
        # print("input:", x.shape)
        if self.use_pde:
            x = self.patch_partition(x)
            for regular_block, shifted_block in self.layers:
                x, zquvtw = regular_block(x, zquvtw)
                x, zquvtw = shifted_block(x, zquvtw)
            x = x.permute(0, 3, 1, 2) # [B, D, H, W]
            # print("output:", x.shape)
            return x, zquvtw
        else:
            x = self.patch_partition(x)
            for regular_block, shifted_block in self.layers:
                x = regular_block(x)
                x = shifted_block(x)
            x = x.permute(0, 3, 1, 2) # [B, D, H, W]
            # print("output:", x.shape)
            return x


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class GFT(nn.Module):
    def __init__(self, 
                hidden_dim=256,
                physics_part_coef=None,
                encoder_layers=[2, 2, 2],
                edcoder_heads=[3, 6, 6],
                encoder_scaling_factors=[0.5, 0.5, 1],
                encoder_dim_factors=[-1, 2, 2],

                body_layers=[4, 4, 4, 4, 4, 4],
                body_heads=[8, 8, 8, 8, 8, 8],
                body_scaling_factors=[1, 1, 1, 1, 1, 1],
                body_dim_factors=[1, 1, 1, 1, 1, 1],

                decoder_layers=[2, 2, 2],
                decoder_heads=[6, 6, 3],
                decoder_scaling_factors=[1, 2, 1],
                decoder_dim_factors=[1, 0.5, 1],

                channels=69,
                head_dim=128,
                window_size=[4,8],
                relative_pos_embedding=False,
                out_kernel=[2,2],
                
                pde_block_depth=3, 
                block_dt=300, 
                inverse_time=False,
                use_checkpoint=True):
        super().__init__()

        self.t_emb_dim = 32
        self.out_layer = [0, 2, 5]
        self.PDE_block_seconds_list = self.get_block_seconds(body_layers, block_dt*pde_block_depth)

        self.downscaling_factor_all = 1
        for factor in encoder_scaling_factors:
            self.downscaling_factor_all = self.downscaling_factor_all // factor
        self.downscaling_factor_all = int(self.downscaling_factor_all)

        encoder_dim_list = [channels, hidden_dim] # first encoder_block, the first block dim is 69 --> 256
        for factor in encoder_dim_factors[1:]:
            encoder_dim_list.append(int(encoder_dim_list[-1]*factor))
        
        body_dim_list = [encoder_dim_list[-1]]
        for factor in body_dim_factors:
            body_dim_list.append(int(encoder_dim_list[-1]*factor))
        
        decoder_dim_list = [encoder_dim_list[-1]]
        for factor in decoder_dim_factors:
            decoder_dim_list.append(int(decoder_dim_list[-1]*factor))

        self.encoder = nn.ModuleList()
        for i_layer in range(len(encoder_layers)):
            layer = StageModule(in_channels=encoder_dim_list[i_layer], hidden_dimension=encoder_dim_list[i_layer+1], layers=encoder_layers[i_layer],
                                physics_part_coef=physics_part_coef, 
                                scaling_factors=encoder_scaling_factors[i_layer], num_heads=edcoder_heads[i_layer], head_dim=head_dim,
                                window_size=window_size, relative_pos_embedding=relative_pos_embedding)
            self.encoder.append(layer)
        
        self.body = nn.ModuleList()
        for i_layer in range(len(body_layers)):
            if use_checkpoint:
                layer = checkpoint_wrapper(StageModule(in_channels=body_dim_list[i_layer], hidden_dimension=body_dim_list[i_layer+1], layers=body_layers[i_layer],
                                            physics_part_coef=physics_part_coef, 
                                            scaling_factors=body_scaling_factors[i_layer], num_heads=body_heads[i_layer], head_dim=head_dim,
                                            window_size=window_size, relative_pos_embedding=relative_pos_embedding, 
                                            use_pde=True, zquvtw_channel=13, depth=pde_block_depth, block_dt=block_dt, inverse_time=inverse_time))
            else:
                layer = StageModule(in_channels=body_dim_list[i_layer], hidden_dimension=body_dim_list[i_layer+1], layers=body_layers[i_layer],
                                    physics_part_coef=physics_part_coef,
                                    scaling_factors=body_scaling_factors[i_layer], num_heads=body_heads[i_layer], head_dim=head_dim,
                                    window_size=window_size, relative_pos_embedding=relative_pos_embedding, 
                                    use_pde=True, zquvtw_channel=13, depth=pde_block_depth, block_dt=block_dt, inverse_time=inverse_time)
            self.body.append(layer)
        
        self.time_mlp = nn.Sequential(
            RandomOrLearnedSinusoidalPosEmb(16, True),
            nn.Linear(17, self.t_emb_dim),
            nn.GELU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        for dim_i in range(len(decoder_dim_list)-1):
            decoder_dim_list[dim_i] += self.t_emb_dim

        self.decoder = nn.ModuleList()
        for i_layer in range(len(decoder_layers)):
            layer = StageModule(in_channels=decoder_dim_list[i_layer], hidden_dimension=decoder_dim_list[i_layer+1], layers=decoder_layers[i_layer],
                                  physics_part_coef=physics_part_coef,
                                  scaling_factors=decoder_scaling_factors[i_layer], num_heads=decoder_heads[i_layer], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
            self.decoder.append(layer)

        self.decoder.append(nn.ConvTranspose2d(in_channels=decoder_dim_list[-1], out_channels=channels, kernel_size=out_kernel, stride=min(out_kernel)))


    def get_block_seconds(self, block_nums, second_per_block=900):
        block_seconds = [block_nums[0]]
        for i in range(1, len(block_nums)):
            block_seconds.append(block_seconds[i-1]+block_nums[i])
        block_seconds = [l*second_per_block for l in block_seconds]
        return block_seconds


    def x_to_zquvtw(self, x):
        zquvtw = x[:,4:] # B, 65, 128, 256
        _, _, self.H, self.W = zquvtw.shape
        zquvtw = torch.nn.functional.interpolate(zquvtw, size=(self.H//self.downscaling_factor_all, self.W//self.downscaling_factor_all), mode='bilinear')
        zquvtw = zquvtw.permute(0, 2, 3, 1) # B, 32, 64, 65
        return zquvtw
    

    def cat_t_emb(self, x, layer_idx):
        B, _, H, W = x.shape
        total_seconds = self.PDE_block_seconds_list[layer_idx]
        t = torch.tensor([total_seconds]*B).to(x.device)
        t_emb = self.time_mlp(t)
        t_emb = t_emb.reshape(B,self.t_emb_dim,1,1).expand(B,self.t_emb_dim, H, W)
        x_t_emb = torch.cat([x, t_emb], dim=1)
        return x_t_emb
    
    
    def get_learnable_physics_importance(self):
        """
        Function that returns learned physics_importance coefs from all layers that contain PDE_kernels.
        Use only if physics_importance_coef was not specified. 

        Returns:
            List[Tensor[Layer, C, H, W]]: List of learned physics importance coefficients 
            from the layers with PDE_kernels.

        Notes:
            - Physics importance coefficients are used to balance between pure neural
              network predictions and physics-based predictions.
            - Higher values (closer to 1) give more weight to the physics-based model.
            - Lower values (closer to 0) give more weight to the neural network.
            - The model learns these values during training to optimize performance.
            - After the enhancements to the physics kernel (RK4 integration, variable 
              Coriolis parameter, turbulent mixing, radiative cooling, and convective
              precipitation), the physics part may become more reliable and these 
              coefficients may trend higher.
        """
        total_coefs = []

        for layer in self.encoder: 
            if isinstance(layer, StageModule):
                for inner_layer_pack in layer.layers:
                    for hybrid_block in inner_layer_pack:
                        if hybrid_block.use_pde:
                            for pde_kernel in hybrid_block.pde_block.PDE_kernels:
                                total_coefs.append(pde_kernel.physics_part_coef.data)

        for layer in self.body: 
            if isinstance(layer, StageModule):
                for inner_layer_pack in layer.layers:
                    for hybrid_block in inner_layer_pack:
                        if hybrid_block.use_pde:
                            for pde_kernel in hybrid_block.pde_block.PDE_kernels:
                                total_coefs.append(pde_kernel.physics_part_coef.data)

        for layer in self.decoder: 
            if isinstance(layer, StageModule):
                for inner_layer_pack in layer.layers:
                    for hybrid_block in inner_layer_pack:
                        if hybrid_block.use_pde:
                            for pde_kernel in hybrid_block.pde_block.PDE_kernels:
                                total_coefs.append(pde_kernel.physics_part_coef.data)

        return torch.squeeze(torch.stack(total_coefs, dim=0))

    def forward(self, x):
        x = x.squeeze(1)
        
        GFT.gft_name = ""
        
        output = []
        zquvtw = self.x_to_zquvtw(x)
        for idx, layer in enumerate(self.encoder):
            GFT.gft_name = f"{idx}_encoder"
            x = layer(x)
        for layer_idx, layer in enumerate(self.body):
            GFT.gft_name = f"{layer_idx}_body"
            x, zquvtw = layer(x, zquvtw)

            if layer_idx in self.out_layer:
                
                x_t_emb = self.cat_t_emb(x, layer_idx)
                for layer in self.decoder:
                    GFT.gft_name = f"{layer_idx}_body_decoder"
                    x_t_emb = layer(x_t_emb)
                output.append(x_t_emb)

        if len(output) == 1:
            return output[0]
        else:
            return torch.stack(output, dim=1)



if __name__ == "__main__":
    import os
    import json
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    # from thop import profile

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)


    model = GFT(hidden_dim=256,
                encoder_layers=[2, 2, 2],
                edcoder_heads=[3, 6, 6],
                encoder_scaling_factors=[0.5, 0.5, 1], # [128, 256] --> [64, 128] --> [32, 64] --> [32, 64], that is, patch size = 4 (128/32)
                encoder_dim_factors=[-1, 2, 2],

                body_layers=[4], # A total of 4x6=24 HybridBlock, corresponding to 6 hours (24x15min) of time evolution
                body_heads=[8],
                body_scaling_factors=[1],
                body_dim_factors=[1],

                decoder_layers=[2, 2, 2],
                decoder_heads=[6, 6, 3],
                decoder_scaling_factors=[1, 2, 1],
                decoder_dim_factors=[1, 0.5, 1],

                channels=69,
                head_dim=128,
                window_size=[4,8],
                relative_pos_embedding=False,
                out_kernel=[2,2],

                pde_block_depth=3, # 1 HybridBlock contains 3 PDE kernels, corresponding to 15 minutes (3x300s) of time evolution
                block_dt=300, # One PDE kernel corresponds to 300s of time evolution
                inverse_time=False).to(device)

    
    # if os.path.exists('../checkpoints/gft.ckpt'):
    #     ckpt = torch.load('../checkpoints/gft.ckpt', map_location=torch.device('cpu'))
    #     model.load_state_dict(ckpt, strict=True)
    #     print('[complete loading model]')
    # else:
    #     print('[checkpoint does not exist]')

    # if os.path.exists('../example_data/input.npy') and os.path.exists('../example_data/target.npy'):
    #     inp = torch.tensor(np.load('../example_data/input.npy')).float().to(device)
    #     target = torch.tensor(np.load('../example_data/target.npy')).float().to(device)
    # else:
    #     inp = torch.randn(1, 69, 128, 256).to(device)
    #     target = None
    
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"total parameters: {total_params}")
    # flops, params = profile(model, inputs=(inp,))
    # print(f"flops: {round(flops/10**9, 2)} G, params: {round(params/10**6, 2)} M")
    inp = torch.randn(1, 69, 128, 256).to(device)

    pred = model(inp)
    print(pred.shape)
    # torch.Size([1, 3, 69, 128, 256]), the prediction results of lead time=[1,3,6]h respectively

    model.out_layer = [5] # decode only the last layer
    pred = model(inp)
    # torch.Size([1, 69, 128, 256]), the prediction results of lead time=[1,3,6]h respectively
    print(pred.shape)


def visualize_physics_contributions(model, input_data, variable_idx=0, level_idx=0):
    """
    Visualizes the relative contributions of physics vs. neural network
    across the model's layers.
    
    Args:
        model: The GFT model
        input_data: Input tensor for the model [B, 69, 128, 256]
        variable_idx: Index of the variable to visualize (0-4: z, t, q, u, v)
        level_idx: Pressure level index to visualize (0-12)
    """
    # Set model to evaluation mode
    model.eval()
    device = input_data.device
    
    # Get the physics importance coefficients
    coefs = model.get_learnable_physics_importance()
    
    # Create a figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot the average coefficient values
    avg_coefs = coefs.mean(dim=(-1, -2, -3))
    ax1.plot(avg_coefs.cpu().numpy())
    ax1.set_xlabel('Layer index')
    ax1.set_ylabel('Average physics importance')
    ax1.set_title('Physics vs. Neural Network Contribution by Layer')
    ax1.grid(True)
    
    # Run a forward pass and capture intermediate activations
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for i, layer in enumerate(model.body):
        if isinstance(layer, StageModule):
            for j, inner_pack in enumerate(layer.layers):
                for k, block in enumerate(inner_pack):
                    if hasattr(block, 'pde_block'):
                        for l, pde_kernel in enumerate(block.pde_block.PDE_kernels):
                            name = f"body_{i}_layer_{j}_block_{k}_kernel_{l}"
                            hooks.append(pde_kernel.register_forward_hook(hook_fn(name)))
    
    # Run the model
    with torch.no_grad():
        model(input_data)
    
    # Collect the variable we're interested in from activations
    var_values = []
    for name, act in sorted(activations.items()):
        if "block_0_kernel_0" in name:  # Select one kernel per layer
            # Extract the variable we want (z, t, q, u, v)
            zquvt = act.permute(0, 2, 3, 1)  # [B, H, W, C]
            variables = zquvt.chunk(5, dim=-1)
            var_values.append(variables[variable_idx][:, level_idx].cpu().numpy())
    
    # Plot heatmap of the selected variable across layers
    if var_values:
        vmin = min(v.min() for v in var_values)
        vmax = max(v.max() for v in var_values)
        
        for i, var in enumerate(var_values):
            ax = plt.subplot(3, len(var_values), i + len(var_values) + 1)
            im = ax.imshow(var[0], vmin=vmin, vmax=vmax, cmap='viridis')
            ax.set_title(f'Layer {i}')
            if i == 0:
                ax.set_ylabel('Variable Evolution')
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    plt.tight_layout()
    return fig


def compare_physics_schemes(model, input_data, old_physics=False):
    """
    Compare the enhanced physics scheme with the original one.
    
    Args:
        model: The GFT model
        input_data: Input tensor for the model [B, 69, 128, 256]
        old_physics: If True, use the original physics implementation
                    If False, use the enhanced physics (RK4, variable Coriolis, etc.)
    """
    # Convert model to CPU for easier manipulation if needed
    model = model.cpu()
    input_data = input_data.cpu()
    
    # Create temporary models with different physics settings
    if old_physics:
        # Temporarily modify model to use original physics
        for module in model.modules():
            if isinstance(module, PDE_kernel):
                # Save the original methods
                original_uv_evolution = module.uv_evolution
                original_t_evolution = module.t_evolution
                original_z_evolution = module.z_evolution
                original_q_evolution = module.q_evolution
                original_get_uv_dt = module.get_uv_dt
                
                # Override with simplified methods (similar to original implementation)
                def simple_uv_evolution(self, u, v, w):
                    u_t, v_t = self.get_uv_dt(u, v, w)
                    u = u + self.scale_diff(u_t*self.block_dt, u).detach()
                    v = v + self.scale_diff(v_t*self.block_dt, v).detach()
                    return u, v
                
                def simple_get_uv_dt(self, u, v, w):
                    u_x = self.u_x
                    u_y = d_y(u)
                    u_z = d_z(u)
                    v_x = d_x(v)
                    v_y = self.v_y
                    v_z = d_z(v)
                    
                    # Use constant Coriolis parameter
                    f_coriolis = 7.29e-5
                    
                    self.u_t = -u*u_x - v*u_y - w*u_z + f_coriolis*v - self.z_x
                    self.v_t = -u*v_x - v*v_y - w*v_z - f_coriolis*u - self.z_y
                    return self.u_t, self.v_t
                
                # Apply the simple methods
                module.uv_evolution = types.MethodType(simple_uv_evolution, module)
                module.get_uv_dt = types.MethodType(simple_get_uv_dt, module)
        
        # Run prediction with original physics
        with torch.no_grad():
            old_pred = model(input_data)
        
        # Restore original methods
        for module in model.modules():
            if isinstance(module, PDE_kernel):
                module.uv_evolution = original_uv_evolution
                module.t_evolution = original_t_evolution
                module.z_evolution = original_z_evolution
                module.q_evolution = original_q_evolution
                module.get_uv_dt = original_get_uv_dt
        
        return old_pred
    
    else:
        # Run prediction with enhanced physics (default)
        with torch.no_grad():
            new_pred = model(input_data)
        
        return new_pred


def analyze_stability(model, input_data, dt_values=[60, 300, 600, 900]):
    """
    Analyze the stability of the physics integration for different time steps.
    
    Args:
        model: The GFT model
        input_data: Input tensor for the model [B, 69, 128, 256]
        dt_values: List of time step sizes to test (in seconds)
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    results = {}
    original_dt = model.body[0].layers[0][0].pde_block.PDE_kernels[0].block_dt
    
    for dt in dt_values:
        # Update all PDE kernels with the new dt
        for module in model.modules():
            if isinstance(module, PDE_kernel):
                module.block_dt = dt
        
        # Run the model
        with torch.no_grad():
            try:
                pred = model(input_data)
                # Check for NaN values
                if torch.isnan(pred).any():
                    stability = "Unstable (NaN values)"
                else:
                    # Simple heuristic: check if values are within reasonable bounds
                    if torch.max(torch.abs(pred)) > 1e5:
                        stability = "Unstable (extreme values)"
                    else:
                        stability = "Stable"
            except Exception as e:
                stability = f"Unstable (error: {str(e)})"
        
        results[dt] = stability
    
    # Restore original dt
    for module in model.modules():
        if isinstance(module, PDE_kernel):
            module.block_dt = original_dt
    
    # Print results
    print("Stability analysis results:")
    for dt, status in results.items():
        print(f"  dt = {dt}s: {status}")
    
    return results

def analyze_physics_components(model, input_data):
    """
    Analyze the relative contributions of different physical processes 
    in the enhanced physics module.
    
    Args:
        model: The GFT model
        input_data: Input tensor for the model [B, 69, 128, 256]
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Dictionary to store contributions from different processes
    contributions = {
        'advection': [],
        'coriolis': [],
        'geopotential_gradient': [],
        'turbulent_mixing': [],
        'radiative_cooling': [],
        'latent_heating': [],
    }
    
    # Register hooks to capture intermediate values
    handles = []
    
    def hook_uv_dt(module, input, output):
        u, v, w = input
        # Extract contributions from the u momentum equation
        u_x = module.u_x
        u_y = d_y(u)
        u_z = d_z(u)
        
        # Get latitude-dependent Coriolis parameter
        B, C, H, W = u.shape
        lat_grid = latitudes.to(u.device).reshape(1, 1, H, 1).expand(B, C, H, W)
        f_coriolis = module.get_coriolis_parameter(lat_grid)
        
        # Calculate magnitudes
        advection_u = torch.abs(-u*u_x - v*u_y - w*u_z).mean().item()
        coriolis_u = torch.abs(f_coriolis*v).mean().item()
        geopotential_u = torch.abs(-module.z_x).mean().item()
        mixing_u = torch.abs(module.apply_turbulent_mixing(u)).mean().item()
        
        # Store values
        contributions['advection'].append(advection_u)
        contributions['coriolis'].append(coriolis_u)
        contributions['geopotential_gradient'].append(geopotential_u)
        contributions['turbulent_mixing'].append(mixing_u)
    
    def hook_t_dt(module, input, output):
        u, v, w, t, q, *_ = input
        
        # Calculate magnitudes of different terms in temperature tendency
        t_x = d_x(t)
        t_y = d_y(t)
        t_z = d_z(t)
        
        # Basic terms
        advection_t = torch.abs(-u*t_x - v*t_y - w*t_z).mean().item()
        adiabatic_t = torch.abs((-module.L*module.z_z*w - module.z_z*w)/module.c_p).mean().item()
        
        # Enhanced physics terms
        mixing_t = torch.abs(module.apply_turbulent_mixing(t)).mean().item()
        radiative_t = torch.abs(module.calculate_radiative_cooling(t)).mean().item()
        
        # Calculate saturation specific humidity
        rho = - 1/module.avoid_inf(module.z_z)
        p = rho*module.R*t
        t_celsius = t - 273.15
        e_s = 6.112 * torch.exp(module.scale_tensor(17.67 * t_celsius / module.avoid_inf(t_celsius + 243.5), -3.47, 3.01)) * 100
        q_s = 0.622 * e_s / module.avoid_inf(p - 0.378 * e_s)
        q_s = torch.maximum(q_s, torch.ones_like(q_s)*1e-6)
        
        # Get latent heating
        _, latent_t = module.calculate_convective_precipitation(q, q_s)
        latent_t = torch.abs(latent_t).mean().item()
        
        # Update appropriate contribution values
        contributions['advection'][-1] = (contributions['advection'][-1] + advection_t) / 2
        contributions['turbulent_mixing'][-1] = (contributions['turbulent_mixing'][-1] + mixing_t) / 2
        contributions['radiative_cooling'].append(radiative_t)
        contributions['latent_heating'].append(latent_t)
    
    # Register hooks for all PDE_kernels
    for module in model.modules():
        if isinstance(module, PDE_kernel):
            handles.append(module.get_uv_dt.__func__.__get__(module, PDE_kernel).register_forward_hook(hook_uv_dt))
            handles.append(module.get_t_t.__func__.__get__(module, PDE_kernel).register_forward_hook(hook_t_dt))
    
    # Run the model
    with torch.no_grad():
        model(input_data)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Calculate averages for each contribution
    avg_contributions = {k: sum(v)/len(v) if v else 0 for k, v in contributions.items()}
    
    # Print results
    print("\nAverage contributions of physical processes:")
    for process, value in avg_contributions.items():
        print(f"  {process}: {value:.6f}")
    
    # Create visualization
    if contributions['advection']:
        plt.figure(figsize=(10, 6))
        processes = list(contributions.keys())
        values = [avg_contributions[p] for p in processes]
        
        # Normalize to percentages
        total = sum(values)
        percentages = [100 * v/total for v in values]
        
        plt.bar(processes, percentages)
        plt.title('Relative Importance of Physical Processes')
        plt.ylabel('Contribution (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    else:
        return None