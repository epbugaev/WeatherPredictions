import torch
import numpy as np
import xarray as xr
import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from torch.cuda import amp
import math

# Import components from FourCastNet
# Complex operations
@torch.jit.script
def compl_mul2d_fwd_c(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,io->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res

@torch.jit.script
def compl_muladd2d_fwd_c(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    tmpcc = torch.view_as_complex(compl_mul2d_fwd_c(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)

@torch.jit.script
def compl_mul2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixys,ior->srboxy", a, b)
    res = torch.stack(
        [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1
    )
    return res

@torch.jit.script
def compl_muladd2d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    res = compl_mul2d_fwd(a, b) + c
    return res

# Complex activation
class ComplexReLU(nn.Module):
    def __init__(self, negative_slope=0.0, mode="cartesian", bias_shape=None):
        super(ComplexReLU, self).__init__()

        # store parameters
        self.mode = mode
        if self.mode in ["modulus", "halfplane"]:
            if bias_shape is not None:
                self.bias = nn.Parameter(torch.zeros(bias_shape, dtype=torch.float32))
            else:
                self.bias = nn.Parameter(torch.zeros((1), dtype=torch.float32))
        else:
            bias = torch.zeros((1), dtype=torch.float32)
            self.register_buffer("bias", bias)

        self.negative_slope = negative_slope
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)
        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
            out = self.act(zabs + self.bias) * torch.exp(1.0j * z.angle())
        elif self.mode == "halfplane":
            # bias is an angle parameter in this case
            modified_angle = torch.angle(z) - self.bias
            condition = torch.logical_and(
                (0.0 <= modified_angle), (modified_angle < torch.pi / 2.0)
            )
            out = torch.where(condition, z, self.negative_slope * z)
        elif self.mode == "real":
            zr = torch.view_as_real(z)
            outr = zr.clone()
            outr[..., 0] = self.act(zr[..., 0])
            out = torch.view_as_complex(outr)
        else:
            # identity
            out = z

        return out

# FFT transforms
class RealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):
        super(RealFFT2, self).__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1
        self.num_batches = 1

        assert self.lmax % 2 == 0

    def forward(self, x):
        # do batched FFT
        xs = torch.split(x, x.shape[1] // self.num_batches, dim=1)

        ys = []
        for xt in xs:
            yt = torch.fft.rfft2(xt, dim=(-2, -1), norm="ortho")
            ys.append(yt)

        return torch.cat(ys, dim=1)

class InverseRealFFT2(nn.Module):
    """
    Helper routine to wrap IFFT similarly to the inverse SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):
        super(InverseRealFFT2, self).__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1
        self.num_batches = 1

    def forward(self, x):
        # do batched IFFT
        xs = torch.split(x, x.shape[1] // self.num_batches, dim=1)

        ys = []
        for xt in xs:
            yt = torch.fft.irfft2(xt, dim=(-2, -1), norm="ortho")
            ys.append(yt)

        return torch.cat(ys, dim=1)

# SpectralAttention for patches
class SpectralAttentionPatches(nn.Module):
    """
    Adapted SpectralAttention2d for patch-based data in PredFormer
    """

    def __init__(
        self,
        num_patches_h,
        num_patches_w,
        embed_dim,
        sparsity_threshold=0.0,
        hidden_size_factor=2,
        use_complex_network=True,
        use_complex_kernels=False,
        complex_activation="real",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
    ):
        super(SpectralAttentionPatches, self).__init__()

        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        self.hidden_size = int(hidden_size_factor * self.embed_dim)
        self.scale = 0.02
        self.spectral_layers = spectral_layers
        self.mul_add_handle = (
            compl_muladd2d_fwd_c if use_complex_kernels else compl_muladd2d_fwd
        )
        self.mul_handle = compl_mul2d_fwd_c if use_complex_kernels else compl_mul2d_fwd

        # Create transforms for patch grid
        self.forward_transform = RealFFT2(num_patches_h, num_patches_w)
        self.inverse_transform = InverseRealFFT2(num_patches_h, num_patches_w)
        
        self.modes_lat = self.forward_transform.lmax
        self.modes_lon = self.forward_transform.mmax

        # weights
        w = [self.scale * torch.randn(self.embed_dim, self.hidden_size, 2)]
        for l in range(1, self.spectral_layers):
            w.append(self.scale * torch.randn(self.hidden_size, self.hidden_size, 2))
        self.w = nn.ParameterList(w)

        if bias:
            self.b = nn.ParameterList(
                [
                    self.scale * torch.randn(self.hidden_size, 1, 2)
                    for _ in range(self.spectral_layers)
                ]
            )

        self.wout = nn.Parameter(
            self.scale * torch.randn(self.hidden_size, self.embed_dim, 2)
        )

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self.activation = ComplexReLU(
            mode=complex_activation, bias_shape=(self.hidden_size, 1, 1)
        )

    def forward_mlp(self, xr):
        for l in range(self.spectral_layers):
            if hasattr(self, "b"):
                xr = self.mul_add_handle(
                    xr, self.w[l].to(xr.dtype), self.b[l].to(xr.dtype)
                )
            else:
                xr = self.mul_handle(xr, self.w[l].to(xr.dtype))
            xr = torch.view_as_complex(xr)
            xr = self.activation(xr)
            xr = self.drop(xr)
            xr = torch.view_as_real(xr)

        xr = self.mul_handle(xr, self.wout)
        return xr

    def forward(self, x):
        # x shape: [batch, num_patches, embed_dim]  
        # Need to reshape to 2D grid for spectral operations
        batch_size, num_patches, embed_dim = x.shape
        
        # Reshape patches to 2D grid
        x = x.view(batch_size, self.num_patches_h, self.num_patches_w, embed_dim)
        x = x.permute(0, 3, 1, 2)  # [batch, embed_dim, h, w]
        
        dtype = x.dtype

        # FWD transform
        with amp.autocast(enabled=False):
            x = x.to(torch.float32)
            x = self.forward_transform(x)
            x = torch.view_as_real(x)

        # MLP
        x = self.forward_mlp(x)

        # BWD transform
        with amp.autocast(enabled=False):
            x = torch.view_as_complex(x)
            x = self.inverse_transform(x)
            x = x.to(dtype)

        # Reshape back to patches
        x = x.permute(0, 2, 3, 1)  # [batch, h, w, embed_dim]
        x = x.view(batch_size, num_patches, embed_dim)

        return x

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

class SpectralGatedTransformer(nn.Module):
    def __init__(self, dim, depth, num_patches_h, num_patches_w, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SpectralAttentionPatches(
                    num_patches_h=num_patches_h,
                    num_patches_w=num_patches_w,
                    embed_dim=dim,
                    drop_rate=attn_dropout
                )),
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
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)

class SpectralPredFormerLayer(nn.Module):
    def __init__(self, dim, depth, num_patches_h, num_patches_w, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super(SpectralPredFormerLayer, self).__init__()

        # For temporal attention, we use regular attention since temporal dimension is typically small
        from Models.PredFormer import GatedTransformer
        self.temporal_transformer_first = GatedTransformer(dim, depth, 8, 32, 
                                                      mlp_dim, dropout, attn_dropout, drop_path)
        self.space_transformer = SpectralGatedTransformer(dim, depth, num_patches_h, num_patches_w, 
                                             mlp_dim, dropout, attn_dropout, drop_path)
        self.temporal_transformer_second = GatedTransformer(dim, depth, 8, 32, 
                                                       mlp_dim, dropout, attn_dropout, drop_path)

    def forward(self, x):
        b, t, n, _ = x.shape        
        x_t, x_ori = x, x 
        
        # t branch (first temporal) - use 1x1 spatial attention
        x_t = rearrange(x_t, 'b t n d -> b n t d')
        x_t = rearrange(x_t, 'b n t d -> (b n) t d')
        x_t = self.temporal_transformer_first(x_t)
        
        # s branch (space) - use full spatial attention
        x_ts = rearrange(x_t, '(b n) t d -> b n t d', b=b)
        x_ts = rearrange(x_ts, 'b n t d -> b t n d')
        x_ts = rearrange(x_ts, 'b t n d -> (b t) n d') 
        x_ts = self.space_transformer(x_ts)
        
        # t branch (second temporal) - use 1x1 spatial attention  
        x_tst = rearrange(x_ts, '(b t) n d -> b t n d', b=b)
        x_tst = rearrange(x_tst, 'b t n d -> b n t d')
        x_tst = rearrange(x_tst, 'b n t d -> (b n) t d')
        x_tst = self.temporal_transformer_second(x_tst)

        # ts output branch     
        x_tst = rearrange(x_tst, '(b n) t d -> b n t d', b=b)
        x_tst = rearrange(x_tst, 'b n t d -> b t n d', b=b) 
        
        return x_tst

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')
      
class PredFormerSpectral_Model(nn.Module):
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
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']
        self.depth = model_config['depth']
        self.path_to_constants = model_config['path_to_constants']
        self.ds = xr.open_dataset(self.path_to_constants)
        self.orography_mask = torch.Tensor(np.array(self.ds.orography))
        self.soil_mask = torch.Tensor(np.array(self.ds.slt))
        self.lsm_mask = torch.Tensor(np.array(self.ds.lsm))
        self.static_masks = torch.stack([self.orography_mask, self.soil_mask, self.lsm_mask], dim=0).unsqueeze(0)
        self.num_masks = 3

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
            SpectralPredFormerLayer(self.dim, self.depth, self.num_patches_h, self.num_patches_w, self.dim * self.scale_dim, self.dropout, self.attn_dropout, self.drop_path)
            for i in range(self.Ndepth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_channels * self.patch_size ** 2)
            ) 
                
    def forward(self, x):
        B, T, C, H, W = x.shape
        assert C == self.num_channels

        mask_patches = self.rearrange_masks(self.static_masks.to(x.device))
        mask_embed = self.mask_embedding(mask_patches)
        mask_embed = mask_embed.unsqueeze(1)
        mask_embed = mask_embed.to(x.device) 

        # Patch Embedding
        x_embed = self.to_patch_embedding(x)
        
        x_combined = x_embed + mask_embed       

        # Position Embedding
        x_combined += self.pos_embedding.to(x.device)
        
        # PredFormer Encoder with Spectral Attention
        for idx, blk in enumerate(self.blocks):
            x_combined = blk(x_combined)
            
        # MLP head        
        x = self.mlp_head(x_combined.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)
        
        return x 