import math

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum, nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class SpectralMixer2D(nn.Module):
    """AFNO/F-FNO-lite spatial token mixer as a drop-in for spatial Attention.

    Operates on a token sequence of shape ``(B', N, D)`` where ``N = num_patches_h *
    num_patches_w``. The forward pass:

      1. reshape to ``(B', D, h, w)``,
      2. cast to fp32, perform rFFT2 with autocast disabled,
      3. block-diagonal complex linear on the low ``(modes_h, modes_w)`` spectrum,
      4. optional complex soft-thresholding (AFNO sparsity),
      5. zero-pad the spectrum back to full size, iRFFT2,
      6. optional parallel 3x3 depthwise conv branch (recovers short-wavelength
         detail lost to mode truncation, important on cropped non-periodic domains),
      7. flatten to ``(B', N, D)`` and project with a final linear.

    The output has the same shape as the input and is intended to be wrapped in
    ``PreNorm`` so the residual addition happens externally, mirroring how
    ``Attention`` is wrapped in :class:`GatedTransformer`.

    Args:
        dim: token feature dimension D.
        num_patches_h: spatial patch grid height h.
        num_patches_w: spatial patch grid width w.
        modes_h: number of low frequencies kept along height. Defaults to h.
        modes_w: number of low frequencies kept along width. Defaults to w//2+1
            (the full rFFT width).
        channel_blocks: number of block-diagonal blocks in the complex linear.
            ``dim`` must be divisible by ``channel_blocks``.
        soft_thresh_lambda: AFNO complex soft-threshold magnitude. 0 disables it.
        use_local_branch: whether to add a parallel 3x3 depthwise conv branch.
        dropout: dropout applied after the output projection.

    Notes:
        Complex parameters are stored as a real tensor with leading dim 2
        ``(re, im)`` to keep AMP/optimizer behavior simple.
    """

    def __init__(
        self,
        dim,
        num_patches_h,
        num_patches_w,
        modes_h=None,
        modes_w=None,
        channel_blocks=8,
        soft_thresh_lambda=0.01,
        use_local_branch=True,
        dropout=0.0,
    ):
        super().__init__()
        if dim % channel_blocks != 0:
            raise ValueError(f"dim={dim} must be divisible by channel_blocks={channel_blocks}")
        self.dim = dim
        self.h = num_patches_h
        self.w = num_patches_w
        self.num_blocks = channel_blocks
        self.block_dim = dim // channel_blocks

        full_modes_h = num_patches_h
        full_modes_w = num_patches_w // 2 + 1
        self.modes_h = full_modes_h if modes_h is None else min(int(modes_h), full_modes_h)
        self.modes_w = full_modes_w if modes_w is None else min(int(modes_w), full_modes_w)
        self.soft_thresh_lambda = float(soft_thresh_lambda)

        scale = 1.0 / math.sqrt(self.block_dim)
        self.weight = nn.Parameter(
            scale
            * torch.randn(
                2, self.num_blocks, self.modes_h, self.modes_w, self.block_dim, self.block_dim
            )
        )
        self.bias = nn.Parameter(
            torch.zeros(2, self.num_blocks, self.modes_h, self.modes_w, self.block_dim)
        )

        self.local_branch = (
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim) if use_local_branch else None
        )

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @staticmethod
    def _complex_soft_thresh(re, im, lam):
        if lam <= 0.0:
            return re, im
        mag = torch.sqrt(re * re + im * im + 1e-12)
        scale = torch.relu(mag - lam) / mag
        return re * scale, im * scale

    def _spectral_branch(self, x_spatial):
        """Spectral path on input of shape (B', D, h, w)."""
        B_, D, H, W = x_spatial.shape
        orig_dtype = x_spatial.dtype

        with torch.amp.autocast(device_type="cuda", enabled=False):
            x_fft = torch.fft.rfft2(x_spatial.float(), dim=(-2, -1), norm="ortho")
            x_re = x_fft.real
            x_im = x_fft.imag
            mh, mw = self.modes_h, self.modes_w

            x_re_low = x_re[..., :mh, :mw].reshape(B_, self.num_blocks, self.block_dim, mh, mw)
            x_im_low = x_im[..., :mh, :mw].reshape(B_, self.num_blocks, self.block_dim, mh, mw)

            w_re = self.weight[0].float()
            w_im = self.weight[1].float()
            b_re = self.bias[0].float()
            b_im = self.bias[1].float()

            out_re = torch.einsum("bkihw, khwio -> bkohw", x_re_low, w_re) - torch.einsum(
                "bkihw, khwio -> bkohw", x_im_low, w_im
            )
            out_im = torch.einsum("bkihw, khwio -> bkohw", x_re_low, w_im) + torch.einsum(
                "bkihw, khwio -> bkohw", x_im_low, w_re
            )

            out_re = out_re + b_re.permute(0, 3, 1, 2).unsqueeze(0)
            out_im = out_im + b_im.permute(0, 3, 1, 2).unsqueeze(0)

            out_re, out_im = self._complex_soft_thresh(out_re, out_im, self.soft_thresh_lambda)

            full_re = torch.zeros(
                B_,
                self.num_blocks,
                self.block_dim,
                x_re.shape[-2],
                x_re.shape[-1],
                dtype=out_re.dtype,
                device=out_re.device,
            )
            full_im = torch.zeros_like(full_re)
            full_re[..., :mh, :mw] = out_re
            full_im[..., :mh, :mw] = out_im
            full_re = full_re.reshape(B_, D, x_re.shape[-2], x_re.shape[-1])
            full_im = full_im.reshape(B_, D, x_im.shape[-2], x_im.shape[-1])

            out_spatial = torch.fft.irfft2(
                torch.complex(full_re, full_im),
                s=(H, W),
                dim=(-2, -1),
                norm="ortho",
            )
        return out_spatial.to(orig_dtype)

    def forward(self, x):
        """Apply the spectral mixer to a token sequence.

        Args:
            x: torch.Tensor of shape ``(B', N, D)`` where ``N = h * w``.

        Returns:
            torch.Tensor of shape ``(B', N, D)``.
        """
        B_, N, D = x.shape
        if self.h * self.w != N:
            raise ValueError(f"token count N={N} does not match h*w={self.h * self.w}")

        x_spatial = x.transpose(1, 2).reshape(B_, D, self.h, self.w)
        out = self._spectral_branch(x_spatial)
        if self.local_branch is not None:
            out = out + self.local_branch(x_spatial)

        out = out.reshape(B_, D, N).transpose(1, 2)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class DualOperatorBridge(nn.Module):
    """Bridge that blends attention and a spectral mixer with a scheduled alpha.

    Forward: ``out = alpha * attn(x) + (1 - alpha) * spectral(x)``.

    With ``alpha == 1`` only the attention branch is executed, so the bridge is
    numerically identical to a plain :class:`Attention` block and the spectral
    branch parameters are inert. This is the intended initial state when loading
    a pretrained attention-only PredFormer checkpoint: load the attention
    weights into ``self.attn``, then anneal ``alpha`` from 1 toward 0 over
    training to fade attention out and bring the spectral mixer in. With
    ``alpha == 0`` only the spectral branch runs and attention compute is
    skipped, so after annealing the bridge can be exported back to a pure
    spectral schedule.

    Args:
        attn: an Attention module (or any (B', N, D) -> (B', N, D) mixer).
        spectral: a SpectralMixer2D module with matching IO shape.
        alpha: initial blend weight in [0, 1].

    Notes:
        ``alpha`` is stored as a plain Python float, not a buffer, so it does
        not participate in autograd and is not affected by ``model.to(device)``.
        It is intended to be set externally (e.g. from a trainer callback) via
        :meth:`PredFormer_Model.set_dual_bridge_alpha`.
    """

    def __init__(self, attn, spectral, alpha=1.0):
        super().__init__()
        self.attn = attn
        self.spectral = spectral
        self._alpha = 1.0
        self.alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {v}")
        self._alpha = v

    def forward(self, x):
        a = self._alpha
        if a >= 1.0:
            return self.attn(x)
        if a <= 0.0:
            return self.spectral(x)
        return a * self.attn(x) + (1.0 - a) * self.spectral(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        norm_layer=None,
        bias=True,
        drop=0.0,
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
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        attn_dropout=0.0,
        drop_path=0.1,
        op_type="attn",
        mixer_kwargs=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        mixer_kwargs = dict(mixer_kwargs) if mixer_kwargs else {}
        for _ in range(depth):
            if op_type == "attn":
                mixer = Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)
            elif op_type == "spectral":
                mixer = SpectralMixer2D(dim, dropout=attn_dropout, **mixer_kwargs)
            elif op_type == "dual_bridge":
                attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)
                spectral = SpectralMixer2D(dim, dropout=attn_dropout, **mixer_kwargs)
                mixer = DualOperatorBridge(attn, spectral, alpha=1.0)
            else:
                raise ValueError(
                    f"Unknown op_type={op_type!r}, expected one of "
                    f"{{'attn','spectral','dual_bridge'}}"
                )
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, mixer),
                        PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                        DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
                        DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
                    ]
                )
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
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


class PredFormerLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        attn_dropout=0.0,
        drop_path=0.1,
        space_op_type="attn",
        spectral_kwargs=None,
    ):
        super().__init__()

        self.temporal_transformer_first = GatedTransformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            attn_dropout,
            drop_path,
            op_type="attn",
        )
        self.space_transformer = GatedTransformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            attn_dropout,
            drop_path,
            op_type=space_op_type,
            mixer_kwargs=spectral_kwargs,
        )
        self.temporal_transformer_second = GatedTransformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            attn_dropout,
            drop_path,
            op_type="attn",
        )

    def forward(self, x):
        b, t, n, _ = x.shape
        x_t = x

        # t branch (first temporal)
        x_t = rearrange(x_t, "b t n d -> b n t d")
        x_t = rearrange(x_t, "b n t d -> (b n) t d")
        x_t = self.temporal_transformer_first(x_t)

        # s branch (space)
        x_ts = rearrange(x_t, "(b n) t d -> b n t d", b=b)
        x_ts = rearrange(x_ts, "b n t d -> b t n d")
        x_ts = rearrange(x_ts, "b t n d -> (b t) n d")
        x_ts = self.space_transformer(x_ts)

        # t branch (second temporal)
        x_tst = rearrange(x_ts, "(b t) n d -> b t n d", b=b)
        x_tst = rearrange(x_tst, "b t n d -> b n t d")
        x_tst = rearrange(x_tst, "b n t d -> (b n) t d")
        x_tst = self.temporal_transformer_second(x_tst)

        # ts output branch
        x_tst = rearrange(x_tst, "(b n) t d -> b n t d", b=b)
        x_tst = rearrange(x_tst, "b n t d -> b t n d", b=b)

        # add residual connection, we only add this for human3.6m
        # x_tst += x_ori

        return x_tst


def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor(
        [[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(n_channels)]
    )
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, "... -> 1 ...")


def _fourier_features_1d(coords: torch.Tensor, num_freqs: int, base: float = 2.0) -> torch.Tensor:
    """Build NeRF-style sin/cos Fourier features for a 1D coordinate array.

    Args:
        coords: torch.Tensor of shape (M,), values expected in their natural range
            (radians for lat/lon, [0, 1] for time).
        num_freqs: number of frequency bands.
        base: geometric base for frequencies, freq_l = base**l * pi.

    Returns:
        torch.Tensor of shape (M, 2 * num_freqs) = concat([sin(args), cos(args)]).
    """
    freqs = (base ** torch.arange(num_freqs, dtype=coords.dtype, device=coords.device)) * math.pi
    args = coords.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


def build_latlon_time_fourier_pe(
    num_frames: int,
    num_patches_h: int,
    num_patches_w: int,
    dim: int,
    lat_pixel_rad: torch.Tensor,
    lon_pixel_rad: torch.Tensor,
    patch_size: int,
    K_lat: int = 16,
    K_lon: int = 16,
    K_t: int = 8,
    base: float = 2.0,
    seed: int = 0,
) -> torch.Tensor:
    """Build a runtime 2D lat-lon + time Fourier positional encoding.

    Lat/lon for patch centers are aggregated as the mean of the underlying pixel
    coordinates inside the patch. Lat is normalized to [-1, 1] by pi/2, lon to
    [-1, 1] by pi; time index is normalized to [0, 1]. Features for the three
    axes are concatenated and projected to ``dim`` by a frozen orthogonally
    initialised linear map.

    Args:
        num_frames: T, number of input time steps.
        num_patches_h: H / patch_size.
        num_patches_w: W / patch_size.
        dim: model hidden dim.
        lat_pixel_rad: torch.Tensor of shape (H,), latitude in radians per
            pixel row of the (cropped) input grid.
        lon_pixel_rad: torch.Tensor of shape (W,), longitude in radians per
            pixel col of the (cropped) input grid.
        patch_size: spatial patch size in pixels.
        K_lat, K_lon, K_t: number of Fourier frequency bands per axis.
        base: geometric base for frequencies.
        seed: RNG seed for the frozen orthogonal projection.

    Returns:
        torch.Tensor of shape (1, T * N, dim) where N = num_patches_h * num_patches_w,
        suitable to wrap into an nn.Parameter(requires_grad=False) just like
        ``sinusoidal_embedding`` does.
    """
    assert lat_pixel_rad.shape[0] == num_patches_h * patch_size, (
        f"lat_pixel_rad shape {tuple(lat_pixel_rad.shape)} != {num_patches_h * patch_size}"
    )
    assert lon_pixel_rad.shape[0] == num_patches_w * patch_size, (
        f"lon_pixel_rad shape {tuple(lon_pixel_rad.shape)} != {num_patches_w * patch_size}"
    )

    lat_patches = lat_pixel_rad.view(num_patches_h, patch_size).mean(dim=1)
    lon_patches = lon_pixel_rad.view(num_patches_w, patch_size).mean(dim=1)
    t_norm = torch.linspace(0.0, 1.0, num_frames, dtype=lat_patches.dtype)

    lat_norm = lat_patches / (math.pi / 2.0)
    lon_norm = lon_patches / math.pi

    f_lat = _fourier_features_1d(lat_norm, K_lat, base)
    f_lon = _fourier_features_1d(lon_norm, K_lon, base)
    f_t = _fourier_features_1d(t_norm, K_t, base)

    h, w, T = num_patches_h, num_patches_w, num_frames
    F_lat, F_lon, F_t = 2 * K_lat, 2 * K_lon, 2 * K_t

    f_lat_e = f_lat.view(1, h, 1, F_lat).expand(T, h, w, F_lat)
    f_lon_e = f_lon.view(1, 1, w, F_lon).expand(T, h, w, F_lon)
    f_t_e = f_t.view(T, 1, 1, F_t).expand(T, h, w, F_t)
    features = torch.cat([f_lat_e, f_lon_e, f_t_e], dim=-1)
    features = features.reshape(T * h * w, F_lat + F_lon + F_t)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    proj_weight = torch.empty(F_lat + F_lon + F_t, dim, dtype=features.dtype)
    nn.init.orthogonal_(proj_weight, generator=gen)

    pe = features @ proj_weight
    return pe.unsqueeze(0)


class PredFormer_Model(nn.Module):
    """Видеопредиктор PredFormer: ViT-блоки с patch-эмбеддингом + статичная маска ландшафта.

    Конфигурация передаётся одним dict’ом `model_config` (для совместимости
    с YAML-фабрикой). Patch-эмбеддинг разбивает каждый кадр на сетку
    `num_patches_h × num_patches_w` патчей размером `patch_size`. Глобальная
    «маска ландшафта» (z, lat, lon с `path_to_constants`) добавляется к
    патч-эмбеддингам как обучаемый landscape prior.

    Args:
        model_config: dict с ключами `height`, `width`, `num_channels`,
            `pre_seq`, `after_seq`, `patch_size`, `dim`, `heads`, `dim_head`,
            `dropout`, `attn_dropout`, `drop_path`, `scale_dim`, `depth`,
            `Ndepth`, `path_to_constants`.
    """

    def __init__(self, model_config, **kwargs):
        super().__init__()
        self.image_height = model_config["height"]
        self.image_width = model_config["width"]
        self.patch_size = model_config["patch_size"]
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = model_config["pre_seq"]
        self.dim = model_config["dim"]
        self.num_channels = model_config["num_channels"]
        self.num_classes = self.num_channels
        self.heads = model_config["heads"]
        self.dim_head = model_config["dim_head"]
        self.dropout = model_config["dropout"]
        self.attn_dropout = model_config["attn_dropout"]
        self.drop_path = model_config["drop_path"]
        self.scale_dim = model_config["scale_dim"]
        self.Ndepth = model_config["Ndepth"]  # Ensure this is defined
        self.depth = model_config["depth"]  # Ensure this is defined
        self.path_to_constants = model_config["path_to_constants"]
        self.ds = xr.open_dataset(self.path_to_constants)
        self.orography_mask = torch.Tensor(np.array(self.ds.orography))  # shape = [H, W]
        self.soil_mask = torch.Tensor(np.array(self.ds.slt))  # shape = [H, W]
        self.lsm_mask = torch.Tensor(np.array(self.ds.lsm))  # shape = [H, W]
        self.static_masks = torch.stack(
            [self.orography_mask, self.soil_mask, self.lsm_mask], dim=0
        ).unsqueeze(0)  # [1, 3, H, W]
        cut = model_config.get("cut")
        if cut is not None:
            self.static_masks = self.static_masks[..., cut[0][0] : cut[0][1], cut[1][0] : cut[1][1]]
        if self.static_masks.shape[-2:] != (self.image_height, self.image_width):
            raise ValueError(
                f"Static mask shape {tuple(self.static_masks.shape[-2:])} does not match "
                f"configured image shape {(self.image_height, self.image_width)}"
            )
        self.num_masks = 3  # Определяем количество масок

        assert self.image_height % self.patch_size == 0, (
            "Image height must be divisible by the patch size."
        )
        assert self.image_width % self.patch_size == 0, (
            "Image width must be divisible by the patch size."
        )
        self.patch_dim = self.num_channels * self.patch_size**2

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size
            ),
            nn.Linear(self.patch_dim, self.dim),
        )

        self.mask_patch_dim = self.num_masks * self.patch_size**2
        self.rearrange_masks = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size
        )
        self.mask_embedding = nn.Linear(self.mask_patch_dim, self.dim)

        pos_encoding_type = model_config.get("pos_encoding_type", "sinusoidal")
        if pos_encoding_type == "fourier_2d":
            K_lat = model_config.get("pe_K_lat", 16)
            K_lon = model_config.get("pe_K_lon", 16)
            K_t = model_config.get("pe_K_t", 8)
            pe_freq_base = float(model_config.get("pe_freq_base", 2.0))
            pe_seed = int(model_config.get("pe_seed", 0))

            lat_key = "lat" if "lat" in self.ds.coords else "latitude"
            lon_key = "lon" if "lon" in self.ds.coords else "longitude"
            lat_full = torch.tensor(np.array(self.ds[lat_key]), dtype=torch.float32)
            lon_full = torch.tensor(np.array(self.ds[lon_key]), dtype=torch.float32)
            if cut is not None:
                lat_pixel = lat_full[cut[0][0] : cut[0][1]]
                lon_pixel = lon_full[cut[1][0] : cut[1][1]]
            else:
                lat_pixel = lat_full
                lon_pixel = lon_full
            if lat_pixel.shape[0] != self.image_height or lon_pixel.shape[0] != self.image_width:
                raise ValueError(
                    f"Lat/lon coords after cut have shape "
                    f"({lat_pixel.shape[0]}, {lon_pixel.shape[0]}), "
                    f"expected ({self.image_height}, {self.image_width})"
                )
            lat_pixel_rad = lat_pixel * math.pi / 180.0
            lon_pixel_rad = lon_pixel * math.pi / 180.0

            pe_flat = build_latlon_time_fourier_pe(
                num_frames=self.num_frames_in,
                num_patches_h=self.num_patches_h,
                num_patches_w=self.num_patches_w,
                dim=self.dim,
                lat_pixel_rad=lat_pixel_rad,
                lon_pixel_rad=lon_pixel_rad,
                patch_size=self.patch_size,
                K_lat=K_lat,
                K_lon=K_lon,
                K_t=K_t,
                base=pe_freq_base,
                seed=pe_seed,
            )
        elif pos_encoding_type == "sinusoidal":
            pe_flat = sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim)
        else:
            raise ValueError(
                f"Unknown pos_encoding_type={pos_encoding_type!r}, "
                f"expected one of {{'sinusoidal','fourier_2d'}}"
            )

        self.pos_embedding = nn.Parameter(pe_flat, requires_grad=False).view(
            1,
            self.num_frames_in,
            self.num_patches,
            self.dim,
        )

        space_op_schedule = model_config.get("space_op_schedule")
        if space_op_schedule is None:
            space_op_schedule = ["attn"] * self.Ndepth
        if len(space_op_schedule) != self.Ndepth:
            raise ValueError(
                f"space_op_schedule has length {len(space_op_schedule)} but Ndepth={self.Ndepth}"
            )

        spectral_kwargs = dict(model_config.get("spectral_kwargs", {}))
        spectral_kwargs.setdefault("num_patches_h", self.num_patches_h)
        spectral_kwargs.setdefault("num_patches_w", self.num_patches_w)

        self.blocks = nn.ModuleList(
            [
                PredFormerLayer(
                    self.dim,
                    self.depth,
                    self.heads,
                    self.dim_head,
                    self.dim * self.scale_dim,
                    self.dropout,
                    self.attn_dropout,
                    self.drop_path,
                    space_op_type=space_op_schedule[i],
                    spectral_kwargs=spectral_kwargs,
                )
                for i in range(self.Ndepth)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, self.num_channels * self.patch_size**2)
        )

    def set_dual_bridge_alpha(self, value):
        """Set the blend alpha on every :class:`DualOperatorBridge` in the model.

        Args:
            value: float in [0, 1]. 1.0 = pure attention (matches pretrained
                attention-only checkpoint), 0.0 = pure spectral mixer.

        Returns:
            int: number of bridges updated.
        """
        count = 0
        for m in self.modules():
            if isinstance(m, DualOperatorBridge):
                m.alpha = value
                count += 1
        return count

    def forward(self, x):
        """Прогон клипа `(B, T, C, H, W)` через стек PredFormer-блоков.

        Args:
            x: входной клип формы `(B, T, C, H, W)`, `C == self.num_channels`.

        Returns:
            Прогноз формы `(B, T, C, H, W)`.
        """
        B, T, C, H, W = x.shape
        assert self.num_channels == C

        mask_patches = self.rearrange_masks(
            self.static_masks.to(x.device)
        )  # [1, num_patches, 3*ps*ps]
        mask_embed = self.mask_embedding(mask_patches)  # [1, num_patches, dim]
        mask_embed = mask_embed.unsqueeze(1)  # [1, 1, num_patches, dim]
        mask_embed = mask_embed.to(x.device)

        # Patch Embedding для входа x
        x_embed = self.to_patch_embedding(x)  # [B, T, num_patches, dim]

        x_combined = x_embed + mask_embed

        # Position Embedding
        x_combined += self.pos_embedding.to(x.device)

        # PredFormer Encoder
        for idx, blk in enumerate(self.blocks):
            x_combined = blk(x_combined)
        # MLP head
        x = self.mlp_head(x_combined.reshape(-1, self.dim))
        x = x.view(
            B, T, self.num_patches_h, self.num_patches_w, C, self.patch_size, self.patch_size
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)

        return x


# print(output.shape)  # [B, T, C, H, W]
