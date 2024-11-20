import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import math
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from WeatherGFT import (
    lat, 
    latents_size, 
    radius, 
    num_lat, 
    lat_t, 
    latitudes, 
    c_lats, 
    pixel_x, pixel_y, pixel_z,
    pressure, pressure_level_num, 
    M_z, 
    integral_z, d_x, d_y, d_z, 
    PDE_kernel, PDE_block, 
    CyclicShift, Residual, PreNorm, MLP, 
    create_mask, get_relative_distances, 
    PatchMerging, PatchRemaining, PatchExpanding, 
    RandomOrLearnedSinusoidalPosEmb
)


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    

class ConvNeXtV2Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + x
        return x
    

class WindowConvModule(nn.Module):
    """ ConvNeXtV2 Block to adjust for errors in physical variable 
    calculation in PDE kernels. 
    
    Args:
        dim (int): Number of input channels.
        shifted (bool): whether this block is shifted (Shifted Windows). 
        windows_size (int): size of windows (Shifted Windows).
    """
    def __init__(self, dim, shifted, window_size):
        super().__init__()
        self.window_size = window_size
        self.shifted = shifted
        if self.shifted:
            displacement = [window_size[0] // 2, window_size[1] // 2]
            displacement_ = [-window_size[0] // 2, -window_size[1] // 2]
            self.cyclic_shift = CyclicShift(displacement_)
            self.cyclic_back_shift = CyclicShift(displacement)
        self.conv_block = ConvNeXtV2Block(dim=dim)
    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)
        x = x.permute(0, 3, 1, 2).contiguous() # Switch to [BS, Channels, H, W]
        x = self.conv_block(x)
        
        out = x.permute(0, 2, 3, 1).contiguous() #[1, 180, 360, 96], switching back to [BS, H, W, Channels]
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out

class HybridBlock(nn.Module):
    def __init__(self, dim, mlp_dim, shifted, window_size, use_pde, zquvtw_channel, depth, block_dt, inverse_time):
        super().__init__()
        self.conv_block = Residual(PreNorm(dim, WindowConvModule(dim=dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size)))
        
        self.use_pde = use_pde
        if use_pde:
            self.pde_block = PDE_block(dim, zquvtw_channel, depth, block_dt, inverse_time)
            self.router_weight = nn.Parameter(torch.zeros(1, 1, 1, dim), requires_grad=True)

        self.router_MLP = Residual(PreNorm(dim, MLP(dim, hidden_dim=mlp_dim)))

    def forward(self, x, zquvtw=None):
        if self.use_pde:
            # AI & Physics
            feat_att = self.conv_block(x)
            feat_pde, zquvtw = self.pde_block(x, zquvtw)
            
            # Adaptive Router
            GFT.layer_weights[f"{self.__class__.__name__}_{GFT.gft_name}"] = self.router_weight.detach().cpu().numpy()
            weight_AI = 0.5*torch.ones_like(x)+self.router_weight
            weight_Physics = 0.5*torch.ones_like(x)-self.router_weight
            x = weight_AI*feat_att + weight_Physics*feat_pde
            # x = weight_AI*feat_att
            x = self.router_MLP(x)
            return x, zquvtw
        else:
            x = self.conv_block(x)
            x = self.router_MLP(x)
            return x

class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, scaling_factors, window_size,
                 use_pde=False, zquvtw_channel=None, depth=3, block_dt=300, inverse_time=False):
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
                HybridBlock(dim=hidden_dimension, mlp_dim=hidden_dimension * 4,
                            shifted=False, window_size=window_size,
                            use_pde=use_pde, zquvtw_channel=zquvtw_channel, depth=depth, block_dt=block_dt, inverse_time=inverse_time),
                HybridBlock(dim=hidden_dimension, mlp_dim=hidden_dimension * 4,
                            shifted=True, window_size=window_size,
                            use_pde=use_pde, zquvtw_channel=zquvtw_channel, depth=depth, block_dt=block_dt, inverse_time=inverse_time),
            ]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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


class GFT(nn.Module):
    
    layer_weights = {}  # Глобальный словарь для хранения значений router_weight
    
    gft_name = ""
    
    def __init__(self, 
                hidden_dim=256,
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
                window_size=[4,8],
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
                                scaling_factors=encoder_scaling_factors[i_layer],
                                window_size=window_size)
            self.encoder.append(layer)
        
        self.body = nn.ModuleList()
        for i_layer in range(len(body_layers)):
            if use_checkpoint:
                layer = checkpoint_wrapper(StageModule(in_channels=body_dim_list[i_layer], hidden_dimension=body_dim_list[i_layer+1], layers=body_layers[i_layer],
                                            scaling_factors=body_scaling_factors[i_layer],
                                            window_size=window_size,
                                            use_pde=True, zquvtw_channel=13, depth=pde_block_depth, block_dt=block_dt, inverse_time=inverse_time))
            else:
                layer = StageModule(in_channels=body_dim_list[i_layer], hidden_dimension=body_dim_list[i_layer+1], layers=body_layers[i_layer],
                                    scaling_factors=body_scaling_factors[i_layer],
                                    window_size=window_size,
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
                                  scaling_factors=decoder_scaling_factors[i_layer],
                                  window_size=window_size)
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
    
    
    def forward(self, x):
        x = x.squeeze(1)
        
        GFT.layer_weights.clear()
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

                body_layers=[4, 4, 4, 4, 4, 4], # A total of 4x6=24 HybridBlock, corresponding to 6 hours (24x15min) of time evolution
                body_heads=[8, 8, 8, 8, 8, 8],
                body_scaling_factors=[1, 1, 1, 1, 1, 1],
                body_dim_factors=[1, 1, 1, 1, 1, 1],

                decoder_layers=[2, 2, 2],
                decoder_heads=[6, 6, 3],
                decoder_scaling_factors=[1, 2, 1],
                decoder_dim_factors=[1, 0.5, 1],

                channels=69,
                window_size=[4,8],
                out_kernel=[2,2],
                
                pde_block_depth=3, # 1 HybridBlock contains 3 PDE kernels, corresponding to 15 minutes (3x300s) of time evolution
                block_dt=300, # One PDE kernel corresponds to 300s of time evolution
                inverse_time=False).to(device)

    
    if os.path.exists('../checkpoints/gft.ckpt'):
        ckpt = torch.load('../checkpoints/gft.ckpt', map_location=torch.device('cpu'))
        model.load_state_dict(ckpt, strict=True)
        print('[complete loading model]')
    else:
        print('[checkpoint does not exist]')

    if os.path.exists('../example_data/input.npy') and os.path.exists('../example_data/target.npy'):
        inp = torch.tensor(np.load('../example_data/input.npy')).float().to(device)
        target = torch.tensor(np.load('../example_data/target.npy')).float().to(device)
    else:
        inp = torch.randn(1, 69, 128, 256).to(device)
        target = None

    pred = model(inp)
    print(pred.shape)
    # torch.Size([1, 3, 69, 128, 256]), the prediction results of lead time=[1,3,6]h respectively

    model.out_layer = [5] # decode only the last layer
    pred = model(inp)
    # torch.Size([1, 69, 128, 256]), the prediction results of lead time=[1,3,6]h respectively
    print(pred.shape)

    if target is not None:
        print('prediction MSE:', ((target-pred)**2).mean().item())

        with open('../example_data/mean_std.json', 'r') as json_file:
            mean_std = json.load(json_file)
        mean = torch.tensor(mean_std['mean']).reshape(1, 69, 1, 1).to(inp.device)
        std = torch.tensor(mean_std['std']).reshape(1, 69, 1, 1).to(inp.device)

        pred = pred*std+mean # Denormalization
        target = target*std+mean # Denormalization

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        a0 = axs[0].imshow(pred[0, 0].detach().cpu().flip(0).numpy())
        axs[0].set_title('prediction t2m')
        axs[0].axis('off')
        fig.colorbar(a0, ax=axs[0], orientation='horizontal', shrink=0.8, aspect=16, extend='both')

        a1 = axs[1].imshow(target[0, 0].detach().cpu().flip(0).numpy())
        axs[1].set_title('ground truth t2m')
        axs[1].axis('off')
        fig.colorbar(a1, ax=axs[1], orientation='horizontal', shrink=0.8, aspect=16, extend='both')

        error = pred[0, 0]-target[0, 0]
        a2 = axs[2].imshow(error.detach().cpu().flip(0).numpy(), cmap='RdBu_r', norm = colors.Normalize(-10, 10))
        axs[2].set_title('prediction error t2m')
        axs[2].axis('off')
        fig.colorbar(a2, ax=axs[2], orientation='horizontal', shrink=0.8, aspect=16, extend='both')
        
        plt.tight_layout()
        plt.savefig('visualization.png', dpi=300)
        plt.close()