import torch
import math
import random
from torch import nn
import torch.nn.functional as F

from .imvp_modules import CircularConvSC, ConvNeXt_block, Learnable_Filter, Attention, ConvNeXt_bottle

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Time_MLP(nn.Module):
    def __init__(self, dim):
        super(Time_MLP, self).__init__()
        self.sinusoidaposemb = SinusoidalPosEmb(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.sinusoidaposemb(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            CircularConvSC(C_in, C_hid, stride=strides[0]),
            *[CircularConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*10, 2, 32, 64
        enc1 = self.enc[0](x)
        latent = enc1

        latent_1 = self.enc[1](latent)
        latent_2 = self.enc[2](latent_1)
        latent_3 = self.enc[3](latent_2)

        return latent_3, enc1, latent_1, latent_2


class LP(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(LP, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            CircularConvSC(C_in, C_hid, stride=strides[0]),
            *[CircularConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*10, 2, 32, 64
        enc1 = self.enc[0](x)
        latent = enc1

        latent_1 = self.enc[1](latent)
        latent_2 = self.enc[2](latent_1)
        latent_3 = self.enc[3](latent_2)

        return latent_3, enc1, latent_1, latent_2
    
    
class Predictor(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T):
        super(Predictor, self).__init__()

        self.N_T = N_T
        st_block = [ConvNeXt_bottle(dim=channel_in)]
        for i in range(0, N_T):
            st_block.append(ConvNeXt_block(dim=channel_in))

        self.st_block = nn.Sequential(*st_block)

    def forward(self, x, time_emb):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        z = self.st_block[0](x, time_emb)
        for i in range(1, self.N_T):
            z = self.st_block[i](z, time_emb)

        y = z.reshape(B, int(T / 2), C, H, W)
        return y


class IAM4VP(nn.Module):
    def __init__(self, T_data=6, C_data=69, H_data=128, W_data=256, hid_S=64, hid_T=256, N_S=4, N_T=6):
        super(IAM4VP, self).__init__()
        
        self.time_mlp = Time_MLP(dim=hid_S)
        self.enc = Encoder(C_data, hid_S, N_S)
        self.hid = Predictor(T_data * hid_S, hid_T, N_T)
        
        # Use our new ADRDecoder instead of the regular Decoder
        self.dec = ADRDecoder(hid_S, C_data, N_S, T_data, imsz=[H_data, W_data], adv_layers=4)
        
        # No need for projection layer anymore as ADRDecoder directly outputs C_data channels
        self.attn = Attention(hid_S)
        self.readout = nn.Conv2d(hid_S, C_data, 1)
        
        self.mask_token = nn.Parameter(torch.zeros(T_data, hid_S, H_data // 4, W_data // 4))
        self.lp = LP(C_data, hid_S, N_S)
        
        self.skip_mask_token = nn.Parameter(torch.zeros(T_data, hid_S, H_data, W_data))
        self.embed_1_mask_token = nn.Parameter(torch.zeros(T_data, hid_S, H_data // 2, W_data // 2))
        self.embed_2_mask_token = nn.Parameter(torch.zeros(T_data, hid_S, H_data // 2, W_data // 2))

    def forward(self, x_raw, y_raw=None, t=None):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)
        time_emb = self.time_mlp(t)
        
        embed, skip, embed_1, embed_2 = self.enc(x)
        
        mask_token = self.mask_token.repeat(B, 1, 1, 1, 1)

        skip_mask_token = self.skip_mask_token.repeat(B, 1, 1, 1, 1)
        embed_1_mask_token = self.embed_1_mask_token.repeat(B, 1, 1, 1, 1)
        embed_2_mask_token = self.embed_2_mask_token.repeat(B, 1, 1, 1, 1)

        if y_raw is not None:
            for idx, pred in enumerate(y_raw):
                embed2, skip_lp, embed_1_lp, embed_2_lp = self.lp(pred)

                mask_token[:, idx, :, :, :] = embed2

                skip_mask_token[:, idx, :, :, :] = skip_lp
                embed_1_mask_token[:, idx, :, :, :] = embed_1_lp
                embed_2_mask_token[:, idx, :, :, :] = embed_2_lp

        _, C_, H_, W_ = embed.shape
        
        skip = skip + skip_mask_token.view(B * T, C_, H_ * 4, W_ * 4)
        embed_1 = embed_1 + embed_1_mask_token.view(B * T, C_, H_ * 2, W_ * 2)
        embed_2 = embed_2 + embed_2_mask_token.view(B * T, C_, H_ * 2, W_ * 2)

        z = embed.view(B, T, C_, H_, W_)
        z2 = mask_token
        z = torch.cat([z, z2], dim=1)
        
        hid = self.hid(z, time_emb)
        hid = hid.reshape(B * T, C_, H_, W_)
        
        # Pass the time embedding to the ADRDecoder
        Y = self.dec(hid, skip, embed_1, embed_2, embed, time_emb, T=T, H=H, W=W)
        
        return Y

class ADRDecoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, T, imsz=[128, 256], adv_layers=4, device='cuda'):
        super(ADRDecoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        
        # Create upsampling blocks with circular convolution
        self.upsample_blocks = nn.ModuleList([
            CircularConvSC(C_hid, C_hid, stride=strides[0], transpose=True),
            CircularConvSC(C_hid, C_hid, stride=strides[1], transpose=True),
            CircularConvSC(C_hid, C_hid, stride=strides[2], transpose=True)
        ])
        
        # Final upsampling with concat from encoder
        self.final_upsample = CircularConvSC(2 * C_hid, C_hid, stride=strides[3], transpose=True)
        
        # ADRnet integration
        self.adv_layers = adv_layers
        self.h = 1.0 / imsz[0]  # Step size for residual connection
        
        # Create advection blocks and diffusion-reaction blocks
        self.adv_blocks = nn.ModuleList()
        self.dr_blocks = nn.ModuleList()
        
        for i in range(adv_layers):
            adv_block = advectionColorBlock(C_hid, imsz, device)
            dr_block = CLP(C_hid, C_hid, imsz, kernel_size=[5, 5])
            
            self.adv_blocks.append(adv_block)
            self.dr_blocks.append(dr_block)
            
        # Final projection to output channels
        self.readout = nn.Conv2d(C_hid * T, C_out, 1)
        
    def forward(self, hid, enc1, latent_1, latent_2, latent_3, t, T=10, H=128, W=256):
        # Progressive upsampling with residual connections
        x = self.upsample_blocks[0](hid + latent_3)
        x = self.upsample_blocks[1](x + latent_2)
        x = self.upsample_blocks[2](x + latent_1)
        
        # Concatenate with encoder features for skip connection
        x = self.final_upsample(torch.cat([x, enc1], dim=1))
        
        # Extract batch size for reshaping time embedding
        batch_size = int(x.shape[0] / T)
        
        # Reshape time embedding to match batch size of x
        if len(t.shape) == 2:  # [batch, embedding_dim]
            # Duplicate time embedding for each time step in sequence
            t_expanded = t.unsqueeze(1).expand(-1, T, -1).reshape(batch_size * T, -1)
        else:
            t_expanded = t
        
        # Process through advection-diffusion-reaction layers
        z = x.clone()
        for i in range(self.adv_layers):
            # Advection Layer with expanded time embedding
            dz = self.adv_blocks[i](z, t_expanded)
            
            # Diffusion and Reaction Layer
            dz = self.dr_blocks[i](dz)
            
            # Residual Connection with step size scaling
            z = z + self.h * dz
        
        # Reshape for readout
        ys = z.shape
        z = z.reshape(int(ys[0]/T), int(ys[1]*T), H, W)
        
        # Final projection to output channels
        Y = self.readout(z)
        
        return Y

# Add CLP and advectionColorBlock from adr_net
def CLP(dim_in, dim_out, shape=[256, 256], kernel_size=[3, 3]):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=kernel_size[0]//2),
        nn.LayerNorm(shape),
        nn.SiLU(),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, padding=kernel_size[0]//2)
    )

class advectionColorBlock(nn.Module):
    def __init__(self, c, mesh_size, device='cuda'):
        super(advectionColorBlock, self).__init__()
        self.Adv = color_preserving_advection(mesh_size, device)
        
        self.ConvU1 = nn.Conv2d(c, c, kernel_size=[3, 3], padding=1)
        self.LNU1 = nn.LayerNorm(normalized_shape=mesh_size)
        self.ConvU2 = nn.Conv2d(c, c, kernel_size=[3, 3], padding=1, bias=False)
        self.ConvU2.weight = nn.Parameter(1e-4*torch.randn(c, c, 3, 3))

        self.TimeEmbedU = nn.Parameter(1e-4*torch.randn(1, c, mesh_size[0], mesh_size[1]))
        self.TNU = CLP(c, c, mesh_size)
        self.ConvV1 = nn.Conv2d(c, c, kernel_size=[3, 3], padding=1)
        self.LNV1 = nn.LayerNorm(normalized_shape=mesh_size)
        self.ConvV2 = nn.Conv2d(c, c, kernel_size=[3, 3], padding=1, bias=False)
        self.ConvV2.weight = nn.Parameter(1e-4*torch.randn(c, c, 3, 3))

        self.TimeEmbedV = nn.Parameter(1e-4*torch.randn(1, c, mesh_size[0], mesh_size[1]))
        self.TNV = CLP(c, c, mesh_size)
        
    def forward(self, x, t):
        nw, nh = x.shape[2], x.shape[3]
        batch_size = x.shape[0]
        
        # Handle batch dimension properly
        # Expand time embedding to match batch size
        if len(t.shape) == 2:  # [batch, embedding_dim]
            t = t.mean(dim=1)  # Take mean across embedding dimension to get scalar per batch
            
        # Reshape t to [batch, 1, 1, 1] for proper broadcasting
        t_reshaped = t.view(-1, 1, 1, 1)
        
        # Broadcast TimeEmbedU and TimeEmbedV to match batch size
        expanded_TimeEmbedU = self.TimeEmbedU.expand(batch_size, -1, -1, -1)
        expanded_TimeEmbedV = self.TimeEmbedV.expand(batch_size, -1, -1, -1)
        
        # Apply time modulation
        teU = t_reshaped * expanded_TimeEmbedU
        teU = self.TNU(teU) 
        teV = t_reshaped * expanded_TimeEmbedV
        teV = self.TNV(teV) 
        
        # Calculate flow fields
        U = self.ConvU1(x) + teU
        U = self.LNU1(U)
        U = F.silu(U)
        U = self.ConvU2(U)
        
        V = self.ConvV1(x) + teV
        V = self.LNV1(V)
        V = F.silu(V)
        V = self.ConvV2(V)
        
        # Normalize flow field
        U, V = U/nw, V/nw 

        # Reshape for advection
        xr = x.reshape(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        Ur = U.reshape(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        Vr = V.reshape(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        
        # Apply color conserving advection
        xr = self.Adv(xr, Ur, Vr)
        
        # Reshape back
        x = xr.reshape(x.shape)

        return x

class color_preserving_advection(nn.Module):
    def __init__(self, shape, device='cuda'):
        super(color_preserving_advection, self).__init__()
        
        grid_h, grid_w = shape[0], shape[1]
        y, x = torch.meshgrid(torch.linspace(-1, 1, grid_h), torch.linspace(-1, 1, grid_w))
        self.grid = torch.stack((x, y), dim=-1).unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, T, U, V):
        # Перемещаем self.grid на то же устройство, что и входные тензоры
        grid = self.grid.to(T.device)
        UV = torch.stack((U, V), dim=-1)
        transformation_grid = grid + UV
        Th = F.grid_sample(T, transformation_grid.squeeze(1), align_corners=True)

        return Th