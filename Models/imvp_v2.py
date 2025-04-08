import torch
import math
import random
from torch import nn

from .imvp_modules import CircularConvSC, ConvNeXt_block, Learnable_Filter, Attention, ConvNeXt_bottle
from .FedorPredFormerGFT import HybridBlock

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
    
    
class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, T):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[CircularConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            CircularConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(64 * T, 64, 1)

    def forward(self, hid, enc1, latent_1, latent_2, latent_3, T = 10, H = 32, W = 64):
        
        hid = self.dec[0](hid + latent_3)
        hid = self.dec[1](hid + latent_2)
        hid = self.dec[2](hid + latent_1)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        ys = Y.shape
        Y = Y.reshape(int(ys[0]/T), int(ys[1]*T), H, W)
        Y = self.readout(Y)
        return Y


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
    def __init__(self, T_data=12, C_data=69, H_data=128, W_data=256, hid_S=64, hid_T=512, N_S=4, N_T=6):
        super(IAM4VP, self).__init__()
        self.time_mlp = Time_MLP(dim=hid_S)
        self.enc = Encoder(C_data, hid_S, N_S)
        self.hid = Predictor(T_data * hid_S, hid_T, N_T)
        self.dec = Decoder(hid_S, C_data, N_S, T_data)
        self.attn = Attention(hid_S)
        self.readout = nn.Conv2d(hid_S, C_data, 1)
        self.mask_token = nn.Parameter(torch.zeros(T_data, hid_S, H_data // 4, W_data // 4)) #for 1_4 and 5_6
        self.lp = LP(C_data, hid_S, N_S)
        self.lp_phys = LP(C_data, hid_S, N_S)
        self.hybrid_block = HybridBlock(dim=C_data-4, zquvtw_channel=13, depth=3, block_dt=1200, inverse_time=False, physics_part_coef=0.5)
        
        self.skip_mask_token = nn.Parameter(torch.zeros(T_data, hid_S,  H_data, W_data))
        self.embed_1_mask_token = nn.Parameter(torch.zeros(T_data, hid_S,  H_data // 2, W_data // 2))
        self.embed_2_mask_token = nn.Parameter(torch.zeros(T_data, hid_S,  H_data // 2, W_data // 2))
        self.downscaling_factor_all = 4

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


    def forward(self, x_raw, y_raw=None, t=None):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)
        time_emb = self.time_mlp(t)
        
        embed, skip, embed_1, embed_2 = self.enc(x)
        mask_token = self.mask_token.repeat(B, 1, 1, 1, 1)

        skip_mask_token = self.skip_mask_token.repeat(B, 1, 1, 1, 1)
        embed_1_mask_token = self.embed_1_mask_token.repeat(B, 1, 1, 1, 1)
        embed_2_mask_token = self.embed_2_mask_token.repeat(B, 1, 1, 1, 1)

        for idx, pred in enumerate(y_raw):

            embed2, skip_lp, embed_1_lp, embed_2_lp = self.lp(pred)

            if idx == 0:
                prev_pred = x_raw[:, -1]

            pred_phys = prev_pred[:, 4:, :, :]
            zquvtw = self.x_to_zquvtw(pred_phys)
            pred_phys = zquvtw

            for j in range(3):
                # Получаем физические эмбеддинги через hybrid_block
                pred_phys, zquvtw = self.hybrid_block(pred_phys, zquvtw)  # Используем одинаковые данные для обоих входов
        
            # Возвращаем к исходному формату
            pred_phys = pred_phys.permute(0, 3, 1, 2)  # [B, C, H//4, W//4]
            
            # Масштабируем обратно до исходного размера
            pred_phys = torch.nn.functional.interpolate(pred_phys, size=(H, W), mode='bilinear')
            
            pred_to_hybrid = torch.cat([pred[:, :4, :, :], pred_phys], dim=1)

            embed2_phys, skip_lp_phys, embed_1_lp_phys, embed_2_lp_phys = self.lp_phys(pred_to_hybrid)


            mask_token[:, idx, :, :, :] = embed2 + 0.1 * embed2_phys

            skip_mask_token[:, idx, :, :, :] = skip_lp + 0.1 * skip_lp_phys
            embed_1_mask_token[:, idx, :, :, :] = embed_1_lp + 0.1 * embed_1_lp_phys
            embed_2_mask_token[:, idx, :, :, :] = embed_2_lp + 0.1 * embed_2_lp_phys

            prev_pred = pred

        _, C_, H_, W_ = embed.shape
        
        skip = skip + skip_mask_token.view(B * T, C_, H_ * 4, W_ * 4)
        embed_1 = embed_1 + embed_1_mask_token.view(B * T, C_, H_ * 2, W_ * 2)
        embed_2 = embed_2 + embed_2_mask_token.view(B * T, C_, H_ * 2, W_ * 2)

        z = embed.view(B, T, C_, H_, W_)
        z2 = mask_token
        z = torch.cat([z, z2], dim=1)
        hid = self.hid(z, time_emb)
        hid = hid.reshape(B * T, C_, H_, W_)
        
        Y = self.dec(hid, skip, embed_1, embed_2, embed, T = T, H = H, W = W)
        
        Y = self.attn(Y)
        Y = self.readout(Y)
        return Y