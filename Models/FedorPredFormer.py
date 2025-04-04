import torch
import numpy as np
import xarray as xr
import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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
                
     
    def forward(self, x):
        B, T, C, H, W = x.shape
        assert C == self.num_channels

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
